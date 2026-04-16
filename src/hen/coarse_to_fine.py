from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ShuffleNet_V2_X0_5_Weights,
    ShuffleNet_V2_X1_0_Weights,
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
)

from .hierarchy import HierarchySpec
from .models import (
    JointHierarchicalOutput,
    ResidualAdapter,
    _copy_linear_rows_by_name,
    _copy_module_if_compatible,
    _set_trainable,
    build_resnet_backbone,
)


def _zero_init_linear(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class HighResMidBranch(nn.Module):
    def __init__(
        self,
        image_size: int,
        num_classes: int,
        base_width: int = 24,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_width * 4, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        high_res = F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        features = self.proj(self.encoder(high_res))
        return self.head(features)


class MidFeatureResidualBranch(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        adapter_dim: int = 256,
    ) -> None:
        super().__init__()
        self.adapter = ResidualAdapter(feature_dim, adapter_dim)
        self.head = nn.Linear(feature_dim, num_classes)
        self.residual_head = nn.Linear(feature_dim, num_classes)
        _zero_init_linear(self.residual_head)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        adapted = self.adapter(features)
        return self.head(features) + self.residual_head(adapted)


class AttentionCropMidBranch(nn.Module):
    def __init__(
        self,
        image_size: int,
        num_classes: int,
        base_width: int = 24,
        hidden_dim: int = 256,
        threshold_scale: float = 0.6,
        min_crop_ratio: float = 0.45,
        margin_ratio: float = 0.15,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.threshold_scale = threshold_scale
        self.min_crop_ratio = min_crop_ratio
        self.margin_ratio = margin_ratio
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_width * 4, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def _build_crops(self, images: torch.Tensor, heatmaps: torch.Tensor) -> torch.Tensor:
        batch_size, _, image_h, image_w = images.shape
        crops = []
        for index in range(batch_size):
            heatmap = heatmaps[index]
            max_value = float(heatmap.max().item())
            if max_value <= 0.0:
                crop = images[index : index + 1]
            else:
                threshold = max_value * self.threshold_scale
                mask = heatmap >= threshold
                if not mask.any():
                    mask = heatmap >= heatmap.mean()
                coordinates = mask.nonzero(as_tuple=False)
                if coordinates.numel() == 0:
                    crop = images[index : index + 1]
                else:
                    y_min = int(coordinates[:, 0].min().item())
                    y_max = int(coordinates[:, 0].max().item()) + 1
                    x_min = int(coordinates[:, 1].min().item())
                    x_max = int(coordinates[:, 1].max().item()) + 1

                    scale_y = image_h / heatmap.size(0)
                    scale_x = image_w / heatmap.size(1)
                    y0 = int(y_min * scale_y)
                    y1 = int((y_max + 1) * scale_y)
                    x0 = int(x_min * scale_x)
                    x1 = int((x_max + 1) * scale_x)

                    box_h = max(y1 - y0, int(image_h * self.min_crop_ratio))
                    box_w = max(x1 - x0, int(image_w * self.min_crop_ratio))
                    center_y = (y0 + y1) // 2
                    center_x = (x0 + x1) // 2

                    margin_y = int(box_h * self.margin_ratio)
                    margin_x = int(box_w * self.margin_ratio)
                    box_h = min(image_h, box_h + 2 * margin_y)
                    box_w = min(image_w, box_w + 2 * margin_x)

                    y0 = max(0, center_y - box_h // 2)
                    x0 = max(0, center_x - box_w // 2)
                    y1 = min(image_h, y0 + box_h)
                    x1 = min(image_w, x0 + box_w)
                    y0 = max(0, y1 - box_h)
                    x0 = max(0, x1 - box_w)
                    crop = images[index : index + 1, :, y0:y1, x0:x1]

            crop = F.interpolate(crop, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            crops.append(crop)

        return torch.cat(crops, dim=0)

    def forward(self, images: torch.Tensor, feature_maps: torch.Tensor) -> torch.Tensor:
        heatmaps = feature_maps.detach().abs().mean(dim=1)
        crops = self._build_crops(images, heatmaps)
        features = self.proj(self.encoder(crops))
        return self.head(features)


class TinyHierarchicalRouter(nn.Module):
    def __init__(
        self,
        hierarchy: HierarchySpec,
        backbone: str = "tiny",
        pretrained: bool = True,
        image_size: int = 64,
        base_width: int = 32,
        hidden_dim: int = 256,
        mid_highres_level1: str | None = None,
        mid_highres_image_size: int | None = None,
        mid_highres_base_width: int = 24,
        mid_highres_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.image_size = image_size
        if backbone == "tiny":
            self.encoder = nn.Sequential(
                nn.Conv2d(3, base_width, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_width, base_width * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_width * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_width * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_width * 4),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(base_width * 4, hidden_dim),
                nn.ReLU(inplace=True),
            )
            feature_dim = hidden_dim
        elif backbone in {"shufflenet_v2_x0_5", "shufflenet_v2_x1_0"}:
            if backbone == "shufflenet_v2_x0_5":
                weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None
                router_model = shufflenet_v2_x0_5(weights=weights)
            else:
                weights = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
                router_model = shufflenet_v2_x1_0(weights=weights)
            self.encoder = nn.Sequential(
                router_model.conv1,
                router_model.maxpool,
                router_model.stage2,
                router_model.stage3,
                router_model.stage4,
                router_model.conv5,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.proj = nn.Identity()
            feature_dim = router_model.fc.in_features
        else:
            raise ValueError(f"Unsupported router backbone: {backbone}")

        self.level1_head = nn.Linear(feature_dim, hierarchy.num_level1)
        self.level2_adapters = nn.ModuleDict(
            {
                str(level1_id): ResidualAdapter(feature_dim, min(feature_dim, hidden_dim))
                for level1_id in hierarchy.level1_to_level2
            }
        )
        self.level2_heads = nn.ModuleDict(
            {
                str(level1_id): nn.Linear(feature_dim, len(children))
                for level1_id, children in hierarchy.level1_to_level2.items()
            }
        )
        self.level2_residual_heads = nn.ModuleDict(
            {
                str(level1_id): nn.Linear(feature_dim, len(children))
                for level1_id, children in hierarchy.level1_to_level2.items()
            }
        )
        for head in self.level2_residual_heads.values():
            _zero_init_linear(head)
        self.level1_to_level2 = {key: list(value) for key, value in hierarchy.level1_to_level2.items()}
        self.level2_highres_branches = nn.ModuleDict()
        if mid_highres_level1 is not None:
            if mid_highres_level1 not in hierarchy.level1_name_to_id:
                raise ValueError(f"Unknown mid_highres_level1: {mid_highres_level1}")
            level1_id = hierarchy.level1_name_to_id[mid_highres_level1]
            self.level2_highres_branches[str(level1_id)] = HighResMidBranch(
                image_size=mid_highres_image_size or image_size,
                num_classes=len(hierarchy.level1_to_level2[level1_id]),
                base_width=mid_highres_base_width,
                hidden_dim=mid_highres_hidden_dim,
            )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low_res = F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        features = self.proj(self.encoder(low_res))
        level1_logits = self.level1_head(features)
        level1_log_probs = F.log_softmax(level1_logits, dim=1)

        level2_count = sum(len(children) for children in self.level1_to_level2.values())
        level2_log_probs = torch.full(
            (features.size(0), level2_count),
            float("-inf"),
            device=features.device,
            dtype=level1_log_probs.dtype,
        )
        for level1_id, level2_ids in self.level1_to_level2.items():
            adapted = self.level2_adapters[str(level1_id)](features)
            base_logits = self.level2_heads[str(level1_id)](features)
            residual_logits = self.level2_residual_heads[str(level1_id)](adapted)
            conditional_logits = base_logits + residual_logits
            if str(level1_id) in self.level2_highres_branches:
                conditional_logits = conditional_logits + self.level2_highres_branches[str(level1_id)](images)
            conditional_log_probs = F.log_softmax(conditional_logits, dim=1)
            level2_log_probs[:, level2_ids] = level1_log_probs[:, level1_id].unsqueeze(1) + conditional_log_probs

        return level1_logits, level1_log_probs, level2_log_probs


class Level1ExpertTower(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        level2_to_leaf: dict[int, tuple[int, ...]],
        feature_dim: int,
        leaf_adapter_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layer3 = copy.deepcopy(base_model.layer3)
        self.layer4 = copy.deepcopy(base_model.layer4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.level2_to_leaf = {key: list(value) for key, value in level2_to_leaf.items()}
        self.leaf_adapters = nn.ModuleDict(
            {
                str(level2_id): ResidualAdapter(feature_dim, leaf_adapter_dim)
                for level2_id in self.level2_to_leaf
            }
        )
        self.leaf_heads = nn.ModuleDict(
            {
                str(level2_id): nn.Linear(feature_dim, len(leaf_ids))
                for level2_id, leaf_ids in self.level2_to_leaf.items()
            }
        )
        self.leaf_residual_heads = nn.ModuleDict(
            {
                str(level2_id): nn.Linear(feature_dim, len(leaf_ids))
                for level2_id, leaf_ids in self.level2_to_leaf.items()
            }
        )
        for head in self.leaf_residual_heads.values():
            _zero_init_linear(head)

    def extract_features(self, shared_features: torch.Tensor) -> torch.Tensor:
        features = self.layer3(shared_features)
        features = self.layer4(features)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        return self.dropout(features)


@dataclass
class CoarseToFineMetadata:
    backbone: str
    router_backbone: str
    router_image_size: int
    router_base_width: int
    router_hidden_dim: int
    mid_highres_level1: str | None
    mid_highres_image_size: int | None
    mid_highres_base_width: int
    mid_highres_hidden_dim: int
    mid_feature_level1: str | None
    mid_feature_adapter_dim: int
    mid_attention_level1: str | None
    mid_attention_image_size: int | None
    mid_attention_base_width: int
    mid_attention_hidden_dim: int
    leaf_adapter_dim: int
    dropout: float


class CoarseToFineModularHEN(nn.Module):
    def __init__(
        self,
        backbone: str,
        hierarchy: HierarchySpec,
        pretrained: bool = True,
        router_backbone: str = "tiny",
        router_image_size: int = 64,
        router_base_width: int = 32,
        router_hidden_dim: int = 256,
        mid_highres_level1: str | None = None,
        mid_highres_image_size: int | None = None,
        mid_highres_base_width: int = 24,
        mid_highres_hidden_dim: int = 256,
        mid_feature_level1: str | None = None,
        mid_feature_adapter_dim: int = 256,
        mid_attention_level1: str | None = None,
        mid_attention_image_size: int | None = None,
        mid_attention_base_width: int = 24,
        mid_attention_hidden_dim: int = 256,
        leaf_adapter_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hierarchy = hierarchy
        self.metadata = CoarseToFineMetadata(
            backbone=backbone,
            router_backbone=router_backbone,
            router_image_size=router_image_size,
            router_base_width=router_base_width,
            router_hidden_dim=router_hidden_dim,
            mid_highres_level1=mid_highres_level1,
            mid_highres_image_size=mid_highres_image_size,
            mid_highres_base_width=mid_highres_base_width,
            mid_highres_hidden_dim=mid_highres_hidden_dim,
            mid_feature_level1=mid_feature_level1,
            mid_feature_adapter_dim=mid_feature_adapter_dim,
            mid_attention_level1=mid_attention_level1,
            mid_attention_image_size=mid_attention_image_size,
            mid_attention_base_width=mid_attention_base_width,
            mid_attention_hidden_dim=mid_attention_hidden_dim,
            leaf_adapter_dim=leaf_adapter_dim,
            dropout=dropout,
        )

        base_model = build_resnet_backbone(backbone=backbone, pretrained=pretrained)
        self.shared_stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
        )
        self.feature_dim = base_model.fc.in_features
        self.router = TinyHierarchicalRouter(
            hierarchy=hierarchy,
            backbone=router_backbone,
            pretrained=pretrained,
            image_size=router_image_size,
            base_width=router_base_width,
            hidden_dim=router_hidden_dim,
            mid_highres_level1=mid_highres_level1,
            mid_highres_image_size=mid_highres_image_size,
            mid_highres_base_width=mid_highres_base_width,
            mid_highres_hidden_dim=mid_highres_hidden_dim,
        )
        self.level1_experts = nn.ModuleDict(
            {
                str(level1_id): Level1ExpertTower(
                    base_model=base_model,
                    level2_to_leaf={level2_id: hierarchy.level2_to_leaf[level2_id] for level2_id in children},
                    feature_dim=self.feature_dim,
                    leaf_adapter_dim=leaf_adapter_dim,
                    dropout=dropout,
                )
                for level1_id, children in hierarchy.level1_to_level2.items()
            }
        )
        self.mid_feature_branches = nn.ModuleDict()
        if mid_feature_level1 is not None:
            if mid_feature_level1 not in hierarchy.level1_name_to_id:
                raise ValueError(f"Unknown mid_feature_level1: {mid_feature_level1}")
            level1_id = hierarchy.level1_name_to_id[mid_feature_level1]
            self.mid_feature_branches[str(level1_id)] = MidFeatureResidualBranch(
                feature_dim=self.feature_dim,
                num_classes=len(hierarchy.level1_to_level2[level1_id]),
                adapter_dim=mid_feature_adapter_dim,
            )
        self.mid_attention_branches = nn.ModuleDict()
        if mid_attention_level1 is not None:
            if mid_attention_level1 not in hierarchy.level1_name_to_id:
                raise ValueError(f"Unknown mid_attention_level1: {mid_attention_level1}")
            level1_id = hierarchy.level1_name_to_id[mid_attention_level1]
            self.mid_attention_branches[str(level1_id)] = AttentionCropMidBranch(
                image_size=mid_attention_image_size or router_image_size,
                num_classes=len(hierarchy.level1_to_level2[level1_id]),
                base_width=mid_attention_base_width,
                hidden_dim=mid_attention_hidden_dim,
            )
        self.level1_to_level2 = {key: list(value) for key, value in hierarchy.level1_to_level2.items()}
        self.register_buffer("level2_to_level1", torch.tensor(hierarchy.level2_to_level1, dtype=torch.long))

    def _refine_level2_with_branch_features(
        self,
        images: torch.Tensor,
        level1_log_probs: torch.Tensor,
        level2_log_probs: torch.Tensor,
        shared_features: torch.Tensor | None = None,
        compute_leaf: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expert_features_by_level1: dict[str, torch.Tensor] = {}
        needs_shared_features = compute_leaf or bool(self.mid_feature_branches) or bool(self.mid_attention_branches)
        if needs_shared_features and shared_features is None:
            shared_features = self.shared_stem(images)

        for level1_id, level2_ids in self.level1_to_level2.items():
            branch_key = str(level1_id)
            conditional_log_probs = level2_log_probs[:, level2_ids] - level1_log_probs[:, level1_id].unsqueeze(1)

            if branch_key in self.mid_attention_branches:
                attention_logits = self.mid_attention_branches[branch_key](images, shared_features)
                conditional_log_probs = F.log_softmax(conditional_log_probs + attention_logits, dim=1)

            needs_expert_features = compute_leaf or branch_key in self.mid_feature_branches
            if needs_expert_features:
                branch = self.level1_experts[branch_key]
                expert_features = branch.extract_features(shared_features)
                expert_features_by_level1[branch_key] = expert_features

                if branch_key in self.mid_feature_branches:
                    feature_logits = self.mid_feature_branches[branch_key](expert_features)
                    conditional_log_probs = F.log_softmax(conditional_log_probs + feature_logits, dim=1)

            level2_log_probs[:, level2_ids] = level1_log_probs[:, level1_id].unsqueeze(1) + conditional_log_probs

        return level2_log_probs, expert_features_by_level1

    def forward(self, images: torch.Tensor, compute_leaf: bool = True) -> JointHierarchicalOutput:
        level1_logits, level1_log_probs, level2_log_probs = self.router(images)
        shared_features = None
        level2_log_probs, expert_features_by_level1 = self._refine_level2_with_branch_features(
            images=images,
            level1_log_probs=level1_log_probs,
            level2_log_probs=level2_log_probs,
            shared_features=shared_features,
            compute_leaf=compute_leaf,
        )

        if not compute_leaf:
            return JointHierarchicalOutput(
                level1_logits=level1_logits,
                level1_log_probs=level1_log_probs,
                level2_log_probs=level2_log_probs,
                leaf_log_probs=None,
            )

        leaf_log_probs = torch.full(
            (images.size(0), self.hierarchy.num_leaf),
            float("-inf"),
            device=images.device,
            dtype=level2_log_probs.dtype,
        )
        for level1_id, level2_ids in self.level1_to_level2.items():
            branch_key = str(level1_id)
            branch = self.level1_experts[branch_key]
            expert_features = expert_features_by_level1[branch_key]
            for level2_id in level2_ids:
                leaf_ids = branch.level2_to_leaf[level2_id]
                adapted = branch.leaf_adapters[str(level2_id)](expert_features)
                base_logits = branch.leaf_heads[str(level2_id)](adapted)
                residual_logits = branch.leaf_residual_heads[str(level2_id)](adapted)
                conditional_logits = base_logits + residual_logits
                conditional_log_probs = F.log_softmax(conditional_logits, dim=1)
                leaf_log_probs[:, leaf_ids] = level2_log_probs[:, level2_id].unsqueeze(1) + conditional_log_probs

        return JointHierarchicalOutput(
            level1_logits=level1_logits,
            level1_log_probs=level1_log_probs,
            level2_log_probs=level2_log_probs,
            leaf_log_probs=leaf_log_probs,
        )

    def freeze_all(self) -> None:
        _set_trainable(self, False)

    def unfreeze_router(self) -> None:
        _set_trainable(self.router, True)

    def unfreeze_shared_stem(self) -> None:
        _set_trainable(self.shared_stem, True)

    def unfreeze_mid_level1_branch(self, level1_id: int, train_top_head: bool = False) -> None:
        _set_trainable(self.router.level2_adapters[str(level1_id)], True)
        _set_trainable(self.router.level2_heads[str(level1_id)], True)
        _set_trainable(self.router.level2_residual_heads[str(level1_id)], True)
        if str(level1_id) in self.router.level2_highres_branches:
            _set_trainable(self.router.level2_highres_branches[str(level1_id)], True)
        if str(level1_id) in self.mid_feature_branches:
            _set_trainable(self.mid_feature_branches[str(level1_id)], True)
        if str(level1_id) in self.mid_attention_branches:
            _set_trainable(self.mid_attention_branches[str(level1_id)], True)
        if train_top_head:
            _set_trainable(self.router.level1_head, True)

    def unfreeze_mid_feature_branches(self) -> None:
        _set_trainable(self.mid_feature_branches, True)

    def unfreeze_mid_attention_branches(self) -> None:
        _set_trainable(self.mid_attention_branches, True)

    def unfreeze_level1_branch(self, level1_id: int) -> None:
        _set_trainable(self.level1_experts[str(level1_id)], True)

    def unfreeze_level2_branch(self, level2_id: int) -> None:
        level1_id = self.hierarchy.level2_to_level1[level2_id]
        branch = self.level1_experts[str(level1_id)]
        _set_trainable(branch.leaf_adapters[str(level2_id)], True)
        _set_trainable(branch.leaf_heads[str(level2_id)], True)
        _set_trainable(branch.leaf_residual_heads[str(level2_id)], True)

    def configure_trainable(
        self,
        scope: str,
        level1_id: int | None = None,
        level2_id: int | None = None,
        train_shared_stem: bool = False,
        train_router: bool = False,
    ) -> None:
        self.freeze_all()
        if scope == "full":
            _set_trainable(self, True)
            return
        if train_shared_stem:
            self.unfreeze_shared_stem()
        if train_router or scope in {"top_router", "router"}:
            self.unfreeze_router()
        if scope == "router":
            self.unfreeze_mid_feature_branches()
            self.unfreeze_mid_attention_branches()
        if scope in {"top_router", "router"}:
            return
        if scope == "mid_level1_branch":
            if level1_id is None:
                raise ValueError("level1_id is required for mid_level1_branch scope.")
            self.unfreeze_mid_level1_branch(level1_id, train_top_head=train_router)
            return
        if scope == "level1_branch":
            if level1_id is None:
                raise ValueError("level1_id is required for level1_branch scope.")
            self.unfreeze_level1_branch(level1_id)
            return
        if scope == "level2_branch":
            if level2_id is None:
                raise ValueError("level2_id is required for level2_branch scope.")
            self.unfreeze_level2_branch(level2_id)
            return
        raise ValueError(f"Unsupported training scope: {scope}")


def transfer_coarse_to_fine_weights(
    target_model: CoarseToFineModularHEN,
    source_model: CoarseToFineModularHEN,
    target_hierarchy: HierarchySpec,
    source_hierarchy: HierarchySpec,
) -> None:
    _copy_module_if_compatible(target_model.shared_stem, source_model.shared_stem)
    _copy_module_if_compatible(target_model.router.encoder, source_model.router.encoder)
    _copy_module_if_compatible(target_model.router.proj, source_model.router.proj)
    _copy_linear_rows_by_name(
        target_linear=target_model.router.level1_head,
        source_linear=source_model.router.level1_head,
        target_names=target_hierarchy.level1_names,
        source_names=source_hierarchy.level1_names,
    )

    source_level1_by_name = source_hierarchy.level1_name_to_id
    target_level1_by_name = target_hierarchy.level1_name_to_id
    source_level2_by_name = source_hierarchy.level2_name_to_id
    target_level2_by_name = target_hierarchy.level2_name_to_id

    for level1_name in target_hierarchy.level1_names:
        if level1_name not in source_level1_by_name:
            continue
        target_level1_id = target_level1_by_name[level1_name]
        source_level1_id = source_level1_by_name[level1_name]

        target_level2_names = [target_hierarchy.level2_names[idx] for idx in target_hierarchy.level1_to_level2[target_level1_id]]
        source_level2_names = [source_hierarchy.level2_names[idx] for idx in source_hierarchy.level1_to_level2[source_level1_id]]
        _copy_linear_rows_by_name(
            target_linear=target_model.router.level2_heads[str(target_level1_id)],
            source_linear=source_model.router.level2_heads[str(source_level1_id)],
            target_names=target_level2_names,
            source_names=source_level2_names,
        )
        _copy_module_if_compatible(
            target_model.router.level2_adapters[str(target_level1_id)],
            source_model.router.level2_adapters[str(source_level1_id)],
        )
        _copy_linear_rows_by_name(
            target_linear=target_model.router.level2_residual_heads[str(target_level1_id)],
            source_linear=source_model.router.level2_residual_heads[str(source_level1_id)],
            target_names=target_level2_names,
            source_names=source_level2_names,
        )
        if (
            str(target_level1_id) in target_model.router.level2_highres_branches
            and str(source_level1_id) in source_model.router.level2_highres_branches
        ):
            target_highres = target_model.router.level2_highres_branches[str(target_level1_id)]
            source_highres = source_model.router.level2_highres_branches[str(source_level1_id)]
            _copy_module_if_compatible(target_highres.encoder, source_highres.encoder)
            _copy_module_if_compatible(target_highres.proj, source_highres.proj)
            _copy_linear_rows_by_name(
                target_linear=target_highres.head,
                source_linear=source_highres.head,
                target_names=target_level2_names,
                source_names=source_level2_names,
            )
        if (
            str(target_level1_id) in target_model.mid_feature_branches
            and str(source_level1_id) in source_model.mid_feature_branches
        ):
            target_mid_feature = target_model.mid_feature_branches[str(target_level1_id)]
            source_mid_feature = source_model.mid_feature_branches[str(source_level1_id)]
            _copy_module_if_compatible(target_mid_feature.adapter, source_mid_feature.adapter)
            _copy_linear_rows_by_name(
                target_linear=target_mid_feature.head,
                source_linear=source_mid_feature.head,
                target_names=target_level2_names,
                source_names=source_level2_names,
            )
            _copy_linear_rows_by_name(
                target_linear=target_mid_feature.residual_head,
                source_linear=source_mid_feature.residual_head,
                target_names=target_level2_names,
                source_names=source_level2_names,
            )
        if (
            str(target_level1_id) in target_model.mid_attention_branches
            and str(source_level1_id) in source_model.mid_attention_branches
        ):
            target_mid_attention = target_model.mid_attention_branches[str(target_level1_id)]
            source_mid_attention = source_model.mid_attention_branches[str(source_level1_id)]
            _copy_module_if_compatible(target_mid_attention.encoder, source_mid_attention.encoder)
            _copy_module_if_compatible(target_mid_attention.proj, source_mid_attention.proj)
            _copy_linear_rows_by_name(
                target_linear=target_mid_attention.head,
                source_linear=source_mid_attention.head,
                target_names=target_level2_names,
                source_names=source_level2_names,
            )

        target_branch = target_model.level1_experts[str(target_level1_id)]
        source_branch = source_model.level1_experts[str(source_level1_id)]
        _copy_module_if_compatible(target_branch.layer3, source_branch.layer3)
        _copy_module_if_compatible(target_branch.layer4, source_branch.layer4)

    for level2_name in target_hierarchy.level2_names:
        if level2_name not in source_level2_by_name:
            continue
        target_level2_id = target_level2_by_name[level2_name]
        source_level2_id = source_level2_by_name[level2_name]
        target_level1_id = target_hierarchy.level2_to_level1[target_level2_id]
        source_level1_id = source_hierarchy.level2_to_level1[source_level2_id]

        target_branch = target_model.level1_experts[str(target_level1_id)]
        source_branch = source_model.level1_experts[str(source_level1_id)]
        _copy_module_if_compatible(
            target_branch.leaf_adapters[str(target_level2_id)],
            source_branch.leaf_adapters[str(source_level2_id)],
        )

        target_leaf_names = [target_hierarchy.leaf_names[idx] for idx in target_hierarchy.level2_to_leaf[target_level2_id]]
        source_leaf_names = [source_hierarchy.leaf_names[idx] for idx in source_hierarchy.level2_to_leaf[source_level2_id]]
        _copy_linear_rows_by_name(
            target_linear=target_branch.leaf_heads[str(target_level2_id)],
            source_linear=source_branch.leaf_heads[str(source_level2_id)],
            target_names=target_leaf_names,
            source_names=source_leaf_names,
        )
        _copy_linear_rows_by_name(
            target_linear=target_branch.leaf_residual_heads[str(target_level2_id)],
            source_linear=source_branch.leaf_residual_heads[str(source_level2_id)],
            target_names=target_leaf_names,
            source_names=source_leaf_names,
        )


def build_coarse_to_fine_hen(
    backbone: str,
    hierarchy: HierarchySpec,
    pretrained: bool = True,
    router_backbone: str = "tiny",
    router_image_size: int = 64,
    router_base_width: int = 32,
    router_hidden_dim: int = 256,
    mid_highres_level1: str | None = None,
    mid_highres_image_size: int | None = None,
    mid_highres_base_width: int = 24,
    mid_highres_hidden_dim: int = 256,
    mid_feature_level1: str | None = None,
    mid_feature_adapter_dim: int = 256,
    mid_attention_level1: str | None = None,
    mid_attention_image_size: int | None = None,
    mid_attention_base_width: int = 24,
    mid_attention_hidden_dim: int = 256,
    leaf_adapter_dim: int = 128,
    dropout: float = 0.0,
) -> CoarseToFineModularHEN:
    return CoarseToFineModularHEN(
        backbone=backbone,
        hierarchy=hierarchy,
        pretrained=pretrained,
        router_backbone=router_backbone,
        router_image_size=router_image_size,
        router_base_width=router_base_width,
        router_hidden_dim=router_hidden_dim,
        mid_highres_level1=mid_highres_level1,
        mid_highres_image_size=mid_highres_image_size,
        mid_highres_base_width=mid_highres_base_width,
        mid_highres_hidden_dim=mid_highres_hidden_dim,
        mid_feature_level1=mid_feature_level1,
        mid_feature_adapter_dim=mid_feature_adapter_dim,
        mid_attention_level1=mid_attention_level1,
        mid_attention_image_size=mid_attention_image_size,
        mid_attention_base_width=mid_attention_base_width,
        mid_attention_hidden_dim=mid_attention_hidden_dim,
        leaf_adapter_dim=leaf_adapter_dim,
        dropout=dropout,
    )
