from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ShuffleNet_V2_X0_5_Weights,
    ShuffleNet_V2_X1_0_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
    resnet18,
    resnet34,
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
)

from .hierarchy import HierarchySpec


def build_resnet(backbone: str, num_classes: int, pretrained: bool = True, dropout: float = 0.0) -> nn.Module:
    model = build_resnet_backbone(backbone=backbone, pretrained=pretrained)
    in_features = model.fc.in_features
    if dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    else:
        model.fc = nn.Linear(in_features, num_classes)
    return model


def build_resnet_backbone(backbone: str, pretrained: bool = True):
    if backbone == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    elif backbone == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model


def build_feature_backbone(backbone: str, pretrained: bool = True) -> tuple[nn.Sequential, int]:
    if backbone == "resnet18":
        model = build_resnet_backbone(backbone=backbone, pretrained=pretrained)
        feature_extractor = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
        )
        return feature_extractor, model.fc.in_features

    if backbone == "resnet34":
        model = build_resnet_backbone(backbone=backbone, pretrained=pretrained)
        feature_extractor = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
        )
        return feature_extractor, model.fc.in_features

    if backbone == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        feature_extractor = nn.Sequential(
            model.features,
            model.avgpool,
        )
        return feature_extractor, model.classifier[0].in_features

    if backbone == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        feature_extractor = nn.Sequential(
            model.features,
            model.avgpool,
        )
        return feature_extractor, model.classifier[0].in_features

    if backbone == "shufflenet_v2_x0_5":
        weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None
        model = shufflenet_v2_x0_5(weights=weights)
        feature_extractor = nn.Sequential(
            model.conv1,
            model.maxpool,
            model.stage2,
            model.stage3,
            model.stage4,
            model.conv5,
            nn.AdaptiveAvgPool2d(1),
        )
        return feature_extractor, model.fc.in_features

    if backbone == "shufflenet_v2_x1_0":
        weights = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        model = shufflenet_v2_x1_0(weights=weights)
        feature_extractor = nn.Sequential(
            model.conv1,
            model.maxpool,
            model.stage2,
            model.stage3,
            model.stage4,
            model.conv5,
            nn.AdaptiveAvgPool2d(1),
        )
        return feature_extractor, model.fc.in_features

    raise ValueError(f"Unsupported backbone: {backbone}")


@dataclass
class JointHierarchicalOutput:
    level1_logits: torch.Tensor
    level1_log_probs: torch.Tensor
    level2_log_probs: torch.Tensor
    leaf_log_probs: torch.Tensor


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def _copy_module_if_compatible(target_module: nn.Module, source_module: nn.Module) -> None:
    target_state = target_module.state_dict()
    source_state = source_module.state_dict()
    if target_state.keys() != source_state.keys():
        return
    if any(target_state[key].shape != source_state[key].shape for key in target_state):
        return
    target_module.load_state_dict(source_state)


def _copy_linear_rows_by_name(
    target_linear: nn.Linear,
    source_linear: nn.Linear,
    target_names: Iterable[str],
    source_names: Iterable[str],
) -> None:
    if target_linear.weight.shape[1] != source_linear.weight.shape[1]:
        return

    source_index = {name: idx for idx, name in enumerate(source_names)}
    with torch.no_grad():
        for target_idx, name in enumerate(target_names):
            source_idx = source_index.get(name)
            if source_idx is None:
                continue
            target_linear.weight[target_idx].copy_(source_linear.weight[source_idx])
            if target_linear.bias is not None and source_linear.bias is not None:
                target_linear.bias[target_idx].copy_(source_linear.bias[source_idx])


class ResidualAdapter(nn.Module):
    def __init__(self, feature_dim: int, adapter_dim: int) -> None:
        super().__init__()
        self.down = nn.Linear(feature_dim, adapter_dim)
        self.activation = nn.ReLU(inplace=True)
        self.up = nn.Linear(adapter_dim, feature_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.up(self.activation(self.down(features)))


class JointHierarchicalResNet(nn.Module):
    def __init__(
        self,
        backbone: str,
        hierarchy: HierarchySpec,
        pretrained: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_extractor, self.feature_dim = build_feature_backbone(backbone=backbone, pretrained=pretrained)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.level1_head = nn.Linear(self.feature_dim, hierarchy.num_level1)
        self.level2_heads = nn.ModuleDict(
            {
                str(level1_id): nn.Linear(self.feature_dim, len(children))
                for level1_id, children in hierarchy.level1_to_level2.items()
            }
        )
        self.leaf_heads = nn.ModuleDict(
            {
                str(level2_id): nn.Linear(self.feature_dim, len(children))
                for level2_id, children in hierarchy.level2_to_leaf.items()
            }
        )

        self.level1_to_level2 = {key: list(value) for key, value in hierarchy.level1_to_level2.items()}
        self.level2_to_leaf = {key: list(value) for key, value in hierarchy.level2_to_leaf.items()}
        self.register_buffer("level2_to_level1", torch.tensor(hierarchy.level2_to_level1, dtype=torch.long))
        self.register_buffer("leaf_to_level1", torch.tensor(hierarchy.leaf_to_level1, dtype=torch.long))
        self.register_buffer("leaf_to_level2", torch.tensor(hierarchy.leaf_to_level2, dtype=torch.long))

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        features = torch.flatten(features, 1)
        return self.dropout(features)

    def forward(self, images: torch.Tensor) -> JointHierarchicalOutput:
        features = self.extract_features(images)
        batch_size = features.size(0)

        level1_logits = self.level1_head(features)
        level1_log_probs = F.log_softmax(level1_logits, dim=1)

        level2_log_probs = torch.full(
            (batch_size, self.level2_to_level1.numel()),
            float("-inf"),
            device=features.device,
            dtype=level1_log_probs.dtype,
        )
        for level1_id, level2_ids in self.level1_to_level2.items():
            conditional_logits = self.level2_heads[str(level1_id)](features)
            conditional_log_probs = F.log_softmax(conditional_logits, dim=1)
            level2_log_probs[:, level2_ids] = level1_log_probs[:, level1_id].unsqueeze(1) + conditional_log_probs

        leaf_log_probs = torch.full(
            (batch_size, self.leaf_to_level2.numel()),
            float("-inf"),
            device=features.device,
            dtype=level2_log_probs.dtype,
        )
        for level2_id, leaf_ids in self.level2_to_leaf.items():
            conditional_logits = self.leaf_heads[str(level2_id)](features)
            conditional_log_probs = F.log_softmax(conditional_logits, dim=1)
            leaf_log_probs[:, leaf_ids] = level2_log_probs[:, level2_id].unsqueeze(1) + conditional_log_probs

        return JointHierarchicalOutput(
            level1_logits=level1_logits,
            level1_log_probs=level1_log_probs,
            level2_log_probs=level2_log_probs,
            leaf_log_probs=leaf_log_probs,
        )


class ModularHierarchicalResNet(nn.Module):
    def __init__(
        self,
        backbone: str,
        hierarchy: HierarchySpec,
        pretrained: bool = True,
        dropout: float = 0.0,
        adapter_dim: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hierarchy = hierarchy
        self.adapter_dim = adapter_dim

        self.feature_extractor, self.feature_dim = build_feature_backbone(backbone=backbone, pretrained=pretrained)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.level1_head = nn.Linear(self.feature_dim, hierarchy.num_level1)
        self.level2_adapters = nn.ModuleDict(
            {
                str(level1_id): ResidualAdapter(self.feature_dim, adapter_dim)
                for level1_id in hierarchy.level1_to_level2
            }
        )
        self.level2_heads = nn.ModuleDict(
            {
                str(level1_id): nn.Linear(self.feature_dim, len(children))
                for level1_id, children in hierarchy.level1_to_level2.items()
            }
        )
        self.leaf_adapters = nn.ModuleDict(
            {
                str(level2_id): ResidualAdapter(self.feature_dim, adapter_dim)
                for level2_id in hierarchy.level2_to_leaf
            }
        )
        self.leaf_heads = nn.ModuleDict(
            {
                str(level2_id): nn.Linear(self.feature_dim, len(children))
                for level2_id, children in hierarchy.level2_to_leaf.items()
            }
        )

        self.level1_to_level2 = {key: list(value) for key, value in hierarchy.level1_to_level2.items()}
        self.level2_to_leaf = {key: list(value) for key, value in hierarchy.level2_to_leaf.items()}
        self.register_buffer("level2_to_level1", torch.tensor(hierarchy.level2_to_level1, dtype=torch.long))
        self.register_buffer("leaf_to_level1", torch.tensor(hierarchy.leaf_to_level1, dtype=torch.long))
        self.register_buffer("leaf_to_level2", torch.tensor(hierarchy.leaf_to_level2, dtype=torch.long))

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        features = torch.flatten(features, 1)
        return self.dropout(features)

    def forward(self, images: torch.Tensor) -> JointHierarchicalOutput:
        features = self.extract_features(images)
        batch_size = features.size(0)

        level1_logits = self.level1_head(features)
        level1_log_probs = F.log_softmax(level1_logits, dim=1)

        level2_log_probs = torch.full(
            (batch_size, self.level2_to_level1.numel()),
            float("-inf"),
            device=features.device,
            dtype=level1_log_probs.dtype,
        )
        for level1_id, level2_ids in self.level1_to_level2.items():
            branch_features = self.level2_adapters[str(level1_id)](features)
            conditional_logits = self.level2_heads[str(level1_id)](branch_features)
            conditional_log_probs = F.log_softmax(conditional_logits, dim=1)
            level2_log_probs[:, level2_ids] = level1_log_probs[:, level1_id].unsqueeze(1) + conditional_log_probs

        leaf_log_probs = torch.full(
            (batch_size, self.leaf_to_level2.numel()),
            float("-inf"),
            device=features.device,
            dtype=level2_log_probs.dtype,
        )
        for level2_id, leaf_ids in self.level2_to_leaf.items():
            branch_features = self.leaf_adapters[str(level2_id)](features)
            conditional_logits = self.leaf_heads[str(level2_id)](branch_features)
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

    def unfreeze_backbone(self) -> None:
        _set_trainable(self.feature_extractor, True)

    def unfreeze_top(self) -> None:
        _set_trainable(self.level1_head, True)

    def unfreeze_level1_branch(self, level1_id: int, level2_id: int | None = None) -> None:
        _set_trainable(self.level2_adapters[str(level1_id)], True)
        _set_trainable(self.level2_heads[str(level1_id)], True)
        if level2_id is None:
            return
        if level2_id not in self.level1_to_level2[level1_id]:
            raise ValueError(f"level2_id {level2_id} does not belong to level1_id {level1_id}.")
        _set_trainable(self.leaf_adapters[str(level2_id)], True)
        _set_trainable(self.leaf_heads[str(level2_id)], True)

    def unfreeze_level1_with_children(self, level1_id: int) -> None:
        self.unfreeze_level1_branch(level1_id)
        for level2_id in self.level1_to_level2[level1_id]:
            _set_trainable(self.leaf_adapters[str(level2_id)], True)
            _set_trainable(self.leaf_heads[str(level2_id)], True)

    def unfreeze_level2_branch(self, level2_id: int) -> None:
        _set_trainable(self.leaf_adapters[str(level2_id)], True)
        _set_trainable(self.leaf_heads[str(level2_id)], True)

    def configure_trainable(
        self,
        scope: str,
        level1_id: int | None = None,
        level2_id: int | None = None,
        train_backbone: bool = False,
        include_leaf_branch: bool = False,
    ) -> None:
        self.freeze_all()
        if scope == "full":
            _set_trainable(self, True)
            return
        if train_backbone:
            self.unfreeze_backbone()
        if scope == "top":
            self.unfreeze_top()
            return
        if scope == "level1_branch":
            if level1_id is None:
                raise ValueError("level1_id is required for level1_branch scope.")
            self.unfreeze_level1_branch(level1_id, level2_id if include_leaf_branch else None)
            return
        if scope == "level2_branch":
            if level2_id is None:
                raise ValueError("level2_id is required for level2_branch scope.")
            self.unfreeze_level2_branch(level2_id)
            return
        raise ValueError(f"Unsupported training scope: {scope}")


def transfer_modular_hen_weights(
    target_model: ModularHierarchicalResNet,
    source_model: ModularHierarchicalResNet,
    target_hierarchy: HierarchySpec,
    source_hierarchy: HierarchySpec,
) -> None:
    _copy_module_if_compatible(target_model.feature_extractor, source_model.feature_extractor)

    _copy_linear_rows_by_name(
        target_linear=target_model.level1_head,
        source_linear=source_model.level1_head,
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

        _copy_module_if_compatible(
            target_model.level2_adapters[str(target_level1_id)],
            source_model.level2_adapters[str(source_level1_id)],
        )

        target_level2_names = [target_hierarchy.level2_names[idx] for idx in target_hierarchy.level1_to_level2[target_level1_id]]
        source_level2_names = [source_hierarchy.level2_names[idx] for idx in source_hierarchy.level1_to_level2[source_level1_id]]
        _copy_linear_rows_by_name(
            target_linear=target_model.level2_heads[str(target_level1_id)],
            source_linear=source_model.level2_heads[str(source_level1_id)],
            target_names=target_level2_names,
            source_names=source_level2_names,
        )

    for level2_name in target_hierarchy.level2_names:
        if level2_name not in source_level2_by_name:
            continue
        target_level2_id = target_level2_by_name[level2_name]
        source_level2_id = source_level2_by_name[level2_name]

        _copy_module_if_compatible(
            target_model.leaf_adapters[str(target_level2_id)],
            source_model.leaf_adapters[str(source_level2_id)],
        )

        target_leaf_names = [target_hierarchy.leaf_names[idx] for idx in target_hierarchy.level2_to_leaf[target_level2_id]]
        source_leaf_names = [source_hierarchy.leaf_names[idx] for idx in source_hierarchy.level2_to_leaf[source_level2_id]]
        _copy_linear_rows_by_name(
            target_linear=target_model.leaf_heads[str(target_level2_id)],
            source_linear=source_model.leaf_heads[str(source_level2_id)],
            target_names=target_leaf_names,
            source_names=source_leaf_names,
        )


def build_joint_hen(
    backbone: str,
    hierarchy: HierarchySpec,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> JointHierarchicalResNet:
    return JointHierarchicalResNet(
        backbone=backbone,
        hierarchy=hierarchy,
        pretrained=pretrained,
        dropout=dropout,
    )


def build_modular_hen(
    backbone: str,
    hierarchy: HierarchySpec,
    pretrained: bool = True,
    dropout: float = 0.0,
    adapter_dim: int = 128,
) -> ModularHierarchicalResNet:
    return ModularHierarchicalResNet(
        backbone=backbone,
        hierarchy=hierarchy,
        pretrained=pretrained,
        dropout=dropout,
        adapter_dim=adapter_dim,
    )
