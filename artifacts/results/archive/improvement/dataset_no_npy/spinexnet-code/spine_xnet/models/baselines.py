"""Black-box and simple interpretable baselines."""

from __future__ import annotations

import torch
from torch import nn

from spine_xnet.models.heads import ConditionLevelEmbedding, mlp


class BaselineClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        num_classes: int = 3,
        meta_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        ordinal: bool = True,
    ) -> None:
        super().__init__()
        import timm

        self.backbone_name = backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        feature_dim = self.backbone.num_features
        self.meta = ConditionLevelEmbedding(dim=meta_dim)
        head_in = feature_dim + self.meta.output_dim
        self.classifier = mlp(head_in, hidden_dim, num_classes, dropout=dropout)
        self.ordinal_head = mlp(head_in, hidden_dim, num_classes - 1, dropout=dropout) if ordinal else None

    def forward(self, image: torch.Tensor, condition_idx: torch.Tensor, level_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(image)
        meta = self.meta(condition_idx, level_idx)
        fused = torch.cat([features, meta], dim=1)
        out = {
            "logits": self.classifier(fused),
            "embedding": fused,
        }
        if self.ordinal_head is not None:
            out["ordinal_logits"] = self.ordinal_head(fused)
        return out


class ConceptBottleneckBaseline(nn.Module):
    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        num_concepts: int = 15,
        num_classes: int = 3,
        meta_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        ordinal: bool = True,
    ) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        feature_dim = self.backbone.num_features
        self.meta = ConditionLevelEmbedding(dim=meta_dim)
        concept_in = feature_dim + self.meta.output_dim
        self.concept_head = mlp(concept_in, hidden_dim, num_concepts, dropout=dropout)
        classifier_in = num_concepts + self.meta.output_dim
        self.classifier = mlp(classifier_in, hidden_dim, num_classes, dropout=dropout)
        self.ordinal_head = mlp(classifier_in, hidden_dim, num_classes - 1, dropout=dropout) if ordinal else None

    def forward(self, image: torch.Tensor, condition_idx: torch.Tensor, level_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(image)
        meta = self.meta(condition_idx, level_idx)
        concept_logits = self.concept_head(torch.cat([features, meta], dim=1))
        concepts = torch.sigmoid(concept_logits)
        fused = torch.cat([concepts, meta], dim=1)
        out = {
            "logits": self.classifier(fused),
            "concept_logits": concept_logits,
            "concept_probs": concepts,
            "embedding": fused,
        }
        if self.ordinal_head is not None:
            out["ordinal_logits"] = self.ordinal_head(fused)
        return out

