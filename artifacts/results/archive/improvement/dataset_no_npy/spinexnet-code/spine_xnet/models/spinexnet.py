"""SpineXNet: concept bottleneck + prototype evidence + ordinal grading."""

from __future__ import annotations

import torch
from torch import nn

from spine_xnet.models.heads import ConditionLevelEmbedding, mlp
from spine_xnet.models.prototypes import PrototypeLayer


class SpineXNet(nn.Module):
    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        num_concepts: int = 15,
        num_classes: int = 3,
        num_prototypes: int = 50,
        meta_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        prototype_temperature: float = 0.25,
        residual_features: bool = True,
        residual_dim: int = 128,
        ordinal: bool = True,
        residual_warmup_epochs: int = 0,
    ) -> None:
        super().__init__()
        import timm

        self.backbone_name = backbone
        self.features = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(-1,))
        feature_dim = self.features.feature_info.channels()[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.meta = ConditionLevelEmbedding(dim=meta_dim)

        self.num_concepts = num_concepts
        self.num_prototypes = num_prototypes
        concept_in = feature_dim + self.meta.output_dim
        self.concept_head = mlp(concept_in, hidden_dim, num_concepts, dropout=dropout) if num_concepts > 0 else None
        self.prototype_layer = PrototypeLayer(feature_dim, num_prototypes, temperature=prototype_temperature) if num_prototypes > 0 else None

        self.residual_features = residual_features
        self.residual_projector = (
            nn.Sequential(nn.Linear(feature_dim, residual_dim), nn.GELU(), nn.Dropout(dropout))
            if residual_features
            else None
        )
        residual_out = residual_dim if residual_features else 0
        classifier_in = num_concepts + num_prototypes + self.meta.output_dim + residual_out
        self.classifier = mlp(classifier_in, hidden_dim, num_classes, dropout=dropout)
        self.ordinal_head = mlp(classifier_in, hidden_dim, num_classes - 1, dropout=dropout) if ordinal else None

        # Residual weight annealing: start at 0.0 and ramp to 1.0 over warmup epochs
        # This forces the model to learn through concepts and prototypes first.
        self.residual_warmup_epochs = residual_warmup_epochs
        self._residual_weight = 1.0 if residual_warmup_epochs <= 0 else 0.0

    @property
    def residual_weight(self) -> float:
        return self._residual_weight

    def set_residual_weight_for_epoch(self, epoch: int) -> None:
        """Set residual weight based on current epoch. Call at the start of each epoch."""
        if self.residual_warmup_epochs <= 0:
            self._residual_weight = 1.0
        else:
            self._residual_weight = min(1.0, float(epoch) / float(self.residual_warmup_epochs))

    def forward(self, image: torch.Tensor, condition_idx: torch.Tensor, level_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.features(image)[-1]
        pooled_features = self.pool(feature_map).flatten(1)
        meta = self.meta(condition_idx, level_idx)

        if self.concept_head is not None:
            concept_logits = self.concept_head(torch.cat([pooled_features, meta], dim=1))
            concept_probs = torch.sigmoid(concept_logits)
        else:
            concept_logits = None
            concept_probs = pooled_features.new_zeros((pooled_features.shape[0], 0))

        if self.prototype_layer is not None:
            proto_activations, proto_maps, proto_distances = self.prototype_layer(feature_map)
        else:
            b, _, h, w = feature_map.shape
            proto_activations = pooled_features.new_zeros((b, 0))
            proto_maps = pooled_features.new_zeros((b, 0, h, w))
            proto_distances = pooled_features.new_zeros((b, 0, h, w))

        parts = [concept_probs, proto_activations, meta]
        if self.residual_projector is not None:
            residual = self.residual_projector(pooled_features)
            # Scale residual by annealing weight — during early epochs this is 0,
            # forcing the classifier to learn through concepts and prototypes.
            if self._residual_weight < 1.0:
                residual = residual * self._residual_weight
            parts.append(residual)
        fused = torch.cat(parts, dim=1)

        out = {
            "logits": self.classifier(fused),
            "concept_probs": concept_probs,
            "prototype_activations": proto_activations,
            "embedding": fused,
        }
        # Only return expensive maps during inference (eval mode), not during training
        if not self.training:
            out["prototype_maps"] = proto_maps
            out["prototype_distances"] = proto_distances
            out["feature_map"] = feature_map
        if concept_logits is not None:
            out["concept_logits"] = concept_logits
        if self.ordinal_head is not None:
            out["ordinal_logits"] = self.ordinal_head(fused)
        return out

    def prototype_diversity_loss(self, margin: float = 0.15) -> torch.Tensor:
        if self.prototype_layer is None:
            return next(self.parameters()).new_tensor(0.0)
        return self.prototype_layer.diversity_loss(margin=margin)
