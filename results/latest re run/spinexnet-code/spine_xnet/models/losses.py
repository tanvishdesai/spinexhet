"""Loss functions for multi-class ordinal severity and explanations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from spine_xnet.models.prototypes import prototype_cluster_loss


def coral_targets(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    ranks = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)
    return (labels.unsqueeze(1) > ranks).float()


def coral_loss(ordinal_logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    targets = coral_targets(labels, num_classes=num_classes)
    return F.binary_cross_entropy_with_logits(ordinal_logits, targets)


def ordinal_probs_from_logits(ordinal_logits: torch.Tensor) -> torch.Tensor:
    """Convert K-1 CORAL logits to K class probabilities."""

    rank_probs = torch.sigmoid(ordinal_logits)
    p0 = 1.0 - rank_probs[:, :1]
    middle = rank_probs[:, :-1] - rank_probs[:, 1:]
    plast = rank_probs[:, -1:]
    probs = torch.cat([p0, middle, plast], dim=1)
    return probs.clamp_min(1e-6)


@dataclass
class LossWeights:
    classification: float = 1.0
    concept: float = 0.3
    prototype_cluster: float = 0.05
    prototype_diversity: float = 0.02
    ordinal: float = 0.2


class SpineLoss(nn.Module):
    def __init__(
        self,
        class_weights: list[float] | torch.Tensor | None = None,
        loss_weights: LossWeights | None = None,
        num_classes: int = 3,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.loss_weights = loss_weights or LossWeights()
        self.label_smoothing = label_smoothing
        if class_weights is None:
            self.register_buffer("class_weights", torch.ones(num_classes))
        else:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], model: nn.Module | None = None) -> dict[str, torch.Tensor]:
        labels = batch["label"]
        losses: dict[str, torch.Tensor] = {}
        losses["classification"] = F.cross_entropy(
            outputs["logits"], labels, weight=self.class_weights, label_smoothing=self.label_smoothing
        )

        if "concept_logits" in outputs and batch.get("concept_targets") is not None:
            targets = batch["concept_targets"]
            mask = batch.get("concept_mask", torch.ones_like(targets))
            raw = F.binary_cross_entropy_with_logits(outputs["concept_logits"], targets, reduction="none")
            losses["concept"] = (raw * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            losses["concept"] = outputs["logits"].new_tensor(0.0)

        if "prototype_activations" in outputs:
            losses["prototype_cluster"] = prototype_cluster_loss(outputs["prototype_activations"])
            if model is not None and hasattr(model, "prototype_diversity_loss"):
                losses["prototype_diversity"] = model.prototype_diversity_loss()
            else:
                losses["prototype_diversity"] = outputs["logits"].new_tensor(0.0)
        else:
            losses["prototype_cluster"] = outputs["logits"].new_tensor(0.0)
            losses["prototype_diversity"] = outputs["logits"].new_tensor(0.0)

        if "ordinal_logits" in outputs:
            losses["ordinal"] = coral_loss(outputs["ordinal_logits"], labels, num_classes=self.num_classes)
        else:
            losses["ordinal"] = outputs["logits"].new_tensor(0.0)

        total = (
            self.loss_weights.classification * losses["classification"]
            + self.loss_weights.concept * losses["concept"]
            + self.loss_weights.prototype_cluster * losses["prototype_cluster"]
            + self.loss_weights.prototype_diversity * losses["prototype_diversity"]
            + self.loss_weights.ordinal * losses["ordinal"]
        )
        losses["total"] = total
        return losses

