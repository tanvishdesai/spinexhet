"""Prototype evidence layer used by SpineXNet."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PrototypeLayer(nn.Module):
    """1x1 spatial prototypes over convolutional feature maps.

    Each prototype lives in the feature-channel space. The layer compares every
    spatial feature vector to each prototype and returns both max activations and
    dense activation maps, which can be overlaid as local explanations.
    """

    def __init__(self, channels: int, num_prototypes: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.channels = channels
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        # Initialize prototypes on the unit sphere for stable cosine similarity
        init = torch.randn(num_prototypes, channels)
        init = F.normalize(init, dim=-1) * 0.1
        self.prototypes = nn.Parameter(init)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = features.shape
        feat = F.normalize(features.flatten(2).transpose(1, 2), dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        similarity = torch.einsum("bnc,pc->bnp", feat, proto)
        distance = 1.0 - similarity
        # Clamp temperature to prevent saturation (min 0.5)
        temp = max(self.temperature, 0.5)
        activation = torch.exp(-distance / temp)
        maps = activation.transpose(1, 2).reshape(b, self.num_prototypes, h, w)
        pooled = maps.flatten(2).max(dim=-1).values
        distances = distance.transpose(1, 2).reshape(b, self.num_prototypes, h, w)
        return pooled, maps, distances

    def diversity_loss(self, margin: float = 0.15) -> torch.Tensor:
        proto = F.normalize(self.prototypes, dim=-1)
        sim = proto @ proto.t()
        eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        off_diag = sim.masked_fill(eye, -1.0)
        return F.relu(off_diag - margin).mean()


def prototype_cluster_loss(proto_activations: torch.Tensor) -> torch.Tensor:
    """Encourage every sample to be close to at least one prototype."""

    if proto_activations.shape[1] == 0:
        return proto_activations.new_tensor(0.0)
    return 1.0 - proto_activations.max(dim=1).values.mean()

