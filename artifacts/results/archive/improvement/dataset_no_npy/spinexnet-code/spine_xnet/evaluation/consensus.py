"""Faithfulness-Weighted Consensus and Disagreement Maps.

This module implements the paper's novel methodological contribution:
a per-sample adaptive consensus saliency map whose weights are
proportional to each XAI method's faithfulness (Insertion AUC) on
that specific input.

It also provides pixel-wise disagreement uncertainty maps that
highlight image regions where methods disagree — a clinical
communication tool.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from spine_xnet.evaluation.metrics import normalize_map


# ── Faithfulness-Weighted Consensus ──────────────────────────────────────────


class FaithfulnessWeightedConsensus:
    """Produce a consensus saliency map weighted by per-sample faithfulness.

    Each XAI method's contribution is scaled by its Insertion AUC score
    for that specific input, measured on-the-fly.

    Parameters
    ----------
    model : torch.nn.Module
        The classification model (already in eval mode).
    xai_methods : dict[str, Callable]
        ``{method_name: fn(input_tensor, target_class) -> np.ndarray}``.
    n_steps : int
        Resolution of the insertion curve approximation.
    temperature : float
        Softmax temperature for faithfulness-to-weight conversion.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        xai_methods: dict[str, Callable],
        n_steps: int = 20,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        self.xai_methods = xai_methods
        self.n_steps = n_steps
        self.temperature = temperature

    @torch.no_grad()
    def _compute_insertion_auc(
        self,
        saliency_map: np.ndarray,
        input_tensor: torch.Tensor,
        target_class: int,
        forward_fn: Callable,
    ) -> float:
        """Fast approximation of Insertion AUC for a single sample."""
        device = input_tensor.device
        _, _, H, W = input_tensor.shape
        flat_saliency = saliency_map.flatten()
        sorted_idx = np.argsort(flat_saliency)[::-1]  # high to low

        step_size = max(1, len(sorted_idx) // self.n_steps)
        baseline = torch.zeros_like(input_tensor)
        confidences: list[float] = []

        for step in range(self.n_steps):
            mask = torch.zeros(H * W, device=device)
            end = min((step + 1) * step_size, len(sorted_idx))
            insert_idx = torch.from_numpy(sorted_idx[:end].copy()).long().to(device)
            mask[insert_idx] = 1.0
            mask = mask.reshape(1, 1, H, W)

            partial_input = input_tensor * mask + baseline * (1 - mask)
            logits = forward_fn(partial_input)
            conf = torch.softmax(logits, dim=-1)[0, target_class].item()
            confidences.append(conf)

        return float(np.trapz(confidences) / self.n_steps)

    def get_consensus_map(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        forward_fn: Callable,
    ) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
        """Compute the consensus saliency map for one input.

        Returns
        -------
        consensus_map : np.ndarray, shape (H, W)
        weights : dict  {method_name: weight}
        saliency_maps : dict  {method_name: normalized_map}
        """
        saliency_maps: dict[str, np.ndarray] = {}
        faithfulness_scores: dict[str, float] = {}

        for name, method_fn in self.xai_methods.items():
            try:
                smap = method_fn(input_tensor, target_class)
                smap_normalized = normalize_map(smap)
                saliency_maps[name] = smap_normalized
                faithfulness_scores[name] = self._compute_insertion_auc(
                    smap_normalized, input_tensor, target_class, forward_fn,
                )
            except Exception:
                continue

        if not saliency_maps:
            H, W = input_tensor.shape[-2:]
            return np.zeros((H, W), dtype=np.float32), {}, {}

        # Softmax-normalize faithfulness scores → weights
        scores = np.array(list(faithfulness_scores.values()))
        scores_shifted = (scores - scores.mean()) / max(float(self.temperature), 1e-6)
        weights_arr = np.exp(scores_shifted) / np.exp(scores_shifted).sum()

        # Weighted average
        consensus = sum(
            w * saliency_maps[name]
            for name, w in zip(faithfulness_scores.keys(), weights_arr)
        )
        consensus = normalize_map(consensus)

        weights = dict(zip(faithfulness_scores.keys(), weights_arr.tolist()))
        return consensus, weights, saliency_maps


# ── Disagreement / Uncertainty Maps ──────────────────────────────────────────


def compute_disagreement_map(
    saliency_maps: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-wise disagreement and confidence maps.

    Parameters
    ----------
    saliency_maps : {method_name: (H, W) saliency array}

    Returns
    -------
    disagreement : (H, W) — std across methods; high = methods disagree
    confidence   : (H, W) — high where methods agree *and* saliency is high
    """
    maps = np.stack([normalize_map(m) for m in saliency_maps.values()])  # (K, H, W)
    disagreement = maps.std(axis=0)
    mean_saliency = maps.mean(axis=0)
    confidence = mean_saliency * (1.0 - disagreement)
    return disagreement, confidence
