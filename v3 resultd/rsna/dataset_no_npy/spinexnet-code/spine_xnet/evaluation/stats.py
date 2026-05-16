"""Statistical significance utilities for the XAI benchmark.

Provides bootstrap confidence intervals, Wilcoxon signed-rank tests,
and random-baseline agreement computation.  Every table and figure in
the revised manuscript uses this module to add CIs and p-values.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap, wilcoxon


# ── Bootstrap Confidence Intervals ────────────────────────────────────────────


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval for any scalar statistic."""
    data = np.asarray(data).ravel()
    if len(data) < 2:
        val = float(statistic(data))
        return val, val
    result = bootstrap(
        (data,),
        statistic,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def bootstrap_spearman_ci(
    saliency_a: np.ndarray,
    saliency_b: np.ndarray,
    n_resamples: int = 2000,
) -> tuple[float, float]:
    """Bootstrap 95% CI for the Spearman correlation between two saliency map sets.

    Parameters
    ----------
    saliency_a, saliency_b : (N, H, W) or (N, H*W) arrays
        Each row is one flattened or 2-D saliency map.
    """
    n = len(saliency_a)
    a_flat = saliency_a.reshape(n, -1)
    b_flat = saliency_b.reshape(n, -1)
    correlations: list[float] = []
    for _ in range(n_resamples):
        idx = np.random.randint(0, n, n)
        rho, _ = stats.spearmanr(a_flat[idx].mean(axis=0), b_flat[idx].mean(axis=0))
        correlations.append(float(rho) if np.isfinite(rho) else 0.0)
    return float(np.percentile(correlations, 2.5)), float(np.percentile(correlations, 97.5))


# ── Pairwise Wilcoxon Tests ──────────────────────────────────────────────────


def pairwise_wilcoxon_tests(
    metric_dict: dict[str, np.ndarray],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """All pairwise Wilcoxon signed-rank tests.

    Parameters
    ----------
    metric_dict : {model_name: array_of_per_sample_scores}
    """
    results: list[dict] = []
    models = sorted(metric_dict.keys())
    for m1, m2 in combinations(models, 2):
        a = np.asarray(metric_dict[m1])
        b = np.asarray(metric_dict[m2])
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
        if np.allclose(a, b):
            stat, p = 0.0, 1.0
        else:
            try:
                stat, p = wilcoxon(a, b)
            except ValueError:
                stat, p = 0.0, 1.0
        results.append({
            "model_A": m1,
            "model_B": m2,
            "statistic": float(stat),
            "p_value": float(p),
            "significant": p < alpha,
            "effect_size": float(stat) / max(len(a), 1),
        })
    return pd.DataFrame(results)


# ── Random Baseline Agreement ────────────────────────────────────────────────


def random_baseline_agreement(
    n_samples: int = 300,
    n_bootstrap: int = 500,
    map_size: tuple[int, int] = (224, 224),
) -> tuple[float, float]:
    """Expected Spearman ρ between two *random* saliency maps.

    Returns (mean, std) of the null distribution.
    """
    correlations: list[float] = []
    for _ in range(n_bootstrap):
        random_a = np.random.rand(n_samples, *map_size)
        random_b = np.random.rand(n_samples, *map_size)
        rho, _ = stats.spearmanr(
            random_a.reshape(n_samples, -1).mean(axis=0),
            random_b.reshape(n_samples, -1).mean(axis=0),
        )
        correlations.append(float(rho) if np.isfinite(rho) else 0.0)
    return float(np.mean(correlations)), float(np.std(correlations))


# ── Convenience: format value with CI ─────────────────────────────────────────


def format_with_ci(
    values: np.ndarray,
    statistic: Callable = np.mean,
    n_resamples: int = 2000,
    decimals: int = 3,
) -> str:
    """Return e.g. '0.489 [0.421, 0.556]'."""
    val = statistic(values)
    lo, hi = bootstrap_ci(values, statistic=statistic, n_resamples=n_resamples)
    return f"{val:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"
