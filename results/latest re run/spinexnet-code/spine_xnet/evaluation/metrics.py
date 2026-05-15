"""Classification and explanation metrics."""

from __future__ import annotations

import numpy as np


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def weighted_log_loss(y_true: np.ndarray, probs: np.ndarray, class_weights: list[float] | None = None) -> float:
    class_weights = np.asarray(class_weights or [1.0, 2.0, 4.0], dtype=np.float64)
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    weights = class_weights[y_true.astype(int)]
    losses = -np.log(probs[np.arange(len(y_true)), y_true.astype(int)]) * weights
    return float(losses.sum() / weights.sum())


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

    pred = probs.argmax(axis=1)
    metrics = {
        "weighted_log_loss": weighted_log_loss(y_true, probs),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "ordinal_mae": float(np.abs(pred - y_true).mean()),
        "skip_error_rate": float((np.abs(pred - y_true) >= 2).mean()),
        "accuracy": float((pred == y_true).mean()),
    }
    try:
        metrics["auc_ovr"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except Exception:
        metrics["auc_ovr"] = float("nan")
    return metrics


def per_group_metrics(y_true: np.ndarray, probs: np.ndarray, groups: np.ndarray, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        if mask.sum() == 0:
            continue
        sub = classification_metrics(y_true[mask], probs[mask])
        for key, value in sub.items():
            out[f"{prefix}_{group}_{key}"] = value
    return out


def normalize_map(attr: np.ndarray) -> np.ndarray:
    attr = np.asarray(attr, dtype=np.float32)
    attr = np.nan_to_num(attr, nan=0.0, posinf=0.0, neginf=0.0)
    attr = attr - attr.min()
    denom = attr.max() - attr.min()
    if denom < 1e-8:
        return np.zeros_like(attr, dtype=np.float32)
    return attr / denom


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    corr = spearmanr(a.reshape(-1), b.reshape(-1)).correlation
    return float(0.0 if np.isnan(corr) else corr)


def topk_iou(a: np.ndarray, b: np.ndarray, top_fraction: float = 0.20) -> float:
    a = normalize_map(a)
    b = normalize_map(b)
    k = max(1, int(a.size * top_fraction))
    ath = np.partition(a.reshape(-1), -k)[-k]
    bth = np.partition(b.reshape(-1), -k)[-k]
    am = a >= ath
    bm = b >= bth
    inter = np.logical_and(am, bm).sum()
    union = np.logical_or(am, bm).sum()
    return float(inter / max(union, 1))

