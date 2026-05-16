"""Feature-map coherence analysis for the XAI disagreement paper.

The old Gradient Locality Score measured entropy of input gradients. That was
too close to the shattered-gradient regime, so values collapsed across
architectures. This script measures the final spatial feature map instead:

  - feature_autocorrelation: lag-1 spatial autocorrelation of a
    class-conditioned final feature energy map. Higher means smoother,
    spatially coherent evidence.
  - feature_total_variation: normalized neighbor-to-neighbor variation.
    Higher means more fragmented evidence.
  - feature_coherence_score: autocorrelation divided by one plus total
    variation. Higher is the proposed predictor of XAI agreement.

Usage:
    python scripts/feature_map_smoothness.py \
        --configs configs/baselines/resnet50.yaml configs/baselines/densenet121.yaml \
        --checkpoints checkpoints/resnet50_best.pt checkpoints/densenet121_best.pt \
        --names resnet50 densenet121 \
        --manifest manifests/manifest_v2.csv \
        --cache-dir image_cache_224 \
        --output-dir /kaggle/working/feature_map_smoothness \
        --agreement-csv /kaggle/working/figures/fold_0/cross_model/cross_model_agreement.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spine_xnet.config import load_config_with_base
from spine_xnet.constants import DEFAULT_CONCEPTS
from spine_xnet.data.dataset import RSNACropDataset, load_manifest_for_fold
from spine_xnet.evaluation.metrics import normalize_map
from spine_xnet.models import build_model
from spine_xnet.utils import seed_everything, to_device, unwrap_model, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature-map coherence analysis.")
    p.add_argument("--configs", type=Path, nargs="+", required=True)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--names", nargs="+", default=None)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--crop-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument(
        "--agreement-csv",
        type=Path,
        default=None,
        help="CSV with columns [model, mean_spearman] for the scatter plot.",
    )
    return p.parse_args()


def _as_spatial_map(tensor: torch.Tensor, backbone: torch.nn.Module) -> torch.Tensor:
    """Convert final features or token features to (B, H, W) energy maps."""
    if isinstance(tensor, (tuple, list)):
        tensor = tensor[-1]
    if tensor.ndim == 4:
        # Most CNNs return BCHW. Some timm ConvNeXt variants expose BHWC.
        if tensor.shape[1] <= 32 and tensor.shape[-1] > tensor.shape[1]:
            tensor = tensor.permute(0, 3, 1, 2)
        return tensor.abs().mean(dim=1)
    if tensor.ndim == 3:
        b, n_tokens, channels = tensor.shape
        num_prefix_tokens = int(getattr(backbone, "num_prefix_tokens", 1))
        spatial_tokens = n_tokens - num_prefix_tokens
        grid = int(spatial_tokens ** 0.5)
        if grid * grid != spatial_tokens:
            # Some models return no prefix token. Try all tokens before giving up.
            num_prefix_tokens = 0
            spatial_tokens = n_tokens
            grid = int(spatial_tokens ** 0.5)
        if grid * grid != spatial_tokens:
            raise ValueError(f"Cannot reshape {n_tokens} tokens into a square grid.")
        token_energy = tensor[:, num_prefix_tokens:, :].abs().mean(dim=2)
        return token_energy.reshape(b, grid, grid)
    if tensor.ndim == 2:
        raise ValueError("Backbone returned pooled features with no spatial dimension.")
    raise ValueError(f"Unsupported feature tensor shape: {tuple(tensor.shape)}")


def _pool_features(backbone: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Pool timm forward_features output into the classifier feature vector."""
    if hasattr(backbone, "forward_head"):
        try:
            pooled = backbone.forward_head(features, pre_logits=True)
        except TypeError:
            pooled = backbone.forward_head(features)
        if pooled.ndim == 2:
            return pooled

    if features.ndim == 4:
        if features.shape[1] <= 32 and features.shape[-1] > features.shape[1]:
            features = features.permute(0, 3, 1, 2)
        return features.mean(dim=(2, 3))
    if features.ndim == 3:
        num_prefix_tokens = int(getattr(backbone, "num_prefix_tokens", 1))
        if num_prefix_tokens > 0 and features.shape[1] > num_prefix_tokens:
            return features[:, num_prefix_tokens:, :].mean(dim=1)
        return features.mean(dim=1)
    return features


def _forward_from_spatial_features(
    model: torch.nn.Module,
    image: torch.Tensor,
    condition_idx: torch.Tensor,
    level_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
    """Run a Baseline/CBM model while retaining final spatial features."""
    raw = unwrap_model(model)
    backbone = getattr(raw, "backbone", None)
    if backbone is None or not hasattr(backbone, "forward_features"):
        raise ValueError("Feature-map coherence requires a timm backbone with forward_features().")

    features = backbone.forward_features(image)
    if isinstance(features, (tuple, list)):
        features = features[-1]
    features.retain_grad()
    pooled = _pool_features(backbone, features)
    meta = raw.meta(condition_idx, level_idx)

    if hasattr(raw, "concept_head") and hasattr(raw, "classifier"):
        concept_logits = raw.concept_head(torch.cat([pooled, meta], dim=1))
        concepts = torch.sigmoid(concept_logits)
        logits = raw.classifier(torch.cat([concepts, meta], dim=1))
    elif hasattr(raw, "classifier"):
        logits = raw.classifier(torch.cat([pooled, meta], dim=1))
    else:
        raise ValueError(f"Unsupported model class for feature coherence: {type(raw).__name__}")

    return logits, features, backbone


def class_conditioned_feature_energy(
    model: torch.nn.Module,
    image: torch.Tensor,
    condition_idx: torch.Tensor,
    level_idx: torch.Tensor,
) -> np.ndarray:
    """Return a class-conditioned final feature energy map for one sample."""
    model.zero_grad(set_to_none=True)
    logits, features, backbone = _forward_from_spatial_features(model, image, condition_idx, level_idx)
    target = int(logits.detach().softmax(dim=1).argmax(dim=1).item())
    logits[:, target].sum().backward()

    if features.grad is None:
        energy_tensor = _as_spatial_map(features.detach(), backbone)
    else:
        energy_tensor = _as_spatial_map((features.grad.detach().abs() * features.detach().abs()), backbone)

    return normalize_map(energy_tensor[0].detach().float().cpu().numpy())


def spatial_autocorrelation(feature_map: np.ndarray) -> float:
    """Lag-1 spatial autocorrelation over horizontal and vertical neighbors."""
    arr = np.asarray(feature_map, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    z = arr - arr.mean()
    var = float(np.mean(z * z)) + 1e-12
    products = []
    if arr.shape[1] > 1:
        products.append(np.mean(z[:, :-1] * z[:, 1:]))
    if arr.shape[0] > 1:
        products.append(np.mean(z[:-1, :] * z[1:, :]))
    if not products:
        return 0.0
    return float(np.mean(products) / var)


def normalized_total_variation(feature_map: np.ndarray) -> float:
    """Mean neighbor difference normalized by feature-map standard deviation."""
    arr = np.asarray(feature_map, dtype=np.float64)
    diffs = []
    if arr.shape[1] > 1:
        diffs.append(np.abs(arr[:, 1:] - arr[:, :-1]).mean())
    if arr.shape[0] > 1:
        diffs.append(np.abs(arr[1:, :] - arr[:-1, :]).mean())
    if not diffs:
        return 0.0
    return float(np.mean(diffs) / (arr.std() + 1e-12))


def normalized_spatial_entropy(feature_map: np.ndarray) -> float:
    arr = normalize_map(feature_map).reshape(-1).astype(np.float64)
    p = arr / (arr.sum() + 1e-12)
    entropy = -float(np.sum(p * np.log(p + 1e-12)))
    return entropy / max(float(np.log(len(p))), 1e-12)


def map_metrics(feature_map: np.ndarray) -> dict[str, float]:
    autocorr = spatial_autocorrelation(feature_map)
    tv = normalized_total_variation(feature_map)
    entropy = normalized_spatial_entropy(feature_map)
    return {
        "feature_autocorrelation": autocorr,
        "feature_total_variation": tv,
        "feature_entropy_norm": entropy,
        "feature_coherence_score": autocorr / (1.0 + tv),
        "feature_fragmentation_score": tv * entropy,
    }


def compute_for_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> list[dict[str, float]]:
    model.eval()
    rows: list[dict[str, float]] = []
    for i, batch in enumerate(tqdm(loader, desc="feature coherence", leave=False)):
        if i >= max_samples:
            break
        batch = to_device(batch, device)
        image = batch["image"]
        fmap = class_conditioned_feature_energy(
            model,
            image,
            batch["condition_idx"],
            batch["level_idx"],
        )
        rows.append(map_metrics(fmap))
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if len(args.configs) != len(args.checkpoints):
        raise ValueError("--configs and --checkpoints must have the same length")
    names = args.names or [c.stem for c in args.configs]
    if len(names) != len(args.configs):
        raise ValueError("--names must match --configs length")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)

    cfg0 = load_config_with_base(args.configs[0])
    data_cfg = cfg0.get("data", {})
    if args.cache_dir:
        data_cfg["cache_dir"] = str(args.cache_dir)
    if args.crop_root:
        data_cfg["crop_root"] = str(args.crop_root)
    _, val_df = load_manifest_for_fold(args.manifest, fold=args.fold)
    val_df = val_df.head(args.max_samples).reset_index(drop=True)

    concept_columns = data_cfg.get("concept_columns", DEFAULT_CONCEPTS)
    dataset = RSNACropDataset(
        val_df,
        image_size=int(data_cfg.get("image_size", 224)),
        train=False,
        concept_columns=concept_columns,
        crop_root=data_cfg.get("crop_root"),
        manifest_path=args.manifest,
        require_crops=False,
        cache_dir=data_cfg.get("cache_dir"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    summary_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for name, cfg_path, ckpt_path in zip(names, args.configs, args.checkpoints):
        print(f"\nComputing feature-map coherence for {name}", flush=True)
        cfg = load_config_with_base(cfg_path)
        model = build_model(cfg, num_concepts=len(concept_columns)).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)

        rows = compute_for_model(model, loader, device, args.max_samples)
        for sample_idx, row in enumerate(rows):
            sample_rows.append({"model": name, "sample_idx": sample_idx, **row})
        df = pd.DataFrame(rows)
        summary = {"model": name, "n_samples": len(df)}
        for col in [
            "feature_autocorrelation",
            "feature_total_variation",
            "feature_entropy_norm",
            "feature_coherence_score",
            "feature_fragmentation_score",
        ]:
            summary[col] = float(df[col].mean()) if col in df else float("nan")
            summary[f"{col}_std"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
        summary_rows.append(summary)
        print(
            {
                "model": name,
                "feature_autocorrelation": round(summary["feature_autocorrelation"], 4),
                "feature_total_variation": round(summary["feature_total_variation"], 4),
                "feature_coherence_score": round(summary["feature_coherence_score"], 4),
            },
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summary_rows)
    sample_df = pd.DataFrame(sample_rows)
    summary_df.to_csv(args.output_dir / "feature_map_smoothness_scores.csv", index=False)
    sample_df.to_csv(args.output_dir / "feature_map_smoothness_per_sample.csv", index=False)
    write_json(summary_rows, args.output_dir / "feature_map_smoothness_scores.json")

    if args.agreement_csv and args.agreement_csv.exists():
        _plot_feature_coherence_vs_agreement(summary_df, args.agreement_csv, args.output_dir)

    print(f"\nFeature-map smoothness results saved to {args.output_dir}")


def _plot_feature_coherence_vs_agreement(
    smooth_df: pd.DataFrame,
    agreement_csv: Path,
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    agree_df = pd.read_csv(agreement_csv)
    merged = smooth_df.merge(agree_df, on="model", how="inner")
    if merged.empty:
        print("No matching models between feature coherence and agreement data.")
        return

    x = merged["feature_coherence_score"].to_numpy(dtype=float)
    y = merged["mean_spearman"].to_numpy(dtype=float)
    labels = merged["model"].astype(str).to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=100, c="#2f80ed", edgecolors="#1f4f8f", zorder=5)
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(8, 5), fontsize=9)

    title = "Feature Coherence vs XAI Agreement"
    if len(x) > 2:
        r, p = pearsonr(x, y)
        rho, rho_p = spearmanr(x, y)
        z = np.polyfit(x, y, 1)
        xline = np.linspace(x.min() * 0.95, x.max() * 1.05, 50)
        ax.plot(xline, np.polyval(z, xline), "--", color="gray", alpha=0.65)
        title = f"{title} (r={r:.2f}, p={p:.3f}; rho={rho:.2f}, p={rho_p:.3f})"

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Feature Coherence Score (higher = smoother class evidence)")
    ax.set_ylabel("Mean Spearman rho (inter-method agreement)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "feature_coherence_vs_agreement.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved feature coherence scatter plot to {output_dir / 'feature_coherence_vs_agreement.png'}")


if __name__ == "__main__":
    main()
