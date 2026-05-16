"""Gradient Locality Score (GLS) Analysis.

Computes the Gradient Locality Score for all architectures and produces
the GLS-vs-agreement scatter plot — the paper's theoretical contribution.

GLS = 1 / gradient_spatial_entropy.  High GLS means concentrated gradients,
which predicts high inter-method XAI agreement.

Usage (Kaggle):
    python scripts/gradient_locality.py \
        --configs configs/baselines/resnet50.yaml configs/baselines/densenet121.yaml ... \
        --checkpoints checkpoints/resnet50_best.pt checkpoints/densenet121_best.pt ... \
        --manifest manifests/manifest_v2.csv \
        --cache-dir image_cache_224 \
        --output-dir /kaggle/working/gradient_locality \
        --max-samples 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from spine_xnet.models import build_model
from spine_xnet.utils import seed_everything, to_device, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradient Locality Score analysis.")
    p.add_argument("--configs", type=Path, nargs="+", required=True,
                    help="Config YAML files, one per architecture.")
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                    help="Checkpoint files, one per architecture (same order as --configs).")
    p.add_argument("--names", nargs="+", default=None,
                    help="Short names for each model. Derived from config stem if omitted.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--crop-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--agreement-csv", type=Path, default=None,
                    help="CSV with columns [model, mean_spearman] for the scatter plot.")
    return p.parse_args()


def compute_gradient_entropy(
    model: torch.nn.Module,
    image: torch.Tensor,
    condition_idx: int,
    level_idx: int,
    target: int,
) -> float:
    """Spatial entropy of the input-gradient magnitude distribution.

    Low entropy  → concentrated (localized) gradients → high XAI agreement.
    High entropy → diffuse gradients → low XAI agreement.
    """
    image = image.detach().clone().requires_grad_(True)
    b = image.shape[0]
    cond = torch.full((b,), condition_idx, dtype=torch.long, device=image.device)
    level = torch.full((b,), level_idx, dtype=torch.long, device=image.device)

    outputs = model(image, cond, level)
    logits = outputs["logits"]
    loss = logits[0, target]
    loss.backward()

    grad = image.grad.abs().squeeze().mean(dim=0)  # (H, W)
    grad_flat = grad.reshape(-1)
    grad_norm = grad_flat / (grad_flat.sum() + 1e-12)
    entropy = -(grad_norm * torch.log(grad_norm + 1e-12)).sum()
    return float(entropy.item())


def compute_gls_for_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 300,
) -> list[float]:
    """Compute per-sample gradient entropy for a model."""
    model.eval()
    entropies: list[float] = []

    for i, batch in enumerate(tqdm(loader, desc="GLS", leave=False)):
        if i >= max_samples:
            break
        batch = to_device(batch, device)
        image = batch["image"]
        condition_idx = int(batch["condition_idx"].item())
        level_idx = int(batch["level_idx"].item())

        with torch.no_grad():
            outputs = model(image, batch["condition_idx"], batch["level_idx"])
            target = int(torch.softmax(outputs["logits"], dim=1).argmax(dim=1).item())

        model.zero_grad()
        ent = compute_gradient_entropy(model, image, condition_idx, level_idx, target)
        entropies.append(ent)

    return entropies


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assert len(args.configs) == len(args.checkpoints), \
        "--configs and --checkpoints must have the same number of entries"
    names = args.names or [c.stem for c in args.configs]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)

    # Build dataset once (shared across models)
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

    # Compute GLS per model
    results: list[dict] = []
    for name, cfg_path, ckpt_path in zip(names, args.configs, args.checkpoints):
        print(f"\n{'='*60}\nComputing GLS for {name}\n{'='*60}", flush=True)
        cfg = load_config_with_base(cfg_path)
        model = build_model(cfg, num_concepts=len(concept_columns)).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        entropies = compute_gls_for_model(model, loader, device, args.max_samples)
        mean_entropy = float(np.mean(entropies))
        gls = 1.0 / (mean_entropy + 1e-8)

        results.append({
            "model": name,
            "mean_gradient_entropy": round(mean_entropy, 4),
            "gradient_locality_score": round(gls, 6),
            "n_samples": len(entropies),
        })
        print(f"  {name}: entropy={mean_entropy:.4f}, GLS={gls:.6f}", flush=True)

        del model
        torch.cuda.empty_cache()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output_dir / "gradient_locality_scores.csv", index=False)
    write_json(results, args.output_dir / "gradient_locality_scores.json")

    # Generate scatter plot if agreement data is provided
    if args.agreement_csv and args.agreement_csv.exists():
        _plot_gls_vs_agreement(df, args.agreement_csv, args.output_dir)

    print(f"\nGLS results saved to {args.output_dir}")


def _plot_gls_vs_agreement(
    gls_df: pd.DataFrame,
    agreement_csv: Path,
    output_dir: Path,
) -> None:
    """Scatter plot: GLS vs mean Spearman inter-method agreement."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    agree_df = pd.read_csv(agreement_csv)
    merged = gls_df.merge(agree_df, on="model", how="inner")
    if merged.empty:
        print("No matching models between GLS and agreement data.")
        return

    x = merged["gradient_locality_score"].values
    y = merged["mean_spearman"].values
    labels = merged["model"].values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=100, c="#2ecc71", edgecolors="#27ae60", zorder=5)
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(8, 5),
                    fontsize=9, fontweight="bold")

    if len(x) > 2:
        r, p = pearsonr(x, y)
        z = np.polyfit(x, y, 1)
        xline = np.linspace(x.min() * 0.9, x.max() * 1.1, 50)
        ax.plot(xline, np.polyval(z, xline), "--", color="gray", alpha=0.6)
        ax.set_title(f"Gradient Locality Predicts XAI Agreement (r={r:.2f}, p={p:.3f})",
                     fontsize=12, fontweight="bold")
    else:
        ax.set_title("Gradient Locality vs XAI Agreement", fontsize=12, fontweight="bold")

    ax.set_xlabel("Gradient Locality Score (1 / entropy)", fontsize=11)
    ax.set_ylabel("Mean Spearman ρ (inter-method agreement)", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "gls_vs_agreement_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved GLS scatter plot to {output_dir / 'gls_vs_agreement_scatter.png'}")


if __name__ == "__main__":
    main()
