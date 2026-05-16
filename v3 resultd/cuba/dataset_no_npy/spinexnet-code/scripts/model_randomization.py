"""Model Randomization Sanity Check (Adebayo et al., NeurIPS 2018).

Progressively randomizes model weights from the output layer toward the
input, and checks whether saliency maps change.  A faithful XAI method's
saliency should degrade as learned features are destroyed.  If saliency
is invariant to weight randomization, the method is not using the model.

Usage (Kaggle):
    python scripts/model_randomization.py \
        --config configs/baselines/convnext_blackbox.yaml \
        --checkpoint checkpoints/convnext_blackbox_best.pt \
        --manifest manifests/manifest_v2.csv \
        --cache-dir image_cache_224 \
        --output-dir /kaggle/working/randomization/convnext \
        --methods gradcam integrated_gradients gradient_shap occlusion \
        --max-samples 50 --n-levels 5
"""

from __future__ import annotations

import argparse
import copy
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
from spine_xnet.evaluation.xai import run_attribution_method
from spine_xnet.models import build_model
from spine_xnet.utils import seed_everything, to_device, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model randomization sanity check.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--crop-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--methods", nargs="+",
                    default=["gradcam", "integrated_gradients", "gradient_shap", "occlusion"])
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--n-levels", type=int, default=5,
                    help="Number of cascading randomization levels.")
    return p.parse_args()


def _ssim_2d(a: np.ndarray, b: np.ndarray) -> float:
    """Structural Similarity Index between two 2-D maps."""
    try:
        from skimage.metrics import structural_similarity as ssim
        data_range = max(a.max() - a.min(), b.max() - b.min(), 1e-8)
        return float(ssim(a, b, data_range=data_range))
    except ImportError:
        # Fallback: Pearson correlation
        from scipy.stats import pearsonr
        r, _ = pearsonr(a.ravel(), b.ravel())
        return float(r) if np.isfinite(r) else 0.0


def cascading_randomization(
    model: torch.nn.Module,
    image: torch.Tensor,
    condition_idx: int,
    level_idx: int,
    target: int,
    method: str,
    n_levels: int = 5,
) -> list[float]:
    """Run cascading randomization and return SSIM at each level.

    Low SSIM  = method is sensitive to model weights (GOOD).
    High SSIM = method ignores model weights (BAD).
    """
    # Baseline saliency from the trained model
    original_map = run_attribution_method(method, model, image, condition_idx, level_idx, target)

    model_copy = copy.deepcopy(model)
    params = list(model_copy.named_parameters())
    layer_step = max(1, len(params) // n_levels)

    ssim_scores: list[float] = []
    for level in range(1, n_levels + 1):
        # Randomize from the output (deepest) layers backward
        start_idx = max(0, len(params) - level * layer_step)
        for i in range(start_idx, len(params)):
            _, param = params[i]
            param.data = torch.randn_like(param.data)

        try:
            randomized_map = run_attribution_method(
                method, model_copy, image, condition_idx, level_idx, target,
            )
            score = _ssim_2d(original_map, randomized_map)
        except Exception:
            score = float("nan")
        ssim_scores.append(score)

    return ssim_scores


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(42)

    cfg = load_config_with_base(args.config)
    data_cfg = cfg.get("data", {})
    if args.cache_dir:
        data_cfg["cache_dir"] = str(args.cache_dir)
    if args.crop_root:
        data_cfg["crop_root"] = str(args.crop_root)

    manifest = args.manifest
    _, val_df = load_manifest_for_fold(manifest, fold=args.fold)
    val_df = val_df.head(args.max_samples).reset_index(drop=True)

    concept_columns = data_cfg.get("concept_columns", DEFAULT_CONCEPTS)
    dataset = RSNACropDataset(
        val_df,
        image_size=int(data_cfg.get("image_size", 224)),
        train=False,
        concept_columns=concept_columns,
        crop_root=data_cfg.get("crop_root"),
        manifest_path=manifest,
        require_crops=False,
        cache_dir=data_cfg.get("cache_dir"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, num_concepts=len(concept_columns)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    rows: list[dict] = []
    for sample_idx, batch in enumerate(tqdm(loader, desc="randomization")):
        if sample_idx >= args.max_samples:
            break
        batch = to_device(batch, device)
        with torch.no_grad():
            outputs = model(batch["image"], batch["condition_idx"], batch["level_idx"])
            target = int(torch.softmax(outputs["logits"], dim=1).argmax(dim=1).item())

        condition_idx = int(batch["condition_idx"].item())
        level_idx = int(batch["level_idx"].item())

        for method in args.methods:
            try:
                scores = cascading_randomization(
                    model, batch["image"], condition_idx, level_idx, target,
                    method, n_levels=args.n_levels,
                )
                for lvl, ssim_val in enumerate(scores, start=1):
                    rows.append({
                        "sample_id": batch["sample_id"][0],
                        "method": method,
                        "randomization_level": lvl,
                        "ssim": ssim_val,
                    })
            except Exception as e:
                print(f"  Skipped {method}: {e}")

    # Save per-sample results
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "randomization_results.csv", index=False)

    # Aggregate summary
    if not df.empty:
        summary = (
            df.groupby(["method", "randomization_level"])["ssim"]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary.to_csv(args.output_dir / "randomization_summary.csv", index=False)
        _plot_randomization(summary, args.output_dir)

    print(f"Randomization results saved to {args.output_dir}")


def _plot_randomization(summary: pd.DataFrame, output_dir: Path) -> None:
    """Line plot: SSIM vs randomization depth per method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, grp in summary.groupby("method"):
        ax.errorbar(
            grp["randomization_level"], grp["mean"], yerr=grp["std"],
            marker="o", capsize=3, label=method,
        )
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="SSIM=0.5 (moderate)")
    ax.set_xlabel("Randomization Level (output → input)", fontsize=11)
    ax.set_ylabel("SSIM with Original Saliency", fontsize=11)
    ax.set_title("Model Randomization Sanity Check (Adebayo et al. 2018)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "randomization_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved randomization plot to {output_dir / 'randomization_plot.png'}")


if __name__ == "__main__":
    main()
