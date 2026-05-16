"""Generate qualitative disagreement gallery figure.

Creates a 4-row, N-column figure showing:
  Row 1: High-agreement case  (e.g. DenseNet-121)
  Row 2: Low-agreement case   (e.g. ConvNeXt-Tiny)
  Row 3: Near-zero agreement  (e.g. ViT-Small)
  Row 4: FW-Consensus vs individual methods

Usage (Kaggle):
    python scripts/generate_gallery.py \
        --xai-dir /kaggle/working/xai \
        --models densenet121 convnext_blackbox vit_small \
        --manifest manifests/manifest_v2.csv \
        --cache-dir image_cache_224 \
        --output-dir /kaggle/working/figures/gallery
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from spine_xnet.evaluation.metrics import normalize_map


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qualitative disagreement gallery.")
    p.add_argument("--xai-dir", type=Path, required=True,
                    help="Directory containing per-model XAI result subdirectories.")
    p.add_argument("--models", nargs="+", required=True,
                    help="Model subdirectory names for rows 1-3 (high, low, near-zero agreement).")
    p.add_argument("--manifest", type=Path, default=None,
                    help="Manifest CSV for loading crop images.")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--crop-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=1,
                    help="Number of extreme cases to display per row.")
    return p.parse_args()


def find_extreme_samples(
    agreement_csv: Path,
    metric_col: str = "mean_spearman",
) -> tuple[list[str], list[str]]:
    """Find sample IDs with highest and lowest agreement."""
    df = pd.read_csv(agreement_csv)
    if metric_col not in df.columns:
        return [], []
    df = df.dropna(subset=[metric_col])
    sorted_df = df.sort_values(metric_col)
    worst = sorted_df.head(3)["sample_id"].tolist()
    best = sorted_df.tail(3)["sample_id"].tolist()
    return best, worst


def load_attribution_maps(
    maps_dir: Path,
    sample_id: str,
) -> dict[str, np.ndarray]:
    """Load cached .npy attribution maps for a sample."""
    maps: dict[str, np.ndarray] = {}
    for npy_file in maps_dir.glob(f"{sample_id}_*.npy"):
        method = npy_file.stem.replace(f"{sample_id}_", "")
        maps[method] = np.load(npy_file)
    return maps


def make_gallery_figure(
    rows_data: list[dict],
    output_path: Path,
) -> None:
    """Create the gallery figure.

    Each entry in rows_data is:
        {"title": str, "maps": dict[str, np.ndarray], "input": np.ndarray | None}
    """
    if not rows_data:
        print("No data to plot.")
        return

    # Determine methods (columns) from the union of all maps
    all_methods = sorted(set(m for row in rows_data for m in row["maps"]))
    if not all_methods:
        print("No attribution maps found.")
        return

    n_rows = len(rows_data)
    n_cols = len(all_methods) + 1  # +1 for the input image column

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    saliency_cmap = plt.cm.jet

    for row_idx, row in enumerate(rows_data):
        # Column 0: input image (placeholder if not available)
        ax = axes[row_idx, 0]
        if row.get("input") is not None:
            ax.imshow(row["input"], cmap="gray")
        else:
            ax.text(0.5, 0.5, "Input", ha="center", va="center", fontsize=10,
                    transform=ax.transAxes)
        ax.set_ylabel(row["title"], fontsize=10, fontweight="bold", rotation=90,
                      labelpad=10)
        if row_idx == 0:
            ax.set_title("Input", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Columns 1..N: saliency maps
        for col_idx, method in enumerate(all_methods):
            ax = axes[row_idx, col_idx + 1]
            if method in row["maps"]:
                smap = normalize_map(row["maps"][method])
                ax.imshow(smap, cmap=saliency_cmap, vmin=0, vmax=1)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=9,
                        transform=ax.transAxes)
            if row_idx == 0:
                ax.set_title(method.replace("_", " ").title(), fontsize=9, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Qualitative XAI Disagreement Gallery", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved gallery to {output_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    row_labels = [
        "High Agreement",
        "Low Agreement",
        "Near-Zero Agreement",
    ]

    rows_data: list[dict] = []
    for model_idx, model_name in enumerate(args.models[:3]):
        model_dir = args.xai_dir / model_name
        agreement_csv = model_dir / "agreement_metrics.csv"
        maps_dir = model_dir / "attribution_maps"

        label = row_labels[model_idx] if model_idx < len(row_labels) else model_name
        title = f"{label}\n({model_name})"

        if not agreement_csv.exists():
            print(f"Skipping {model_name} — no agreement_metrics.csv")
            rows_data.append({"title": title, "maps": {}, "input": None})
            continue

        best_ids, worst_ids = find_extreme_samples(agreement_csv)

        # For high-agreement model (row 0), pick the best sample
        # For low/near-zero models (rows 1-2), pick the worst sample
        if model_idx == 0:
            target_id = best_ids[0] if best_ids else None
        else:
            target_id = worst_ids[0] if worst_ids else None

        if target_id is None or not maps_dir.exists():
            rows_data.append({"title": title, "maps": {}, "input": None})
            continue

        maps = load_attribution_maps(maps_dir, target_id)
        rows_data.append({"title": title, "maps": maps, "input": None})

    make_gallery_figure(rows_data, args.output_dir / "disagreement_gallery.png")
    print(f"Gallery saved to {args.output_dir}")


if __name__ == "__main__":
    main()
