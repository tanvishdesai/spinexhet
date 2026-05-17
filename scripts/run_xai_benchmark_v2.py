"""XAI Disagreement Benchmark v2 (BMVC Revision).

Extended benchmark script that evaluates:
  1. Faithfulness (deletion AUC, insertion AUC)
  2. Agreement (pairwise Spearman, Top-20% IoU)  — with bootstrap CIs
  3. Consistency (augmentation stability)
  4. Clinical alignment (expert ROI + proxy ROI)
  5. Faithfulness-Weighted Consensus (novel contribution)
  6. Statistical significance (bootstrap CI, Wilcoxon, random baseline)

XAI methods supported:
  gradcam, gradcam++, integrated_gradients, gradient_shap,
  occlusion, guided_backprop, attention_rollout (ViT only)

Outputs per-sample CSV results and aggregate summaries for paper figures.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spine_xnet.config import load_config_with_base
from spine_xnet.constants import DEFAULT_CONCEPTS
from spine_xnet.data.dataset import RSNACropDataset, load_manifest_for_fold
from spine_xnet.evaluation.metrics import normalize_map, spearman_corr, topk_iou
from spine_xnet.evaluation.xai import (
    FixedMetaModel,
    attribution_agreement,
    coord_to_crop_roi,
    deletion_insertion_auc,
    expert_roi_alignment,
    make_expert_roi_mask,
    proxy_roi_alignment,
    run_attribution_method,
)
from spine_xnet.evaluation.consensus import (
    compute_disagreement_map,
    faithfulness_weighted_consensus_map,
    topk_faithfulness_consensus_map,
    uniform_consensus_map,
)
from spine_xnet.evaluation.stats import bootstrap_ci, format_with_ci
from spine_xnet.models import build_model
from spine_xnet.utils import seed_everything, to_device, write_json


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XAI Disagreement Benchmark v2 (BMVC Revision).")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument(
        "--methods", nargs="+",
        default=["gradcam", "gradcam++", "integrated_gradients",
                 "gradient_shap", "occlusion", "guided_backprop"],
    )
    p.add_argument("--faithfulness-steps", type=int, default=20)
    p.add_argument("--crop-root", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--save-maps", action="store_true", help="Save attribution maps as .npy")
    p.add_argument("--consistency-augmentations", type=int, default=3,
                    help="Number of augmented copies per image for consistency eval")
    p.add_argument("--skip-consistency", action="store_true")
    p.add_argument("--skip-faithfulness", action="store_true")
    p.add_argument("--skip-consensus", action="store_true",
                    help="Skip FW-Consensus computation (saves time)")
    p.add_argument("--consensus-every", type=int, default=10,
                    help="Evaluate consensus maps every N samples.")
    p.add_argument("--consensus-temperature", type=float, default=1.0,
                    help="Softmax temperature for faithfulness consensus weights.")
    p.add_argument("--topk-consensus-methods", type=int, default=3,
                    help="Number of most faithful methods for top-k consensus baseline.")
    p.add_argument("--enable-attention-rollout", action="store_true",
                    help="Add attention_rollout to methods (ViT only)")
    return p.parse_args()


# ── Consistency helpers ──────────────────────────────────────────────────────

CONSISTENCY_AUGMENTATIONS = [
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
]


def augment_image(image: torch.Tensor, aug_idx: int) -> torch.Tensor:
    """Apply a deterministic augmentation to a CHW tensor."""
    aug = CONSISTENCY_AUGMENTATIONS[aug_idx % len(CONSISTENCY_AUGMENTATIONS)]
    return aug(image)


def compute_consistency_for_sample(
    model: torch.nn.Module,
    image: torch.Tensor,
    condition_idx: int,
    level_idx: int,
    target: int,
    method: str,
    n_augmentations: int = 3,
) -> float:
    """Average Top-20% IoU between original and augmented attributions."""
    original_attr = run_attribution_method(method, model, image, condition_idx, level_idx, target)
    ious = []
    for i in range(n_augmentations):
        aug_image = augment_image(image.squeeze(0), i).unsqueeze(0).to(image.device)
        try:
            aug_attr = run_attribution_method(method, model, aug_image, condition_idx, level_idx, target)
            ious.append(topk_iou(original_attr, aug_attr, top_fraction=0.20))
        except Exception:
            continue
    return float(np.mean(ious)) if ious else float("nan")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    data_cfg = cfg.get("data", {})
    if args.crop_root is not None:
        data_cfg["crop_root"] = str(args.crop_root)
    if args.cache_dir is not None:
        data_cfg["cache_dir"] = str(args.cache_dir)

    manifest = args.manifest or Path(data_cfg["manifest"])
    fold = args.fold if args.fold is not None else int(data_cfg.get("fold", 0))
    _, val_df = load_manifest_for_fold(manifest, fold=fold, fold_col=data_cfg.get("fold_col", "fold"))
    val_df = val_df.head(args.max_samples).reset_index(drop=True)

    concept_columns = data_cfg.get("concept_columns", DEFAULT_CONCEPTS)
    dataset = RSNACropDataset(
        val_df,
        image_size=int(data_cfg.get("image_size", 224)),
        train=False,
        concept_columns=concept_columns,
        crop_root=data_cfg.get("crop_root"),
        manifest_path=manifest,
        require_crops=bool(data_cfg.get("require_crops", False)),
        cache_dir=data_cfg.get("cache_dir"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, num_concepts=len(dataset.concept_columns)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # Add attention_rollout for ViT models if requested
    methods = list(args.methods)
    if args.enable_attention_rollout and "attention_rollout" not in methods:
        methods.append("attention_rollout")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = args.output_dir / "attribution_maps" if args.save_maps else None
    if maps_dir:
        maps_dir.mkdir(parents=True, exist_ok=True)

    faithfulness_rows = []
    agreement_rows = []
    consistency_rows = []
    clinical_rows = []
    consensus_rows = []
    skipped: dict[str, str] = {}
    start_time = time.monotonic()

    for sample_idx, batch in enumerate(tqdm(loader, desc="xai_v2")):
        batch = to_device(batch, device)
        sample_id = batch["sample_id"][0]
        condition = batch["condition"][0] if "condition" in batch else None
        condition_idx = int(batch["condition_idx"].item())
        level_idx = int(batch["level_idx"].item())
        label = int(batch["label"].item())

        with torch.no_grad():
            outputs = model(batch["image"], batch["condition_idx"], batch["level_idx"])
            target = int(torch.softmax(outputs["logits"], dim=1).argmax(dim=1).item())

        attributions: dict[str, np.ndarray] = {}
        sample_insertions: dict[str, float] = {}

        # ── Compute attributions ─────────────────────────────────────────
        for method in methods:
            try:
                attr = run_attribution_method(
                    method, model, batch["image"], condition_idx, level_idx, target,
                )
                attributions[method] = attr
                if maps_dir:
                    np.save(maps_dir / f"{sample_id}_{method}.npy", attr)
            except Exception as exc:
                if method not in skipped:
                    skipped[method] = str(exc)

        if not attributions:
            continue

        # ── Faithfulness ─────────────────────────────────────────────────
        if not args.skip_faithfulness:
            for method, attr in attributions.items():
                try:
                    del_auc = deletion_insertion_auc(
                        model, batch["image"], attr, condition_idx, level_idx, target,
                        mode="deletion", steps=args.faithfulness_steps,
                    )
                    ins_auc = deletion_insertion_auc(
                        model, batch["image"], attr, condition_idx, level_idx, target,
                        mode="insertion", steps=args.faithfulness_steps,
                    )
                    faithfulness_rows.append({
                        "sample_id": sample_id, "method": method,
                        "condition": condition, "label": label, "target": target,
                        "deletion_auc": del_auc, "insertion_auc": ins_auc,
                    })
                    sample_insertions[method] = ins_auc
                except Exception:
                    pass

        # ── Agreement ────────────────────────────────────────────────────
        if len(attributions) > 1:
            agreement = attribution_agreement(attributions)
            agreement_rows.append({
                "sample_id": sample_id, "condition": condition,
                "label": label, **agreement,
            })

        # ── Clinical alignment ───────────────────────────────────────────
        # Expert ROI: crops are centered on RSNA coordinates, so the expert
        # annotation is at the center of the crop (rel_x=0.5, rel_y=0.5).
        image_size = int(data_cfg.get("image_size", 224))
        expert_mask = make_expert_roi_mask(0.5, 0.5, radius_fraction=0.15,
                                           H=image_size, W=image_size)
        for method, attr in attributions.items():
            proxy_score = proxy_roi_alignment(attr, condition=condition)
            expert_score = expert_roi_alignment(attr, expert_mask)
            clinical_rows.append({
                "sample_id": sample_id, "method": method,
                "condition": condition, "label": label,
                "proxy_roi_alignment": proxy_score,
                "expert_roi_alignment": expert_score,
            })

        # ── Consistency ──────────────────────────────────────────────────
        if not args.skip_consistency and sample_idx < 200:
            for method in attributions:
                try:
                    cons = compute_consistency_for_sample(
                        model, batch["image"], condition_idx, level_idx, target,
                        method, n_augmentations=args.consistency_augmentations,
                    )
                    consistency_rows.append({
                        "sample_id": sample_id, "method": method,
                        "condition": condition,
                        "augmentation_iou": cons,
                    })
                except Exception:
                    pass

        # ── FW-Consensus (every 10th sample to save time) ────────────────
        if (
            not args.skip_consensus
            and args.consensus_every > 0
            and sample_idx % args.consensus_every == 0
            and len(attributions) >= 3
        ):
            try:
                if not sample_insertions:
                    for method, attr in attributions.items():
                        sample_insertions[method] = deletion_insertion_auc(
                            model, batch["image"], attr, condition_idx, level_idx, target,
                            mode="insertion", steps=args.faithfulness_steps,
                        )

                fw_map, fw_weights = faithfulness_weighted_consensus_map(
                    attributions,
                    sample_insertions,
                    temperature=args.consensus_temperature,
                )
                uniform_map, _ = uniform_consensus_map(attributions)
                topk_map, topk_weights = topk_faithfulness_consensus_map(
                    attributions,
                    sample_insertions,
                    top_k=args.topk_consensus_methods,
                    temperature=args.consensus_temperature,
                )

                fw_ins_auc = deletion_insertion_auc(
                    model, batch["image"], fw_map, condition_idx, level_idx, target,
                    mode="insertion", steps=args.faithfulness_steps,
                )
                uniform_ins_auc = deletion_insertion_auc(
                    model, batch["image"], uniform_map, condition_idx, level_idx, target,
                    mode="insertion", steps=args.faithfulness_steps,
                )
                topk_ins_auc = deletion_insertion_auc(
                    model, batch["image"], topk_map, condition_idx, level_idx, target,
                    mode="insertion", steps=args.faithfulness_steps,
                )
                fw_expert = expert_roi_alignment(fw_map, expert_mask)
                uniform_expert = expert_roi_alignment(uniform_map, expert_mask)
                topk_expert = expert_roi_alignment(topk_map, expert_mask)
                consensus_rows.append({
                    "sample_id": sample_id,
                    "consensus_insertion_auc": fw_ins_auc,
                    "consensus_expert_roi": fw_expert,
                    "fw_consensus_insertion_auc": fw_ins_auc,
                    "fw_consensus_expert_roi": fw_expert,
                    "uniform_consensus_insertion_auc": uniform_ins_auc,
                    "uniform_consensus_expert_roi": uniform_expert,
                    "topk_consensus_insertion_auc": topk_ins_auc,
                    "topk_consensus_expert_roi": topk_expert,
                    "topk_methods": " ".join(topk_weights),
                    **{f"weight_{k}": v for k, v in fw_weights.items()},
                })
            except Exception:
                pass

        # Periodic checkpoint
        if (sample_idx + 1) % 100 == 0:
            elapsed = (time.monotonic() - start_time) / 60.0
            print({"event": "xai_progress", "samples": sample_idx + 1,
                    "elapsed_min": round(elapsed, 1)}, flush=True)

    # ── Save results ─────────────────────────────────────────────────────
    _save_df(faithfulness_rows, args.output_dir / "faithfulness_metrics.csv")
    _save_df(agreement_rows, args.output_dir / "agreement_metrics.csv")
    _save_df(consistency_rows, args.output_dir / "consistency_metrics.csv")
    _save_df(clinical_rows, args.output_dir / "clinical_alignment.csv")
    _save_df(consensus_rows, args.output_dir / "consensus_metrics.csv")

    # ── Aggregate summary with bootstrap CIs ─────────────────────────────
    summary = _build_summary(faithfulness_rows, agreement_rows, consistency_rows,
                             clinical_rows, consensus_rows)
    write_json({"summary": summary, "skipped": skipped, "args": _args_dict(args)},
               args.output_dir / "xai_summary_v2.json")
    print({"summary": summary, "skipped": skipped})


def _save_df(rows: list[dict], path: Path) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {len(rows)} rows to {path}")


def _build_summary(faith, agree, consist, clinical, consensus) -> dict:
    s: dict = {}
    if faith:
        df = pd.DataFrame(faith)
        per_method = df.groupby("method")[["deletion_auc", "insertion_auc"]].mean().to_dict(orient="index")
        # Add bootstrap CIs
        for method in per_method:
            method_data = df[df["method"] == method]
            for metric in ["deletion_auc", "insertion_auc"]:
                vals = method_data[metric].dropna().values
                if len(vals) >= 2:
                    lo, hi = bootstrap_ci(vals)
                    per_method[method][f"{metric}_ci_lo"] = round(lo, 4)
                    per_method[method][f"{metric}_ci_hi"] = round(hi, 4)
        s["faithfulness"] = per_method
    if agree:
        df = pd.DataFrame(agree)
        for col in ["mean_spearman", "mean_top20_iou"]:
            if col in df.columns:
                vals = df[col].dropna().values
                s[col] = float(vals.mean())
                if len(vals) >= 2:
                    lo, hi = bootstrap_ci(vals)
                    s[f"{col}_ci"] = [round(lo, 4), round(hi, 4)]
    if consist:
        df = pd.DataFrame(consist)
        s["consistency"] = df.groupby("method")["augmentation_iou"].mean().to_dict()
    if clinical:
        df = pd.DataFrame(clinical)
        s["clinical_alignment_proxy"] = df.groupby("method")["proxy_roi_alignment"].mean().to_dict()
        s["clinical_alignment_expert"] = df.groupby("method")["expert_roi_alignment"].mean().to_dict()
        s["clinical_alignment_by_condition"] = (
            df.groupby(["method", "condition"])["expert_roi_alignment"].mean()
            .unstack(level="condition").to_dict(orient="index")
        )
    if consensus:
        df = pd.DataFrame(consensus)
        fw_col = "fw_consensus_insertion_auc" if "fw_consensus_insertion_auc" in df else "consensus_insertion_auc"
        fw_expert_col = "fw_consensus_expert_roi" if "fw_consensus_expert_roi" in df else "consensus_expert_roi"
        s["consensus_insertion_auc_mean"] = float(df[fw_col].mean())
        s["consensus_expert_roi_mean"] = float(df[fw_expert_col].mean())
        s["fw_consensus_insertion_auc_mean"] = float(df[fw_col].mean())
        s["fw_consensus_expert_roi_mean"] = float(df[fw_expert_col].mean())
        for prefix in ["uniform_consensus", "topk_consensus"]:
            ins_col = f"{prefix}_insertion_auc"
            expert_col = f"{prefix}_expert_roi"
            if ins_col in df:
                s[f"{ins_col}_mean"] = float(df[ins_col].mean())
            if expert_col in df:
                s[f"{expert_col}_mean"] = float(df[expert_col].mean())
    return s


def _args_dict(args) -> dict:
    return {k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()}


if __name__ == "__main__":
    main()
