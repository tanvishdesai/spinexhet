"""Create targeted V4 repair notebooks for Kaggle.

The generated notebooks avoid retraining. They load existing checkpoints and
repair only the remaining numerical gaps flagged by the V4 mentor reviews.
"""

from __future__ import annotations

import json
from pathlib import Path


def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip().splitlines(keepends=True)}


DEPENDENCY_SETUP = r'''
%pip install -q timm pydicom captum grad-cam scikit-image scipy seaborn
'''


RSNA_NOTEBOOK = r'''
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PY = sys.executable
WORK = Path("/kaggle/working")

def run(cmd):
    cmd = [str(x) for x in cmd]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def input_roots():
    roots = [WORK]
    root = Path("/kaggle/input")
    if root.exists():
        roots += [p for p in root.iterdir() if p.is_dir()]
        datasets = root / "datasets"
        if datasets.exists():
            for owner in datasets.iterdir():
                if owner.is_dir():
                    roots += [p for p in owner.iterdir() if p.is_dir()]
    return roots

def find_code_source():
    candidates = [WORK / "spinexnet-code", Path.cwd()]
    for r in input_roots():
        candidates += [r / "spinexnet-code", r]
        candidates += list(r.glob("**/spinexnet-code"))
    for p in candidates:
        if (p / "scripts" / "run_xai_benchmark_v2.py").exists():
            return p
    raise FileNotFoundError("Attach the updated spinexnet-code package/dataset.")

def first_existing(candidates, label, required=True):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    if required:
        raise FileNotFoundError(f"Could not find {label}. Checked: {[str(c) for c in candidates]}")
    return None

SRC = find_code_source()
CODE = WORK / "spinexnet-code"
if SRC.resolve() != CODE.resolve():
    shutil.copytree(SRC, CODE, dirs_exist_ok=True)

def find_manifest():
    candidates = [
        Path("/kaggle/input") / "manifests-of-spinexnet" / "manifests" / "manifest_v2.csv",
        Path("/kaggle/input") / "datasets" / "vasuaashadesai" / "manifests-of-spinexnet" / "manifests" / "manifest_v2.csv",
    ]
    for r in input_roots():
        candidates += [
            r / "manifest_v2.csv",
            r / "manifests" / "manifest_v2.csv",
            r / "manifest_with_concepts.csv",
            r / "concepts" / "manifests" / "manifest_with_concepts.csv",
        ]
    return first_existing(candidates, "RSNA manifest_v2.csv")

def find_cache():
    candidates = [
        Path("/kaggle/input") / "pre-processed-crop-224" / "image_cache_224",
        Path("/kaggle/input") / "datasets" / "vasuaashadesai" / "pre-processed-crop-224" / "image_cache_224",
    ]
    for r in input_roots():
        candidates += [
            r / "image_cache_224",
            r / "cache_224",
            r / "image_cache",
            r / "pre-processed-crop-224" / "image_cache_224",
        ]
    cache_candidates = [p for p in candidates if (Path(p) / "images_uint8.npy").exists()]
    return first_existing(cache_candidates, "RSNA image_cache_224", required=False)

def find_rsna_result_root():
    candidates = []
    for r in input_roots():
        candidates += [
            r / "rsna" / "dataset_no_npy",
            r / "v4 results" / "rsna" / "dataset_no_npy",
            r,
        ]
    candidates = [
        p for p in candidates
        if (Path(p) / "combined_xai_multifold").exists()
        or (Path(p) / "combined_eval_multifold").exists()
    ]
    return first_existing(candidates, "original V4 RSNA results root", required=False)

def find_tree(name):
    candidates = []
    if ORIGINAL_RSNA_ROOT:
        candidates.append(ORIGINAL_RSNA_ROOT / name)
    for r in input_roots():
        candidates += [r / name, r / "rsna" / "dataset_no_npy" / name, r / "v4 results" / "rsna" / "dataset_no_npy" / name]
    return first_existing(candidates, name, required=False)

MANIFEST = find_manifest()
CACHE = find_cache()
ORIGINAL_RSNA_ROOT = find_rsna_result_root()
ORIGINAL_XAI_ROOT = find_tree("combined_xai_multifold")
EVAL_ROOT = find_tree("combined_eval_multifold")

REPAIR_ROOT = WORK / "v4_repair_rsna"
REPAIR_XAI_ROOT = REPAIR_ROOT / "combined_xai_multifold"
AUDIT_ROOT = REPAIR_ROOT / "efficientnet_gradient_audit"
AGG_ROOT = REPAIR_ROOT / "rsna_aggregate"
FIG_ROOT = REPAIR_ROOT / "figures"
for p in [REPAIR_XAI_ROOT, AUDIT_ROOT, AGG_ROOT, FIG_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES_XAI = 300
AUDIT_SAMPLES = 48
FAITHFULNESS_STEPS = 20
GRADIENT_BASELINE_MODE = "mean"      # historical neutral baseline in normalized space
FAITHFULNESS_BASELINE_MODE = "mean"  # keep old metric baseline unless you intentionally change the protocol

print("CODE:", CODE)
print("MANIFEST:", MANIFEST)
print("CACHE:", CACHE)
print("ORIGINAL_RSNA_ROOT:", ORIGINAL_RSNA_ROOT)
print("REPAIR_ROOT:", REPAIR_ROOT)
'''


RSNA_CHECKPOINTS = r'''
EXP = {
    "efficientnet_b4": "baseline_efficientnet_b4",
    "vit_small": "baseline_vit_small",
    "densenet121": "baseline_densenet121",
    "deit_small": "baseline_deit_small",
}

def find_checkpoint(model, fold):
    rels = [
        Path("outputs") / EXP.get(model, model) / f"fold_{fold}" / "best.pt",
        Path(EXP.get(model, model)) / f"fold_{fold}" / "best.pt",
        Path("checkpoints") / f"{model}_fold_{fold}_best.pt",
        Path("checkpoints") / f"{model}_fold{fold}_best.pt",
        Path("combined_outputs") / model / f"fold_{fold}" / "best.pt",
    ]
    candidates = []
    for root in input_roots():
        candidates += [root / rel for rel in rels]
    candidates += [WORK / rel for rel in rels]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find checkpoint for {model} fold {fold}")

for model, folds in {"efficientnet_b4": range(5), "vit_small": [0, 1], "densenet121": [0]}.items():
    for fold in folds:
        print(model, "fold", fold, "=>", find_checkpoint(model, fold))
'''


RSNA_AUDIT = r'''
for fold in range(5):
    out = AUDIT_ROOT / f"fold_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    run([
        PY, CODE / "scripts" / "audit_efficientnet_gradients.py",
        "--config", CODE / "configs" / "baselines" / "efficientnet_b4.yaml",
        "--checkpoint", find_checkpoint("efficientnet_b4", fold),
        "--manifest", MANIFEST,
        "--fold", fold,
        "--output-dir", out,
        "--max-samples", AUDIT_SAMPLES,
        "--faithfulness-steps", FAITHFULNESS_STEPS,
        "--baseline-modes", "mean", "black", "gray",
        *(["--cache-dir", CACHE] if CACHE else []),
    ])

audit_parts = []
for path in AUDIT_ROOT.glob("fold_*/efficientnet_gradient_audit_summary_fold_*.csv"):
    df = pd.read_csv(path)
    df["fold"] = path.parent.name
    audit_parts.append(df)
audit_summary = pd.concat(audit_parts, ignore_index=True)
audit_summary.to_csv(AUDIT_ROOT / "efficientnet_gradient_audit_summary_all_folds.csv", index=False)
display(audit_summary)
'''


RSNA_RERUN = r'''
METHODS = ["gradcam", "gradcam++", "integrated_gradients", "gradient_shap", "occlusion", "guided_backprop"]

def run_rsna_xai(model, fold, enable_attention=False):
    out = REPAIR_XAI_ROOT / f"fold_{fold}" / model
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, CODE / "scripts" / "run_xai_benchmark_v2.py",
        "--config", CODE / "configs" / "baselines" / f"{model}.yaml",
        "--checkpoint", find_checkpoint(model, fold),
        "--manifest", MANIFEST,
        "--fold", fold,
        "--output-dir", out,
        "--max-samples", MAX_SAMPLES_XAI,
        "--methods", *METHODS,
        "--faithfulness-steps", FAITHFULNESS_STEPS,
        "--skip-consistency",
        "--gradient-baseline-mode", GRADIENT_BASELINE_MODE,
        "--faithfulness-baseline-mode", FAITHFULNESS_BASELINE_MODE,
        "--consensus-every", 10,
        "--topk-consensus-methods", 3,
    ]
    if CACHE:
        cmd += ["--cache-dir", CACHE]
    if enable_attention:
        cmd += ["--enable-attention-rollout"]
    run(cmd)

# Full EfficientNet rerun: no retraining, just corrected/audited XAI artifacts.
for fold in range(5):
    run_rsna_xai("efficientnet_b4", fold, enable_attention=False)

# ViT repair: only folds 0 and 1 had stale Attention Rollout failures.
for fold in [0, 1]:
    run_rsna_xai("vit_small", fold, enable_attention=True)
'''


RSNA_AGG_GALLERY = r'''
xai_roots = []
if ORIGINAL_XAI_ROOT and ORIGINAL_XAI_ROOT.exists():
    xai_roots.append(ORIGINAL_XAI_ROOT)
xai_roots.append(REPAIR_XAI_ROOT)

eval_root_for_agg = EVAL_ROOT if EVAL_ROOT and EVAL_ROOT.exists() else (REPAIR_ROOT / "missing_eval_root")
run([
    PY, CODE / "scripts" / "aggregate_rsna_results.py",
    "--eval-root", eval_root_for_agg,
    "--xai-roots", *xai_roots,
    "--output-dir", AGG_ROOT,
])
if (AGG_ROOT / "xai_multifold_summary_by_fold.csv").exists():
    display(pd.read_csv(AGG_ROOT / "xai_multifold_summary_by_fold.csv").query("model in ['efficientnet_b4', 'vit_small']"))
if (AGG_ROOT / "faithfulness_mean_std_by_method_flat.csv").exists():
    display(pd.read_csv(AGG_ROOT / "faithfulness_mean_std_by_method_flat.csv").query("model == 'efficientnet_b4'"))
if not (EVAL_ROOT and EVAL_ROOT.exists()):
    print("Note: original eval root was not found, so classification tables were not refreshed.")

run([
    PY, CODE / "scripts" / "generate_disagreement_gallery_v4.py",
    "--manifest", MANIFEST,
    "--fold", 0,
    "--output-dir", FIG_ROOT,
    "--models", "densenet121", "vit_small",
    "--configs",
    CODE / "configs" / "baselines" / "densenet121.yaml",
    CODE / "configs" / "baselines" / "vit_small.yaml",
    "--checkpoints",
    find_checkpoint("densenet121", 0),
    find_checkpoint("vit_small", 0),
    "--xai-root", REPAIR_XAI_ROOT,
    "--gradient-baseline-mode", GRADIENT_BASELINE_MODE,
    *(["--cache-dir", CACHE] if CACHE else []),
])

print("Repair outputs:")
print("  XAI:", REPAIR_XAI_ROOT)
print("  Audit:", AUDIT_ROOT)
print("  Aggregate:", AGG_ROOT)
print("  Figures:", FIG_ROOT)
'''


CUB_NOTEBOOK = r'''
import shutil
import subprocess
import sys
from pathlib import Path

PY = sys.executable
WORK = Path("/kaggle/working")

def run(cmd):
    cmd = [str(x) for x in cmd]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def input_roots():
    roots = [WORK]
    root = Path("/kaggle/input")
    if root.exists():
        roots += [p for p in root.iterdir() if p.is_dir()]
        datasets = root / "datasets"
        if datasets.exists():
            for owner in datasets.iterdir():
                if owner.is_dir():
                    roots += [p for p in owner.iterdir() if p.is_dir()]
    return roots

def find_code_source():
    for r in input_roots() + [Path.cwd()]:
        for p in [r, r / "spinexnet-code", *r.glob("**/spinexnet-code")]:
            if (p / "cub_200_generalization" / "run_cub_xai.py").exists():
                return p
    raise FileNotFoundError("Attach the updated spinexnet-code package/dataset.")

def first_existing(candidates, label, required=True):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    if required:
        raise FileNotFoundError(f"Could not find {label}")
    return None

SRC = find_code_source()
CODE = WORK / "spinexnet-code"
if SRC.resolve() != CODE.resolve():
    shutil.copytree(SRC, CODE, dirs_exist_ok=True)
CUB_CODE = CODE / "cub_200_generalization"

CUB_ROOT = first_existing(
    [r / "CUB_200_2011" for r in input_roots()] + [p for r in input_roots() for p in r.glob("**/CUB_200_2011")],
    "CUB_200_2011 dataset",
)
ORIGINAL_CUB_RESULT_ROOT = first_existing(
    [r / "cuba 200" / "dataset_no_npy" for r in input_roots()]
    + [r / "cub_200" / "dataset_no_npy" for r in input_roots()]
    + [r / "v4 results" / "cuba 200" / "dataset_no_npy" for r in input_roots()]
    + [
        r for r in input_roots()
        if (r / "cub_xai").exists() or (r / "cub_eval").exists() or (r / "cub_aggregate").exists()
    ],
    "original V4 CUB result root",
    required=False,
)
REPAIR_ROOT = WORK / "v4_repair_cub_consensus"
REPAIR_XAI_ROOT = REPAIR_ROOT / "cub_xai"
AGG_ROOT = REPAIR_ROOT / "cub_aggregate"
REPAIR_XAI_ROOT.mkdir(parents=True, exist_ok=True)
AGG_ROOT.mkdir(parents=True, exist_ok=True)

MODELS_TO_REPAIR = ["resnet50", "densenet121", "efficientnet_b4", "vit_small"]
FOLDS = [0, 1, 2, 3, 4]
MAX_SAMPLES_XAI = 300
FAITHFULNESS_STEPS = 20
GRADIENT_BASELINE_MODE = "mean"
FAITHFULNESS_BASELINE_MODE = "mean"

def find_checkpoint(model, fold):
    rels = [
        Path("cub_outputs") / model / f"fold_{fold}" / "best.pt",
        Path(model) / f"fold_{fold}" / "best.pt",
        Path("checkpoints") / f"{model}_fold_{fold}_best.pt",
    ]
    candidates = []
    for root in input_roots():
        candidates += [root / rel for rel in rels]
    candidates += [WORK / rel for rel in rels]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find CUB checkpoint for {model} fold {fold}")

print("CODE:", CODE)
print("CUB_ROOT:", CUB_ROOT)
print("ORIGINAL_CUB_RESULT_ROOT:", ORIGINAL_CUB_RESULT_ROOT)
print("REPAIR_ROOT:", REPAIR_ROOT)
for model in MODELS_TO_REPAIR:
    print(model, "fold_0 checkpoint:", find_checkpoint(model, 0))
'''


CUB_RERUN = r'''
METHODS = ["gradcam", "gradcam++", "integrated_gradients", "gradient_shap", "occlusion", "guided_backprop"]

for model in MODELS_TO_REPAIR:
    for fold in FOLDS:
        out = REPAIR_XAI_ROOT / f"fold_{fold}" / model
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            PY, CUB_CODE / "run_cub_xai.py",
            "--model", model,
            "--checkpoint", find_checkpoint(model, fold),
            "--cub-root", CUB_ROOT,
            "--fold", fold,
            "--output-dir", out,
            "--max-samples", MAX_SAMPLES_XAI,
            "--methods", *METHODS,
            "--faithfulness-steps", FAITHFULNESS_STEPS,
            "--consensus-every", 10,
            "--topk-consensus-methods", 3,
            "--gradient-baseline-mode", GRADIENT_BASELINE_MODE,
            "--faithfulness-baseline-mode", FAITHFULNESS_BASELINE_MODE,
        ]
        if model in {"vit_small", "deit_small"}:
            cmd += ["--enable-attention-rollout"]
        run(cmd)
'''


CUB_AGG = r'''
input_roots_for_agg = [REPAIR_ROOT]
if ORIGINAL_CUB_RESULT_ROOT and ORIGINAL_CUB_RESULT_ROOT.exists():
    input_roots_for_agg.insert(0, ORIGINAL_CUB_RESULT_ROOT)

run([
    PY, CUB_CODE / "aggregate_cub_results.py",
    "--input-roots", *input_roots_for_agg,
    "--copy-inputs",
    "--output-dir", AGG_ROOT,
])

print("CUB repair outputs:")
print("  XAI:", REPAIR_XAI_ROOT)
print("  Aggregate:", AGG_ROOT)
'''


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def main() -> None:
    rsna = nb(
        [
            md(
                """
                # V4 RSNA XAI Repair

                This notebook does not retrain models. It repairs the V4 numerical blockers:

                1. Audits EfficientNet Integrated Gradients vs GradientSHAP across folds and baseline modes.
                2. Recomputes EfficientNet XAI artifacts from existing checkpoints.
                3. Recomputes ViT-Small folds 0 and 1 with Attention Rollout enabled.
                4. Aggregates original V4 outputs plus repaired folders.
                5. Generates a Figure-1 style qualitative disagreement gallery.
                """
            ),
            code(DEPENDENCY_SETUP),
            code(RSNA_NOTEBOOK),
            md("## Locate Checkpoints"),
            code(RSNA_CHECKPOINTS),
            md("## EfficientNet IG/GradientSHAP Audit"),
            code(RSNA_AUDIT),
            md("## Recompute EfficientNet and ViT Repair XAI"),
            code(RSNA_RERUN),
            md("## Aggregate and Generate Figure 1 Gallery"),
            code(RSNA_AGG_GALLERY),
        ]
    )
    write_json(Path("kaggle_parallel_notebooks/repair_v4_rsna_xai.ipynb"), rsna)

    cub = nb(
        [
            md(
                """
                # V4 CUB Consensus Repair

                Optional but recommended. This notebook does not retrain CUB models. It reruns the four older CUB XAI
                jobs with the updated consensus code so the CUB consensus table has all six model rows.
                """
            ),
            code(DEPENDENCY_SETUP),
            code(CUB_NOTEBOOK),
            md("## Rerun CUB XAI With Consensus"),
            code(CUB_RERUN),
            md("## Aggregate Original + Repaired CUB Outputs"),
            code(CUB_AGG),
        ]
    )
    write_json(Path("cub_200_generalization/notebooks/repair_v4_cub_consensus.ipynb"), cub)


if __name__ == "__main__":
    main()
