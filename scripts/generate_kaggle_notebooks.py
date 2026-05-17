"""Generate Kaggle notebooks for the 5-fold append and CUB-200 tracks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SPINE_MODELS = {
    "convnext_blackbox": "ConvNeXt Black-Box",
    "cbm_nonleaky": "CBM Non-Leaky",
    "cbm_leaky": "CBM Leaky",
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b4": "EfficientNet-B4",
    "vit_small": "ViT-Small",
    "deit_small": "DeiT-Small",
}

CUB_MODELS = {
    "convnext_blackbox": "ConvNeXt Black-Box",
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
    "efficientnet_b4": "EfficientNet-B4",
    "vit_small": "ViT-Small",
    "deit_small": "DeiT-Small",
}


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


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


SPINE_SETUP_SUFFIX = r'''
METHODS = [
    "gradcam",
    "gradcam++",
    "integrated_gradients",
    "gradient_shap",
    "occlusion",
    "guided_backprop",
]

CFG_NAME = {
    "convnext_blackbox": "configs/baselines/convnext_blackbox.yaml",
    "cbm_nonleaky": "configs/baselines/cbm_nonleaky.yaml",
    "cbm_leaky": "configs/baselines/cbm_leaky.yaml",
    "resnet50": "configs/baselines/resnet50.yaml",
    "densenet121": "configs/baselines/densenet121.yaml",
    "efficientnet_b4": "configs/baselines/efficientnet_b4.yaml",
    "vit_small": "configs/baselines/vit_small.yaml",
    "deit_small": "configs/baselines/deit_small.yaml",
}
EXP = {
    "convnext_blackbox": "baseline_convnext_tiny_blackbox",
    "cbm_nonleaky": "cbm_nonleaky_7concepts",
    "cbm_leaky": "cbm_leaky_5concepts",
    "resnet50": "baseline_resnet50",
    "densenet121": "baseline_densenet121",
    "efficientnet_b4": "baseline_efficientnet_b4",
    "vit_small": "baseline_vit_small",
    "deit_small": "baseline_deit_small",
}
BATCH = {
    "convnext_blackbox": 32,
    "cbm_nonleaky": 32,
    "cbm_leaky": 32,
    "resnet50": 32,
    "densenet121": 32,
    "efficientnet_b4": 16,
    "vit_small": 16,
    "deit_small": 16,
}
ACCUM = {m: max(1, 32 // BATCH[m]) for m in BATCH}

def run(cmd, env=None):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, env=env)

def input_roots():
    roots = [WORK]
    if INPUT.exists():
        roots += [p for p in INPUT.iterdir() if p.is_dir()]
        datasets = INPUT / "datasets"
        if datasets.exists():
            for owner in datasets.iterdir():
                if owner.is_dir():
                    roots += [p for p in owner.iterdir() if p.is_dir()]
    return roots

def looks_like_code(p):
    return (
        (p / "scripts" / "train.py").exists()
        and (p / "configs").exists()
        and (p / "scripts" / "feature_map_smoothness.py").exists()
    )

def find_code_source():
    candidates = [
        INPUT / "spinexnet-code",
        INPUT / "spinexnet-code" / "het-spine",
        INPUT / "datasets" / "vasuaashadesai" / "spinexnet-code",
    ]
    for r in input_roots():
        candidates += [r, r / "het-spine", r / "spinexnet-code"]
    for p in candidates:
        if looks_like_code(p):
            return p
    raise FileNotFoundError("Could not find spinexnet-code with scripts/train.py")

def first_existing(candidates, label):
    for p in map(Path, candidates):
        if p.exists():
            return p
    preview = [str(p) for p in candidates[:10]]
    raise FileNotFoundError(f"Could not find {label}. Tried: {preview}")

def find_manifest():
    candidates = [
        INPUT / "manifests-of-spinexnet" / "manifests" / "manifest_v2.csv",
        INPUT / "datasets" / "vasuaashadesai" / "manifests-of-spinexnet" / "manifests" / "manifest_v2.csv",
    ]
    for r in input_roots():
        candidates += [r / "manifest_v2.csv", r / "manifests" / "manifest_v2.csv"]
    return first_existing(candidates, "manifest_v2.csv")

def find_cache():
    candidates = [
        INPUT / "pre-processed-crop-224" / "image_cache_224",
        INPUT / "datasets" / "vasuaashadesai" / "pre-processed-crop-224" / "image_cache_224",
    ]
    for r in input_roots():
        candidates += [r / "image_cache_224", r / "pre-processed-crop-224" / "image_cache_224"]
    candidates = [p for p in candidates if (Path(p) / "images_uint8.npy").exists()]
    return first_existing(candidates, "image_cache_224")

def gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0

def run_jobs(jobs, parallel_if_2gpu=True):
    """Run [(name, cmd), ...]. On multi-GPU Kaggle, run up to one job per GPU."""
    n_gpu = gpu_count()
    if parallel_if_2gpu and n_gpu >= 2 and len(jobs) > 1:
        print(f"Detected {n_gpu} GPUs; running up to {n_gpu} jobs concurrently.", flush=True)
        for start in range(0, len(jobs), n_gpu):
            group = jobs[start:start+n_gpu]
            procs = []
            for local_idx, (name, cmd) in enumerate(group):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(local_idx)
                env["PYTHONUNBUFFERED"] = "1"
                print("\n$", " ".join(map(str, cmd)), f"  # {name} on visible GPU {local_idx}", flush=True)
                procs.append((name, subprocess.Popen(list(map(str, cmd)), env=env)))
            failures = []
            for name, proc in procs:
                rc = proc.wait()
                if rc != 0:
                    failures.append((name, rc))
            if failures:
                raise subprocess.CalledProcessError(failures[0][1], failures[0][0])
    else:
        if len(jobs) > 1:
            print(f"Detected {n_gpu} GPU(s); running jobs sequentially.", flush=True)
        for name, cmd in jobs:
            run(cmd)

def safe_same_path(a, b):
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return False

def merge_previous_outputs():
    """Copy prior per-model Kaggle output datasets into /kaggle/working.

    Attach the previous output dataset for this same notebook. Existing files in
    /kaggle/working win, so reruns can resume safely.
    """
    for tree_name in ["outputs", "eval_multifold", "xai_multifold"]:
        dest = WORK / tree_name
        for root in input_roots():
            src = root / tree_name
            if src.exists() and not safe_same_path(src, dest):
                print(f"Merging prior {tree_name}: {src} -> {dest}", flush=True)
                shutil.copytree(src, dest, dirs_exist_ok=True)
    for pattern in ["classification_metrics_*", "xai_multifold_summary_*"]:
        for root in input_roots():
            for src in root.glob(pattern):
                if src.is_file() and not safe_same_path(src, WORK / src.name):
                    shutil.copy2(src, WORK / src.name)

SRC = find_code_source()
CODE = WORK / "spinexnet-code"
if SRC.resolve() != CODE.resolve():
    shutil.copytree(SRC, CODE, dirs_exist_ok=True)

MANIFEST = find_manifest()
CACHE = find_cache()
CFG = {m: CODE / rel for m, rel in CFG_NAME.items()}
merge_previous_outputs()

def find_checkpoint(model, fold):
    rels = [
        Path("outputs") / EXP[model] / f"fold_{fold}" / "best.pt",
        Path(EXP[model]) / f"fold_{fold}" / "best.pt",
        Path("checkpoints") / f"{model}_fold_{fold}_best.pt",
        Path("checkpoints") / f"{model}_fold{fold}_best.pt",
    ]
    if fold == 0:
        rels.append(Path("checkpoints") / f"{model}_best.pt")
    candidates = []
    for r in input_roots():
        candidates += [r / rel for rel in rels]
    candidates += [WORK / rel for rel in rels]
    return first_existing(candidates, f"{model} fold {fold} checkpoint")

print("MODEL:", SELECTED_MODEL)
print("CODE:", CODE)
print("MANIFEST:", MANIFEST)
print("CACHE:", CACHE)
print("GPUs:", gpu_count())
'''


SPINE_TRAIN_CELL = r'''
model = SELECTED_MODEL
jobs = []
for fold in FOLDS_TO_TRAIN:
    out = WORK / "outputs" / EXP[model] / f"fold_{fold}"
    if (out / "best.pt").exists():
        print("Skip existing:", out / "best.pt")
        continue
    jobs.append((
        f"train {model} fold {fold}",
        [
            PY, CODE / "scripts/train.py",
            "--config", CFG[model],
            "--manifest", MANIFEST,
            "--fold", fold,
            "--cache-dir", CACHE,
            "--output-dir", out,
            "--batch-size", BATCH[model],
            "--grad-accum", ACCUM[model],
            "--time-limit-minutes", 500,
            "--checkpoint-every-minutes", 20,
            "--heartbeat-every-minutes", 5,
            "--no-data-parallel",
        ],
    ))

run_jobs(jobs, parallel_if_2gpu=PARALLEL_FOLDS_IF_2GPU)
'''


SPINE_EVAL_CELL = r'''
model = SELECTED_MODEL
eval_root = WORK / "eval_multifold"
for fold in EVAL_FOLDS:
    ckpt = find_checkpoint(model, fold)
    out = eval_root / model / f"fold_{fold}"
    if (out / "metrics_val.json").exists():
        print("Skip existing:", out / "metrics_val.json")
        continue
    run([
        PY, CODE / "scripts/evaluate.py",
        "--config", CFG[model],
        "--checkpoint", ckpt,
        "--manifest", MANIFEST,
        "--fold", fold,
        "--cache-dir", CACHE,
        "--output-dir", out,
    ])

rows = []
for path in (eval_root / model).glob("fold_*/metrics_val.json"):
    with open(path) as f:
        d = json.load(f)
    rows.append({"model": model, "fold": path.parent.name, **d})

df = pd.DataFrame(rows)
if not df.empty:
    df["fold_idx"] = df["fold"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["model", "fold_idx"]).drop(columns=["fold_idx"])
    df.to_csv(WORK / f"classification_metrics_{model}_by_fold.csv", index=False)
    metrics = ["weighted_log_loss", "balanced_accuracy", "macro_f1", "accuracy", "auc_ovr"]
    summary = df.groupby("model")[metrics].agg(["mean", "std"]).round(4)
    summary.to_csv(WORK / f"classification_metrics_{model}_mean_std.csv")
    display(summary)
else:
    print("No evaluation metrics found yet.")
'''


SPINE_XAI_CELL = r'''
model = SELECTED_MODEL
jobs = []
for fold in XAI_FOLDS:
    ckpt = find_checkpoint(model, fold)
    out = WORK / "xai_multifold" / f"fold_{fold}" / model
    if (out / "xai_summary_v2.json").exists():
        print("Skip existing:", out / "xai_summary_v2.json")
        continue
    cmd = [
        PY, CODE / "scripts/run_xai_benchmark_v2.py",
        "--config", CFG[model],
        "--checkpoint", ckpt,
        "--manifest", MANIFEST,
        "--fold", fold,
        "--cache-dir", CACHE,
        "--output-dir", out,
        "--max-samples", MAX_SAMPLES,
        "--methods", *METHODS,
        "--skip-consistency",
    ]
    if SAVE_MAPS:
        cmd.append("--save-maps")
    if not RUN_CONSENSUS:
        cmd.append("--skip-consensus")
    if model in {"vit_small", "deit_small"}:
        cmd.append("--enable-attention-rollout")
    jobs.append((f"xai {model} fold {fold}", cmd))

run_jobs(jobs, parallel_if_2gpu=PARALLEL_FOLDS_IF_2GPU)

rows = []
for path in (WORK / "xai_multifold").glob(f"fold_*/{model}/xai_summary_v2.json"):
    with open(path) as f:
        d = json.load(f)["summary"]
    rows.append({
        "fold": path.parents[1].name,
        "model": model,
        "mean_spearman": d.get("mean_spearman"),
        "mean_top20_iou": d.get("mean_top20_iou"),
        "consensus_insertion_auc_mean": d.get("consensus_insertion_auc_mean"),
        "consensus_expert_roi_mean": d.get("consensus_expert_roi_mean"),
    })

xai_summary = pd.DataFrame(rows)
if not xai_summary.empty:
    xai_summary["fold_idx"] = xai_summary["fold"].str.extract(r"(\d+)").astype(int)
    xai_summary = xai_summary.sort_values(["model", "fold_idx"]).drop(columns=["fold_idx"])
    xai_summary.to_csv(WORK / f"xai_multifold_summary_{model}.csv", index=False)
    display(xai_summary)
else:
    print("No XAI summaries found yet.")
'''


AGG_SETUP = r'''%pip install -q timm pydicom captum grad-cam scikit-image scipy seaborn

from pathlib import Path
import json, os, shutil, subprocess, sys
import pandas as pd

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
PY = sys.executable

MODELS = ["convnext_blackbox", "cbm_nonleaky", "cbm_leaky", "resnet50", "densenet121", "efficientnet_b4", "vit_small", "deit_small"]
THEORY_MODELS = ["resnet50", "densenet121", "convnext_blackbox", "efficientnet_b4", "vit_small", "deit_small"]

CFG_NAME = {
    "convnext_blackbox": "configs/baselines/convnext_blackbox.yaml",
    "cbm_nonleaky": "configs/baselines/cbm_nonleaky.yaml",
    "cbm_leaky": "configs/baselines/cbm_leaky.yaml",
    "resnet50": "configs/baselines/resnet50.yaml",
    "densenet121": "configs/baselines/densenet121.yaml",
    "efficientnet_b4": "configs/baselines/efficientnet_b4.yaml",
    "vit_small": "configs/baselines/vit_small.yaml",
    "deit_small": "configs/baselines/deit_small.yaml",
}
EXP = {
    "convnext_blackbox": "baseline_convnext_tiny_blackbox",
    "cbm_nonleaky": "cbm_nonleaky_7concepts",
    "cbm_leaky": "cbm_leaky_5concepts",
    "resnet50": "baseline_resnet50",
    "densenet121": "baseline_densenet121",
    "efficientnet_b4": "baseline_efficientnet_b4",
    "vit_small": "baseline_vit_small",
    "deit_small": "baseline_deit_small",
}

def run(cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)

def input_roots():
    roots = [WORK]
    if INPUT.exists():
        roots += [p for p in INPUT.iterdir() if p.is_dir()]
        datasets = INPUT / "datasets"
        if datasets.exists():
            for owner in datasets.iterdir():
                if owner.is_dir():
                    roots += [p for p in owner.iterdir() if p.is_dir()]
    return roots

def looks_like_code(p):
    return (
        (p / "scripts" / "train.py").exists()
        and (p / "configs").exists()
        and (p / "scripts" / "feature_map_smoothness.py").exists()
    )

def first_existing(candidates, label):
    for p in map(Path, candidates):
        if p.exists():
            return p
    preview = [str(p) for p in candidates[:10]]
    raise FileNotFoundError(f"Could not find {label}. Tried: {preview}")

def find_code_source():
    candidates = [INPUT / "spinexnet-code", INPUT / "spinexnet-code" / "het-spine"]
    for r in input_roots():
        candidates += [r, r / "het-spine", r / "spinexnet-code"]
    for p in candidates:
        if looks_like_code(p):
            return p
    raise FileNotFoundError("Could not find spinexnet-code")

def find_manifest():
    candidates = []
    for r in input_roots():
        candidates += [r / "manifest_v2.csv", r / "manifests" / "manifest_v2.csv"]
    return first_existing(candidates, "manifest_v2.csv")

def find_cache():
    candidates = []
    for r in input_roots():
        candidates += [r / "image_cache_224", r / "pre-processed-crop-224" / "image_cache_224"]
    candidates = [p for p in candidates if (p / "images_uint8.npy").exists()]
    return first_existing(candidates, "image_cache_224")

SRC = find_code_source()
CODE = WORK / "spinexnet-code"
if SRC.resolve() != CODE.resolve():
    shutil.copytree(SRC, CODE, dirs_exist_ok=True)
MANIFEST = find_manifest()
CACHE = find_cache()
CFG = {m: CODE / rel for m, rel in CFG_NAME.items()}

def find_checkpoint(model, fold):
    rels = [
        Path("outputs") / EXP[model] / f"fold_{fold}" / "best.pt",
        Path(EXP[model]) / f"fold_{fold}" / "best.pt",
        Path("checkpoints") / f"{model}_fold_{fold}_best.pt",
        Path("checkpoints") / f"{model}_fold{fold}_best.pt",
    ]
    if fold == 0:
        rels.append(Path("checkpoints") / f"{model}_best.pt")
    candidates = []
    for r in input_roots():
        candidates += [r / rel for rel in rels]
    candidates += [WORK / rel for rel in rels]
    return first_existing(candidates, f"{model} fold {fold} checkpoint")

def merge_tree_named(name):
    dest = WORK / f"combined_{name}"
    for r in input_roots():
        src = r / name
        if src.exists():
            try:
                if src.resolve() == dest.resolve():
                    continue
            except Exception:
                pass
            shutil.copytree(src, dest, dirs_exist_ok=True)
    return dest

EVAL_ROOT = merge_tree_named("eval_multifold")
XAI_ROOT = merge_tree_named("xai_multifold")
print("CODE:", CODE)
print("MANIFEST:", MANIFEST)
print("CACHE:", CACHE)
print("EVAL_ROOT:", EVAL_ROOT)
print("XAI_ROOT:", XAI_ROOT)
'''


AGG_CLASSIFICATION = r'''
rows = []
for path in EVAL_ROOT.glob("*/fold_*/metrics_val.json"):
    with open(path) as f:
        d = json.load(f)
    rows.append({"model": path.parents[1].name, "fold": path.parent.name, **d})

classification = pd.DataFrame(rows)
if classification.empty:
    print("No classification metrics found.")
else:
    classification["fold_idx"] = classification["fold"].str.extract(r"(\d+)").astype(int)
    classification = classification.sort_values(["model", "fold_idx"]).drop(columns=["fold_idx"])
    classification.to_csv(WORK / "classification_metrics_by_fold.csv", index=False)
    metrics = ["weighted_log_loss", "balanced_accuracy", "macro_f1", "accuracy", "auc_ovr"]
    summary = classification.groupby("model")[metrics].agg(["mean", "std"]).round(4)
    summary.to_csv(WORK / "classification_metrics_mean_std.csv")
    display(summary)
'''


AGG_XAI_SUMMARY = r'''
rows = []
for path in XAI_ROOT.glob("fold_*/*/xai_summary_v2.json"):
    with open(path) as f:
        d = json.load(f)["summary"]
    rows.append({
        "fold": path.parents[1].name,
        "model": path.parent.name,
        "mean_spearman": d.get("mean_spearman"),
        "mean_top20_iou": d.get("mean_top20_iou"),
        "consensus_insertion_auc_mean": d.get("consensus_insertion_auc_mean"),
        "consensus_expert_roi_mean": d.get("consensus_expert_roi_mean"),
    })

xai_summary = pd.DataFrame(rows)
if xai_summary.empty:
    print("No XAI summaries found.")
else:
    xai_summary["fold_idx"] = xai_summary["fold"].str.extract(r"(\d+)").astype(int)
    xai_summary = xai_summary.sort_values(["model", "fold_idx"]).drop(columns=["fold_idx"])
    xai_summary.to_csv(WORK / "xai_multifold_summary_by_fold.csv", index=False)
    xai_mean_std = xai_summary.groupby("model")[[
        "mean_spearman",
        "mean_top20_iou",
        "consensus_insertion_auc_mean",
        "consensus_expert_roi_mean",
    ]].agg(["mean", "std"]).round(4)
    xai_mean_std.to_csv(WORK / "xai_multifold_summary_mean_std.csv")
    display(xai_mean_std)
'''


AGG_FIGURES = r'''
FIG = WORK / "figures"
XAI_FOLD0 = XAI_ROOT / "fold_0"

for model in MODELS:
    if not (XAI_FOLD0 / model / "xai_summary_v2.json").exists():
        print("Missing XAI fold_0 for", model)
        continue
    run([PY, CODE / "scripts/visualize_xai.py", "--results-dir", XAI_FOLD0 / model, "--output-dir", FIG / "fold_0" / model])

run([PY, CODE / "scripts/visualize_xai.py", "--cross-model-dir", XAI_FOLD0, "--output-dir", FIG / "fold_0" / "cross_model"])
run([PY, CODE / "scripts/generate_gallery.py", "--xai-dir", XAI_FOLD0, "--models", "densenet121", "deit_small", "vit_small", "--manifest", MANIFEST, "--cache-dir", CACHE, "--output-dir", FIG / "fold_0" / "gallery"])
'''


AGG_THEORY = r'''
run([
    PY, CODE / "scripts/feature_map_smoothness.py",
    "--configs", *[CFG[m] for m in THEORY_MODELS],
    "--checkpoints", *[find_checkpoint(m, 0) for m in THEORY_MODELS],
    "--names", *THEORY_MODELS,
    "--manifest", MANIFEST,
    "--fold", 0,
    "--cache-dir", CACHE,
    "--output-dir", WORK / "feature_map_smoothness",
    "--max-samples", 300,
    "--agreement-csv", FIG / "fold_0" / "cross_model" / "cross_model_agreement.csv",
])
'''


AGG_RANDOMIZATION_INTERVENTION = r'''
for model in ["convnext_blackbox", "vit_small", "deit_small"]:
    run([
        PY, CODE / "scripts/model_randomization.py",
        "--config", CFG[model],
        "--checkpoint", find_checkpoint(model, 0),
        "--manifest", MANIFEST,
        "--fold", 0,
        "--cache-dir", CACHE,
        "--output-dir", WORK / "randomization" / model,
        "--methods", "gradcam", "integrated_gradients", "gradient_shap", "occlusion",
        "--max-samples", 50,
        "--n-levels", 5,
    ])

for model in ["cbm_nonleaky", "cbm_leaky"]:
    run([
        PY, CODE / "scripts/concept_intervention.py",
        "--config", CFG[model],
        "--checkpoint", find_checkpoint(model, 0),
        "--manifest", MANIFEST,
        "--fold", 0,
        "--cache-dir", CACHE,
        "--output-dir", WORK / "intervention" / model,
        "--max-samples", 1000,
    ])
'''


CUB_SETUP_SUFFIX = r'''
from pathlib import Path
import json, os, shutil, subprocess, sys
import pandas as pd

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
PY = sys.executable

FOLDS = [0, 1, 2, 3, 4]
XAI_FOLDS = [0, 1, 2, 3, 4]
MAX_SAMPLES = 300
PARALLEL_FOLDS_IF_2GPU = True
SAVE_MAPS = True
RUN_FAITHFULNESS = True

BATCH = {
    "convnext_blackbox": 32,
    "resnet50": 32,
    "densenet121": 32,
    "efficientnet_b4": 16,
    "vit_small": 16,
    "deit_small": 16,
}
EPOCHS = {
    "convnext_blackbox": 35,
    "resnet50": 15,
    "densenet121": 15,
    "efficientnet_b4": 15,
    "vit_small": 15,
    "deit_small": 30,
}
LR = {
    "convnext_blackbox": 1e-4,
    "resnet50": 3e-4,
    "densenet121": 3e-4,
    "efficientnet_b4": 3e-4,
    "vit_small": 3e-4,
    "deit_small": 1e-4,
}
BACKBONE_LR = {
    "convnext_blackbox": 5e-5,
    "deit_small": 5e-5,
}
HEAD_LR = {
    "convnext_blackbox": 5e-4,
    "deit_small": 5e-4,
}
WARMUP_EPOCHS = {
    "convnext_blackbox": 3,
    "deit_small": 3,
}
PATIENCE = {
    "convnext_blackbox": 8,
    "deit_small": 8,
}
DROP_PATH = {
    "convnext_blackbox": 0.1,
    "deit_small": 0.1,
}
METHODS = [
    "gradcam",
    "gradcam++",
    "integrated_gradients",
    "gradient_shap",
    "occlusion",
    "guided_backprop",
]

def run(cmd, env=None):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, env=env)

def input_roots():
    roots = [WORK]
    if INPUT.exists():
        roots += [p for p in INPUT.iterdir() if p.is_dir()]
        datasets = INPUT / "datasets"
        if datasets.exists():
            for owner in datasets.iterdir():
                if owner.is_dir():
                    roots += [p for p in owner.iterdir() if p.is_dir()]
    return roots

def looks_like_code(p):
    return (p / "cub_200_generalization" / "train_cub_model.py").exists()

def find_code_source():
    candidates = [INPUT / "spinexnet-code", INPUT / "spinexnet-code" / "het-spine"]
    for r in input_roots():
        candidates += [r, r / "het-spine", r / "spinexnet-code"]
    for p in candidates:
        if looks_like_code(p):
            return p
    raise FileNotFoundError("Could not find spinexnet-code with cub_200_generalization/train_cub_model.py")

def find_cub_root():
    candidates = []
    for r in input_roots():
        candidates += [r / "CUB_200_2011"]
        candidates += list(r.glob("**/CUB_200_2011"))
    for p in candidates:
        if (p / "images.txt").exists() and (p / "images").exists():
            return p
    raise FileNotFoundError("Could not find CUB_200_2011. Attach the CUB dataset to this notebook.")

def gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0

def run_jobs(jobs, parallel_if_2gpu=True):
    n_gpu = gpu_count()
    if parallel_if_2gpu and n_gpu >= 2 and len(jobs) > 1:
        for start in range(0, len(jobs), n_gpu):
            group = jobs[start:start+n_gpu]
            procs = []
            for local_idx, (name, cmd) in enumerate(group):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(local_idx)
                env["PYTHONUNBUFFERED"] = "1"
                print("\n$", " ".join(map(str, cmd)), f"  # {name} on visible GPU {local_idx}", flush=True)
                procs.append((name, subprocess.Popen(list(map(str, cmd)), env=env)))
            failures = []
            for name, proc in procs:
                rc = proc.wait()
                if rc != 0:
                    failures.append((name, rc))
            if failures:
                raise subprocess.CalledProcessError(failures[0][1], failures[0][0])
    else:
        for name, cmd in jobs:
            run(cmd)

def safe_same_path(a, b):
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return False

def merge_previous_cub_outputs():
    for tree_name in ["cub_outputs", "cub_eval", "cub_xai"]:
        dest = WORK / tree_name
        for root in input_roots():
            src = root / tree_name
            if src.exists() and not safe_same_path(src, dest):
                print(f"Merging prior {tree_name}: {src} -> {dest}", flush=True)
                shutil.copytree(src, dest, dirs_exist_ok=True)

SRC = find_code_source()
CODE = WORK / "spinexnet-code"
if SRC.resolve() != CODE.resolve():
    shutil.copytree(SRC, CODE, dirs_exist_ok=True)
CUB_CODE = CODE / "cub_200_generalization"
CUB_ROOT = find_cub_root()
merge_previous_cub_outputs()

def find_checkpoint(model, fold):
    candidates = []
    rels = [
        Path("cub_outputs") / model / f"fold_{fold}" / "best.pt",
        Path(model) / f"fold_{fold}" / "best.pt",
    ]
    for r in input_roots():
        candidates += [r / rel for rel in rels]
    candidates += [WORK / rel for rel in rels]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find CUB checkpoint for {model} fold {fold}")

print("MODEL:", SELECTED_MODEL)
print("CODE:", CODE)
print("CUB_ROOT:", CUB_ROOT)
print("GPUs:", gpu_count())
'''


CUB_TRAIN = r'''
model = SELECTED_MODEL
jobs = []
for fold in FOLDS:
    out = WORK / "cub_outputs" / model / f"fold_{fold}"
    if (out / "best.pt").exists():
        print("Skip existing:", out / "best.pt")
        continue
    jobs.append((
        f"train {model} fold {fold}",
        [
            PY, CUB_CODE / "train_cub_model.py",
            "--model", model,
            "--cub-root", CUB_ROOT,
            "--fold", fold,
            "--output-dir", out,
            "--epochs", EPOCHS[model],
            "--batch-size", BATCH[model],
            "--lr", LR[model],
            "--weight-decay", 0.05,
            "--time-limit-minutes", 500,
        ],
    ))
    if model in BACKBONE_LR:
        jobs[-1][1].extend(["--backbone-lr", BACKBONE_LR[model]])
    if model in HEAD_LR:
        jobs[-1][1].extend(["--head-lr", HEAD_LR[model]])
    if model in WARMUP_EPOCHS:
        jobs[-1][1].extend(["--warmup-epochs", WARMUP_EPOCHS[model]])
    if model in PATIENCE:
        jobs[-1][1].extend(["--patience", PATIENCE[model]])
    if model in DROP_PATH:
        jobs[-1][1].extend(["--drop-path-rate", DROP_PATH[model]])

run_jobs(jobs, parallel_if_2gpu=PARALLEL_FOLDS_IF_2GPU)
'''


CUB_EVAL = r'''
model = SELECTED_MODEL
for fold in FOLDS:
    out = WORK / "cub_eval" / model / f"fold_{fold}"
    if (out / "metrics_val.json").exists():
        print("Skip existing:", out / "metrics_val.json")
        continue
    run([
        PY, CUB_CODE / "evaluate_cub_model.py",
        "--model", model,
        "--checkpoint", find_checkpoint(model, fold),
        "--cub-root", CUB_ROOT,
        "--fold", fold,
        "--output-dir", out,
    ])

rows = []
for path in (WORK / "cub_eval" / model).glob("fold_*/metrics_val.json"):
    with open(path) as f:
        rows.append({"model": model, "fold": path.parent.name, **json.load(f)})
df = pd.DataFrame(rows)
if not df.empty:
    df["fold_idx"] = df["fold"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values(["model", "fold_idx"]).drop(columns=["fold_idx"])
    df.to_csv(WORK / f"cub_classification_{model}_by_fold.csv", index=False)
    display(df.groupby("model")[["log_loss", "accuracy", "balanced_accuracy", "macro_f1", "top5_accuracy"]].agg(["mean", "std"]).round(4))
'''


CUB_XAI = r'''
model = SELECTED_MODEL
jobs = []
for fold in XAI_FOLDS:
    out = WORK / "cub_xai" / f"fold_{fold}" / model
    if (out / "xai_summary_cub.json").exists():
        print("Skip existing:", out / "xai_summary_cub.json")
        continue
    cmd = [
        PY, CUB_CODE / "run_cub_xai.py",
        "--model", model,
        "--checkpoint", find_checkpoint(model, fold),
        "--cub-root", CUB_ROOT,
        "--fold", fold,
        "--output-dir", out,
        "--max-samples", MAX_SAMPLES,
        "--methods", *METHODS,
    ]
    if SAVE_MAPS:
        cmd.append("--save-maps")
    if not RUN_FAITHFULNESS:
        cmd.append("--skip-faithfulness")
    if model in {"vit_small", "deit_small"}:
        cmd.append("--enable-attention-rollout")
    jobs.append((f"cub xai {model} fold {fold}", cmd))

run_jobs(jobs, parallel_if_2gpu=PARALLEL_FOLDS_IF_2GPU)

rows = []
for path in (WORK / "cub_xai").glob(f"fold_*/{model}/xai_summary_cub.json"):
    with open(path) as f:
        payload = json.load(f)
    s = payload["summary"]
    rows.append({
        "fold": path.parents[1].name,
        "model": model,
        "mean_spearman": s.get("mean_spearman"),
        "mean_top20_iou": s.get("mean_top20_iou"),
        "skipped": json.dumps(s.get("skipped", {}), sort_keys=True),
    })
xai_summary = pd.DataFrame(rows)
if not xai_summary.empty:
    xai_summary["fold_idx"] = xai_summary["fold"].str.extract(r"(\d+)").astype(int)
    xai_summary = xai_summary.sort_values(["model", "fold_idx"]).drop(columns=["fold_idx"])
    xai_summary.to_csv(WORK / f"cub_xai_summary_{model}_by_fold.csv", index=False)
    display(xai_summary)
'''


CUB_AGG = r'''%pip install -q timm captum grad-cam scikit-learn scipy seaborn

from pathlib import Path
import shutil, subprocess, sys

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
PY = sys.executable

def input_roots():
    roots = [WORK]
    if INPUT.exists():
        roots += [p for p in INPUT.iterdir() if p.is_dir()]
        datasets = INPUT / "datasets"
        if datasets.exists():
            for owner in datasets.iterdir():
                if owner.is_dir():
                    roots += [p for p in owner.iterdir() if p.is_dir()]
    return roots

def looks_like_code(p):
    return (p / "cub_200_generalization" / "aggregate_cub_results.py").exists()

def find_code_source():
    candidates = [INPUT / "spinexnet-code", INPUT / "spinexnet-code" / "het-spine"]
    for r in input_roots():
        candidates += [r, r / "het-spine", r / "spinexnet-code"]
    for p in candidates:
        if looks_like_code(p):
            return p
    raise FileNotFoundError("Could not find spinexnet-code with cub_200_generalization/aggregate_cub_results.py")

SRC = find_code_source()
CODE = WORK / "spinexnet-code"
if SRC.resolve() != CODE.resolve():
    shutil.copytree(SRC, CODE, dirs_exist_ok=True)

cmd = [
    PY,
    CODE / "cub_200_generalization" / "aggregate_cub_results.py",
    "--copy-inputs",
    "--output-dir",
    WORK / "cub_aggregate",
]
print("$", " ".join(map(str, cmd)), flush=True)
subprocess.run(list(map(str, cmd)), check=True)
'''


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def generate_spine_notebooks() -> None:
    out_dir = ROOT / "kaggle_parallel_notebooks"
    for model, title in SPINE_MODELS.items():
        setup = f'''%pip install -q timm pydicom captum grad-cam scikit-image scipy seaborn

from pathlib import Path
import json, os, shutil, subprocess, sys, time
import pandas as pd

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
PY = sys.executable

# Attach this model's previous output dataset, then run the notebook again.
SELECTED_MODEL = "{model}"
FOLDS_TO_TRAIN = [3, 4]
EVAL_FOLDS = [0, 1, 2, 3, 4]
XAI_FOLDS = [0, 1, 2, 3, 4]  # set to [3, 4] if you only want the append pass
MAX_SAMPLES = 300
RUN_CONSENSUS = True
SAVE_MAPS = True
PARALLEL_FOLDS_IF_2GPU = True
''' + SPINE_SETUP_SUFFIX
        notebook = nb([
            md(f"# {title}: Append Folds 3 and 4\n\nRun this notebook after attaching the previous output dataset for `{model}`. It merges the old folds into `/kaggle/working`, trains folds 3 and 4, then leaves one output dataset containing folds 0-4."),
            code(setup),
            code(SPINE_TRAIN_CELL),
            code(SPINE_EVAL_CELL),
            code(SPINE_XAI_CELL),
        ])
        write_json(out_dir / f"train_eval_xai_{model}.ipynb", notebook)

    aggregate = nb([
        md("# Aggregate Parallel Model Results\n\nAttach all seven per-model Kaggle output datasets. This notebook merges `eval_multifold/` and `xai_multifold/`, creates 5-fold tables, renders figures, and runs the rethought feature-map coherence metric."),
        code(AGG_SETUP),
        code(AGG_CLASSIFICATION),
        code(AGG_XAI_SUMMARY),
        code(AGG_FIGURES),
        code(AGG_THEORY),
        code(AGG_RANDOMIZATION_INTERVENTION),
    ])
    write_json(out_dir / "aggregate_parallel_results.ipynb", aggregate)


def generate_cub_notebooks() -> None:
    out_dir = ROOT / "cub_200_generalization" / "notebooks"
    for model, title in CUB_MODELS.items():
        setup = f'''%pip install -q timm captum grad-cam scikit-learn scipy seaborn

# CUB-200 per-model notebook. Run one model per Kaggle session.
SELECTED_MODEL = "{model}"
''' + CUB_SETUP_SUFFIX
        notebook = nb([
            md(f"# CUB-200: {title}\n\nTrains `{model}` on five stratified CUB-200 folds, evaluates each fold, and runs attribution agreement. Attach the updated `spinexnet-code` dataset and the CUB dataset."),
            code(setup),
            code(CUB_TRAIN),
            code(CUB_EVAL),
            code(CUB_XAI),
        ])
        write_json(out_dir / f"train_eval_xai_cub_{model}.ipynb", notebook)

    aggregate = nb([
        md("# CUB-200 Aggregate Results\n\nAttach the five per-model CUB output datasets, then run this notebook to produce the generalization tables and figures."),
        code(CUB_AGG),
    ])
    write_json(out_dir / "aggregate_cub_results.ipynb", aggregate)


def main() -> None:
    generate_spine_notebooks()
    generate_cub_notebooks()
    print("Generated Kaggle notebooks.")


if __name__ == "__main__":
    main()
