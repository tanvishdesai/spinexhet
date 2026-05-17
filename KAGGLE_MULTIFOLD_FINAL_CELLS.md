# Final Kaggle Cells: Multi-Fold SpineXNet Revision

This is the strict Tier 1 path. It skips BiomedCLIP/CBM Visual and implements
multi-fold training/evaluation for the models that are actually in the current
codebase.

Key decision:

- Do not run BiomedCLIP for the main paper. The current codebase has no
  `configs/baselines/cbm_visual.yaml`, and the project report explicitly keeps
  deterministic non-leaky concepts because BiomedCLIP was noisy.
- For the append pass, attach each model's previous output dataset, then train folds
  3 and 4. Folds 0, 1, and 2 should already exist in the attached output dataset.
- For XAI stability, run the XAI benchmark on folds 0 through 4. Existing
  summaries are skipped, so this also fills any missing fold-2 XAI output.

## Cell 0: Common setup

Run this at the top of every Kaggle notebook.

```python
%pip install -q timm pydicom captum grad-cam scikit-image scipy seaborn

from pathlib import Path
import json, os, shutil, subprocess, sys
import pandas as pd

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

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
    return (p / "scripts" / "train.py").exists() and (p / "configs").exists()

def find_code_source():
    candidates = [
        INPUT / "spinexnet-code",
        INPUT / "spinexnet-code" / "het-spine",
        INPUT / "datasets" / "vasuaashadesai" / "spinexnet-code",
    ]
    for root in input_roots():
        candidates += [root, root / "het-spine", root / "spinexnet-code"]
    for p in candidates:
        if looks_like_code(p):
            return p
    raise FileNotFoundError("Could not find spinexnet-code with scripts/train.py")

def first_existing(candidates, label):
    for p in map(Path, candidates):
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {label}. Tried: {candidates[:5]}")

def find_manifest():
    candidates = [
        INPUT / "manifests-of-spinexnet" / "manifests" / "manifest_v2.csv",
        INPUT / "datasets" / "vasuaashadesai" / "manifests-of-spinexnet" / "manifests" / "manifest_v2.csv",
    ]
    for root in input_roots():
        candidates += [root / "manifest_v2.csv", root / "manifests" / "manifest_v2.csv"]
    return first_existing(candidates, "manifest_v2.csv")

def find_cache():
    candidates = [
        INPUT / "pre-processed-crop-224" / "image_cache_224",
        INPUT / "datasets" / "vasuaashadesai" / "pre-processed-crop-224" / "image_cache_224",
    ]
    for root in input_roots():
        candidates += [root / "image_cache_224", root / "pre-processed-crop-224" / "image_cache_224"]
    candidates = [p for p in candidates if (Path(p) / "images_uint8.npy").exists()]
    return first_existing(candidates, "image_cache_224")

SRC = find_code_source()
CODE = WORK / "spinexnet-code"
if SRC.resolve() != CODE.resolve():
    shutil.copytree(SRC, CODE, dirs_exist_ok=True)

MANIFEST = find_manifest()
CACHE = find_cache()

CFG = {
    "convnext_blackbox": CODE / "configs/baselines/convnext_blackbox.yaml",
    "cbm_nonleaky": CODE / "configs/baselines/cbm_nonleaky.yaml",
    "cbm_leaky": CODE / "configs/baselines/cbm_leaky.yaml",
    "resnet50": CODE / "configs/baselines/resnet50.yaml",
    "densenet121": CODE / "configs/baselines/densenet121.yaml",
    "efficientnet_b4": CODE / "configs/baselines/efficientnet_b4.yaml",
    "vit_small": CODE / "configs/baselines/vit_small.yaml",
    "deit_small": CODE / "configs/baselines/deit_small.yaml",
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
MODELS = list(CFG)

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
    for root in input_roots():
        candidates += [root / rel for rel in rels]
    candidates += [WORK / rel for rel in rels]
    return first_existing(candidates, f"{model} fold {fold} checkpoint")

print("CODE:", CODE)
print("MANIFEST:", MANIFEST)
print("CACHE:", CACHE)
```

## Cell 1: Train missing folds

For full Tier 1, run with all models. For parallel Kaggle notebooks, edit
`TRAIN_MODELS` so each notebook owns a different subset.

```python
TRAIN_MODELS = [
    "convnext_blackbox",
    "cbm_nonleaky",
    "cbm_leaky",
    "resnet50",
    "densenet121",
    "efficientnet_b4",
    "vit_small",
    "deit_small",
]
FOLDS_TO_TRAIN = [3, 4]

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
ACCUM = {m: (32 // BATCH[m]) for m in BATCH}

for model in TRAIN_MODELS:
    for fold in FOLDS_TO_TRAIN:
        out = WORK / "outputs" / EXP[model] / f"fold_{fold}"
        if (out / "best.pt").exists():
            print("Skip existing:", out / "best.pt")
            continue
        run([
            sys.executable, CODE / "scripts/train.py",
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
        ])
```

## Cell 2: Evaluate folds 0 to 4 and create the classification table

Run this after the previous output dataset has been merged and fold 3/4 checkpoints are available in `/kaggle/working`.

```python
EVAL_MODELS = MODELS
EVAL_FOLDS = [0, 1, 2, 3, 4]
eval_root = WORK / "eval_multifold"

for model in EVAL_MODELS:
    for fold in EVAL_FOLDS:
        ckpt = find_checkpoint(model, fold)
        out = eval_root / model / f"fold_{fold}"
        if (out / "metrics_val.json").exists():
            continue
        run([
            sys.executable, CODE / "scripts/evaluate.py",
            "--config", CFG[model],
            "--checkpoint", ckpt,
            "--manifest", MANIFEST,
            "--fold", fold,
            "--cache-dir", CACHE,
            "--output-dir", out,
        ])

rows = []
for path in eval_root.glob("*/fold_*/metrics_val.json"):
    with open(path) as f:
        d = json.load(f)
    rows.append({"model": path.parents[1].name, "fold": path.parent.name, **d})

df = pd.DataFrame(rows).sort_values(["model", "fold"])
df.to_csv(WORK / "classification_metrics_by_fold.csv", index=False)

metrics = ["weighted_log_loss", "balanced_accuracy", "macro_f1", "accuracy", "auc_ovr"]
summary = df.groupby("model")[metrics].agg(["mean", "std"]).round(4)
summary.to_csv(WORK / "classification_metrics_mean_std.csv")
display(summary)
```

## Cell 3: XAI benchmark v2 on folds 0 to 4

Use `XAI_FOLDS = [3, 4]` only for a quick append pass. The default below creates a complete 5-fold XAI table and skips existing summaries.

```python
XAI_MODELS = MODELS
XAI_FOLDS = [0, 1, 2, 3, 4]
MAX_SAMPLES = 300
METHODS = [
    "gradcam",
    "gradcam++",
    "integrated_gradients",
    "gradient_shap",
    "occlusion",
    "guided_backprop",
]
RUN_CONSENSUS = True
SAVE_MAPS = True

for fold in XAI_FOLDS:
    for model in XAI_MODELS:
        ckpt = find_checkpoint(model, fold)
        out = WORK / "xai_multifold" / f"fold_{fold}" / model
        if (out / "xai_summary_v2.json").exists():
            print("Skip existing:", out)
            continue
        cmd = [
            sys.executable, CODE / "scripts/run_xai_benchmark_v2.py",
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
        run(cmd)

rows = []
for path in (WORK / "xai_multifold").glob("fold_*/*/xai_summary_v2.json"):
    fold = path.parents[1].name
    model = path.parent.name
    with open(path) as f:
        d = json.load(f)["summary"]
    rows.append({
        "fold": fold,
        "model": model,
        "mean_spearman": d.get("mean_spearman"),
        "mean_top20_iou": d.get("mean_top20_iou"),
        "consensus_insertion_auc_mean": d.get("consensus_insertion_auc_mean"),
        "consensus_expert_roi_mean": d.get("consensus_expert_roi_mean"),
    })

xai_summary = pd.DataFrame(rows).sort_values(["model", "fold"])
xai_summary.to_csv(WORK / "xai_multifold_summary.csv", index=False)
display(xai_summary)
```

## Cell 4: Figures for fold 0 and qualitative gallery

```python
FIG = WORK / "figures"
XAI_FOLD0 = WORK / "xai_multifold" / "fold_0"

for model in XAI_MODELS:
    run([
        sys.executable, CODE / "scripts/visualize_xai.py",
        "--results-dir", XAI_FOLD0 / model,
        "--output-dir", FIG / "fold_0" / model,
    ])

run([
    sys.executable, CODE / "scripts/visualize_xai.py",
    "--cross-model-dir", XAI_FOLD0,
    "--output-dir", FIG / "fold_0" / "cross_model",
])

run([
    sys.executable, CODE / "scripts/generate_gallery.py",
    "--xai-dir", XAI_FOLD0,
    "--models", "densenet121", "convnext_blackbox", "vit_small",
    "--manifest", MANIFEST,
    "--cache-dir", CACHE,
    "--output-dir", FIG / "fold_0" / "gallery",
])
```

## Cell 5: Feature-Map Coherence

```python
THEORY_MODELS = ["resnet50", "densenet121", "convnext_blackbox", "efficientnet_b4", "vit_small", "deit_small"]
fold = 0

run([
    sys.executable, CODE / "scripts/feature_map_smoothness.py",
    "--configs", *[CFG[m] for m in THEORY_MODELS],
    "--checkpoints", *[find_checkpoint(m, fold) for m in THEORY_MODELS],
    "--names", *THEORY_MODELS,
    "--manifest", MANIFEST,
    "--fold", fold,
    "--cache-dir", CACHE,
    "--output-dir", WORK / "feature_map_smoothness",
    "--max-samples", 300,
    "--agreement-csv", FIG / "fold_0" / "cross_model" / "cross_model_agreement.csv",
])
```

## Cell 6: Randomization sanity check

```python
for model in ["convnext_blackbox", "vit_small", "deit_small"]:
    run([
        sys.executable, CODE / "scripts/model_randomization.py",
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
```

## Cell 7: CBM concept intervention

```python
for model in ["cbm_nonleaky", "cbm_leaky"]:
    run([
        sys.executable, CODE / "scripts/concept_intervention.py",
        "--config", CFG[model],
        "--checkpoint", find_checkpoint(model, 0),
        "--manifest", MANIFEST,
        "--fold", 0,
        "--cache-dir", CACHE,
        "--output-dir", WORK / "intervention" / model,
        "--max-samples", 1000,
    ])
```

## Parallel execution plan

Use separate Kaggle GPU notebooks for parallelism. Do not launch many GPU
processes inside one notebook unless you manually pin each process to a
different GPU.

Recommended parallel split:

1. Notebook A: `TRAIN_MODELS = ["convnext_blackbox", "cbm_nonleaky", "cbm_leaky"]`
2. Notebook B: `TRAIN_MODELS = ["resnet50", "densenet121"]`
3. Notebook C: `TRAIN_MODELS = ["efficientnet_b4", "vit_small", "deit_small"]`
4. After those outputs are saved as a Kaggle dataset, run Cell 2.
5. Run Cell 3 either sequentially or split by `XAI_MODELS` across notebooks.
6. Cells 4 to 7 can run after the needed XAI/checkpoint inputs exist. Cell 4 is
   CPU-safe. Cells 5 to 7 use GPU but are much shorter than training/XAI.
