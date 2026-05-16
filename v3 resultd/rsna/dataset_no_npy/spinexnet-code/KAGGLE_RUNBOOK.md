# SpineXNet BMVC Revision — Kaggle Runbook v2

> **Last updated:** 2026-05-16
> **Goal:** Complete the full Tier 1–3 improvement pipeline per the BMVC revision roadmap.

---

## Overview of Changes from v1

This runbook implements all improvements from `SpineXNet_BMVC_Improvement_Plan.md`:

| Tier | Improvement | Status |
|------|-------------|--------|
| **T1** | Expert ROI from RSNA coordinates | 🆕 Implemented |
| **T1** | 7 XAI methods (4 families) | 🆕 Implemented |
| **T1** | Bootstrap CIs + Wilcoxon tests | 🆕 Implemented |
| **T1** | Multi-fold training (folds 0-4 append) | Updated |
| **T1** | Qualitative disagreement gallery | 🆕 New script |
| **T2** | Faithfulness-Weighted Consensus | 🆕 Implemented |
| **T2** | Feature-Map Coherence metric | 🆕 New script |
| **T2** | Visual CBM (BiomedCLIP concepts) | ⚠️ Skipped (noisy — see Session 3b) |
| **T3** | Model randomization sanity check | 🆕 New script |
| **T3** | Disagreement uncertainty maps | 🆕 Implemented |

---

## Prerequisites (Kaggle Datasets)

Upload these as Kaggle Datasets:

1. **`spinexnet-code`** — this entire codebase (updated with v2 changes)
2. **`manifests-of-spinexnet`** — contains `manifests/manifest_v2.csv`
3. **`pre-processed-crop-224`** — contains `image_cache_224/images_uint8.npy`
4. **`spinexnet-checkpoints`** — all `best.pt` files from Sessions 2–4 (keep existing)
5. **RSNA 2024 Competition Dataset** — raw competition data (only if not using cache)

---

## Common Variables (paste at top of every notebook)

```python
import os, shutil

CODE = "/kaggle/working/spinexnet-code"
MANIFEST = "/kaggle/input/datasets/vasuaashadesai/manifests-of-spinexnet/manifests/manifest_v2.csv"
CACHE = "/kaggle/input/datasets/vasuaashadesai/pre-processed-crop-224/image_cache_224"
CKPT = "/kaggle/input/datasets/vasuaashadesai/spinexnet-checkpoints/checkpoints"
WORK = "/kaggle/working"

# Copy code to writable dir
if not os.path.exists(CODE):
    shutil.copytree("/kaggle/input/datasets/vasuaashadesai/spinexnet-code", CODE)
```

---

## Dependency Chain & Parallelism

```
                    ┌─ S2a: ConvNeXt fold 0 ──┐
                    ├─ S2b: ConvNeXt fold 1 ──┤
  S1 (data prep) ──┼─ S2c: ConvNeXt fold 2 ──┼─→ S5 (XAI benchmark v2)
  [DONE]           ├─ S3a: CBM Non-Leaky ─────┤      ↓
                    ├─ S3b: CBM Visual (NEW) ──┤   S6 (Concept Intervention)
                    ├─ S4a: ResNet50 ──────────┤      ↓
                    ├─ S4b: DenseNet121 ───────┤   S7 (Feature coherence + Randomization)
                    ├─ S4c: EfficientNet-B4 ───┤      ↓
                    └─ S4d: ViT-Small ─────────┘   S8 (Visualization + Gallery)
```

### What Can Run in Parallel

| Parallel Group | Sessions | GPU? | Est. Time |
|----------------|----------|------|-----------|
| **Group A** | S2a, S2b, S2c (ConvNeXt folds) | Yes × 3 | ~1.5h each |
| **Group B** | S3a, S3b (CBMs) | Yes × 2 | ~1.5h each |
| **Group C** | S4a, S4b, S4c, S4d (baselines) | Yes × 4 | ~1h each |
| **Group D** | S5 cells (XAI per model) | Yes | ~6h total |
| **Group E** | S7a (Feature coherence), S7b (Randomization) | Yes × 2 | ~2h each |
| **Sequential** | S6 after S3, S8 after S5+S7 | S6:GPU, S8:CPU | ~30min each |

> **Sessions 2–4 are fully independent** — run on separate Kaggle notebooks simultaneously.
> **S5 cells are independent per model** — but fit in one 12h session.
> **S8 can run on CPU** (no GPU needed).

---

## Session 1 — Data Preparation ✅ DONE

Already completed. Outputs uploaded as:
- `manifests-of-spinexnet` → `manifest_v2.csv` with folds and concepts
- `pre-processed-crop-224` → `images_uint8.npy` (48,657 images at 224×224)

---

## Session 2 — Multi-Fold Training (ConvNeXt-Tiny Baseline)

**Purpose:** Train ConvNeXt black-box on folds 0, 1, 2 for multi-fold evaluation.

> **Fold 0 is already done** from the previous round. Only folds 1 and 2 are new.

```bash
# ── Setup ──
!pip install -q timm pydicom scikit-image
!cp -r "/kaggle/input/datasets/vasuaashadesai/spinexnet-code" /kaggle/working/spinexnet-code

# ── Fold 1 ──
!python $CODE/scripts/train.py \
  --config $CODE/configs/baselines/convnext_blackbox.yaml \
  --manifest $MANIFEST \
  --fold 1 \
  --cache-dir $CACHE \
  --output-dir $WORK/outputs/convnext_blackbox/fold_1 \
  --no-data-parallel --batch-size 32 --grad-accum 1 --time-limit-minutes 300

# ── Fold 2 ──
!python $CODE/scripts/train.py \
  --config $CODE/configs/baselines/convnext_blackbox.yaml \
  --manifest $MANIFEST \
  --fold 2 \
  --cache-dir $CACHE \
  --output-dir $WORK/outputs/convnext_blackbox/fold_2 \
  --no-data-parallel --batch-size 32 --grad-accum 1 --time-limit-minutes 300
```

**Action:** Download `fold_1/best.pt` and `fold_2/best.pt`, add to checkpoints dataset.

---

## Session 3 — CBM Training ✅ DONE (fold 0)

### 3a: CBM Non-Leaky (already done for fold 0)

Train folds 1–2 if multi-fold is needed.

### 3b: CBM Visual (BiomedCLIP concepts) — ⚠️ SKIPPED / OPTIONAL

> **Why skipped:** BiomedCLIP zero-shot concept scoring was tested in earlier experiments
> and produced **noisy supervision signals** that made CBM training unstable. The project
> report explicitly notes: *"Programmatic concepts only — deterministic derivation from
> labels/metadata — no noisy BiomedCLIP predictions."* The paper narrative does NOT depend
> on this ablation. Config file `cbm_visual.yaml` exists if you want to revisit later.


---

## Session 4 — Remaining Architecture Baselines ✅ DONE (fold 0)

For multi-fold: repeat Session 4 commands with `--fold 1` and `--fold 2`.

---

## Session 5 — XAI Benchmark v2 (All Models, Extended Methods)

**Purpose:** Run the 6-axis XAI evaluation with 6+ methods on all trained models.

**Estimated time:** ~45-90 min per model, ~6-8 hours total. Fits in one 12-hour GPU session.

> **Key changes from v1:**
> - 6 methods (was 3): `gradcam gradcam++ integrated_gradients gradient_shap occlusion guided_backprop`
> - Expert ROI alignment computed automatically (crops are centered on RSNA coordinates)
> - FW-Consensus evaluated on every 10th sample
> - Bootstrap CIs in summary JSON
> - Use `--save-maps` to cache saliency maps for the disagreement gallery

```bash
# ── Cell 1: Setup ──
!pip install -q timm pydicom captum grad-cam scikit-image
!cp -r "/kaggle/input/datasets/vasuaashadesai/spinexnet-code" /kaggle/working/spinexnet-code

CKPT="/kaggle/input/datasets/vasuaashadesai/spinexnet-checkpoints/checkpoints"
CODE="/kaggle/working/spinexnet-code"
MANIFEST="/kaggle/input/datasets/vasuaashadesai/manifests-of-spinexnet/manifests/manifest_v2.csv"
CACHE="/kaggle/input/datasets/vasuaashadesai/pre-processed-crop-224/image_cache_224"
METHODS="gradcam gradcam++ integrated_gradients gradient_shap occlusion guided_backprop"

# ── Cell 2: ConvNeXt Black-Box ──
!python $CODE/scripts/run_xai_benchmark_v2.py \
  --config $CODE/configs/baselines/convnext_blackbox.yaml \
  --checkpoint $CKPT/convnext_blackbox_best.pt \
  --manifest $MANIFEST --fold 0 --cache-dir $CACHE \
  --output-dir /kaggle/working/xai/convnext_blackbox \
  --max-samples 300 --methods $METHODS \
  --save-maps --skip-consistency

# ── Cell 3: CBM Non-Leaky ──
!python $CODE/scripts/run_xai_benchmark_v2.py \
  --config $CODE/configs/baselines/cbm_nonleaky.yaml \
  --checkpoint $CKPT/cbm_nonleaky_best.pt \
  --manifest $MANIFEST --fold 0 --cache-dir $CACHE \
  --output-dir /kaggle/working/xai/cbm_nonleaky \
  --max-samples 300 --methods $METHODS \
  --save-maps --skip-consistency

# ── Cell 4: ResNet50 ──
!python $CODE/scripts/run_xai_benchmark_v2.py \
  --config $CODE/configs/baselines/resnet50.yaml \
  --checkpoint $CKPT/resnet50_best.pt \
  --manifest $MANIFEST --fold 0 --cache-dir $CACHE \
  --output-dir /kaggle/working/xai/resnet50 \
  --max-samples 300 --methods $METHODS \
  --save-maps --skip-consistency

# ── Cell 5: DenseNet121 ──
!python $CODE/scripts/run_xai_benchmark_v2.py \
  --config $CODE/configs/baselines/densenet121.yaml \
  --checkpoint $CKPT/densenet121_best.pt \
  --manifest $MANIFEST --fold 0 --cache-dir $CACHE \
  --output-dir /kaggle/working/xai/densenet121 \
  --max-samples 300 --methods $METHODS \
  --save-maps --skip-consistency

# ── Cell 6: EfficientNet-B4 ──
!python $CODE/scripts/run_xai_benchmark_v2.py \
  --config $CODE/configs/baselines/efficientnet_b4.yaml \
  --checkpoint $CKPT/efficientnet_b4_best.pt \
  --manifest $MANIFEST --fold 0 --cache-dir $CACHE \
  --output-dir /kaggle/working/xai/efficientnet_b4 \
  --max-samples 300 --methods $METHODS \
  --save-maps --skip-consistency

# ── Cell 7: ViT-Small (with attention_rollout!) ──
!python $CODE/scripts/run_xai_benchmark_v2.py \
  --config $CODE/configs/baselines/vit_small.yaml \
  --checkpoint $CKPT/vit_small_best.pt \
  --manifest $MANIFEST --fold 0 --cache-dir $CACHE \
  --output-dir /kaggle/working/xai/vit_small \
  --max-samples 300 --methods $METHODS \
  --save-maps --skip-consistency \
  --enable-attention-rollout

# ── Cell 8: CBM Visual (if trained in S3b) ──
# !python $CODE/scripts/run_xai_benchmark_v2.py \
#   --config $CODE/configs/baselines/cbm_visual.yaml \
#   --checkpoint $CKPT/cbm_visual_best.pt \
#   --manifest $WORK/manifests/manifest_v2_with_biomedclip.csv --fold 0 --cache-dir $CACHE \
#   --output-dir /kaggle/working/xai/cbm_visual \
#   --max-samples 300 --methods $METHODS \
#   --save-maps --skip-consistency
```

### Output Structure (per model)

```
xai/<model>/
  faithfulness_metrics.csv       # per-sample, per-method
  agreement_metrics.csv          # per-sample pairwise Spearman + IoU
  clinical_alignment.csv         # proxy + expert ROI alignment
  consensus_metrics.csv          # FW-Consensus evaluation (NEW)
  xai_summary_v2.json            # aggregate with bootstrap CIs (NEW)
  attribution_maps/              # .npy saliency maps (for gallery)
```

**Action:** Download all XAI results. Upload as `spinexnet-xai-results-v2`.

---

## Session 6 — Concept Intervention Analysis

**Purpose:** Show that correcting CBM concept predictions fixes misclassifications.

**Can run:** In the same session as Session 5 if time permits, or separately.

```bash
# ── CBM Non-Leaky Intervention ──
!python $CODE/scripts/concept_intervention.py \
  --config $CODE/configs/baselines/cbm_nonleaky.yaml \
  --checkpoint $CKPT/cbm_nonleaky_best.pt \
  --manifest $MANIFEST --fold 0 --cache-dir $CACHE \
  --output-dir /kaggle/working/intervention/cbm_nonleaky \
  --max-samples 1000

# ── CBM Visual Intervention (NEW) ──
# !python $CODE/scripts/concept_intervention.py \
#   --config $CODE/configs/baselines/cbm_visual.yaml \
#   --checkpoint $CKPT/cbm_visual_best.pt \
#   --manifest $WORK/manifests/manifest_v2_with_biomedclip.csv --fold 0 --cache-dir $CACHE \
#   --output-dir /kaggle/working/intervention/cbm_visual \
#   --max-samples 1000
```

---

## Session 7 - Feature-Map Coherence + Model Randomization

**Purpose:** Tier 2 theoretical analysis and Tier 3 randomization sanity check.
The old input-gradient GLS script is kept for provenance, but the rethought
metric is `feature_map_smoothness.py`: it measures class-conditioned final
feature-map spatial autocorrelation and total variation, avoiding the
shattered-gradient failure mode.

### 7a: Feature-Map Coherence

```bash
!python $CODE/scripts/feature_map_smoothness.py \
  --configs \
    $CODE/configs/baselines/resnet50.yaml \
    $CODE/configs/baselines/densenet121.yaml \
    $CODE/configs/baselines/convnext_blackbox.yaml \
    $CODE/configs/baselines/efficientnet_b4.yaml \
    $CODE/configs/baselines/vit_small.yaml \
  --checkpoints \
    $CKPT/resnet50_best.pt \
    $CKPT/densenet121_best.pt \
    $CKPT/convnext_blackbox_best.pt \
    $CKPT/efficientnet_b4_best.pt \
    $CKPT/vit_small_best.pt \
  --names resnet50 densenet121 convnext_blackbox efficientnet_b4 vit_small \
  --manifest $MANIFEST --fold 0 --cache-dir $CACHE \
  --output-dir /kaggle/working/feature_map_smoothness \
  --max-samples 300 \
  --agreement-csv /kaggle/working/figures/cross_model/cross_model_agreement.csv
```

### 7b: Model Randomization Sanity Check

Run on 2-3 representative models (one per architecture family):

```bash
# ConvNeXt (modern CNN)
!python $CODE/scripts/model_randomization.py \
  --config $CODE/configs/baselines/convnext_blackbox.yaml \
  --checkpoint $CKPT/convnext_blackbox_best.pt \
  --manifest $MANIFEST --fold 0 --cache-dir $CACHE \
  --output-dir /kaggle/working/randomization/convnext \
  --methods gradcam integrated_gradients gradient_shap occlusion \
  --max-samples 50 --n-levels 5

# ViT (transformer)
!python $CODE/scripts/model_randomization.py \
  --config $CODE/configs/baselines/vit_small.yaml \
  --checkpoint $CKPT/vit_small_best.pt \
  --manifest $MANIFEST --fold 0 --cache-dir $CACHE \
  --output-dir /kaggle/working/randomization/vit \
  --methods gradcam integrated_gradients gradient_shap occlusion \
  --max-samples 50 --n-levels 5
```

---

## Session 8 — Visualization & Publication Figures

**Purpose:** Generate all publication-quality figures. **Runs on CPU.**

```bash
# ── Setup ──
!pip install -q matplotlib seaborn pandas numpy scipy scikit-image
!cp -r "/kaggle/input/datasets/vasuaashadesai/spinexnet-code" /kaggle/working/spinexnet-code

XAI="/kaggle/input/datasets/vasuaashadesai/spinexnet-xai-results-v2"
CODE="/kaggle/working/spinexnet-code"

# ── Per-model figures ──
for MODEL in convnext_blackbox cbm_nonleaky resnet50 densenet121 efficientnet_b4 vit_small; do
  python $CODE/scripts/visualize_xai.py \
    --results-dir $XAI/$MODEL \
    --output-dir /kaggle/working/figures/$MODEL
done

# ── Cross-model comparison figures (HERO FIGURES) ──
!python $CODE/scripts/visualize_xai.py \
  --cross-model-dir $XAI \
  --output-dir /kaggle/working/figures/cross_model

# ── Disagreement gallery (requires --save-maps from S5) ──
!python $CODE/scripts/generate_gallery.py \
  --xai-dir $XAI \
  --models densenet121 convnext_blackbox vit_small \
  --output-dir /kaggle/working/figures/gallery
```

### All Figures Generated

| Figure | File | Paper Use |
|--------|------|-----------| 
| Agreement heatmap (per model) | `agreement_heatmap.png` | Supplementary |
| Faithfulness bars (per model) | `faithfulness_comparison.png` | Supplementary |
| Expert ROI alignment | `expert_roi_alignment.png` | **Table 4** |
| **Cross-model agreement** | `cross_model_agreement.png` | **Hero Figure 1** |
| **Cross-model faithfulness** | `cross_model_faithfulness.png` | **Figure 2** |
| **Cross-model expert ROI** | `cross_model_expert_roi.png` | **Figure 3** |
| **Disagreement gallery** | `disagreement_gallery.png` | **Figure 4** |
| **Feature coherence scatter plot** | `feature_coherence_vs_agreement.png` | **Figure 5** |
| Randomization plot | `randomization_plot.png` | **Figure 6** |

---

## Complete Model Inventory

| # | Config File | Model Type | Purpose | Folds after append |
|---|-------------|------------|---------|--------------------|
| 1 | `convnext_blackbox.yaml` | Black-box | Primary baseline | 0,1,2,3,4 |
| 2 | `cbm_nonleaky.yaml` | CBM (7 anatomy concepts) | Primary interpretable | 0,1,2,3,4 |
| 3 | `cbm_leaky.yaml` | CBM (label-leaking) | Leakage ablation | 0,1,2,3,4 |
| 4 | `resnet50.yaml` | Black-box | Architecture diversity | 0,1,2,3,4 |
| 5 | `densenet121.yaml` | Black-box | Architecture diversity | 0,1,2,3,4 |
| 6 | `efficientnet_b4.yaml` | Black-box | Architecture diversity | 0,1,2,3,4 |
| 7 | `vit_small.yaml` | Black-box (ViT) | CNN vs Transformer | 0,1,2,3,4 |

---

## Paper Table Reference (Revised)

| Table/Figure | Source | Script |
|-------------|--------|--------|
| Table 1: Classification performance (WLL, AUC, Bal.Acc) | Session 2–4 outputs | `evaluate.py` |
| Table 2: Faithfulness (del/ins AUC) by method × model | `faithfulness_metrics.csv` | `run_xai_benchmark_v2.py` |
| Table 3: Pairwise agreement (Spearman ρ ± CI) | `agreement_metrics.csv` | `run_xai_benchmark_v2.py` |
| **Table 4: Expert ROI alignment (IoU ± CI)** | `clinical_alignment.csv` | `run_xai_benchmark_v2.py` |
| Table 5: Concept intervention fix rates | `intervention_summary.json` | `concept_intervention.py` |
| **Table 6: FW-Consensus vs individual methods** | `consensus_metrics.csv` | `run_xai_benchmark_v2.py` |
| **Table 7: Feature-map coherence scores** | `feature_map_smoothness_scores.csv` | `feature_map_smoothness.py` |
| **Figure 1: Architecture-dependent disagreement (hero)** | cross-model CSVs | `visualize_xai.py --cross-model-dir` |
| **Figure 2: Disagreement gallery** | attribution maps | `generate_gallery.py` |
| **Figure 3: Feature coherence vs agreement scatter** | feature coherence + agreement CSVs | `feature_map_smoothness.py` |
| **Figure 4: Randomization sanity check** | randomization CSVs | `model_randomization.py` |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `attention_rollout` fails on CNN | Only use `--enable-attention-rollout` for ViT models |
| `GuidedBackprop` fails | Some architectures don't support it — check `skipped` in JSON |
| OOM with 6 methods | Reduce `--max-samples 200` or drop `occlusion` (slowest) |
| `scikit-image` not found | Add `!pip install -q scikit-image` to setup cell |
| BiomedCLIP concepts missing | Run `extract_biomedclip_concepts.py` first (Session 3b Step 1) |
| Bootstrap CI slow | Reduce `n_resamples=500` in stats.py for faster iteration |
| `scorecam`/`lime`/`shap` too slow | Not in default methods — add only if time permits |
| Time limit hit | Training auto-saves; reduce `--time-limit-minutes` |

---

## Execution Sequence (Minimum Viable)

If you have limited Kaggle GPU hours, here is the minimum sequence:

1. **Session 5** (one notebook, ~6h): Run XAI benchmark v2 on all 7 existing checkpoints with 6 methods
2. **Session 7a** (one notebook, ~2h): Feature-Map Coherence on all 5 black-box models
3. **Session 8** (CPU, ~10min): Generate all figures

This gives you Tier 1 + most of Tier 2 without retraining.

For the full Tier 1–3:
1. Sessions 2+3+4 in parallel (3 notebooks, ~2h each) — multi-fold + visual CBM
2. Session 5 (1 notebook, ~8h) — extended XAI benchmark
3. Session 6 (same notebook or separate, ~30min) — concept intervention
4. Session 7 (1 notebook, ~3h) — feature coherence + randomization
5. Session 8 (CPU, ~10min) — all figures
