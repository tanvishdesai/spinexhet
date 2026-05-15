# SpineXNet Explainability Disagreement Benchmark — Kaggle Runbook

> **Last updated:** 2026-05-14  
> **Goal:** Complete the full training + XAI evaluation pipeline for a BMVC submission.

---

## Progress Summary

| Session | Description | Status | Best WLL | AUC |
|---------|-------------|--------|----------|-----|
| 1 | Data prep, manifest, image cache | ✅ Done | — | — |
| 2 | ConvNeXt-Tiny black-box baseline | ✅ Done | **0.4916** | 0.931 |
| 3 | CBM Non-Leaky + CBM Leaky | ✅ Done | 0.5257 / 0.5067 | 0.929 / 0.918 |
| 4 | Remaining baselines (ResNet, DenseNet, EfficientNet, ViT) | ✅ Done | 0.503–0.631 | 0.906–0.927 |
| 5 | XAI Benchmark v2 (all 7 checkpoints) | ✅ Done | ρ=0.08–0.49 | — |
| 6 | Concept Intervention (CBM only) | ⚠️ Bug | fix_rate bug | — |
| 7 | Visualization & publication figures | ✅ Done | 21 figures | — |

---

## Prerequisites (Kaggle Datasets)

You need these uploaded as Kaggle Datasets:

1. **`spinexnet-code`** — this entire codebase
2. **`manifests-of-spinexnet`** — contains `manifests/manifest_v2.csv`
3. **`pre-processed-crop-224`** — contains `image_cache_224/images_uint8.npy`
4. **`spinexnet-checkpoints`** — (create after Sessions 2-4) all `best.pt` files
5. **RSNA 2024 Competition Dataset** — the raw competition data (only needed if not using cache)

---

## Common Variables (paste at top of every notebook)

```python
CODE = "/kaggle/working/spinexnet-code"
MANIFEST = "/kaggle/input/datasets/vasuaashadesai/manifests-of-spinexnet/manifests/manifest_v2.csv"
CACHE = "/kaggle/input/datasets/vasuaashadesai/pre-processed-crop-224/image_cache_224"
WORK = "/kaggle/working"
```

---

## Session 1 — Data Preparation ✅ DONE

**What it does:** Generates manifest, preprocesses crops to 224px, builds numpy image cache.

This was completed in earlier sessions. Outputs are uploaded as Kaggle Datasets:
- `manifests-of-spinexnet` → `manifest_v2.csv` with folds, programmatic concepts
- `pre-processed-crop-224` → `images_uint8.npy` (48,657 images at 224×224)

No action needed — move to Session 2.

---

## Session 2 — ConvNeXt-Tiny Black-Box Baseline ✅ DONE

**Results file:** `results/results-logs-4.txt`

```bash
# ── Setup ──
!pip install -q timm pydicom
!cp -r "/kaggle/input/datasets/vasuaashadesai/spinexnet-code" .

# ── Train ConvNeXt Black-Box ──
!python /kaggle/working/spinexnet-code/scripts/train.py \
  --config /kaggle/working/spinexnet-code/configs/baselines/convnext_blackbox.yaml \
  --manifest /kaggle/input/datasets/vasuaashadesai/manifests-of-spinexnet/manifests/manifest_v2.csv \
  --fold 0 \
  --cache-dir /kaggle/input/datasets/vasuaashadesai/pre-processed-crop-224/image_cache_224 \
  --no-data-parallel \
  --batch-size 32 \
  --grad-accum 1 \
  --time-limit-minutes 300
```

### Results Summary

| Metric | Value |
|--------|-------|
| Best val weighted log loss | **0.4916** |
| Balanced accuracy | 73.7% |
| Macro F1 | 0.705 |
| AUC-OVR | **0.931** |
| Best epoch | 9 (early stop at 22) |
| Training time | ~74 min |

**Per-condition:** Spinal canal stenosis easiest (AUC 0.957), subarticular hardest (AUC 0.911).

**Action:** Save `best.pt` checkpoint as part of the checkpoints dataset.

---

## Session 3 — CBM Non-Leaky + CBM Leaky ✅ DONE

**Results file:** `results/results-ligs-5.txt`

```bash
# ── Setup ──
!pip install -q timm pydicom captum grad-cam quantus lime shap open_clip_torch transformers
!cp -r "/kaggle/input/datasets/vasuaashadesai/spinexnet-code" .

# ── Train Non-Leaky CBM (PRIMARY) ──
!python /kaggle/working/spinexnet-code/scripts/train.py \
  --config /kaggle/working/spinexnet-code/configs/baselines/cbm_nonleaky.yaml \
  --manifest /kaggle/input/datasets/vasuaashadesai/manifests-of-spinexnet/manifests/manifest_v2.csv \
  --fold 0 \
  --cache-dir /kaggle/input/datasets/vasuaashadesai/pre-processed-crop-224/image_cache_224 \
  --no-data-parallel \
  --batch-size 32 \
  --grad-accum 1 \
  --time-limit-minutes 300

# ── Train Leaky CBM (ABLATION) ──
!python /kaggle/working/spinexnet-code/scripts/train.py \
  --config /kaggle/working/spinexnet-code/configs/baselines/cbm_leaky.yaml \
  --manifest /kaggle/input/datasets/vasuaashadesai/manifests-of-spinexnet/manifests/manifest_v2.csv \
  --fold 0 \
  --cache-dir /kaggle/input/datasets/vasuaashadesai/pre-processed-crop-224/image_cache_224 \
  --no-data-parallel \
  --batch-size 32 \
  --grad-accum 1 \
  --time-limit-minutes 300
```

### Results Summary

| Model | Best WLL | Bal. Acc | AUC | Best Epoch | Epochs Ran |
|-------|----------|----------|-----|------------|------------|
| CBM Non-Leaky | **0.5257** | 74.3% | 0.929 | 11 | 22 |
| CBM Leaky | **0.5067** | 75.3% | 0.918 | 8 | 19 |

**Key finding:** Leaky CBM outperforms non-leaky by 0.019 WLL — confirms concept leakage inflates performance. Both CBMs are close to the black-box (0.492), showing interpretability doesn't sacrifice much accuracy.

**Action:** Save both `best.pt` checkpoints.

---

## Session 4 — Remaining Architecture Baselines ✅ DONE

**Results file:** `results/results-logs-6.txt`

```bash
# ── Cell 1: Setup ──
!pip install -q timm pydicom captum grad-cam quantus lime shap open_clip_torch transformers
!cp -r "/kaggle/input/datasets/vasuaashadesai/spinexnet-code" .

# ── Cell 2: ResNet50 ──
!python /kaggle/working/spinexnet-code/scripts/train.py \
  --config /kaggle/working/spinexnet-code/configs/baselines/resnet50.yaml \
  --manifest /kaggle/input/datasets/vasuaashadesai/manifests-of-spinexnet/manifests/manifest_v2.csv \
  --fold 0 \
  --cache-dir /kaggle/input/datasets/vasuaashadesai/pre-processed-crop-224/image_cache_224 \
  --no-data-parallel --batch-size 32 --grad-accum 1 --time-limit-minutes 120

# ── Cell 3: DenseNet121 ──
!python /kaggle/working/spinexnet-code/scripts/train.py \
  --config /kaggle/working/spinexnet-code/configs/baselines/densenet121.yaml \
  --manifest /kaggle/input/datasets/vasuaashadesai/manifests-of-spinexnet/manifests/manifest_v2.csv \
  --fold 0 \
  --cache-dir /kaggle/input/datasets/vasuaashadesai/pre-processed-crop-224/image_cache_224 \
  --no-data-parallel --batch-size 32 --grad-accum 1 --time-limit-minutes 120

# ── Cell 4: EfficientNet-B4 ──
!python /kaggle/working/spinexnet-code/scripts/train.py \
  --config /kaggle/working/spinexnet-code/configs/baselines/efficientnet_b4.yaml \
  --manifest /kaggle/input/datasets/vasuaashadesai/manifests-of-spinexnet/manifests/manifest_v2.csv \
  --fold 0 \
  --cache-dir /kaggle/input/datasets/vasuaashadesai/pre-processed-crop-224/image_cache_224 \
  --no-data-parallel --batch-size 32 --grad-accum 1 --time-limit-minutes 120

# ── Cell 5: ViT-Small ──
!python /kaggle/working/spinexnet-code/scripts/train.py \
  --config /kaggle/working/spinexnet-code/configs/baselines/vit_small.yaml \
  --manifest /kaggle/input/datasets/vasuaashadesai/manifests-of-spinexnet/manifests/manifest_v2.csv \
  --fold 0 \
  --cache-dir /kaggle/input/datasets/vasuaashadesai/pre-processed-crop-224/image_cache_224 \
  --no-data-parallel --batch-size 32 --grad-accum 1 --time-limit-minutes 120
```

### Results Summary

| Model | Best WLL | Bal. Acc | Macro F1 | AUC | Best Epoch | Epochs Ran | Time |
|-------|----------|----------|----------|-----|------------|------------|------|
| ResNet50 | **0.5630** | 71.8% | 0.653 | 0.906 | 8 | 19 | ~54 min |
| DenseNet121 | **0.5087** | 71.6% | 0.697 | 0.917 | ~10 | 18 | ~61 min |
| EfficientNet-B4 | **0.6308** | — | — | — | — | 18 | ~55 min |
| ViT-Small | **0.5027** | 75.1% | 0.694 | 0.927 | 18 | 20 | ~55 min |

### Complete Model Ranking (all sessions)

| Rank | Model | Type | Best WLL ↓ | AUC |
|------|-------|------|-----------|-----|
| 1 | **ConvNeXt-Tiny** | Black-box | **0.4916** | 0.931 |
| 2 | **ViT-Small** | Black-box | **0.5027** | 0.927 |
| 3 | CBM Leaky | CBM (ablation) | 0.5067 | 0.918 |
| 4 | DenseNet121 | Black-box | 0.5087 | 0.917 |
| 5 | CBM Non-Leaky | CBM (primary) | 0.5257 | 0.929 |
| 6 | ResNet50 | Black-box | 0.5630 | 0.906 |
| 7 | EfficientNet-B4 | Black-box | 0.6308 | — |

### Analysis

- **ViT-Small is the surprise winner** (WLL 0.503) — nearly matching ConvNeXt (0.492) and outperforming all other models. This is great for the paper as it shows a Transformer baseline is competitive with CNNs on spine imaging.
- **DenseNet121 is solid** (0.509) — nearly identical to CBM Leaky, confirming it's a strong traditional baseline.
- **ResNet50 is respectable** (0.563) — weaker than the modern architectures, but the gap is expected and provides good diversity for XAI disagreement analysis.
- **EfficientNet-B4 underperformed** (0.631) — likely needs longer training or hyperparameter tuning, but still usable for XAI comparison since the model converged.
- **The CBM Non-Leaky (0.526) sits in the middle** — interpretable and competitive, which is the key argument for the paper.

**Action:** Upload ALL 7 checkpoints as a single Kaggle Dataset called `spinexnet-checkpoints`:
  ```
  checkpoints/
    convnext_blackbox_best.pt
    cbm_nonleaky_best.pt
    cbm_leaky_best.pt
    resnet50_best.pt
    densenet121_best.pt
    efficientnet_b4_best.pt
    vit_small_best.pt
  ```

---

## Session 5 — XAI Benchmark v2 ⬜ TODO

**Purpose:** Run the 4-axis XAI evaluation (Faithfulness, Agreement, Consistency, Clinical Alignment) on all trained models.

**Estimated time:** ~30-60 min per model (with 200-500 samples), ~5-7 hours total. Fits in one 12-hour GPU session.

> **Important:** Use `run_xai_benchmark_v2.py` (not v1). Start with gradient methods only (`gradcam gradcam++ integrated_gradients`) to save time. Add `scorecam lime shap` only if time permits.

```bash
# ── Cell 1: Setup ──
!pip install -q timm pydicom captum grad-cam quantus lime shap
!cp -r "/kaggle/input/datasets/vasuaashadesai/spinexnet-code" .

CKPT="/kaggle/input/datasets/vasuaashadesai/spinexnet-checkpoints/checkpoints"
CODE="/kaggle/working/spinexnet-code"
MANIFEST="/kaggle/input/datasets/vasuaashadesai/manifests-of-spinexnet/manifests/manifest_v2.csv"
CACHE="/kaggle/input/datasets/vasuaashadesai/pre-processed-crop-224/image_cache_224"

# ── Cell 2: ConvNeXt Black-Box XAI ──
!python $CODE/scripts/run_xai_benchmark_v2.py \
  --config $CODE/configs/baselines/convnext_blackbox.yaml \
  --checkpoint $CKPT/convnext_blackbox_best.pt \
  --manifest $MANIFEST \
  --fold 0 \
  --cache-dir $CACHE \
  --output-dir /kaggle/working/xai/convnext_blackbox \
  --max-samples 300 \
  --methods gradcam gradcam++ integrated_gradients \
  --skip-consistency

# ── Cell 3: CBM Non-Leaky XAI ──
!python $CODE/scripts/run_xai_benchmark_v2.py \
  --config $CODE/configs/baselines/cbm_nonleaky.yaml \
  --checkpoint $CKPT/cbm_nonleaky_best.pt \
  --manifest $MANIFEST \
  --fold 0 \
  --cache-dir $CACHE \
  --output-dir /kaggle/working/xai/cbm_nonleaky \
  --max-samples 300 \
  --methods gradcam gradcam++ integrated_gradients \
  --skip-consistency

# ── Cell 4: CBM Leaky XAI ──
!python $CODE/scripts/run_xai_benchmark_v2.py \
  --config $CODE/configs/baselines/cbm_leaky.yaml \
  --checkpoint $CKPT/cbm_leaky_best.pt \
  --manifest $MANIFEST \
  --fold 0 \
  --cache-dir $CACHE \
  --output-dir /kaggle/working/xai/cbm_leaky \
  --max-samples 300 \
  --methods gradcam gradcam++ integrated_gradients \
  --skip-consistency

# ── Cell 5: ResNet50 XAI ──
!python $CODE/scripts/run_xai_benchmark_v2.py \
  --config $CODE/configs/baselines/resnet50.yaml \
  --checkpoint $CKPT/resnet50_best.pt \
  --manifest $MANIFEST \
  --fold 0 \
  --cache-dir $CACHE \
  --output-dir /kaggle/working/xai/resnet50 \
  --max-samples 300 \
  --methods gradcam gradcam++ integrated_gradients \
  --skip-consistency

# ── Cell 6: DenseNet121 XAI ──
!python $CODE/scripts/run_xai_benchmark_v2.py \
  --config $CODE/configs/baselines/densenet121.yaml \
  --checkpoint $CKPT/densenet121_best.pt \
  --manifest $MANIFEST \
  --fold 0 \
  --cache-dir $CACHE \
  --output-dir /kaggle/working/xai/densenet121 \
  --max-samples 300 \
  --methods gradcam gradcam++ integrated_gradients \
  --skip-consistency

# ── Cell 7: EfficientNet-B4 XAI ──
!python $CODE/scripts/run_xai_benchmark_v2.py \
  --config $CODE/configs/baselines/efficientnet_b4.yaml \
  --checkpoint $CKPT/efficientnet_b4_best.pt \
  --manifest $MANIFEST \
  --fold 0 \
  --cache-dir $CACHE \
  --output-dir /kaggle/working/xai/efficientnet_b4 \
  --max-samples 300 \
  --methods gradcam gradcam++ integrated_gradients \
  --skip-consistency

# ── Cell 8: ViT-Small XAI ──
!python $CODE/scripts/run_xai_benchmark_v2.py \
  --config $CODE/configs/baselines/vit_small.yaml \
  --checkpoint $CKPT/vit_small_best.pt \
  --manifest $MANIFEST \
  --fold 0 \
  --cache-dir $CACHE \
  --output-dir /kaggle/working/xai/vit_small \
  --max-samples 300 \
  --methods gradcam gradcam++ integrated_gradients \
  --skip-consistency
```

### Output Structure

Each model directory will contain:
```
xai/<model>/
  faithfulness_metrics.csv
  agreement_metrics.csv
  clinical_alignment.csv
  xai_summary_v2.json
```

### After Completion

- Download all XAI result CSVs
- Upload as a Kaggle Dataset `spinexnet-xai-results` for Session 7

---

## Session 6 — Concept Intervention Analysis ⬜ TODO

**Purpose:** Demonstrate that correcting CBM concept predictions at test time can fix misclassifications — a key advantage of ante-hoc interpretability.

**Estimated time:** ~15-30 min. Can run in the same session as Session 5 if time permits.

```bash
# ── Setup (skip if already done in same notebook) ──
!pip install -q timm pydicom
!cp -r "/kaggle/input/datasets/vasuaashadesai/spinexnet-code" .

CKPT="/kaggle/input/datasets/vasuaashadesai/spinexnet-checkpoints/checkpoints"
CODE="/kaggle/working/spinexnet-code"
MANIFEST="/kaggle/input/datasets/vasuaashadesai/manifests-of-spinexnet/manifests/manifest_v2.csv"
CACHE="/kaggle/input/datasets/vasuaashadesai/pre-processed-crop-224/image_cache_224"

# ── Concept Intervention on CBM Non-Leaky ──
!python $CODE/scripts/concept_intervention.py \
  --config $CODE/configs/baselines/cbm_nonleaky.yaml \
  --checkpoint $CKPT/cbm_nonleaky_best.pt \
  --manifest $MANIFEST \
  --fold 0 \
  --cache-dir $CACHE \
  --output-dir /kaggle/working/intervention/cbm_nonleaky \
  --max-samples 1000

# ── Concept Intervention on CBM Leaky (for ablation comparison) ──
!python $CODE/scripts/concept_intervention.py \
  --config $CODE/configs/baselines/cbm_leaky.yaml \
  --checkpoint $CKPT/cbm_leaky_best.pt \
  --manifest $MANIFEST \
  --fold 0 \
  --cache-dir $CACHE \
  --output-dir /kaggle/working/intervention/cbm_leaky \
  --max-samples 1000
```

### Output

```
intervention/<model>/
  concept_intervention.csv      # per-sample, per-concept results
  intervention_summary.json     # aggregate fix rates
```

---

## Session 7 — Visualization & Publication Figures ⬜ TODO

**Purpose:** Generate all publication-quality figures from the XAI benchmark results.

**Estimated time:** ~5 min. Can run locally or on Kaggle CPU.

```bash
# ── Setup ──
!pip install -q matplotlib seaborn pandas numpy
!cp -r "/kaggle/input/datasets/vasuaashadesai/spinexnet-code" .

CODE="/kaggle/working/spinexnet-code"
XAI="/kaggle/input/datasets/vasuaashadesai/spinexnet-xai-results"

# ── Generate figures for each model ──
!python $CODE/scripts/visualize_xai.py \
  --results-dir $XAI/convnext_blackbox \
  --output-dir /kaggle/working/figures/convnext_blackbox

!python $CODE/scripts/visualize_xai.py \
  --results-dir $XAI/cbm_nonleaky \
  --output-dir /kaggle/working/figures/cbm_nonleaky

!python $CODE/scripts/visualize_xai.py \
  --results-dir $XAI/cbm_leaky \
  --output-dir /kaggle/working/figures/cbm_leaky

!python $CODE/scripts/visualize_xai.py \
  --results-dir $XAI/resnet50 \
  --output-dir /kaggle/working/figures/resnet50

!python $CODE/scripts/visualize_xai.py \
  --results-dir $XAI/densenet121 \
  --output-dir /kaggle/working/figures/densenet121

!python $CODE/scripts/visualize_xai.py \
  --results-dir $XAI/efficientnet_b4 \
  --output-dir /kaggle/working/figures/efficientnet_b4

!python $CODE/scripts/visualize_xai.py \
  --results-dir $XAI/vit_small \
  --output-dir /kaggle/working/figures/vit_small
```

### Figures Generated (per model)

| Figure | File | Paper Use |
|--------|------|-----------|
| Agreement heatmap | `agreement_heatmap.png` | **Hero figure** — shows pairwise XAI disagreement |
| Faithfulness bars | `faithfulness_comparison.png` | Deletion/insertion AUC comparison |
| Clinical alignment | `clinical_alignment.png` | Condition-specific ROI overlap |
| Consistency | `consistency_by_method.png` | Augmentation stability |

---

## Complete Model Inventory

| # | Config File | Model Type | Purpose |
|---|-------------|------------|---------|
| 1 | `convnext_blackbox.yaml` | Black-box | Primary baseline |
| 2 | `cbm_nonleaky.yaml` | CBM (7 anatomy concepts) | Primary interpretable model |
| 3 | `cbm_leaky.yaml` | CBM (label-leaking concepts) | Leakage ablation |
| 4 | `resnet50.yaml` | Black-box | Architecture diversity |
| 5 | `densenet121.yaml` | Black-box | Architecture diversity |
| 6 | `efficientnet_b4.yaml` | Black-box | Architecture diversity |
| 7 | `vit_small.yaml` | Black-box (ViT) | CNN vs Transformer comparison |

---

## Dependency Chain

```
Session 1 (data)
    ↓
Session 2 (ConvNeXt) ──→ Session 5 (XAI on ConvNeXt)
Session 3 (CBMs) ──────→ Session 5 (XAI on CBMs) + Session 6 (Intervention)
Session 4 (baselines) ─→ Session 5 (XAI on baselines)
    ↓
Session 5 (all XAI results)
    ↓
Session 7 (figures)
```

> Sessions 2, 3, and 4 are **independent** of each other and can run in parallel on separate Kaggle notebooks.  
> Session 6 can run in the same notebook as Session 5.  
> Session 7 can run on CPU (no GPU needed).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `HF_TOKEN` warning | Set `HF_TOKEN` in Kaggle secrets for faster timm downloads |
| OOM on ViT | Reduce batch size: `--batch-size 16 --grad-accum 2` |
| XAI method fails silently | Check `skipped` dict in `xai_summary_v2.json` |
| `scorecam`/`lime`/`shap` too slow | Use `--max-samples 100` and `--skip-consistency` |
| Checkpoint not loading | Ensure `strict=True` matches; check config matches checkpoint architecture |
| Time limit hit | Reduce `--time-limit-minutes`; training auto-saves `best.pt` |

---

## Paper Table Reference

The final paper needs these tables/figures from the benchmark:

1. **Table 1:** Classification performance (WLL, AUC, Bal.Acc) across all 7 models
2. **Table 2:** Faithfulness (deletion/insertion AUC) by method × model
3. **Table 3:** Pairwise agreement (Spearman ρ) between XAI methods
4. **Table 4:** Clinical alignment (proxy ROI IoU) by condition
5. **Table 5:** Concept intervention fix rates (CBM only)
6. **Figure 1:** Agreement heatmap (hero figure)
7. **Figure 2:** Faithfulness comparison bars
8. **Figure 3:** Clinical alignment by condition
9. **Figure 4:** CBM leaky vs non-leaky performance gap
