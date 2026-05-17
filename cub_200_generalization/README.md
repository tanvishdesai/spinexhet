# CUB-200 Generalization Track

This folder is the natural-image generalization proof requested in `next_steps.md`.
It trains the black-box architecture families on CUB-200, including a second
Transformer added after V3 mentor review:

- `convnext_blackbox`
- `resnet50`
- `densenet121`
- `efficientnet_b4`
- `vit_small`
- `deit_small`

The spine CBM variants are not carried over here because their concepts are
spine metadata concepts. CUB has attributes, but using them would create a
separate concept-learning paper rather than the architecture-level XAI
generalization test. The BMVC/MIDL claim here is: the same architecture-linked
attribution disagreement pattern should reproduce on fine-grained natural
images.

## Kaggle Inputs

Attach these datasets to every CUB notebook:

1. The updated `spinexnet-code` dataset containing this repository.
2. The CUB-200 dataset. The scripts auto-detect a folder named
   `CUB_200_2011` with the structure:
   `images/`, `attributes/`, `parts/`, `classes.txt`,
   `images.txt`, `image_class_labels.txt`, and `train_test_split.txt`.

## Execution Order

1. Run each per-model notebook in `cub_200_generalization/notebooks/`.
   Each notebook trains folds `0..4`, evaluates the validation split for each
   fold, and runs CUB attribution agreement.
2. Save each per-model notebook output as a Kaggle dataset.
3. Attach all per-model output datasets to
   `aggregate_cub_results.ipynb`.
4. Run `aggregate_cub_results.ipynb` to produce final CSV tables and figures.

The aggregator keeps classification metrics for every model, but excludes CUB
XAI summaries for models with mean validation accuracy below `0.70` by default.
This prevents a failed classifier, especially the previous ConvNeXt run, from
contaminating the cross-domain explanation analysis. Rerun ConvNeXt with the
updated training recipe below; if it crosses the threshold, it is included
automatically.

## Direct CLI

Train and evaluate one fold:

```bash
python cub_200_generalization/train_cub_model.py \
  --model resnet50 \
  --cub-root /kaggle/input/<dataset>/CUB_200_2011 \
  --fold 0 \
  --output-dir /kaggle/working/cub_outputs/resnet50/fold_0

python cub_200_generalization/evaluate_cub_model.py \
  --model resnet50 \
  --checkpoint /kaggle/working/cub_outputs/resnet50/fold_0/best.pt \
  --cub-root /kaggle/input/<dataset>/CUB_200_2011 \
  --fold 0 \
  --output-dir /kaggle/working/cub_eval/resnet50/fold_0
```

Run XAI agreement for that fold:

```bash
python cub_200_generalization/run_cub_xai.py \
  --model resnet50 \
  --checkpoint /kaggle/working/cub_outputs/resnet50/fold_0/best.pt \
  --cub-root /kaggle/input/<dataset>/CUB_200_2011 \
  --fold 0 \
  --output-dir /kaggle/working/cub_xai/fold_0/resnet50 \
  --max-samples 300 \
  --methods gradcam gradcam++ integrated_gradients gradient_shap occlusion guided_backprop \
  --save-maps
```

For `vit_small` and `deit_small`, add `--enable-attention-rollout`.

Recommended rerun for ConvNeXt-CUB:

```bash
python cub_200_generalization/train_cub_model.py \
  --model convnext_blackbox \
  --cub-root /kaggle/input/<dataset>/CUB_200_2011 \
  --fold 0 \
  --output-dir /kaggle/working/cub_outputs/convnext_blackbox/fold_0 \
  --epochs 35 \
  --lr 1e-4 \
  --backbone-lr 5e-5 \
  --head-lr 5e-4 \
  --warmup-epochs 3 \
  --patience 8 \
  --drop-path-rate 0.1
```

Recommended run for the second Transformer:

```bash
python cub_200_generalization/train_cub_model.py \
  --model deit_small \
  --cub-root /kaggle/input/<dataset>/CUB_200_2011 \
  --fold 0 \
  --output-dir /kaggle/working/cub_outputs/deit_small/fold_0 \
  --epochs 30 \
  --lr 1e-4 \
  --backbone-lr 5e-5 \
  --head-lr 5e-4 \
  --warmup-epochs 3 \
  --patience 8 \
  --drop-path-rate 0.1
```

Aggregate all per-model outputs:

```bash
python cub_200_generalization/aggregate_cub_results.py \
  --copy-inputs \
  --output-dir /kaggle/working/cub_aggregate
```

## Outputs

Per model/fold:

- `cub_outputs/<model>/fold_<k>/best.pt`
- `cub_eval/<model>/fold_<k>/metrics_val.json`
- `cub_eval/<model>/fold_<k>/predictions_val.csv`
- `cub_xai/fold_<k>/<model>/agreement_metrics.csv`
- `cub_xai/fold_<k>/<model>/faithfulness_metrics.csv`
- `cub_xai/fold_<k>/<model>/consensus_metrics.csv`
- `cub_xai/fold_<k>/<model>/xai_summary_cub.json`

Aggregated:

- `cub_classification_metrics_by_fold.csv`
- `cub_classification_metrics_mean_std.csv`
- `cub_xai_summary_by_fold.csv`
- `cub_xai_summary_mean_std.csv`
- `cub_xai_model_exclusion_report.csv`
- `figures/cub_classification_accuracy.png`
- `figures/cub_xai_agreement.png`
