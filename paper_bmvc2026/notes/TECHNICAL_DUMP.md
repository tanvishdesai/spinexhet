# Technical Dump

Core thesis: post-hoc XAI method disagreement is organized by architecture class across RSNA lumbar spine MRI and CUB-200.

Datasets:
- RSNA 2024 Lumbar Spine Degenerative Classification: approximately 2,697 patients, 48,657 condition-level crops, five conditions, five levels, three severity grades, expert coordinate annotations, 224x224 crops, stratified five-fold patient-level validation.
- CUB-200-2011: 11,788 images, 200 bird species, 224x224 validation images, stratified five-fold validation.

Architectures:
- RSNA: ConvNeXt-Tiny, ResNet-50, DenseNet-121, EfficientNet-B4, ViT-Small, DeiT-Small, CBM non-leaky, CBM leaky.
- CUB-200: ConvNeXt-Tiny, ResNet-50, DenseNet-121, EfficientNet-B4, ViT-Small, DeiT-Small.

Training:
- RSNA: AdamW, batch size 16, gradient accumulation 2, ImageNet normalization, AMP, 5 warmup epochs, early stopping patience 10, seed 42, dropout 0.2, hidden dim 256, ordinal loss weight 0.2.
- RSNA baseline LR: 1e-3, backbone LR 1e-4; DeiT backbone LR 5e-5.
- CUB default: AdamW, 15 epochs, LR 3e-4, weight decay 0.05, label smoothing 0.05.
- CUB ConvNeXt/DeiT rerun: 30-35 epochs, backbone LR 5e-5, head LR 5e-4, warmup 3, patience 8, drop path 0.1.

XAI methods:
- Grad-CAM, Grad-CAM++, Integrated Gradients, GradientSHAP, Occlusion, Guided Backpropagation, Attention Rollout for ViT/DeiT.

Metrics:
- Classification: RSNA weighted log loss, balanced accuracy, macro F1, AUC-OVR; CUB accuracy, top-5 accuracy, balanced accuracy, macro F1.
- Agreement: mean pairwise Spearman rho; top-20 percent saliency IoU.
- Faithfulness: deletion/insertion AUC.
- Clinical alignment: IoU between saliency mask and coordinate-derived expert ROI.
- Statistical tests: paired t-tests and Wilcoxon over folds; Spearman/Kendall cross-domain rank correlation.

Key results:
- RSNA agreement: DenseNet 0.434, ResNet 0.412, ConvNeXt 0.283, EfficientNet 0.223, DeiT 0.203, ViT 0.200.
- CUB agreement: DenseNet 0.518, ConvNeXt 0.441, ResNet 0.430, EfficientNet 0.412, ViT 0.283, DeiT 0.220.
- RSNA-CUB rank correlation: Spearman 0.886, p=0.019.
- Clinical expert IoU best mean: DenseNet 0.275.
- CBM non-leaky intervention: 90/152 wrong predictions fixed, 59.2%.
- EfficientNet audit: IG and GradientSHAP maps not identical; identical_fraction 0 across all folds.
