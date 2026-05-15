# SpineXNet

Kaggle-ready research code for **Multi-Granularity Concept-Grounded Explanations for Lumbar Spine Degenerative Classification** on the RSNA 2024 lumbar spine dataset.

The implementation includes:

- RSNA condition-level manifest creation from label coordinates
- DICOM-to-PNG crop preprocessing
- Stratified group cross-validation and site-holdout stress split
- Programmatic pseudo-concepts from RSNA labels
- Optional BiomedCLIP soft concept extraction
- Black-box baselines: ResNet-50, EfficientNet-B4, ConvNeXt-Tiny, ViT-Small, DenseNet-121
- Concept Bottleneck baseline
- Prototype-only and concept+prototype/no-residual interpretable baselines
- SpineXNet: ConvNeXt feature encoder, concept bottleneck, prototype evidence layer, ordinal severity head
- Ablation configs for concepts, prototypes, ordinal loss, backbone, and prototype count
- Classification evaluation and XAI faithfulness/agreement benchmark

See [KAGGLE_RUNBOOK.md](KAGGLE_RUNBOOK.md) for the exact Kaggle execution sequence.
