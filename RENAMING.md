# MESFNet Naming Map

MESFNet: Multi-scale Edge-supervised Semantic Fusion Network.

This is a naming-only release. Standard ConvNeXt, BatchNorm, GroupNorm and SK technical attribution is retained. Optional experimental modules remain optional.

| Previous identifier | MESFNet identifier |
|---|---|
| `MSKNet` | `MESFNet` |
| `SCPv2` | `GatedDetailEnhancement` |
| `SpatialPyramidAttentionBlock` | `PyramidContextAggregation` |
| `SKAttention` | `SelectiveKernelAttention` |
| `SKBlock` | `SelectiveKernelResidualBlock` |
| `MultiScaleSemanticFusion` | `HierarchicalSemanticFusion` |
| `FlowAlignment` | `LearnedFeatureAlignment` |
| `EdgeGuidedRefinement` | `BoundaryResidualRefinement` |
| `StableEdgeGuidedRefinement` | `ZeroInitBoundaryResidualRefinement` |
| `LightweightEdgeFusion` | `GatedBoundaryFusion` |
| `SingleScaleSemanticFusion` | `ShallowSemanticProjection` |
| `BilinearAlignment` | `BilinearFeatureAlignment` |
| `ConvBNAct` | `ConvNormActivation` |
| `s1` | `encoder_stage1` |
| `s2` | `encoder_stage2` |
| `s3` | `encoder_stage3` |
| `s4` | `encoder_stage4` |
| `scp1` | `detail_enhance1` |
| `scp2` | `detail_enhance2` |
| `scp3` | `detail_enhance3` |
| `spam` | `context_aggregation` |
| `lat3` | `lateral_projection3` |
| `dec4` | `decoder_stage4` |
| `dec3` | `decoder_stage3` |
| `flow_align` | `edge_alignment` |
| `mask_fuse` | `segmentation_fusion` |
| `mask_head` | `segmentation_head` |
| `edge_head` | `boundary_head` |
| `morph_fc` | `morphology_head` |
| `bn` | `norm` |
| `scp_factory` | `detail_factory` |

- `model/model_scpv2.py` -> `model/mesfnet.py`.
- Checkpoint prefix `msknet_scpv2_` -> `mesfnet_`.
- Metrics CSV `train_metrics_msknet_scpv2.csv` -> `train_metrics_mesfnet.csv`.
- `stem` and `semantic_fusion` remain unchanged.
- Existing ablation option strings and output keys `mask`, `edge`, `morph` remain unchanged for compatibility.
- The old `MSKNet_Ultimate` import alias is removed; import `MESFNet` from `model.mesfnet`.
- Attribute names change state_dict keys. Use `tools/migrate_legacy_checkpoint.py` for raw legacy checkpoints; tensor values are preserved.
- Full optimizer/RNG checkpoints from the previous source version require the old engine for exact resumption. They are not converted by this utility.
