# MESFNet Release Verification

Date: 2026-09-06

- Five unit tests passed, including legacy key migration, collision rejection, resume-checkpoint rejection, D4 transforms, hole filling, data/configuration, and model forward/loss backward.
- An existing three-GN ConvNeXt-Base raw checkpoint was migrated: 542 tensors, 523 renamed keys, and all tensor values preserved exactly after serialization.
- Both old and renamed models loaded their respective checkpoints with strict key matching. On the same normalized 1024x1024 image on GPU0, mask, edge and morphology outputs were bitwise identical (maximum absolute difference 0).
- The new inference entrypoint completed D4 TTA with threshold 0.5 and hole area 8192.
- Training configurations remain unchanged. This release does not retrain the model, improve accuracy, or migrate full optimizer/RNG recovery states.

Runtime: PyTorch 2.4.1+cu124, torchvision 0.19.1+cu124, RTX 3090.

Legacy smoke-test checkpoint SHA256: `bc2d18eeca9cffd157b1de79b62e0d375bbf442e2ad68a3aa5b79e0e4c23865f`

Migrated smoke-test checkpoint SHA256: `ac5cd72f8023ad16ad3d59f5f89f7f743590c732b7e8e8e8c281cfdad4d7bf07`

Checkpoints, images and test predictions are kept outside this source release. Full training/resumption and a fresh Linux installation were not rerun.
