# MESFNet

**Multi-scale Edge-supervised Semantic Fusion Network**，多尺度边缘监督语义融合网络。

边缘分支通过辅助监督参与训练，默认配置不把边缘预测直接融合到分割头。名称对照见 `RENAMING.md`。

面向 1024×1024 RGB 遥感影像的水体二值分割模型，使用 ConvNeXt-Base 编码器、GatedDetailEnhancement、多尺度语义融合、SK 解码器，以及边缘和形态辅助监督。

这是一份从实验工程整理的独立源码副本。默认配置为三处 GroupNorm 版本：替换 `decoder_stage4.norm`、`decoder_stage3.norm` 和 `segmentation_fusion.1`，保留 ConvNeXt 的 LayerNorm。**该版本仍包含 SK，不是 no-SK。** 模型实现同时保留其他实验变体，训练和推理必须使用与权重一致的配置。

## 目录

```text
MESFNet/
  train.py                 # 跨平台训练入口、JSON配置、双流日志
  infer.py                 # 单视角/四向/D4八向推理
  configs/
    train.json             # 三处GN训练配置
    infer.json             # 对应推理配置
  model/
    mesfnet.py          # 模型实现
    train.py               # 原始训练引擎，保留损失和断点逻辑
    dataset_load.py        # 数据配对、增强、标签生成
    losses4.py             # 主任务与辅助损失
    postprocess.py         # 孔洞填充等后处理
    prepare_ablation_split.py
    build_hard_case_manifest.py
    gdalTools.py           # 可选TIFF支持，不是JPG/PNG必需依赖
  tools/interpolate_checkpoints.py
  tests/                   # 入口、数据处理、TTA和模型测试
  requirements.txt
  SOURCE_MANIFEST.json      # 核心文件来源和SHA256
```

## 环境

建议 Python 3.10。依赖版本来自原训练环境：PyTorch 2.4.1、torchvision 0.19.1、CUDA 12.4。Windows 和 Linux 均可使用。

```bash
python -m venv .venv
# Linux
source .venv/bin/activate
# Windows PowerShell 使用 .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

CPU 环境可从 PyTorch CPU wheel 源安装相同版本；完整训练建议使用 CUDA GPU。JPG/PNG 流程不需要 GDAL，处理原有 TIFF 输入时需另行安装匹配平台的 GDAL Python 绑定。

## 数据

数据集自行准备，不随源码发布。

```text
data/train/
  image/
    000000.jpg
    000001.jpg
  mask/
    000000_mask.png
    000001_mask.png
```

也支持 `images/labels` 布局。图像按 RGB 读取；掩膜为单通道，0 表示背景，255 表示水体。训练使用 1024×1024；图像双线性缩放，标签最近邻缩放。归一化为 ImageNet mean `[0.485, 0.456, 0.406]` 和 std `[0.229, 0.224, 0.225]`。

## 训练

从 ImageNet 预训练 backbone 开始，其他层重新初始化：

```bash
python train.py --config configs/train.json --data-root data/train --save-dir runs/gn3_150_seed42 --gpu 0
```

默认 150 epoch、batch size 2、梯度累积 2、AdamW、初始 LR=1e-4、余弦衰减到 1e-6、seed=42，每10轮保存。未指定划分时，按水体占比分层生成 85%/15% 训练/验证划分。它不是地理位置分组划分。只检查配置而不训练：在同一命令末尾加入 `--dry-run`。

可先生成所有实验共用的划分：

```bash
python model/prepare_ablation_split.py --data-root data/train --output-dir runs/fixed_split --seed 42
python train.py --config configs/train.json --data-root data/train --save-dir runs/gn3_fixed --split-dir runs/fixed_split --gpu 0
```

索引依赖配对文件排序，复用划分时必须保持相同数据集和文件命名。请保留原始数据版本或单独记录数据哈希。

每次训练使用新的 `--save-dir`，入口拒绝覆盖已有非空目录。配置目录只保留三处GN对应的 `train.json` 和 `infer.json`。若需要全部数据训练，在训练配置中设置 `FULL_TRAIN=true`，此时 train=val，指标仅作训练监控，且不再传 `--split-dir`。

### 断点续训

使用原配置、相同数据路径和划分，向原命令添加 `--resume`：

```bash
python train.py --config configs/train.json --data-root data/train --save-dir runs/gn3_fixed --split-dir runs/fixed_split --gpu 0 --resume
```

恢复模型、AdamW状态、调度器、AMP、RNG和CSV进度。`EPOCHS` 是总目标轮数，不是追加轮数；续训时保持不变。恢复依赖 `training_state.pth`（或缺失时回退 `training_state_prev.pth`）和完整CSV，不能用单独 best 权重代替。

严格签名会核对代码哈希、配置、环境与数据绝对路径。旧实验工程的 resume-state 不能承诺直接跨目录恢复；迁移目录或更换工程后，可用下面的权重微调方式，但这不等价于精确断点续训。

### 权重微调与 bad-case 增强

微调前，在 `configs/train.json` 中设置 `EPOCHS=30`、`LR=1e-5`、`MIN_LR=1e-6`、`PRETRAINED=false`，再执行下面的命令。这会重新建立优化器，不是断点续训；默认发布配置仍为150轮从头训练。

```bash
python train.py --config configs/train.json --data-root data/train --save-dir runs/gn3_ft30 --split-dir runs/fixed_split --init-checkpoint weights/gn3_best.pth --gpu 0
```

Bad-case增强还需设置 `AUG_MODE="targeted"`、`HARD_CASE_SAMPLING=true`，并在命令中添加 `--hard-case-manifest data/hard_case_manifest.csv`，启用最高3倍的加权采样。普通样本仍参与训练；验证集不进入训练采样器。清单列至少包含 `image,sample_weight,augmentation_profile,review_status,failure_type`；已有逐图指标时，可调用 `model/build_hard_case_manifest.py --help` 生成。

## 推理

权重另行放到 `weights/`。仅接受与模型结构一致的原始 state_dict `.pth`，按 strict 模式载入；不会在推理时下载预训练 backbone。

```bash
python infer.py --checkpoint weights/gn3_best.pth --input data/train/image --output-dir predictions/gn3 --config configs/infer.json --device cuda:0
python infer.py --checkpoint weights/gn3_best.pth --input data/train/image/000000.jpg --output-dir predictions/single --tta none --threshold 0.5 --max-hole-area 0
```

默认 D4 八向 sigmoid 概率平均、阈值0.50、填充最多8192像素的封闭空洞。`--tta flip4` 为水平/垂直翻转组合；`--tta none` 为单视角。CLI 的 `--max-hole-area 0` 关闭填洞。所有参数可覆盖，阈值应在同一推理方式的验证集上选择。

输入必须为1024×1024 JPG/PNG。输出直接写入指定目录：`000000_mask.png`，单通道、1024×1024、像素仅0/255，并生成 `inference_summary.json`。此源码包是公开研究工程，不是平台免配置提交包；用于比赛部署时需另行适配平台入口与目录要求。

随附推理配置仅对应三处GN权重；其他结构的权重不能直接套用。历史75/25插值或三处GN模型的线上阈值不可在不同结构之间直接套用。

## 权重插值

### 旧权重迁移

模块属性改名会改变 state_dict 键名。原 MSKNet 原始权重需先转换，再用于 MESFNet 推理、微调或插值：

```bash
python tools/migrate_legacy_checkpoint.py --input weights/legacy_best.pth --output weights/mesfnet_best.pth
```

工具只更换键名，不改变张量值，并生成哈希报告；拒绝覆盖已有文件。不接受完整训练恢复状态，也不绕过旧实验的恢复签名。迁移后需按原权重的结构选择配置，当前随附配置为三处GN。消融选项字符串与输出字典的 `mask/edge/morph` 保留兼容。

### 同结构权重插值

```bash
python tools/interpolate_checkpoints.py --base weights/base.pth --target weights/finetune.pth --target-weight 0.25 --output weights/mix075_025.pth
```

公式为 `0.75 * base + 0.25 * finetune`。两份权重必须键名、形状和类型一致；浮点状态线性插值，整数缓冲区取基线值。工具保存输入、输出哈希和混合系数。插值后的性能需重新评估。

## 输出与复现

训练保存 best IoU、best F1、last、每10轮权重、完整恢复状态、逐轮CSV、stdout/stderr，以及 `run_manifest.json`、索引、环境和代码哈希。根入口额外保存 `resolved_config.json`。训练结束后生成 `training_config.txt`。

原代码启用确定性设置但默认 `STRICT_DETERMINISTIC=false`；部分CUDA反向算子仍可能非确定，不能保证跨硬件逐位一致。梯度累积也不等价于增大BatchNorm统计使用的真实batch。开启严格模式可能因不支持确定性反向的算子报错。

该副本保留已有算法，不包含架构复盘中尚未验证的重构。这里的150轮默认配置是从头训练用配置，不是历史“先30轮、再延长120轮”调度轨迹的精确复刻。源码内不声明新的测试集成绩，也不附带历史实验的权重和原始数据。

公开副本训练入口仅负责训练和日志；原工作区的个人SMTP包装器独立保留，未附入本仓库。

## 验证

```bash
python -m unittest discover -s tests -v
python train.py --help
python infer.py --help
```

测试覆盖D4正逆变换、概率聚合、孔洞边界、数据配对、配置隔离和模型前向/损失反向。

## 预训练与代码来源

ConvNeXt 由 torchvision 提供，`PRETRAINED=true` 使用该版本 `ConvNeXt_Base_Weights.DEFAULT`（ImageNet-1K）。首次训练需可访问权重下载服务或已有缓存。

- torchvision ConvNeXt: https://pytorch.org/vision/0.19/models/convnext.html
- ConvNeXt论文: https://arxiv.org/abs/2201.03545

核心文件来源和SHA256见 `SOURCE_MANIFEST.json`。未擅自添加MIT等开源许可证；对外发布前由作者决定本项目的授权方式，第三方依赖遵循各自许可证。
