# -*- coding: utf-8 -*-
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, __version__ as pillow_version
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from dataset_load import RemoteSensingDataset
from losses4 import TotalLoss
from mesfnet import MESFNet


SUPPORTED_ABLATIONS = {
    "no_scp",
    "no_spam",
    "no_sk",
    "no_semantic_fusion",
    "no_flow_align",
    "no_edge_branch",
    "no_edge_supervision",
    "no_morph_loss",
    "no_aux_loss",
    "no_edge_consistency",
    "light_edge_fusion",
    "fixed_flow_align",
    "edge_guided_refine",
    "edge_guided_refine_v2",
    "groupnorm_decoder",
    "groupnorm_decoder_semantic",
    "groupnorm_all",
    "mask_only",
}


def parse_ablations(value):
    ablations = tuple(sorted({item.strip().lower() for item in value.split(",") if item.strip()}))
    unknown = set(ablations) - SUPPORTED_ABLATIONS
    if unknown:
        raise ValueError(f"Unsupported ablation flags: {sorted(unknown)}")
    if "light_edge_fusion" in ablations and "no_flow_align" in ablations:
        raise ValueError("light_edge_fusion and no_flow_align are mutually exclusive")
    return ablations


class Config:
    seed = int(os.environ.get("SEED", "42"))
    epochs = int(os.environ.get("EPOCHS", "120"))
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    accumulation_steps = int(os.environ.get("ACCUMULATION_STEPS", "4"))
    lr = float(os.environ.get("LR", "1e-4"))
    min_lr = float(os.environ.get("MIN_LR", "1e-6"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "0.05"))
    val_ratio = float(os.environ.get("VAL_RATIO", "0.15"))
    full_train = os.environ.get("FULL_TRAIN", "0") == "1"
    augment_mode = os.environ.get("AUG_MODE", "basic")
    val_every = int(os.environ.get("VAL_EVERY", "1"))
    save_every = int(os.environ.get("SAVE_EVERY", "0"))
    init_ckpt = os.environ.get("INIT_CKPT", "")
    pretrained = os.environ.get("PRETRAINED", "1") == "1"
    convnext_variant = os.environ.get("CONVNEXT_VARIANT", "base")
    deterministic = os.environ.get("DETERMINISTIC", "1") == "1"
    strict_deterministic = os.environ.get("STRICT_DETERMINISTIC", "0") == "1"
    resume = os.environ.get("RESUME", "0") == "1"
    resume_state = os.environ.get("RESUME_STATE", "").strip()
    keep_resume_state = os.environ.get("KEEP_RESUME_STATE", "1") == "1"
    hard_case_manifest = os.environ.get("HARD_CASE_MANIFEST", "").strip()
    hard_case_sampling = os.environ.get("HARD_CASE_SAMPLING", "0") == "1"
    hard_case_max_weight = float(os.environ.get("HARD_CASE_MAX_WEIGHT", "3.0"))
    freeze_nondeterministic_paths = os.environ.get("FREEZE_NONDETERMINISTIC_PATHS", "0") == "1"
    ablations = parse_ablations(os.environ.get("ABLATIONS", ""))
    split_source_dir_value = os.environ.get("SPLIT_SOURCE_DIR", "").strip()
    split_source_dir = Path(split_source_dir_value) if split_source_dir_value else None

    w_mask = float(os.environ.get("W_MASK", "1.0"))
    w_iou = float(os.environ.get("W_IOU", "0.0"))
    w_edge = float(os.environ.get("W_EDGE", "0.2"))
    w_morph = float(os.environ.get("W_MORPH", "0.01"))
    w_aux = float(os.environ.get("W_AUX", "0.02"))
    lambda_cons = float(os.environ.get("LAMBDA_CONS", "0.05"))
    morph_warmup_epochs = int(os.environ.get("MORPH_WARMUP_EPOCHS", "0"))
    morph_ramp_epochs = int(os.environ.get("MORPH_RAMP_EPOCHS", "0"))
    morph_loss_type = os.environ.get("MORPH_LOSS_TYPE", "mse").strip().lower()
    boundary_tolerance = int(os.environ.get("BOUNDARY_TOLERANCE", "3"))
    d4_consistency_weight = float(os.environ.get("D4_CONSISTENCY_WEIGHT", "0.0"))
    d4_consistency_start_epoch = int(os.environ.get("D4_CONSISTENCY_START_EPOCH", "5"))

    if "no_edge_branch" in ablations:
        w_edge = 0.0
    if "no_edge_supervision" in ablations:
        w_edge = 0.0
        lambda_cons = 0.0
    if "no_morph_loss" in ablations:
        w_morph = 0.0
    if "no_aux_loss" in ablations:
        w_aux = 0.0
    if "no_edge_consistency" in ablations:
        lambda_cons = 0.0
    if "mask_only" in ablations:
        w_edge = 0.0
        w_morph = 0.0
        w_aux = 0.0
        lambda_cons = 0.0

    default_data_root = Path(__file__).resolve().parents[1] / "data" / "train"
    data_root = Path(os.environ.get("WATER_DATA_ROOT", str(default_data_root)))

    img_size = (1024, 1024)
    num_workers = int(os.environ.get("NUM_WORKERS", "4"))

    save_dir = Path(os.environ.get("SAVE_DIR", str(Path("runs") / "water_mesfnet")))
    best_ckpt = "mesfnet_best_iou.pth"
    best_f1_ckpt = "mesfnet_best_f1.pth"
    best_hard_ckpt = "mesfnet_best_hard_iou.pth"
    last_ckpt = "mesfnet_last.pth"
    csv_log = "train_metrics_mesfnet.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = Config.deterministic
    torch.backends.cudnn.benchmark = not Config.deterministic
    if Config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=not Config.strict_deterministic)


def environment_info():
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except Exception as exc:
        torchvision_version = f"unavailable: {exc}"
    try:
        import cv2

        opencv_version = cv2.__version__
    except Exception as exc:
        opencv_version = f"unavailable: {exc}"

    cuda_names = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            cuda_names.append(torch.cuda.get_device_name(idx))

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pillow": pillow_version,
        "opencv": opencv_version,
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count_visible": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_device_names_visible": cuda_names,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", ""),
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
    }


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class EpochWeightedSampler(Sampler):
    """Weighted replacement sampler with an epoch-derived deterministic order."""

    def __init__(self, weights, num_samples, seed):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 1_000_003)
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True,
            generator=generator,
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


def capture_rng_state(train_generator):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "train_generator": train_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state, train_generator):
    def cpu_byte_tensor(value):
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(cpu_byte_tensor(state["torch"]))
    train_generator.set_state(cpu_byte_tensor(state["train_generator"]))
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([cpu_byte_tensor(item) for item in state["cuda"]])


def checkpoint_path(cfg):
    if cfg.resume_state:
        return Path(cfg.resume_state)
    if cfg.resume:
        current = cfg.save_dir / "training_state.pth"
        previous = cfg.save_dir / "training_state_prev.pth"
        return current if current.is_file() else previous
    return None


def file_sha256(path):
    path = Path(path)
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hard_case_manifest(path, pairs, max_weight):
    default = {
        "weights": [1.0] * len(pairs),
        "profiles": {},
        "hard_names": set(),
        "counts": {},
    }
    if not path:
        return default

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Hard-case manifest not found: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        rows = list(csv.DictReader(file_obj))

    required = {"image", "sample_weight", "augmentation_profile", "review_status", "failure_type"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Hard-case manifest is missing columns: {sorted(required)}")
    row_by_name = {}
    counts = {}
    for row in rows:
        image_name = row["image"]
        if image_name in row_by_name:
            raise ValueError(f"Duplicate image in hard-case manifest: {image_name}")
        row_by_name[image_name] = row
        counts[row["failure_type"]] = counts.get(row["failure_type"], 0) + 1

    weights = []
    profiles = {}
    hard_names = set()
    for image_path, _ in pairs:
        row = row_by_name.get(image_path.name)
        if row is None or row["review_status"] != "eligible":
            weights.append(1.0)
            continue
        weight = min(max(1.0, float(row["sample_weight"])), max_weight)
        weights.append(weight)
        profile = row["augmentation_profile"].strip() or "default"
        if profile != "default":
            profiles[image_path.name] = profile
        if weight > 1.0:
            hard_names.add(image_path.name)

    return {
        "weights": weights,
        "profiles": profiles,
        "hard_names": hard_names,
        "counts": counts,
    }


def code_hashes():
    return {
        name: file_sha256(Path(__file__).resolve().parent / name)
        for name in [
            "train.py",
            "dataset_load.py",
            "mesfnet.py",
            "losses4.py",
            "postprocess.py",
            "build_hard_case_manifest.py",
        ]
    }


def training_signature(cfg, train_idx, val_idx):
    train_array = np.asarray(train_idx, dtype=np.int64)
    val_array = np.asarray(val_idx, dtype=np.int64)
    return {
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "accumulation_steps": cfg.accumulation_steps,
        "lr": cfg.lr,
        "min_lr": cfg.min_lr,
        "weight_decay": cfg.weight_decay,
        "full_train": cfg.full_train,
        "augment_mode": cfg.augment_mode,
        "val_every": cfg.val_every,
        "save_every": cfg.save_every,
        "deterministic": cfg.deterministic,
        "convnext_variant": cfg.convnext_variant,
        "strict_deterministic": cfg.strict_deterministic,
        "num_workers": cfg.num_workers,
        "ablations": list(cfg.ablations),
        "loss_weights": [cfg.w_mask, cfg.w_iou, cfg.w_edge, cfg.w_morph, cfg.w_aux, cfg.lambda_cons],
        "morph_schedule": [cfg.morph_warmup_epochs, cfg.morph_ramp_epochs, cfg.morph_loss_type],
        "d4_consistency": [cfg.d4_consistency_weight, cfg.d4_consistency_start_epoch],
        "data_root": str(cfg.data_root.resolve()),
        "hard_case_sampling": cfg.hard_case_sampling,
        "hard_case_max_weight": cfg.hard_case_max_weight,
        "freeze_nondeterministic_paths": cfg.freeze_nondeterministic_paths,
        "hard_case_manifest_sha256": file_sha256(cfg.hard_case_manifest) if cfg.hard_case_manifest else "",
        "train_indices_sha256": hashlib.sha256(train_array.tobytes()).hexdigest(),
        "val_indices_sha256": hashlib.sha256(val_array.tobytes()).hexdigest(),
        "code_sha256": code_hashes(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def validate_training_signature(saved, current):
    if saved is None:
        raise ValueError("Resume checkpoint has no training signature; exact reproducibility cannot be guaranteed")
    mismatches = []
    for key in sorted(set(saved) | set(current)):
        if saved.get(key) != current.get(key):
            mismatches.append(f"{key}: checkpoint={saved.get(key)!r}, current={current.get(key)!r}")
    if mismatches:
        details = "\n  ".join(mismatches)
        raise ValueError(f"Resume configuration mismatch:\n  {details}")


def reconcile_resume_csv(csv_path, completed_epochs):
    if completed_epochs == 0:
        return
    if not csv_path.is_file():
        raise FileNotFoundError(f"Resume CSV is missing: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file_obj:
        rows = list(csv.reader(file_obj))
    if not rows:
        raise ValueError(f"Resume CSV is empty: {csv_path}")

    header, data_rows = rows[0], rows[1:]
    kept = [row for row in data_rows if row and int(row[0]) <= completed_epochs]
    epochs = [int(row[0]) for row in kept]
    expected = list(range(1, completed_epochs + 1))
    if epochs != expected:
        raise ValueError(f"Resume CSV epochs do not match checkpoint: expected 1..{completed_epochs}, got {epochs}")
    if len(kept) != len(data_rows):
        temp_path = csv_path.with_suffix(csv_path.suffix + ".resume_tmp")
        with temp_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(header)
            writer.writerows(kept)
        os.replace(temp_path, csv_path)


def save_run_manifest(cfg, train_idx, val_idx, hard_case_info):
    train_idx_array = np.asarray(train_idx, dtype=np.int64)
    val_idx_array = np.asarray(val_idx, dtype=np.int64)
    manifest = {
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "accumulation_steps": cfg.accumulation_steps,
        "lr": cfg.lr,
        "min_lr": cfg.min_lr,
        "weight_decay": cfg.weight_decay,
        "full_train": cfg.full_train,
        "augment_mode": cfg.augment_mode,
        "val_every": cfg.val_every,
        "save_every": cfg.save_every,
        "deterministic": cfg.deterministic,
        "strict_deterministic": cfg.strict_deterministic,
        "pretrained_backbone": cfg.pretrained,
        "convnext_variant": cfg.convnext_variant,
        "resume": cfg.resume,
        "resume_state": cfg.resume_state,
        "keep_resume_state": cfg.keep_resume_state,
        "hard_case_manifest": cfg.hard_case_manifest,
        "hard_case_manifest_sha256": file_sha256(cfg.hard_case_manifest) if cfg.hard_case_manifest else "",
        "hard_case_sampling": cfg.hard_case_sampling,
        "hard_case_max_weight": cfg.hard_case_max_weight,
        "freeze_nondeterministic_paths": cfg.freeze_nondeterministic_paths,
        "hard_case_counts": hard_case_info["counts"],
        "hard_case_weighted_samples": len(hard_case_info["hard_names"]),
        "ablations": list(cfg.ablations),
        "split_source_dir": str(cfg.split_source_dir) if cfg.split_source_dir else "",
        "loss_weights": {
            "w_mask": cfg.w_mask,
            "w_iou": cfg.w_iou,
            "w_edge": cfg.w_edge,
            "w_morph": cfg.w_morph,
            "w_aux": cfg.w_aux,
            "lambda_cons": cfg.lambda_cons,
        },
        "morph_schedule": {
            "warmup_epochs": cfg.morph_warmup_epochs,
            "ramp_epochs": cfg.morph_ramp_epochs,
            "loss_type": cfg.morph_loss_type,
        },
        "boundary_tolerance_px": cfg.boundary_tolerance,
        "d4_consistency": {
            "weight": cfg.d4_consistency_weight,
            "start_epoch": cfg.d4_consistency_start_epoch,
            "transform_schedule": "deterministic_cycle_over_7_non_identity_D4_elements",
            "loss": "probability_mse",
        },
        "initialization": "checkpoint" if cfg.init_ckpt else ("torchvision_pretrained" if cfg.pretrained else "random"),
        "data_root": str(cfg.data_root),
        "img_size": list(cfg.img_size),
        "save_dir": str(cfg.save_dir),
        "init_ckpt": cfg.init_ckpt,
        "init_ckpt_sha256": file_sha256(cfg.init_ckpt) if cfg.init_ckpt else "",
        "code_sha256": code_hashes(),
        "environment": environment_info(),
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "indices_sha256": {
            "train": hashlib.sha256(train_idx_array.tobytes()).hexdigest(),
            "val": hashlib.sha256(val_idx_array.tobytes()).hexdigest(),
        },
    }
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.save_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    np.savetxt(cfg.save_dir / "train_indices.txt", train_idx_array, fmt="%d")
    np.savetxt(cfg.save_dir / "val_indices.txt", val_idx_array, fmt="%d")


def setup_file_logging():
    log_file = os.environ.get("LOG_FILE", "")
    err_file = os.environ.get("ERR_FILE", "")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = open(log_file, "a", encoding="utf-8", buffering=1)
    if err_file:
        Path(err_file).parent.mkdir(parents=True, exist_ok=True)
        sys.stderr = open(err_file, "a", encoding="utf-8", buffering=1)


def mask_foreground_ratio(mask_path):
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)
    return float((arr >= 128).mean())


def make_stratified_split(pairs, val_ratio=0.15, seed=42, num_bins=5):
    ratios = np.array([mask_foreground_ratio(mask_path) for _, mask_path in pairs])
    quantiles = np.linspace(0, 1, num_bins + 1)
    edges = np.quantile(ratios, quantiles)
    rng = random.Random(seed)

    train_indices = []
    val_indices = []
    for bin_idx in range(num_bins):
        left = edges[bin_idx]
        right = edges[bin_idx + 1]
        if bin_idx == num_bins - 1:
            indices = np.where((ratios >= left) & (ratios <= right))[0].tolist()
        else:
            indices = np.where((ratios >= left) & (ratios < right))[0].tolist()

        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_ratio))) if indices else 0
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def load_split_indices(split_dir, dataset_size):
    split_dir = Path(split_dir)
    train_path = split_dir / "train_indices.txt"
    val_path = split_dir / "val_indices.txt"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(f"Split source must contain train_indices.txt and val_indices.txt: {split_dir}")

    train_indices = np.atleast_1d(np.loadtxt(train_path, dtype=np.int64)).tolist()
    val_indices = np.atleast_1d(np.loadtxt(val_path, dtype=np.int64)).tolist()
    all_indices = train_indices + val_indices
    if not all_indices or min(all_indices) < 0 or max(all_indices) >= dataset_size:
        raise ValueError(f"Split indices are outside dataset range [0, {dataset_size - 1}]")
    if len(train_indices) != len(set(train_indices)) or len(val_indices) != len(set(val_indices)):
        raise ValueError("Split source contains duplicate indices")
    if set(train_indices) & set(val_indices):
        raise ValueError("Train and validation indices overlap")
    return train_indices, val_indices


def calculate_metrics(pred_logits, target, threshold=0.5):
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    target = target.float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    return f1, iou, precision, recall, acc


def update_confusion(pred_logits, target, cm, threshold=0.5):
    pred = torch.sigmoid(pred_logits) > threshold
    target = target > 0.5
    cm["tp"] += torch.logical_and(pred, target).sum().item()
    cm["fp"] += torch.logical_and(pred, ~target).sum().item()
    cm["fn"] += torch.logical_and(~pred, target).sum().item()
    cm["tn"] += torch.logical_and(~pred, ~target).sum().item()


def metrics_from_confusion(cm):
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    return f1, iou, precision, recall, acc


def mask_boundary(mask):
    mask = mask > 0.5
    mask_float = mask.float()
    dilated = F.max_pool2d(mask_float, kernel_size=3, stride=1, padding=1) > 0.5
    eroded = -F.max_pool2d(-mask_float, kernel_size=3, stride=1, padding=1) > 0.5
    return torch.logical_xor(dilated, eroded)


def update_boundary_stats(pred_logits, target, stats, tolerance=3):
    pred_boundary = mask_boundary(torch.sigmoid(pred_logits) > 0.5)
    target_boundary = mask_boundary(target > 0.5)
    kernel_size = 2 * tolerance + 1
    pred_region = F.max_pool2d(pred_boundary.float(), kernel_size, stride=1, padding=tolerance) > 0.5
    target_region = F.max_pool2d(target_boundary.float(), kernel_size, stride=1, padding=tolerance) > 0.5

    stats["matched_pred"] += torch.logical_and(pred_boundary, target_region).sum().item()
    stats["pred_count"] += pred_boundary.sum().item()
    stats["matched_target"] += torch.logical_and(target_boundary, pred_region).sum().item()
    stats["target_count"] += target_boundary.sum().item()
    stats["region_intersection"] += torch.logical_and(pred_region, target_region).sum().item()
    stats["region_union"] += torch.logical_or(pred_region, target_region).sum().item()


def boundary_metrics(stats):
    precision = stats["matched_pred"] / (stats["pred_count"] + 1e-7)
    recall = stats["matched_target"] / (stats["target_count"] + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    iou = stats["region_intersection"] / (stats["region_union"] + 1e-7)
    return f1, iou


def scheduled_morph_weight(epoch, cfg):
    """Return the effective morphology weight for a zero-based epoch."""
    if epoch < cfg.morph_warmup_epochs:
        return 0.0
    if cfg.morph_ramp_epochs <= 0:
        return cfg.w_morph
    ramp_step = epoch - cfg.morph_warmup_epochs + 1
    progress = min(1.0, ramp_step / cfg.morph_ramp_epochs)
    return cfg.w_morph * progress


def apply_d4_transform(tensor, transform_index):
    """Apply one of the eight D4 transforms to a BCHW tensor."""
    if transform_index < 0 or transform_index > 7:
        raise ValueError(f"D4 transform index must be in [0, 7], got {transform_index}")
    if transform_index < 4:
        return torch.rot90(tensor, transform_index, dims=(-2, -1))
    return torch.flip(torch.rot90(tensor, transform_index - 4, dims=(-2, -1)), dims=(-1,))


def invert_d4_transform(tensor, transform_index):
    """Undo apply_d4_transform for a BCHW tensor."""
    if transform_index < 4:
        return torch.rot90(tensor, -transform_index, dims=(-2, -1))
    unflipped = torch.flip(tensor, dims=(-1,))
    return torch.rot90(unflipped, -(transform_index - 4), dims=(-2, -1))


def train_one_epoch(model, loader, criterion, optimizer, scaler, cfg, epoch):
    model.train()
    running_loss = 0.0
    running_d4_consistency = 0.0
    use_amp = cfg.device.type == "cuda"

    pbar = tqdm(loader, desc="Training", leave=False, disable=not sys.stderr.isatty())
    optimizer.zero_grad(set_to_none=True)
    for step, (img, mask, edge, morph) in enumerate(pbar):
        img = img.to(cfg.device, non_blocking=True)
        mask = mask.to(cfg.device, non_blocking=True)
        edge = edge.to(cfg.device, non_blocking=True)
        morph = morph.to(cfg.device, non_blocking=True)

        with torch.amp.autocast(device_type=cfg.device.type, enabled=use_amp):
            d4_active = cfg.d4_consistency_weight > 0 and epoch >= cfg.d4_consistency_start_epoch
            if d4_active:
                with torch.no_grad():
                    reference_prob = torch.sigmoid(model(img)["mask"])
                transform_index = 1 + ((epoch * len(loader) + step) % 7)
                transformed_img = apply_d4_transform(img, transform_index)
                transformed_mask = apply_d4_transform(mask, transform_index)
                transformed_edge = apply_d4_transform(edge, transform_index)
                outputs = model(transformed_img)
                raw_loss = criterion(outputs, transformed_mask, transformed_edge, morph)
                restored_logits = invert_d4_transform(outputs["mask"], transform_index)
                d4_consistency = F.mse_loss(
                    reference_prob,
                    torch.sigmoid(restored_logits),
                )
                raw_loss = raw_loss + cfg.d4_consistency_weight * d4_consistency
            else:
                outputs = model(img)
                raw_loss = criterion(outputs, mask, edge, morph)
                d4_consistency = raw_loss.new_zeros(())
            loss = raw_loss / cfg.accumulation_steps

        scaler.scale(loss).backward()

        should_step = (step + 1) % cfg.accumulation_steps == 0 or (step + 1) == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += raw_loss.item() * img.size(0)
        running_d4_consistency += d4_consistency.item() * img.size(0)
        if not pbar.disable:
            pbar.set_postfix({"loss": f"{raw_loss.item():.4f}"})

    return (
        running_loss / len(loader.dataset),
        running_d4_consistency / len(loader.dataset),
    )


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, cfg):
    model.eval()
    use_amp = cfg.device.type == "cuda"

    total_loss = 0.0
    mask_cm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    edge_cm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    boundary_stats = {
        "matched_pred": 0,
        "pred_count": 0,
        "matched_target": 0,
        "target_count": 0,
        "region_intersection": 0,
        "region_union": 0,
    }
    morph_squared_error = 0.0
    morph_abs_error = 0.0
    morph_elements = 0
    per_image_ious = []
    zero_iou_count = 0
    iou_lt_05_count = 0
    small_water_cm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    hard_case_cm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    hard_case_count = 0

    pbar = tqdm(loader, desc="Validating", leave=False, disable=not sys.stderr.isatty())
    for batch in pbar:
        if len(batch) == 5:
            img, mask, edge, morph, image_names = batch
        else:
            img, mask, edge, morph = batch
            image_names = [""] * img.size(0)
        img = img.to(cfg.device, non_blocking=True)
        mask = mask.to(cfg.device, non_blocking=True)
        edge = edge.to(cfg.device, non_blocking=True)
        morph = morph.to(cfg.device, non_blocking=True)

        with torch.amp.autocast(device_type=cfg.device.type, enabled=use_amp):
            outputs = model(img)
            loss = criterion(outputs, mask, edge, morph)

        batch_size = img.size(0)
        total_loss += loss.item() * batch_size
        update_confusion(outputs["mask"], mask, mask_cm)
        update_confusion(outputs["edge"], edge, edge_cm)
        update_boundary_stats(outputs["mask"], mask, boundary_stats, cfg.boundary_tolerance)
        morph_squared_error += F.mse_loss(outputs["morph"], morph, reduction="sum").item()
        morph_abs_error += F.l1_loss(outputs["morph"], morph, reduction="sum").item()
        morph_elements += morph.numel()

        for batch_idx, image_name in enumerate(image_names):
            sample_cm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            update_confusion(
                outputs["mask"][batch_idx : batch_idx + 1],
                mask[batch_idx : batch_idx + 1],
                sample_cm,
            )
            sample_iou = metrics_from_confusion(sample_cm)[1]
            per_image_ious.append(sample_iou)
            if sample_iou < 0.5:
                iou_lt_05_count += 1
            if sample_cm["tp"] == 0 and sample_cm["fn"] > 0:
                zero_iou_count += 1

            gt_ratio = (sample_cm["tp"] + sample_cm["fn"]) / float(mask[batch_idx].numel())
            if gt_ratio <= 0.02:
                for key in small_water_cm:
                    small_water_cm[key] += sample_cm[key]
            if image_name in cfg.hard_case_names:
                hard_case_count += 1
                for key in hard_case_cm:
                    hard_case_cm[key] += sample_cm[key]

        current_mask_metrics = metrics_from_confusion(mask_cm)
        current_edge_metrics = metrics_from_confusion(edge_cm)
        if not pbar.disable:
            pbar.set_postfix(
                {
                    "mask_iou": f"{current_mask_metrics[1]:.4f}",
                    "edge_iou": f"{current_edge_metrics[1]:.4f}",
                }
            )

    mask_metrics = metrics_from_confusion(mask_cm)
    edge_metrics = metrics_from_confusion(edge_cm)
    small_water_metrics = metrics_from_confusion(small_water_cm)
    hard_case_metrics = metrics_from_confusion(hard_case_cm)
    sorted_ious = sorted(per_image_ious)
    bottom_1_count = max(1, int(np.ceil(len(sorted_ious) * 0.01)))
    bottom_5_count = max(1, int(np.ceil(len(sorted_ious) * 0.05)))
    mask_boundary_f1, mask_boundary_iou = boundary_metrics(boundary_stats)
    return {
        "loss": total_loss / len(loader.dataset),
        "mask_f1": mask_metrics[0],
        "mask_iou": mask_metrics[1],
        "mask_precision": mask_metrics[2],
        "mask_recall": mask_metrics[3],
        "mask_acc": mask_metrics[4],
        "mask_boundary_f1": mask_boundary_f1,
        "mask_boundary_iou": mask_boundary_iou,
        "edge_f1": edge_metrics[0],
        "edge_iou": edge_metrics[1],
        "morph_mse": morph_squared_error / max(morph_elements, 1),
        "morph_mae": morph_abs_error / max(morph_elements, 1),
        "bottom_1_mean_iou": float(np.mean(sorted_ious[:bottom_1_count])),
        "bottom_5_mean_iou": float(np.mean(sorted_ious[:bottom_5_count])),
        "iou_lt_05_count": iou_lt_05_count,
        "zero_iou_count": zero_iou_count,
        "small_water_iou": small_water_metrics[1],
        "small_water_recall": small_water_metrics[3],
        "hard_case_iou": hard_case_metrics[1] if hard_case_count else 0.0,
        "hard_case_recall": hard_case_metrics[3] if hard_case_count else 0.0,
        "hard_case_count": hard_case_count,
    }


def main():
    setup_file_logging()
    cfg = Config()
    if cfg.init_ckpt and (cfg.resume or cfg.resume_state):
        raise ValueError("INIT_CKPT is weights-only initialization and cannot be combined with RESUME/RESUME_STATE")
    if cfg.augment_mode == "targeted" and not cfg.hard_case_manifest:
        raise ValueError("AUG_MODE=targeted requires HARD_CASE_MANIFEST")
    if cfg.hard_case_sampling and not cfg.hard_case_manifest:
        raise ValueError("HARD_CASE_SAMPLING=1 requires HARD_CASE_MANIFEST")
    if cfg.strict_deterministic and not cfg.freeze_nondeterministic_paths:
        raise ValueError(
            "STRICT_DETERMINISTIC=1 requires FREEZE_NONDETERMINISTIC_PATHS=1 for mesfnet CUDA training"
        )
    if cfg.strict_deterministic and not cfg.deterministic:
        raise ValueError("STRICT_DETERMINISTIC=1 requires DETERMINISTIC=1")
    if cfg.morph_warmup_epochs < 0 or cfg.morph_ramp_epochs < 0:
        raise ValueError("Morph warmup and ramp epochs must be non-negative")
    if cfg.morph_loss_type not in {"mse", "smooth_l1"}:
        raise ValueError("MORPH_LOSS_TYPE must be 'mse' or 'smooth_l1'")
    if cfg.save_every < 0:
        raise ValueError("SAVE_EVERY must be non-negative")
    if cfg.d4_consistency_weight < 0 or cfg.d4_consistency_start_epoch < 0:
        raise ValueError("D4 consistency weight and start epoch must be non-negative")
    seed_everything(cfg.seed)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    stale_resume_tmp = cfg.save_dir / "training_state.tmp.pth"
    if stale_resume_tmp.is_file():
        stale_resume_tmp.unlink()
        print("removed stale temporary resume state: training_state.tmp.pth")

    print("=" * 60)
    print("Training MESFNet for HRRSI-WS water segmentation")
    print(f"data_root: {cfg.data_root}")
    print(f"device: {cfg.device}")
    print("=" * 60)

    full_ds = RemoteSensingDataset(cfg.data_root, img_size=cfg.img_size, augment=False)
    if cfg.full_train:
        train_idx = list(range(len(full_ds)))
        val_idx = list(range(len(full_ds)))
    elif cfg.split_source_dir:
        train_idx, val_idx = load_split_indices(cfg.split_source_dir, len(full_ds))
    else:
        train_idx, val_idx = make_stratified_split(
            full_ds.pairs,
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
        )
    hard_case_info = load_hard_case_manifest(
        cfg.hard_case_manifest,
        full_ds.pairs,
        cfg.hard_case_max_weight,
    )
    cfg.hard_case_names = set(hard_case_info["hard_names"])
    run_manifest_path = cfg.save_dir / "run_manifest.json"
    is_resume_request = cfg.resume or bool(cfg.resume_state)
    if not (is_resume_request and run_manifest_path.is_file()):
        save_run_manifest(cfg, train_idx, val_idx, hard_case_info)
        if cfg.hard_case_manifest:
            frozen_manifest = cfg.save_dir / "hard_case_manifest_used.csv"
            frozen_manifest.write_bytes(Path(cfg.hard_case_manifest).read_bytes())

    train_ds = RemoteSensingDataset(
        cfg.data_root,
        img_size=cfg.img_size,
        indices=train_idx,
        augment=True,
        augment_mode=cfg.augment_mode,
        augment_profiles=hard_case_info["profiles"],
    )
    val_ds = RemoteSensingDataset(
        cfg.data_root,
        img_size=cfg.img_size,
        indices=val_idx,
        augment=False,
        return_name=True,
    )

    train_generator = torch.Generator().manual_seed(cfg.seed)
    val_generator = torch.Generator().manual_seed(cfg.seed)

    train_sampler = None
    if cfg.hard_case_sampling:
        train_weights = [hard_case_info["weights"][idx] for idx in train_idx]
        train_sampler = EpochWeightedSampler(train_weights, len(train_ds), cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.type == "cuda",
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=val_generator,
    )

    print(f"train samples: {len(train_ds)}")
    print(f"val samples: {len(val_ds)}")
    print(f"full train mode: {cfg.full_train}")
    print(f"augment mode: {cfg.augment_mode}")
    print(f"hard-case manifest: {cfg.hard_case_manifest or 'disabled'}")
    print(f"hard-case sampling: {cfg.hard_case_sampling}")
    print(f"weighted hard samples: {len(cfg.hard_case_names)}")
    print(f"freeze nondeterministic auxiliary paths: {cfg.freeze_nondeterministic_paths}")
    print(f"convnext variant: {cfg.convnext_variant}")
    print(f"pretrained backbone: {cfg.pretrained}")
    print(f"ablations: {list(cfg.ablations) if cfg.ablations else ['full_model']}")
    print(f"split source: {cfg.split_source_dir or 'generated'}")
    print(
        "loss weights: "
        f"mask={cfg.w_mask}, iou={cfg.w_iou}, edge={cfg.w_edge}, morph={cfg.w_morph}, "
        f"aux={cfg.w_aux}, edge_consistency={cfg.lambda_cons}"
    )
    print(
        "morph schedule: "
        f"warmup={cfg.morph_warmup_epochs}, ramp={cfg.morph_ramp_epochs}, "
        f"loss={cfg.morph_loss_type}"
    )
    print(f"deterministic mode: {cfg.deterministic}")
    print(f"strict deterministic mode: {cfg.strict_deterministic}")
    print(f"image size: {cfg.img_size}")
    print(f"batch size: {cfg.batch_size} | accumulation steps: {cfg.accumulation_steps}")
    print(
        "D4 consistency: "
        f"weight={cfg.d4_consistency_weight}, start_epoch={cfg.d4_consistency_start_epoch}"
    )

    model = MESFNet(
        pretrained=cfg.pretrained,
        num_classes=1,
        backbone_variant=cfg.convnext_variant,
        ablations=cfg.ablations,
        freeze_nondeterministic_paths=cfg.freeze_nondeterministic_paths,
    ).to(cfg.device)
    if cfg.init_ckpt:
        state_dict = torch.load(cfg.init_ckpt, map_location=cfg.device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"loaded init checkpoint: {cfg.init_ckpt}")

    criterion = TotalLoss(
        w_mask=cfg.w_mask,
        w_iou=cfg.w_iou,
        w_edge=cfg.w_edge,
        w_morph=cfg.w_morph,
        w_aux=cfg.w_aux,
        lambda_cons=cfg.lambda_cons,
        morph_loss_type=cfg.morph_loss_type,
    ).to(cfg.device)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(
        trainable_parameters,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.device.type == "cuda")

    csv_path = cfg.save_dir / cfg.csv_log
    best_iou = 0.0
    best_f1 = 0.0
    best_hard_iou = 0.0
    start_epoch = 0
    current_signature = training_signature(cfg, train_idx, val_idx)
    resume_path = checkpoint_path(cfg)
    if resume_path is not None:
        if not resume_path.is_file():
            if cfg.resume_state:
                raise FileNotFoundError(f"Resume state not found: {resume_path}")
            print(f"resume requested but no state exists yet; starting fresh: {resume_path}")
        else:
            resume_state = torch.load(resume_path, map_location=cfg.device, weights_only=False)
            validate_training_signature(resume_state.get("training_signature"), current_signature)
            model.load_state_dict(resume_state["model_state"])
            optimizer.load_state_dict(resume_state["optimizer_state"])
            scheduler.load_state_dict(resume_state["scheduler_state"])
            if resume_state.get("scaler_state"):
                scaler.load_state_dict(resume_state["scaler_state"])
            best_iou = float(resume_state.get("best_iou", 0.0))
            best_f1 = float(resume_state.get("best_f1", 0.0))
            best_hard_iou = float(resume_state.get("best_hard_iou", 0.0))
            start_epoch = int(resume_state["epoch"]) + 1
            restore_rng_state(resume_state["rng_state"], train_generator)
            reconcile_resume_csv(csv_path, start_epoch)
            print(f"resumed from {resume_path} at epoch {start_epoch}/{cfg.epochs}")
    start_time = time.time()

    csv_mode = "a" if start_epoch > 0 and csv_path.is_file() else "w"
    with csv_path.open(csv_mode, newline="", encoding="utf-8") as csv_fp:
        writer = csv.writer(csv_fp)
        if csv_mode == "w":
            writer.writerow(
                [
                    "Epoch",
                    "Train_Loss",
                    "Val_Loss",
                    "Mask_F1",
                    "Mask_IoU",
                    "Mask_Precision",
                    "Mask_Recall",
                    "Mask_Acc",
                    "Mask_Boundary_F1",
                    "Mask_Boundary_IoU",
                    "Edge_F1",
                    "Edge_IoU",
                "Morph_MSE",
                "Morph_MAE",
                "Bottom_1_Mean_IoU",
                "Bottom_5_Mean_IoU",
                "IoU_LT_0_5_Count",
                "Zero_IoU_Count",
                "Small_Water_IoU",
                "Small_Water_Recall",
                "Hard_Case_IoU",
                "Hard_Case_Recall",
                "Hard_Case_Count",
                "Effective_Morph_Weight",
                "Train_D4_Consistency",
                "D4_Consistency_Weight",
                "Learning_Rate",
                ]
            )

        for epoch in range(start_epoch, cfg.epochs):
            epoch_start = time.time()
            effective_morph_weight = scheduled_morph_weight(epoch, cfg)
            criterion.set_morph_weight(effective_morph_weight)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss, train_d4_consistency = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, cfg, epoch
            )
            should_validate = (epoch + 1) % cfg.val_every == 0 or (epoch + 1) == cfg.epochs
            val_metrics = validate_one_epoch(model, val_loader, criterion, cfg) if should_validate else None

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            if val_metrics is not None and val_metrics["mask_iou"] > best_iou:
                best_iou = val_metrics["mask_iou"]
                torch.save(model.state_dict(), cfg.save_dir / cfg.best_ckpt)
                print(f"[*] Epoch {epoch + 1}: new best mask IoU = {best_iou:.4f}")
            if val_metrics is not None and val_metrics["mask_f1"] > best_f1:
                best_f1 = val_metrics["mask_f1"]
                torch.save(model.state_dict(), cfg.save_dir / cfg.best_f1_ckpt)
                print(f"[*] Epoch {epoch + 1}: new best mask F1 = {best_f1:.4f}")
            if val_metrics is not None and val_metrics["hard_case_count"] > 0 and val_metrics["hard_case_iou"] > best_hard_iou:
                best_hard_iou = val_metrics["hard_case_iou"]
                torch.save(model.state_dict(), cfg.save_dir / cfg.best_hard_ckpt)
                print(f"[*] Epoch {epoch + 1}: new best hard-case IoU = {best_hard_iou:.4f}")

            torch.save(model.state_dict(), cfg.save_dir / cfg.last_ckpt)
            if cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0:
                periodic_name = f"mesfnet_epoch_{epoch + 1:03d}.pth"
                torch.save(model.state_dict(), cfg.save_dir / periodic_name)
                print(f"saved periodic checkpoint: {periodic_name}")

            writer.writerow(
                [
                    epoch + 1,
                    train_loss,
                    val_metrics["loss"] if val_metrics is not None else "",
                    val_metrics["mask_f1"] if val_metrics is not None else "",
                    val_metrics["mask_iou"] if val_metrics is not None else "",
                    val_metrics["mask_precision"] if val_metrics is not None else "",
                    val_metrics["mask_recall"] if val_metrics is not None else "",
                    val_metrics["mask_acc"] if val_metrics is not None else "",
                    val_metrics["mask_boundary_f1"] if val_metrics is not None else "",
                    val_metrics["mask_boundary_iou"] if val_metrics is not None else "",
                    val_metrics["edge_f1"] if val_metrics is not None else "",
                    val_metrics["edge_iou"] if val_metrics is not None else "",
                    val_metrics["morph_mse"] if val_metrics is not None else "",
                    val_metrics["morph_mae"] if val_metrics is not None else "",
                    val_metrics["bottom_1_mean_iou"] if val_metrics is not None else "",
                    val_metrics["bottom_5_mean_iou"] if val_metrics is not None else "",
                    val_metrics["iou_lt_05_count"] if val_metrics is not None else "",
                    val_metrics["zero_iou_count"] if val_metrics is not None else "",
                    val_metrics["small_water_iou"] if val_metrics is not None else "",
                    val_metrics["small_water_recall"] if val_metrics is not None else "",
                    val_metrics["hard_case_iou"] if val_metrics is not None else "",
                    val_metrics["hard_case_recall"] if val_metrics is not None else "",
                    val_metrics["hard_case_count"] if val_metrics is not None else "",
                    effective_morph_weight,
                    train_d4_consistency,
                    cfg.d4_consistency_weight if epoch >= cfg.d4_consistency_start_epoch else 0.0,
                    current_lr,
                ]
            )
            csv_fp.flush()

            resume_file = cfg.save_dir / "training_state.pth"
            resume_previous = cfg.save_dir / "training_state_prev.pth"
            resume_tmp = cfg.save_dir / "training_state.tmp.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_iou": best_iou,
                    "best_f1": best_f1,
                    "best_hard_iou": best_hard_iou,
                    "rng_state": capture_rng_state(train_generator),
                    "training_signature": current_signature,
                },
                resume_tmp,
            )
            if resume_file.is_file():
                os.replace(resume_file, resume_previous)
            os.replace(resume_tmp, resume_file)
            print(f"saved resume state: epoch {epoch + 1}")

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1}/{cfg.epochs} | time: {epoch_time:.1f}s")
            if val_metrics is None:
                print(f"  train loss: {train_loss:.4f} | validation skipped")
            else:
                print(f"  train loss: {train_loss:.4f} | val loss: {val_metrics['loss']:.4f}")
                print(
                    "  mask: "
                    f"F1 {val_metrics['mask_f1']:.4f}, "
                    f"IoU {val_metrics['mask_iou']:.4f}, "
                    f"P {val_metrics['mask_precision']:.4f}, "
                    f"R {val_metrics['mask_recall']:.4f}"
                )
                print(f"  edge: F1 {val_metrics['edge_f1']:.4f}, IoU {val_metrics['edge_iou']:.4f}")
                print(
                    "  mask boundary: "
                    f"F1 {val_metrics['mask_boundary_f1']:.4f}, "
                    f"IoU {val_metrics['mask_boundary_iou']:.4f}"
                )
                print(
                    "  robust: "
                    f"bottom1% {val_metrics['bottom_1_mean_iou']:.4f}, "
                    f"bottom5% {val_metrics['bottom_5_mean_iou']:.4f}, "
                    f"IoU<0.5 {val_metrics['iou_lt_05_count']}, "
                    f"zero-IoU {val_metrics['zero_iou_count']}"
                )
                print(
                    "  targeted: "
                    f"small-water IoU {val_metrics['small_water_iou']:.4f}, "
                    f"R {val_metrics['small_water_recall']:.4f}, "
                    f"hard IoU {val_metrics['hard_case_iou']:.4f}, "
                    f"R {val_metrics['hard_case_recall']:.4f}"
                )
                print(f"  morph: MSE {val_metrics['morph_mse']:.6f}, MAE {val_metrics['morph_mae']:.6f}")
            print(f"  morph weight: {effective_morph_weight:.6f} | lr: {current_lr:.2e}")
            print("-" * 60)

    total_time = time.time() - start_time
    config_save_path = cfg.save_dir / "training_config.txt"
    with config_save_path.open("w", encoding="utf-8") as f:
        f.write("Training config\n")
        f.write("model: mesfnet.MESFNet\n")
        f.write(f"data_root: {cfg.data_root}\n")
        f.write(f"epochs: {cfg.epochs}\n")
        f.write(f"batch_size: {cfg.batch_size}\n")
        f.write(f"accumulation_steps: {cfg.accumulation_steps}\n")
        f.write(f"img_size: {cfg.img_size}\n")
        f.write(f"learning_rate: {cfg.lr}\n")
        f.write(f"weight_decay: {cfg.weight_decay}\n")
        f.write(f"full_train: {cfg.full_train}\n")
        f.write(f"augment_mode: {cfg.augment_mode}\n")
        f.write(f"hard_case_manifest: {cfg.hard_case_manifest}\n")
        f.write(f"hard_case_manifest_sha256: {file_sha256(cfg.hard_case_manifest) if cfg.hard_case_manifest else ''}\n")
        f.write(f"hard_case_sampling: {cfg.hard_case_sampling}\n")
        f.write(f"hard_case_max_weight: {cfg.hard_case_max_weight}\n")
        f.write(f"freeze_nondeterministic_paths: {cfg.freeze_nondeterministic_paths}\n")
        f.write(f"weighted_hard_samples: {len(cfg.hard_case_names)}\n")
        f.write(f"val_every: {cfg.val_every}\n")
        f.write(f"save_every: {cfg.save_every}\n")
        f.write(f"convnext_variant: {cfg.convnext_variant}\n")
        f.write(f"pretrained_backbone: {cfg.pretrained}\n")
        f.write(f"resume: {cfg.resume}\n")
        f.write(f"resume_state: {cfg.resume_state}\n")
        f.write(f"keep_resume_state: {cfg.keep_resume_state}\n")
        f.write(f"ablations: {','.join(cfg.ablations) if cfg.ablations else 'full_model'}\n")
        f.write(f"split_source_dir: {cfg.split_source_dir or ''}\n")
        f.write(
            "loss_weights: "
            f"mask={cfg.w_mask}, iou={cfg.w_iou}, edge={cfg.w_edge}, morph={cfg.w_morph}, "
            f"aux={cfg.w_aux}, edge_consistency={cfg.lambda_cons}\n"
        )
        f.write(f"morph_warmup_epochs: {cfg.morph_warmup_epochs}\n")
        f.write(f"morph_ramp_epochs: {cfg.morph_ramp_epochs}\n")
        f.write(f"morph_loss_type: {cfg.morph_loss_type}\n")
        f.write(f"boundary_tolerance_px: {cfg.boundary_tolerance}\n")
        f.write(f"d4_consistency_weight: {cfg.d4_consistency_weight}\n")
        f.write(f"d4_consistency_start_epoch: {cfg.d4_consistency_start_epoch}\n")
        f.write(f"init_ckpt: {cfg.init_ckpt}\n")
        f.write(f"initialization: {'checkpoint' if cfg.init_ckpt else ('torchvision_pretrained' if cfg.pretrained else 'random')}\n")
        f.write(f"deterministic: {cfg.deterministic}\n")
        f.write(f"strict_deterministic: {cfg.strict_deterministic}\n")
        f.write(f"best_mask_iou: {best_iou:.4f}\n")
        f.write(f"best_mask_f1: {best_f1:.4f}\n")
        f.write(f"best_hard_case_iou: {best_hard_iou:.4f}\n")
        f.write(f"best_iou_checkpoint: {cfg.best_ckpt}\n")
        f.write(f"best_f1_checkpoint: {cfg.best_f1_ckpt}\n")
        f.write(f"best_hard_checkpoint: {cfg.best_hard_ckpt}\n")
        f.write(f"last_checkpoint: {cfg.last_ckpt}\n")
        f.write("periodic_checkpoint_pattern: mesfnet_epoch_NNN.pth\n")
        f.write("resume_checkpoints: training_state.pth, training_state_prev.pth\n")

    if not cfg.keep_resume_state:
        removed_resume_files = []
        for resume_name in ("training_state.pth", "training_state_prev.pth", "training_state.tmp.pth"):
            resume_file = cfg.save_dir / resume_name
            if resume_file.is_file():
                resume_file.unlink()
                removed_resume_files.append(resume_name)
        if removed_resume_files:
            print(f"removed completed-run resume states: {', '.join(removed_resume_files)}")

    print("=" * 60)
    print("Training complete")
    print(f"total time: {total_time / 60:.1f} min")
    print(f"best mask IoU: {best_iou:.4f}")
    print(f"best mask F1: {best_f1:.4f}")
    print(f"save_dir: {cfg.save_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
