#!/usr/bin/env python3
"""Interpolate two compatible PyTorch checkpoints on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections.abc import MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


STATE_DICT_KEYS = ("model_state_dict", "state_dict", "model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one checkpoint by linearly interpolating two compatible checkpoints."
    )
    parser.add_argument("--base", type=Path, required=True, help="Baseline checkpoint.")
    parser.add_argument("--target", type=Path, required=True, help="Fine-tuned checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Output checkpoint.")
    parser.add_argument(
        "--target-weight",
        type=float,
        default=0.25,
        help="Fine-tuned checkpoint weight. Baseline weight is 1 minus this value.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSON manifest path. Defaults to OUTPUT.manifest.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output and manifest.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint: Any) -> tuple[MutableMapping[str, Any], str]:
    if isinstance(checkpoint, MutableMapping):
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint, "raw_state_dict"
        for key in STATE_DICT_KEYS:
            value = checkpoint.get(key)
            if isinstance(value, MutableMapping):
                return value, key
    raise TypeError("Checkpoint does not contain a recognized state_dict mapping.")


def validate_paths(args: argparse.Namespace, manifest_path: Path) -> None:
    for label, path in (("base", args.base), ("target", args.target)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    if not 0.0 <= args.target_weight <= 1.0:
        raise ValueError("--target-weight must be between 0 and 1.")
    for path in (args.output, manifest_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def interpolate_state_dicts(
    base_state: MutableMapping[str, Any],
    target_state: MutableMapping[str, Any],
    target_weight: float,
) -> dict[str, int]:
    base_keys = set(base_state)
    target_keys = set(target_state)
    if base_keys != target_keys:
        missing = sorted(base_keys - target_keys)
        extra = sorted(target_keys - base_keys)
        raise ValueError(
            "State dict keys differ. "
            f"Missing from target: {missing[:10]}; extra in target: {extra[:10]}"
        )

    base_weight = 1.0 - target_weight
    stats = {
        "total_tensors": 0,
        "interpolated_tensors": 0,
        "interpolated_parameters": 0,
        "base_only_non_floating_tensors": 0,
    }

    with torch.no_grad():
        for key, base_tensor in base_state.items():
            target_tensor = target_state[key]
            if not isinstance(base_tensor, torch.Tensor) or not isinstance(target_tensor, torch.Tensor):
                raise TypeError(f"Non-tensor state value at key: {key}")
            if base_tensor.shape != target_tensor.shape:
                raise ValueError(
                    f"Shape mismatch at {key}: {tuple(base_tensor.shape)} != "
                    f"{tuple(target_tensor.shape)}"
                )
            if base_tensor.dtype != target_tensor.dtype:
                raise ValueError(
                    f"Dtype mismatch at {key}: {base_tensor.dtype} != {target_tensor.dtype}"
                )

            stats["total_tensors"] += 1
            if base_tensor.is_floating_point() or base_tensor.is_complex():
                base_tensor.mul_(base_weight).add_(target_tensor, alpha=target_weight)
                stats["interpolated_tensors"] += 1
                stats["interpolated_parameters"] += base_tensor.numel()
            else:
                stats["base_only_non_floating_tensors"] += 1

    return stats


def atomic_torch_save(checkpoint: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_name(f".{output.name}.tmp")
    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, output)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_json_save(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_name(f".{output.name}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temp_path, output)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> int:
    args = parse_args()
    args.base = args.base.resolve()
    args.target = args.target.resolve()
    args.output = args.output.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else args.output.with_suffix(args.output.suffix + ".manifest.json")
    )
    validate_paths(args, manifest_path)

    torch.set_num_threads(1)
    base_hash = sha256_file(args.base)
    target_hash = sha256_file(args.target)
    base_checkpoint = load_checkpoint(args.base)
    target_checkpoint = load_checkpoint(args.target)
    base_state, base_format = extract_state_dict(base_checkpoint)
    target_state, target_format = extract_state_dict(target_checkpoint)
    stats = interpolate_state_dicts(base_state, target_state, args.target_weight)

    atomic_torch_save(base_checkpoint, args.output)
    output_hash = sha256_file(args.output)
    script_path = Path(__file__).resolve()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "linear_checkpoint_interpolation",
        "formula": "output = base_weight * base + target_weight * target",
        "base_weight": 1.0 - args.target_weight,
        "target_weight": args.target_weight,
        "base": {
            "path": str(args.base),
            "sha256": base_hash,
            "format": base_format,
        },
        "target": {
            "path": str(args.target),
            "sha256": target_hash,
            "format": target_format,
        },
        "output": {
            "path": str(args.output),
            "sha256": output_hash,
            "bytes": args.output.stat().st_size,
            "format": base_format,
        },
        "tensor_stats": stats,
        "non_floating_policy": "copy_from_base",
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "script": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
        },
    }
    atomic_json_save(manifest, manifest_path)

    print(f"Output: {args.output}")
    print(f"Manifest: {manifest_path}")
    print(f"SHA256: {output_hash}")
    print(
        "Weights: "
        f"base={1.0 - args.target_weight:.6f}, target={args.target_weight:.6f}"
    )
    print(f"Interpolated tensors: {stats['interpolated_tensors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
