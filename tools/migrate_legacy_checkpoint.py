"""Migrate a legacy raw MSKNet state_dict to MESFNet without changing tensor values."""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import torch

TOP_LEVEL_NAMES = {
    "s1": "encoder_stage1",
    "s2": "encoder_stage2",
    "s3": "encoder_stage3",
    "s4": "encoder_stage4",
    "scp1": "detail_enhance1",
    "scp2": "detail_enhance2",
    "scp3": "detail_enhance3",
    "spam": "context_aggregation",
    "lat3": "lateral_projection3",
    "dec4": "decoder_stage4",
    "dec3": "decoder_stage3",
    "flow_align": "edge_alignment",
    "mask_fuse": "segmentation_fusion",
    "mask_head": "segmentation_head",
    "edge_head": "boundary_head",
    "morph_fc": "morphology_head"
}

def migrate_state_dict(state):
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a nonempty raw state_dict, not a training resume checkpoint")
    converted = OrderedDict()
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Only raw tensor state_dict checkpoints are supported")
        parts = key.split(".")
        parts[0] = TOP_LEVEL_NAMES.get(parts[0], parts[0])
        parts = ["norm" if part == "bn" else part for part in parts]
        new_key = ".".join(parts)
        if new_key in converted:
            raise ValueError(f"Key collision: {new_key}")
        converted[new_key] = value
    return converted

def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = args.output.with_suffix(args.output.suffix + ".migration.json")
    if args.output.exists() or report.exists():
        raise FileExistsError("Output checkpoint or migration report already exists")
    state = torch.load(args.input, map_location="cpu", weights_only=True)
    converted = migrate_state_dict(state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        torch.save(converted, stream)
    result = {
        "input_sha256": sha256(args.input),
        "output_sha256": sha256(args.output),
        "tensor_count": len(converted),
        "renamed_keys": sum(a != b for a, b in zip(state, converted)),
        "tensor_values_changed": False,
        "note": "Key migration only; optimizer/RNG/resume signatures are not migrated.",
    }
    with report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
