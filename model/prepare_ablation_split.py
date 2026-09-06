import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dataset_load import RemoteSensingDataset
from train import make_stratified_split


def indices_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Create one reusable ablation split")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = RemoteSensingDataset(args.data_root, img_size=(1024, 1024), augment=False)
    train_indices, val_indices = make_stratified_split(
        dataset.pairs,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_dir / "train_indices.txt", np.asarray(train_indices, dtype=np.int64), fmt="%d")
    np.savetxt(args.output_dir / "val_indices.txt", np.asarray(val_indices, dtype=np.int64), fmt="%d")

    manifest = {
        "data_root": str(args.data_root.resolve()),
        "dataset_size": len(dataset),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "indices_sha256": {
            "train": indices_hash(train_indices),
            "val": indices_hash(val_indices),
        },
    }
    with (args.output_dir / "split_manifest.json").open("w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
