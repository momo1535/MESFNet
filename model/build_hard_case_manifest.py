# -*- coding: utf-8 -*-
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dataset_load import RemoteSensingDataset


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_case(iou, precision, recall, gt_ratio, fp, fn):
    if precision < 0.80 and recall > 0.90 and fp > fn:
        return "fp_dominant_review", "default", 1.0, "manual_review"
    if iou < 0.50 and recall < 0.50 and gt_ratio <= 0.02:
        return "missed_small_water", "small_fn", 3.0, "eligible"
    if recall < 0.80:
        profile = "small_fn" if gt_ratio <= 0.02 else "strong"
        return "low_recall", profile, 3.0, "eligible"
    if iou < 0.90 or recall < 0.90:
        profile = "small_fn" if gt_ratio <= 0.02 else "strong"
        return "moderate_hard", profile, 2.0, "eligible"
    if iou < 0.95:
        return "boundary_or_mixed", "strong", 1.5, "eligible"
    return "normal", "default", 1.0, "eligible"


def component_stats(mask):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([], dtype=np.int64)
    if areas.size == 0:
        return 0, 0, 0, 0.0
    return int(areas.size), int(areas.min()), int(areas.max()), float(areas.mean())


def parse_args():
    parser = argparse.ArgumentParser(description="Build a reproducible hard-case manifest from per-image metrics.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = RemoteSensingDataset(args.data_root, augment=False)
    mask_by_image = {image.name: mask for image, mask in dataset.pairs}

    with args.metrics.open("r", newline="", encoding="utf-8-sig") as file_obj:
        metrics_rows = list(csv.DictReader(file_obj))
    metrics_rows.sort(key=lambda row: row["image"])

    output_rows = []
    for row in metrics_rows:
        image_name = row["image"]
        mask_path = mask_by_image.get(image_name)
        if mask_path is None:
            raise FileNotFoundError(f"Metrics image has no matching mask: {image_name}")

        mask = (np.asarray(Image.open(mask_path).convert("L")) >= 128).astype(np.uint8)
        gt_pixels = int(mask.sum())
        gt_ratio = gt_pixels / float(mask.size)
        component_count, min_area, max_area, mean_area = component_stats(mask)

        iou = float(row["iou"])
        precision = float(row["precision"])
        recall = float(row["recall"])
        fp = int(row["fp"])
        fn = int(row["fn"])
        failure_type, profile, weight, review_status = classify_case(
            iou, precision, recall, gt_ratio, fp, fn
        )
        output_rows.append(
            {
                **row,
                "gt_pixels": gt_pixels,
                "gt_ratio": f"{gt_ratio:.10f}",
                "component_count": component_count,
                "min_component_area": min_area,
                "max_component_area": max_area,
                "mean_component_area": f"{mean_area:.4f}",
                "failure_type": failure_type,
                "augmentation_profile": profile,
                "sample_weight": f"{weight:.2f}",
                "review_status": review_status,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    counts = {}
    for row in output_rows:
        key = row["failure_type"]
        counts[key] = counts.get(key, 0) + 1
    metadata = {
        "metrics": str(args.metrics.resolve()),
        "metrics_sha256": file_sha256(args.metrics),
        "data_root": str(args.data_root.resolve()),
        "rows": len(output_rows),
        "classification_counts": counts,
        "manifest_sha256": file_sha256(args.output),
        "policy": {
            "max_sample_weight": 3.0,
            "small_water_gt_ratio_max": 0.02,
            "fp_dominant_cases_require_manual_review": True,
        },
    }
    metadata_path = args.output.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, ensure_ascii=False)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
