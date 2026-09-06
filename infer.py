"""Mask inference with optional D4 probability averaging and small-hole filling."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from model.mesfnet import MESFNet
from model.postprocess import fill_small_holes

ROOT = Path(__file__).resolve().parent
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MODES = {"none": [(0, False)],
         "flip4": [(0, False), (0, True), (2, True), (2, False)],
         "d4": [(k, flip) for flip in (False, True) for k in range(4)]}


def transform(x, k, flip):
    x = torch.rot90(x, k, (-2, -1))
    return x.flip(-1) if flip else x


def inverse_transform(x, k, flip):
    if flip:
        x = x.flip(-1)
    return torch.rot90(x, -k, (-2, -1))


@torch.inference_mode()
def predict_probability(model, rgb, tta="d4"):
    mean = rgb.new_tensor(MEAN)[None, :, None, None]
    std = rgb.new_tensor(STD)[None, :, None, None]
    total = torch.zeros_like(rgb[:, :1], dtype=torch.float32)
    for k, flip in MODES[tta]:
        transformed = transform(rgb, k, flip)
        logits = model((transformed - mean) / std)["mask"]
        total += inverse_transform(logits.sigmoid().float(), k, flip)
    return total / len(MODES[tta])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "infer.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--tta", choices=tuple(MODES))
    parser.add_argument("--max-hole-area", type=int, help="0 disables hole filling in this CLI")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    allowed = {"backbone", "ablations", "threshold", "tta", "max_hole_area"}
    if set(cfg) - allowed:
        raise ValueError(f"Unknown inference keys: {sorted(set(cfg) - allowed)}")
    for name in ("threshold", "tta", "max_hole_area"):
        if getattr(args, name) is not None:
            cfg[name] = getattr(args, name)
    if not 0 <= cfg["threshold"] <= 1 or cfg["max_hole_area"] < 0 or cfg["tta"] not in MODES:
        raise ValueError("Invalid threshold, TTA or hole-area setting")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.input.is_file():
        files = [args.input]
    elif args.input.is_dir():
        files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    else:
        raise FileNotFoundError(args.input)
    if not files:
        raise ValueError("No JPG/PNG images found")
    names = [p.stem + "_mask.png" for p in files]
    if len(names) != len(set(names)):
        raise ValueError("Input image stems must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in names + ["inference_summary.json"]:
        if (args.output_dir / name).exists():
            raise FileExistsError(args.output_dir / name)
    device = torch.device(args.device)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = MESFNet(pretrained=False, backbone_variant=cfg["backbone"],
                   ablations=cfg["ablations"]).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    for index, path in enumerate(files, 1):
        with Image.open(path) as image:
            if image.size != (1024, 1024):
                raise ValueError(f"Expected 1024x1024 input, got {image.size}: {path}")
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        rgb = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
        mask = predict_probability(model, rgb, cfg["tta"])[0, 0].cpu().numpy() >= cfg["threshold"]
        if cfg["max_hole_area"] > 0:
            mask = fill_small_holes(mask, cfg["max_hole_area"])
        Image.fromarray(mask.astype(np.uint8) * 255).save(args.output_dir / (path.stem + "_mask.png"))
        print(f"Processed {index}/{len(files)}: {path.name}", flush=True)
    summary = {"config": cfg, "checkpoint_sha256": sha256(args.checkpoint),
               "images": len(files), "device": str(device), "torch": torch.__version__,
               "output_format": "1024x1024 single-channel PNG, values 0/255"}
    (args.output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
