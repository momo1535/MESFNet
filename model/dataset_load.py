# -*- coding: utf-8 -*-
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from gdalTools import read_img
except Exception:  # pragma: no cover - only needed for optional tif inputs
    read_img = None


class RemoteSensingDataset(Dataset):
    """Binary segmentation dataset for remote-sensing images.

    Supported layouts:
      root/image/*.jpg + root/mask/*_mask.png    (HRRSI-WS water challenge)
      root/images/*    + root/labels/*           (older local datasets)
    """

    def __init__(
        self,
        root_dir,
        img_size=(512, 512),
        indices=None,
        augment=False,
        augment_mode="basic",
        augment_profiles=None,
        return_name=False,
    ):
        self.root_dir = Path(root_dir)
        self.img_size = tuple(img_size)
        self.augment = augment
        self.augment_mode = augment_mode
        self.augment_profiles = dict(augment_profiles or {})
        self.return_name = return_name

        self.img_dir, self.lab_dir = self._find_dirs(self.root_dir)
        self.pairs = self._build_pairs()
        if indices is not None:
            self.pairs = [self.pairs[i] for i in indices]

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @staticmethod
    def _find_dirs(root_dir):
        candidates = [
            (root_dir / "image", root_dir / "mask"),
            (root_dir / "images", root_dir / "labels"),
        ]
        for img_dir, lab_dir in candidates:
            if img_dir.is_dir() and lab_dir.is_dir():
                return img_dir, lab_dir
        raise FileNotFoundError(
            f"Cannot find image/mask or images/labels directories under {root_dir}"
        )

    @staticmethod
    def _image_key(path):
        return path.stem

    @staticmethod
    def _mask_key(path):
        stem = path.stem
        return stem[:-5] if stem.endswith("_mask") else stem

    def _build_pairs(self):
        image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        mask_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

        images = sorted(p for p in self.img_dir.iterdir() if p.suffix.lower() in image_exts)
        masks = sorted(p for p in self.lab_dir.iterdir() if p.suffix.lower() in mask_exts)
        mask_by_key = {self._mask_key(path): path for path in masks}

        pairs = []
        missing = []
        for image_path in images:
            key = self._image_key(image_path)
            mask_path = mask_by_key.get(key)
            if mask_path is None:
                missing.append(image_path.name)
                continue
            pairs.append((image_path, mask_path))

        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"Missing masks for {len(missing)} images: {preview}")
        if not pairs:
            raise FileNotFoundError(f"No paired samples found under {self.root_dir}")
        return pairs

    def _load_image(self, path):
        if path.suffix.lower() in {".tif", ".tiff"}:
            if read_img is None:
                raise RuntimeError("gdalTools.read_img is required for tif inputs")
            _, _, _, _, img_data = read_img(str(path))
            img_hwc = np.transpose(img_data[:3], (1, 2, 0)).astype(np.uint8)
            return Image.fromarray(img_hwc).convert("RGB")
        return Image.open(path).convert("RGB")

    def _load_mask(self, path):
        mask = Image.open(path).convert("L")
        return mask

    @staticmethod
    def _foreground_zoom(image, mask, min_scale=0.45, max_scale=0.80):
        mask_np = np.asarray(mask)
        ys, xs = np.where(mask_np >= 128)
        if len(xs) == 0:
            return image, mask

        width, height = image.size
        scale = float(np.random.uniform(min_scale, max_scale))
        crop_width = min(width, max(64, int(round(width * scale))))
        crop_height = min(height, max(64, int(round(height * scale))))

        foreground_idx = int(np.random.randint(0, len(xs)))
        center_x = int(xs[foreground_idx])
        center_y = int(ys[foreground_idx])
        center_x += int(np.random.uniform(-0.10, 0.10) * crop_width)
        center_y += int(np.random.uniform(-0.10, 0.10) * crop_height)

        left = int(np.clip(center_x - crop_width // 2, 0, width - crop_width))
        top = int(np.clip(center_y - crop_height // 2, 0, height - crop_height))
        box = (left, top, left + crop_width, top + crop_height)
        image = image.crop(box).resize((width, height), resample=Image.Resampling.BILINEAR)
        mask = mask.crop(box).resize((width, height), resample=Image.Resampling.NEAREST)
        return image, mask

    def _augment(self, image, mask, profile="default"):
        if self.augment_mode == "targeted" and profile == "small_fn" and np.random.rand() < 0.5:
            image, mask = self._foreground_zoom(image, mask)
        if self.augment_mode == "enhanced" and np.random.rand() < 0.25:
            image, mask = self._foreground_zoom(image, mask, min_scale=0.70, max_scale=1.00)

        if np.random.rand() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if np.random.rand() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if np.random.rand() < 0.5:
            k = int(np.random.choice([1, 2, 3]))
            image = image.rotate(90 * k, resample=Image.Resampling.BILINEAR, expand=True)
            mask = mask.rotate(90 * k, resample=Image.Resampling.NEAREST, expand=True)

        if self.augment_mode in {"strong", "targeted", "enhanced"}:
            if np.random.rand() < 0.8:
                image = ImageEnhance.Brightness(image).enhance(float(np.random.uniform(0.85, 1.15)))
            if np.random.rand() < 0.8:
                image = ImageEnhance.Contrast(image).enhance(float(np.random.uniform(0.85, 1.15)))
            if np.random.rand() < 0.5:
                image = ImageEnhance.Color(image).enhance(float(np.random.uniform(0.90, 1.10)))
            if np.random.rand() < 0.3:
                image = ImageEnhance.Sharpness(image).enhance(float(np.random.uniform(0.85, 1.25)))
            if np.random.rand() < (0.10 if self.augment_mode == "enhanced" else 0.15):
                image = image.filter(ImageFilter.GaussianBlur(radius=float(np.random.uniform(0.1, 0.4))))
        if self.augment_mode == "enhanced":
            if np.random.rand() < 0.30:
                gamma = float(np.random.uniform(0.85, 1.15))
                lut = [int(round(255.0 * ((value / 255.0) ** gamma))) for value in range(256)]
                image = image.point(lut * 3)
            if np.random.rand() < 0.15:
                image_np = np.asarray(image, dtype=np.float32) / 255.0
                sigma = float(np.random.uniform(0.0, 0.02))
                noise = np.random.normal(0.0, sigma, size=image_np.shape).astype(np.float32)
                image = Image.fromarray(
                    np.clip((image_np + noise) * 255.0, 0, 255).astype(np.uint8),
                    mode="RGB",
                )
        return image, mask

    def generate_target_labels(self, mask_np):
        kernel = np.ones((4, 4), np.uint8)
        dilated = cv2.dilate(mask_np, kernel, iterations=1)
        eroded = cv2.erode(mask_np, kernel, iterations=1)
        target_edge = ((dilated - eroded) > 0).astype(np.float32)

        area = np.sum(mask_np > 0)
        perimeter = np.sum(target_edge > 0)
        total_pix = mask_np.size
        eps = 1e-6

        edge_density = perimeter / total_pix
        shape_index = perimeter / (2 * np.sqrt(np.pi * area) + eps)

        num_labels, _ = cv2.connectedComponents((mask_np > 0).astype(np.uint8))
        num_patches = num_labels - 1
        mean_patch_area = np.log1p(area / (num_patches + eps)) / 10.0

        if area > 1.0 and perimeter > 0:
            fractal_dim = 2 * np.log(perimeter / 4 + eps) / (np.log(area) + eps)
        else:
            fractal_dim = 0.0
        fractal_dim = np.clip(fractal_dim, 0, 2)

        target_morph_vec = np.array(
            [edge_density, shape_index, mean_patch_area, fractal_dim],
            dtype=np.float32,
        )
        return target_edge, target_morph_vec

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        if self.augment:
            profile = self.augment_profiles.get(image_path.name, "default")
            image, mask = self._augment(image, mask, profile=profile)

        image = image.resize(self.img_size, resample=Image.Resampling.BILINEAR)
        mask = mask.resize(self.img_size, resample=Image.Resampling.NEAREST)

        mask_np = (np.array(mask) >= 128).astype(np.uint8)
        target_edge, target_morph_vec = self.generate_target_labels(mask_np)

        image_tensor = self.to_tensor(image)
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)
        edge_tensor = torch.from_numpy(target_edge).float().unsqueeze(0)
        morph_tensor = torch.from_numpy(target_morph_vec).float()

        sample = (image_tensor, mask_tensor, edge_tensor, morph_tensor)
        if self.return_name:
            return sample + (image_path.name,)
        return sample
