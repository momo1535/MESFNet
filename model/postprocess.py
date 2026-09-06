# -*- coding: utf-8 -*-
import cv2
import numpy as np


def fill_small_holes(mask, max_hole_area=4096):
    """Fill background components enclosed by foreground.

    Args:
        mask: bool or 0/1 ndarray, foreground is water.
        max_hole_area: Fill enclosed holes up to this area. Use 0 or None to
            fill every enclosed hole.
    """
    mask_bool = mask.astype(bool)
    background = (~mask_bool).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=8)

    height, width = mask_bool.shape
    filled = mask_bool.copy()
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        touches_border = x == 0 or y == 0 or (x + w) >= width or (y + h) >= height
        if touches_border:
            continue
        if max_hole_area and area > max_hole_area:
            continue
        filled[labels == label] = True
    return filled


def remove_small_objects(mask, min_object_area=0):
    """Remove small foreground components. Disabled when min_object_area <= 0."""
    if min_object_area <= 0:
        return mask.astype(bool)

    mask_bool = mask.astype(bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8), connectivity=8
    )
    cleaned = np.zeros_like(mask_bool)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_object_area:
            cleaned[labels == label] = True
    return cleaned


def close_mask(mask, kernel_size=0):
    """Apply morphological closing to connect tiny cracks. Disabled at 0."""
    if kernel_size <= 1:
        return mask.astype(bool)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed.astype(bool)


def postprocess_mask(
    mask,
    fill_holes=True,
    max_hole_area=4096,
    min_object_area=0,
    close_kernel=0,
):
    """Postprocess a binary water mask.

    The default is conservative for this competition: fill only small enclosed
    holes and keep all predicted water components.
    """
    out = mask.astype(bool)
    out = close_mask(out, close_kernel)
    if fill_holes:
        out = fill_small_holes(out, max_hole_area=max_hole_area)
    out = remove_small_objects(out, min_object_area=min_object_area)
    return out
