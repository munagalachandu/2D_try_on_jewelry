# ---------------------------------------------------------------------------
# overlay.py — Transparent PNG overlay with resize and rotation
# ---------------------------------------------------------------------------

import cv2
import numpy as np


def overlay_image(
    frame: np.ndarray,
    overlay: np.ndarray,
    cx: int,
    cy: int,
    size: int,
    angle: float = 0.0,
) -> None:
    """
    Composite a transparent PNG *overlay* onto *frame* (in-place).

    Parameters
    ----------
    frame   : BGR frame from OpenCV (modified in-place)
    overlay : BGRA image (must have 4 channels)
    cx, cy  : pixel coordinates of the centre of the jewel placement
    size    : target width in pixels; height is scaled proportionally
    angle   : rotation angle in degrees (positive = clockwise)
    """
    if overlay is None or size <= 0:
        return

    # 1. Resize proportionally
    orig_h, orig_w = overlay.shape[:2]
    target_w = max(1, int(size))
    target_h = max(1, int(orig_h * target_w / orig_w))
    resized = cv2.resize(overlay, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 2. Rotate around the image centre (if needed)
    if abs(angle) > 0.1:
        rot_mat = cv2.getRotationMatrix2D(
            (target_w / 2, target_h / 2), -angle, 1.0
        )
        resized = cv2.warpAffine(
            resized, rot_mat, (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    # 3. Compute the top-left corner so that (cx, cy) is the centre
    x1 = cx - target_w // 2
    y1 = cy - target_h // 2
    x2 = x1 + target_w
    y2 = y1 + target_h

    # 4. Clamp to frame bounds
    fh, fw = frame.shape[:2]
    ox1 = max(0, -x1)
    oy1 = max(0, -y1)
    ox2 = target_w - max(0, x2 - fw)
    oy2 = target_h - max(0, y2 - fh)

    fx1 = max(0, x1)
    fy1 = max(0, y1)
    fx2 = fx1 + (ox2 - ox1)
    fy2 = fy1 + (oy2 - oy1)

    if fx1 >= fx2 or fy1 >= fy2:
        return  # completely out of frame

    # 5. Alpha blend
    roi     = frame[fy1:fy2, fx1:fx2]
    patch   = resized[oy1:oy2, ox1:ox2]
    alpha   = patch[:, :, 3:4].astype(np.float32) / 255.0
    bgr     = patch[:, :, :3].astype(np.float32)
    blended = bgr * alpha + roi.astype(np.float32) * (1 - alpha)
    frame[fy1:fy2, fx1:fx2] = blended.astype(np.uint8)


def load_overlay(path: str) -> np.ndarray | None:
    """Load a PNG with transparency (BGRA).  Returns None if the file is missing."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # Ensure 4 channels
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def split_pair(pair_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a single paired-earring image down the centre into two halves.

    The image is assumed to be a standard product shot:
      left half  → earring for the LEFT  ear
      right half → earring for the RIGHT ear

    Returns (left_half, right_half) as BGRA arrays.
    """
    mid = pair_img.shape[1] // 2
    return pair_img[:, :mid], pair_img[:, mid:]
