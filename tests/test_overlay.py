# tests/test_overlay.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from overlay import overlay_image, load_overlay


def _make_bgra(w=64, h=64, alpha=200):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, 0] = 255   # blue channel
    img[:, :, 3] = alpha
    return img


def test_overlay_does_not_crash_on_blank_frame():
    frame   = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay = _make_bgra()
    overlay_image(frame, overlay, cx=320, cy=240, size=64, angle=0)
    # Should modify the frame where overlay lands
    assert frame[240, 320, 0] > 0  # blue channel was composited


def test_overlay_handles_out_of_bounds_gracefully():
    frame   = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay = _make_bgra()
    # Place entirely off-screen — should not crash
    overlay_image(frame, overlay, cx=-500, cy=-500, size=64)
    overlay_image(frame, overlay, cx=9999,  cy=9999,  size=64)


def test_overlay_with_rotation():
    frame   = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay = _make_bgra()
    overlay_image(frame, overlay, cx=320, cy=240, size=64, angle=30)


def test_overlay_zero_size_is_noop():
    frame   = np.zeros((480, 640, 3), dtype=np.uint8)
    original = frame.copy()
    overlay = _make_bgra()
    overlay_image(frame, overlay, cx=320, cy=240, size=0)
    assert np.array_equal(frame, original)


def test_load_overlay_missing_file_returns_none():
    result = load_overlay("does_not_exist.png")
    assert result is None
