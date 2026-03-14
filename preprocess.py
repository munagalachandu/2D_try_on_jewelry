# ---------------------------------------------------------------------------
# preprocess.py — Background removal for jewelry assets
#
# Usage (run once when adding new raw jewelry images):
#   python preprocess.py                        # processes everything in assets/
#   python preprocess.py path/to/image.png      # processes a single file
#
# Outputs transparent PNGs to assets/processed/.
# The app automatically prefers processed/ images over raw ones.
# ---------------------------------------------------------------------------

import os
import sys
from pathlib import Path

from rembg import remove
from PIL import Image

import config

PROCESSED_DIR = os.path.join(config.ASSET_DIR, "processed")
_SUPPORTED = {".png", ".jpg", ".jpeg", ".webp"}


def remove_bg(src: str, out_dir: str = PROCESSED_DIR) -> str:
    """
    Remove background from *src* and save a transparent PNG to *out_dir*.
    Returns the output path.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, Path(src).stem + ".png")

    img = Image.open(src).convert("RGBA")
    result = remove(img)          # rembg returns an RGBA PIL Image
    result.save(out_path)

    size_kb = os.path.getsize(out_path) // 1024
    print(f"  ✓  {Path(src).name}  →  {out_path}  ({size_kb} KB)")
    return out_path


def process_assets(asset_dir: str = config.ASSET_DIR) -> None:
    """Remove backgrounds from every image in *asset_dir* (non-recursive)."""
    files = [
        f for f in Path(asset_dir).iterdir()
        if f.is_file() and f.suffix.lower() in _SUPPORTED
    ]
    if not files:
        print(f"[preprocess] No images found in {asset_dir}")
        return

    print(f"[preprocess] Processing {len(files)} image(s) …")
    for f in files:
        remove_bg(str(f))
    print(f"[preprocess] Done — transparent PNGs saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single-file mode
        for path in sys.argv[1:]:
            remove_bg(path)
    else:
        process_assets()
