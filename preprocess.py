# ---------------------------------------------------------------------------
# preprocess.py — Background removal + Cloudinary upload
# ---------------------------------------------------------------------------

import os
import sys
from pathlib import Path

import numpy as np
from rembg import remove
from PIL import Image
import cloudinary
import cloudinary.uploader

import config

PROCESSED_DIR = os.path.join(config.ASSET_DIR, "processed")
_SUPPORTED = {".png", ".jpg", ".jpeg", ".webp"}

cloudinary.config(
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key    = os.environ.get("CLOUDINARY_API_KEY"),
    api_secret = os.environ.get("CLOUDINARY_API_SECRET"),
)


def remove_bg(src: str, out_dir: str = PROCESSED_DIR) -> str:
    """
    Remove background from *src*, save transparent PNG locally,
    upload to Cloudinary, and return the Cloudinary URL.
    Falls back to local path if Cloudinary is not configured.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, Path(src).stem + ".png")

    orig = Image.open(src).convert("RGB")
    img_rgba = orig.copy()
    img_rgba.putalpha(255)
    result_arr = np.array(remove(img_rgba))
    Image.fromarray(result_arr).save(out_path)

    size_kb = os.path.getsize(out_path) // 1024
    print(f"  ✓  {Path(src).name}  →  {out_path}  ({size_kb} KB)")

    # Upload to Cloudinary
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME")
    if cloud_name:
        try:
            result = cloudinary.uploader.upload(
                out_path,
                folder="jewelar",
                use_filename=True,
                unique_filename=True,
                overwrite=False,
                resource_type="image",
            )
            url = result.get("secure_url", out_path)
            print(f"  ☁  Uploaded to Cloudinary: {url}")
            return url
        except Exception as e:
            print(f"  ⚠  Cloudinary upload failed: {e} — using local path")
            return out_path
    else:
        print("  ⚠  Cloudinary not configured — using local path")
        return out_path


def process_assets(asset_dir: str = config.ASSET_DIR) -> None:
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
        for path in sys.argv[1:]:
            remove_bg(path)
    else:
        process_assets()
