# ---------------------------------------------------------------------------
# generate_assets.py — Creates simple placeholder jewelry PNG assets
#
# Run this ONCE before launching main.py:
#   python generate_assets.py
#
# These are purely placeholder images.  Replace any of them with a real
# high-quality transparent-background jewelry PNG and the app will use
# your image automatically.
# ---------------------------------------------------------------------------

import os
import math
from PIL import Image, ImageDraw

# Always write assets next to this script, regardless of working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR  = os.path.join(SCRIPT_DIR, "assets")
os.makedirs(ASSET_DIR, exist_ok=True)


def _make_teardrop(w: int = 80, h: int = 120, colour: tuple = (212, 175, 55)) -> Image.Image:
    """Draw a simple teardrop / drop earring shape."""
    img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Upper circle (the stud part)
    r = w // 4
    cx, cy = w // 2, r + 4
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*colour, 255))

    # Lower drop
    drop_top    = cy + r
    drop_bottom = h - 4
    drop_left   = cx - w // 4
    drop_right  = cx + w // 4
    # Draw as a rounded polygon (approximate with ellipse + rectangle)
    draw.ellipse(
        [drop_left, drop_top, drop_right, drop_top + (drop_right - drop_left)],
        fill=(*colour, 240),
    )
    mid = (drop_top + drop_bottom) // 2
    draw.polygon(
        [(drop_left, mid), (drop_right, mid), (cx, drop_bottom)],
        fill=(*colour, 240),
    )
    # Highlight
    hl = (min(colour[0] + 60, 255), min(colour[1] + 60, 255), min(colour[2] + 60, 255))
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(*hl, 200))
    return img


def _make_necklace(w: int = 320, h: int = 80, colour: tuple = (212, 175, 55)) -> Image.Image:
    """Draw an arc-shaped chain with a small pendant."""
    img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Chain arc
    chain_colour = (*colour, 220)
    cx, cy = w // 2, 10
    rx, ry = w // 2 - 10, h - 30

    # Draw arc as a sequence of line segments
    n_segments = 60
    pts = []
    for i in range(n_segments + 1):
        angle = math.pi * i / n_segments  # 0 → π (left to right along the top)
        x = int(cx + rx * math.cos(math.pi - angle))
        y = int(cy + ry * math.sin(angle))
        pts.append((x, y))

    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=chain_colour, width=3)

    # Pendant at the bottom centre (small diamond shape)
    px, py = cx, pts[n_segments // 2][1] + 4
    r = 10
    gem_colour = (100, 200, 255)
    draw.polygon(
        [(px, py - r), (px + r, py), (px, py + r), (px - r, py)],
        fill=(*gem_colour, 240),
    )
    draw.line([(pts[n_segments // 2][0], pts[n_segments // 2][1]), (px, py - r)],
              fill=chain_colour, width=2)

    return img


# --------------- Generate & save ---------------
print("Generating placeholder jewelry assets …")

teardrop = _make_teardrop(80, 120)
teardrop.save(os.path.join(ASSET_DIR, "earring_left.png"))

teardrop_r = teardrop.transpose(Image.FLIP_LEFT_RIGHT)
teardrop_r.save(os.path.join(ASSET_DIR, "earring_right.png"))

necklace = _make_necklace(320, 80)
necklace.save(os.path.join(ASSET_DIR, "necklace.png"))

print("Done!  Assets saved to:")
for f in ["earring_left.png", "earring_right.png", "necklace.png"]:
    print(f"  {ASSET_DIR}/{f}")
print("\nTip: Replace any of these files with your own transparent PNG jewelry images.")
