# Jewelry Virtual Try-On 💍

A **real-time virtual jewelry try-on** prototype using your webcam, MediaPipe Face Mesh, and OpenCV.

---

## Project Structure

```
2D_try_on_jewelry/
│
├── config.py            # Tunable constants (scale, smoothing, offsets)
├── landmarks.py         # MediaPipe Face Mesh wrapper — ear, jaw, tilt
├── overlay.py           # Transparent PNG compositing (resize + rotate)
├── smoother.py          # EMA jitter-smoothing per jewelry piece
├── main.py              # Webcam loop — runs the whole system
│
├── generate_assets.py   # One-time helper to create placeholder PNGs
├── requirements.txt     # Python dependencies
│
├── assets/              # Jewelry PNG images (BGRA, transparent background)
│   ├── earring_left.png
│   ├── earring_right.png
│   └── necklace.png
│
└── tests/               # Unit tests (no webcam needed)
    ├── test_overlay.py
    └── test_smoother.py
```

---

## Quick Start

### 1 — Create & activate a virtual environment (already done if you followed the prompt)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2 — Install dependencies

```powershell
pip install -r requirements.txt
```

### 3 — Generate placeholder jewelry assets (first time only)

```powershell
python generate_assets.py
```

> **Tip:** After running this command you'll find simple golden teardrop earrings and an arc necklace in `assets/`.  
> Replace any of the three PNG files with your own transparent-background jewelry images — the app picks them up automatically on the next launch.

### 4 — Run the try-on

```powershell
python main.py
```

Press **Q** to quit.

---

## Using Real Jewelry Images

1. Prepare a PNG image with a **transparent background** (RGBA).
2. Name / place it as one of:
   - `assets/earring_left.png`
   - `assets/earring_right.png`
   - `assets/necklace.png`
3. Re-run `python main.py` — no code changes needed.

---

## Tuning

All tunable parameters live in `config.py`:

| Constant | Default | Effect |
|---|---|---|
| `SMOOTHING_ALPHA` | `0.25` | Lower = smoother but laggier tracking |
| `SCALE_FACTOR_EARRING` | `0.14` | Earring size relative to face width |
| `SCALE_FACTOR_NECKLACE` | `0.60` | Necklace width relative to face width |
| `EARRING_Y_OFFSET_RATIO` | `0.07` | Drop below earlobe (fraction of face width) |
| `NECKLACE_Y_OFFSET_RATIO` | `0.30` | Drop below jaw midpoint |

---

## Running Tests

```powershell
pip install pytest
python -m pytest tests/ -v
```

No webcam needed for the tests.

---

## Requirements

- Python 3.10+
- Webcam
- Windows / macOS / Linux

---

## How It Works

```
Webcam frame
    │
    ▼
landmarks.py  ──▶  left_ear, right_ear, jaw_mid, face_width, tilt_angle
    │
    ▼
smoother.py   ──▶  EMA-smoothed (x, y, size) per piece
    │
    ▼
overlay.py    ──▶  resize → rotate → alpha-blend onto frame
    │
    ▼
cv2.imshow    ──▶  display to user
```
