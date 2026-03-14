# ---------------------------------------------------------------------------
# config.py — Tunable constants for the jewelry try-on system
# ---------------------------------------------------------------------------

# ------------------------------------------------------------------
# Smoothing
# ------------------------------------------------------------------
# Exponential Moving Average alpha (0 = max smoothing, 1 = no smoothing)
SMOOTHING_ALPHA = 0.25

# ------------------------------------------------------------------
# Earring sizing & placement
# ------------------------------------------------------------------
# Earring width as a fraction of the face width
SCALE_FACTOR_EARRING = 0.14

# Vertical drop below the earlobe landmark (fraction of face_width)
# 0.12 positions the earring centre just below the earlobe
EARRING_Y_OFFSET_RATIO = 0.12   # (was 0.07)

# Outward horizontal nudge so earrings sit outside the face edge
EARRING_X_OUTWARD_RATIO = 0.04  # fraction of face_width

# ------------------------------------------------------------------
# Necklace sizing & placement
# ------------------------------------------------------------------
# Necklace width as a fraction of the face width
SCALE_FACTOR_NECKLACE = 0.72

# How far below the chin tip the necklace centre sits
# 0.50 drops it to neck / upper-collarbone level
NECKLACE_Y_OFFSET_RATIO = 0.50  # (was 0.30)

# ------------------------------------------------------------------
# MediaPipe
# ------------------------------------------------------------------
# Maximum number of faces to detect
MAX_NUM_FACES = 1

# Minimum confidence thresholds
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE  = 0.6

import os as _os

# Always resolve assets relative to this file's own directory,
# regardless of the current working directory.
_HERE         = _os.path.dirname(_os.path.abspath(__file__))
ASSET_DIR     = _os.path.join(_HERE, "assets")
PROCESSED_DIR = _os.path.join(ASSET_DIR, "processed")

# Asset paths — runtime prefers processed/ versions (transparent PNGs)
EARRING_PAIR_IMG   = f"{ASSET_DIR}/earring_pair.png"   # single image with both earrings
EARRING_LEFT_IMG   = f"{ASSET_DIR}/earring_left.png"   # individual (optional)
EARRING_RIGHT_IMG  = f"{ASSET_DIR}/earring_right.png"  # individual (optional)
NECKLACE_IMG       = f"{ASSET_DIR}/necklace.png"

# processed/ counterparts
EARRING_PAIR_PROC  = f"{PROCESSED_DIR}/earring_pair.png"
EARRING_LEFT_PROC  = f"{PROCESSED_DIR}/earring_left.png"
EARRING_RIGHT_PROC = f"{PROCESSED_DIR}/earring_right.png"
NECKLACE_PROC      = f"{PROCESSED_DIR}/necklace.png"
