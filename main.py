# ---------------------------------------------------------------------------
# main.py — Real-time jewelry virtual try-on using webcam
# ---------------------------------------------------------------------------

import sys
import time

import cv2
import numpy as np

import config
from landmarks import get_face_landmarks
from overlay import load_overlay, overlay_image
from smoother import PositionSmoother


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_best(proc_path: str, raw_path: str):
    """
    Prefer a background-removed processed image; fall back to the raw asset.
    Prints which version is used so the user always knows.
    """
    img = load_overlay(proc_path)
    if img is not None:
        print(f"[assets] ✓ processed: {proc_path}")
        return img
    img = load_overlay(raw_path)
    if img is not None:
        print(f"[assets] ✓ raw:       {raw_path}")
    return img


def _compute_earring_anchor(
    ear_pt: tuple, face_width: float, side: str
) -> tuple[int, int]:
    """Drop earring below earlobe landmark; nudge outward for realism."""
    drop    = int(face_width * config.EARRING_Y_OFFSET_RATIO)
    outward = int(face_width * config.EARRING_X_OUTWARD_RATIO)
    x_shift = -outward if side == "left" else outward
    return ear_pt[0] + x_shift, ear_pt[1] + drop


def _compute_necklace_anchor(jaw_mid: tuple, face_width: float) -> tuple[int, int]:
    """Drop the necklace below the jaw midpoint."""
    drop = int(face_width * config.NECKLACE_Y_OFFSET_RATIO)
    return jaw_mid[0], jaw_mid[1] + drop


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("[assets] Loading — processed/ preferred, raw/ as fallback")
    earring_l = _load_best(config.EARRING_LEFT_PROC,  config.EARRING_LEFT_IMG)
    earring_r = _load_best(config.EARRING_RIGHT_PROC, config.EARRING_RIGHT_IMG)
    necklace  = _load_best(config.NECKLACE_PROC,      config.NECKLACE_IMG)

    missing = []
    if earring_l is None: missing.append("earring_left")
    if earring_r is None: missing.append("earring_right")
    if necklace  is None: missing.append("necklace")

    if missing:
        print("[WARNING] Could not load:", ", ".join(missing))
        print("          Drop PNGs into assets/ or run: python preprocess.py")

    # Per-piece smoothers
    sm_left     = PositionSmoother()
    sm_right    = PositionSmoother()
    sm_necklace = PositionSmoother()

    # Auto-detect webcam — use DirectShow backend (most stable on Windows)
    # Setting a custom resolution after open can corrupt the frame buffer, so
    # we let the camera run at its native resolution.
    cap = None
    for cam_idx in range(4):
        _cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if _cap.isOpened():
            ok, _frame = _cap.read()
            if ok and _frame is not None:
                cap = _cap
                h, w = _frame.shape[:2]
                print(f"[camera] Using camera index {cam_idx}  ({w}x{h})")
                break
            _cap.release()
    if cap is None:
        print("[ERROR] No working webcam found (tried indices 0-3).")
        print("        Close any app using the camera (Teams, browser, etc.) and retry.")
        sys.exit(1)

    prev_time = time.time()
    face_present_last = False
    fail_count = 0

    print("Jewelry Try-On running — press  Q  to quit.")

    while True:
        try:
            ok, frame = cap.read()
        except cv2.error as e:
            print(f"[WARNING] cam read error: {e}")
            fail_count += 1
            if fail_count > 10:
                print("[ERROR] Too many camera errors. Exiting.")
                break
            time.sleep(0.05)
            continue

        if not ok or frame is None:
            fail_count += 1
            if fail_count > 10:
                print("[ERROR] Camera stopped sending frames. Exiting.")
                break
            time.sleep(0.05)
            continue
        fail_count = 0

        frame = cv2.flip(frame, 1)   # mirror so it feels like a selfie
        data  = get_face_landmarks(frame)

        if data is not None:
            face_present_last  = True
            face_w   = data["face_width"]
            tilt     = data["tilt_angle"]
            left_ear = data["left_ear"]
            right_ear = data["right_ear"]
            jaw_mid  = data["jaw_mid"]

            # Earring sizes
            earring_size  = int(face_w * config.SCALE_FACTOR_EARRING)
            necklace_size = int(face_w * config.SCALE_FACTOR_NECKLACE)

            # Anchor positions
            l_ax, l_ay = _compute_earring_anchor(left_ear,  face_w, "left")
            r_ax, r_ay = _compute_earring_anchor(right_ear, face_w, "right")
            n_ax, n_ay = _compute_necklace_anchor(jaw_mid,  face_w)

            # Smooth
            sl_x, sl_y, sl_s = sm_left.smooth(l_ax, l_ay, earring_size)
            sr_x, sr_y, sr_s = sm_right.smooth(r_ax, r_ay, earring_size)
            sn_x, sn_y, sn_s = sm_necklace.smooth(n_ax, n_ay, necklace_size)

            # Overlay jewels
            if earring_l is not None:
                overlay_image(frame, earring_l, int(sl_x), int(sl_y), int(sl_s), tilt)
            if earring_r is not None:
                overlay_image(frame, earring_r, int(sr_x), int(sr_y), int(sr_s), tilt)
            if necklace is not None:
                overlay_image(frame, necklace,  int(sn_x), int(sn_y), int(sn_s), tilt)

        else:
            # No face — reset smoothers so they reinitialise cleanly next time
            if face_present_last:
                sm_left.reset()
                sm_right.reset()
                sm_necklace.reset()
                face_present_last = False

        # FPS counter
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(
            frame, f"FPS: {fps:.0f}", (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2, cv2.LINE_AA,
        )

        # Status hint
        status = "Face detected" if data else "No face detected"
        cv2.putText(
            frame, status, (12, 68),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA,
        )

        cv2.imshow("Jewelry Try-On  [Q to quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")


if __name__ == "__main__":
    main()
