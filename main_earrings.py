# ---------------------------------------------------------------------------
# main_earrings.py — Real-time earring try-on using a paired earring image
#
# Input: assets/earring_pair.png  (single image, both earrings side by side)
#        OR assets/processed/earring_pair.png  (background-removed version)
#
# The image is split down the centre:
#   left  half → placed on the LEFT  ear
#   right half → placed on the RIGHT ear
# ---------------------------------------------------------------------------

import sys
import time

import cv2

import config
from landmarks import get_face_landmarks
from overlay import load_overlay, overlay_image, split_pair
from smoother import PositionSmoother


def _load_best(proc_path: str, raw_path: str):
    img = load_overlay(proc_path)
    if img is not None:
        print(f"[assets] ✓ processed: {proc_path}")
        return img
    img = load_overlay(raw_path)
    if img is not None:
        print(f"[assets] ✓ raw:       {raw_path}")
    return img


def _earring_anchor(ear_pt, face_width: float, side: str) -> tuple[int, int]:
    drop    = int(face_width * config.EARRING_Y_OFFSET_RATIO)
    outward = int(face_width * config.EARRING_X_OUTWARD_RATIO)
    x_shift = -outward if side == "left" else outward
    return ear_pt[0] + x_shift, ear_pt[1] + drop


def main():
    # Load the earring pair image and split it
    pair = _load_best(config.EARRING_PAIR_PROC, config.EARRING_PAIR_IMG)
    if pair is None:
        print("[ERROR] No earring pair image found.")
        print(f"        Place your file at: {config.EARRING_PAIR_IMG}")
        print("        Then optionally run: python preprocess.py  (background removal)")
        sys.exit(1)

    earring_l, earring_r = split_pair(pair)
    earring_size_ratio   = config.SCALE_FACTOR_EARRING

    sm_left  = PositionSmoother()
    sm_right = PositionSmoother()

    cap = None
    for idx in range(4):
        _cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if _cap.isOpened():
            ok, _f = _cap.read()
            if ok and _f is not None:
                cap = _cap
                h, w = _f.shape[:2]
                print(f"[camera] index {idx}  ({w}x{h})")
                break
            _cap.release()
    if cap is None:
        print("[ERROR] No webcam found (tried indices 0-3).")
        sys.exit(1)

    fail = 0
    prev = time.time()
    print("Earring Try-On running — press  Q  to quit.")

    while True:
        try:
            ok, frame = cap.read()
        except cv2.error as e:
            fail += 1
            if fail > 10: break
            time.sleep(0.05); continue

        if not ok or frame is None:
            fail += 1
            if fail > 10: break
            time.sleep(0.05); continue
        fail = 0

        frame = cv2.flip(frame, 1)
        data  = get_face_landmarks(frame)

        if data:
            fw   = data["face_width"]
            tilt = data["tilt_angle"]
            size = int(fw * earring_size_ratio)

            lx, ly = _earring_anchor(data["left_ear"],  fw, "left")
            rx, ry = _earring_anchor(data["right_ear"], fw, "right")

            lx, ly, ls = sm_left.smooth(lx, ly, size)
            rx, ry, rs = sm_right.smooth(rx, ry, size)

            overlay_image(frame, earring_l, int(lx), int(ly), int(ls), tilt)
            overlay_image(frame, earring_r, int(rx), int(ry), int(rs), tilt)
        else:
            sm_left.reset(); sm_right.reset()

        fps = 1.0 / max(time.time() - prev, 1e-6)
        prev = time.time()
        cv2.putText(frame, f"FPS: {fps:.0f}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2)
        cv2.putText(frame, "Face detected" if data else "No face", (12, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

        cv2.imshow("Earring Try-On  [Q to quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")


if __name__ == "__main__":
    main()
