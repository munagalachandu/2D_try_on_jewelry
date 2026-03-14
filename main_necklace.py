# ---------------------------------------------------------------------------
# main_necklace.py — Real-time necklace try-on
#
# Input: assets/necklace.png
#        OR assets/processed/necklace.png  (background-removed version)
# ---------------------------------------------------------------------------

import sys
import time

import cv2

import config
from landmarks import get_face_landmarks
from overlay import load_overlay, overlay_image
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


def _necklace_anchor(jaw_mid, face_width: float) -> tuple[int, int]:
    drop = int(face_width * config.NECKLACE_Y_OFFSET_RATIO)
    return jaw_mid[0], jaw_mid[1] + drop


def main():
    necklace = _load_best(config.NECKLACE_PROC, config.NECKLACE_IMG)
    if necklace is None:
        print("[ERROR] No necklace image found.")
        print(f"        Place your file at: {config.NECKLACE_IMG}")
        print("        Then optionally run: python preprocess.py  (background removal)")
        sys.exit(1)

    sm = PositionSmoother()

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
    print("Necklace Try-On running — press  Q  to quit.")

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
            size = int(fw * config.SCALE_FACTOR_NECKLACE)

            nx, ny = _necklace_anchor(data["jaw_mid"], fw)
            nx, ny, ns = sm.smooth(nx, ny, size)

            overlay_image(frame, necklace, int(nx), int(ny), int(ns), tilt)
        else:
            sm.reset()

        fps = 1.0 / max(time.time() - prev, 1e-6)
        prev = time.time()
        cv2.putText(frame, f"FPS: {fps:.0f}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2)
        cv2.putText(frame, "Face detected" if data else "No face", (12, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

        cv2.imshow("Necklace Try-On  [Q to quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")


if __name__ == "__main__":
    main()
