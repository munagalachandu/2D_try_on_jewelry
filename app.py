# ---------------------------------------------------------------------------
# app.py — Flask server: seller upload + dashboard, customer try-on
# ---------------------------------------------------------------------------

import os, uuid, json, socket, threading
from datetime import datetime

import cv2
import numpy as np
from flask import (Flask, render_template, request, redirect,
                   url_for, Response, send_file, jsonify)
import cloudinary
import cloudinary.uploader

import config
from overlay   import load_overlay, overlay_image, split_pair
from smoother  import PositionSmoother
from qr_generator import generate_qr

app = Flask(__name__)

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "sessions")
os.makedirs(SESSIONS_DIR,         exist_ok=True)
os.makedirs(config.PROCESSED_DIR, exist_ok=True)

cloudinary.config(
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key    = os.environ.get("CLOUDINARY_API_KEY"),
    api_secret = os.environ.get("CLOUDINARY_API_SECRET"),
)

# ---------------------------------------------------------------------------
# Lazy loaders
# ---------------------------------------------------------------------------

_get_face_landmarks_fn = None
_remove_bg_fn          = None

def get_face_landmarks(frame):
    global _get_face_landmarks_fn
    if _get_face_landmarks_fn is None:
        from landmarks import get_face_landmarks as _fn
        _get_face_landmarks_fn = _fn
    return _get_face_landmarks_fn(frame)

def remove_bg(raw_path, prod_dir):
    global _remove_bg_fn
    if _remove_bg_fn is None:
        from preprocess import remove_bg as _fn
        _remove_bg_fn = _fn
    return _remove_bg_fn(raw_path, prod_dir)

# ---------------------------------------------------------------------------
# Pre-warm models in background
# ---------------------------------------------------------------------------

def _prewarm_models():
    import time
    time.sleep(5)
    print("[prewarm] Loading mediapipe + rembg models in background...")
    try:
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        get_face_landmarks(dummy)
        print("[prewarm] mediapipe ready.")
    except Exception as e:
        print(f"[prewarm] mediapipe note: {e}")
    try:
        from preprocess import remove_bg as _fn
        global _remove_bg_fn
        _remove_bg_fn = _fn
        print("[prewarm] rembg ready.")
    except Exception as e:
        print(f"[prewarm] rembg load failed: {e}")
    print("[prewarm] All models ready.")

threading.Thread(target=_prewarm_models, daemon=True).start()

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _save_product(pid: str, data: dict):
    with open(os.path.join(SESSIONS_DIR, f"{pid}.json"), "w") as f:
        json.dump(data, f, indent=2)


def _load_product(pid: str) -> dict | None:
    path = os.path.join(SESSIONS_DIR, f"{pid}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _all_products() -> list[dict]:
    products = []
    for fname in sorted(os.listdir(SESSIONS_DIR), reverse=True):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(SESSIONS_DIR, fname)) as f:
                    products.append(json.load(f))
            except Exception:
                pass
    return products


def _is_cloudinary_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _upload_qr_to_cloudinary(qr_path: str, pid: str) -> str:
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME")
    if not cloud_name:
        return qr_path
    try:
        result = cloudinary.uploader.upload(
            qr_path,
            folder="jewelar_qr",
            public_id=f"qr_{pid}",
            overwrite=True,
            resource_type="image",
        )
        return result.get("secure_url", qr_path)
    except Exception as e:
        print(f"  ⚠  Cloudinary QR upload failed: {e}")
        return qr_path


# ---------------------------------------------------------------------------
# Background processing job
# ---------------------------------------------------------------------------

def _process_in_background(pid: str, raw_path: str, prod_dir: str,
                            base_url: str, field: str, label: str,
                            original_filename: str):
    """Run bg removal + Cloudinary upload in a background thread."""
    try:
        print(f"[bg-job:{pid}] Starting background removal...")
        processed_url = remove_bg(raw_path, prod_dir)
        print(f"[bg-job:{pid}] Done: {processed_url}")
    except Exception as e:
        print(f"[bg-job:{pid}] remove_bg failed: {e}")
        processed_url = raw_path  # fallback to raw if removal fails

    # Upload QR now that we have the processed URL
    tryon_url = f"{base_url}/tryon/{pid}"
    qr_path   = os.path.join(prod_dir, "qr.png")
    try:
        generate_qr(tryon_url, qr_path)
        qr_url = _upload_qr_to_cloudinary(qr_path, pid)
    except Exception as e:
        print(f"[bg-job:{pid}] QR generation failed: {e}")
        qr_url = ""

    # Update the saved product with final URLs + mark ready
    _save_product(pid, {
        "id":        pid,
        "type":      field,
        "label":     label,
        "name":      original_filename,
        "processed": processed_url,
        "qr":        qr_url,
        "tryon_url": tryon_url,
        "status":    "ready",
        "created":   datetime.now().isoformat(),
    })
    print(f"[bg-job:{pid}] Product ready.")


# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------

def _load_overlay_any(path: str):
    if _is_cloudinary_url(path):
        import urllib.request, tempfile
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                urllib.request.urlretrieve(path, tmp.name)
                return load_overlay(tmp.name)
        except Exception as e:
            print(f"[overlay] Failed to fetch {path}: {e}")
            return None
    return load_overlay(path)


def _apply_overlay(frame: np.ndarray, product: dict, data: dict,
                   smoothers: dict | None = None):
    ptype = product["type"]
    img   = _load_overlay_any(product["processed"])
    if img is None:
        return

    fw, tilt = data["face_width"], data["tilt_angle"]

    if ptype == "earring_pair":
        el, er = split_pair(img)
        drop = int(fw * config.EARRING_Y_OFFSET_RATIO)
        out  = int(fw * config.EARRING_X_OUTWARD_RATIO)
        size = int(fw * config.SCALE_FACTOR_EARRING)
        lx, ly = data["left_ear"]
        rx, ry = data["right_ear"]
        if smoothers:
            lx, ly, size_l = smoothers["l"].smooth(lx - out, ly + drop, size)
            rx, ry, size_r = smoothers["r"].smooth(rx + out, ry + drop, size)
        else:
            lx, ly, size_l = lx - out, ly + drop, size
            rx, ry, size_r = rx + out, ry + drop, size
        overlay_image(frame, el, int(lx), int(ly), int(size_l), tilt)
        overlay_image(frame, er, int(rx), int(ry), int(size_r), tilt)

    elif ptype == "necklace":
        size = int(fw * config.SCALE_FACTOR_NECKLACE)
        nx, ny = data["jaw_mid"]
        drop = int(fw * config.NECKLACE_Y_OFFSET_RATIO)
        if smoothers:
            nx, ny, size = smoothers["n"].smooth(nx, ny + drop, size)
        else:
            ny += drop
        overlay_image(frame, img, int(nx), int(ny), int(size), tilt)


# ---------------------------------------------------------------------------
# Webcam helper
# ---------------------------------------------------------------------------

def _open_cam():
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, f = cap.read()
            if ok and f is not None:
                return cap
            cap.release()
    return None


def _gen_frames(product: dict):
    cap = _open_cam()
    if cap is None:
        return
    sm = {"l": PositionSmoother(), "r": PositionSmoother(), "n": PositionSmoother()}
    fail = 0
    try:
        while True:
            try:
                ok, frame = cap.read()
            except cv2.error:
                fail += 1
                if fail > 10: break
                continue
            if not ok or frame is None:
                fail += 1
                if fail > 10: break
                continue
            fail = 0
            frame = cv2.flip(frame, 1)
            data  = get_face_landmarks(frame)
            if data:
                _apply_overlay(frame, product, data, sm)
            else:
                for s in sm.values(): s.reset()
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Routes — Seller
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return redirect(url_for("seller"))


@app.route("/seller")
def seller():
    return render_template("upload.html")


@app.route("/seller/upload", methods=["POST"])
def upload():
    base_url = request.host_url.rstrip("/")
    product_fields = {
        "earring_pair": ("earring_pair.png", "Earring Pair"),
        "necklace":     ("necklace.png",     "Necklace"),
    }
    created = 0
    for field, (filename, label) in product_fields.items():
        f = request.files.get(field)
        if not f or not f.filename:
            continue

        pid      = str(uuid.uuid4())[:8]
        prod_dir = os.path.join(config.PROCESSED_DIR, pid)
        os.makedirs(prod_dir, exist_ok=True)

        raw_path = os.path.join(prod_dir, f"raw_{filename}")
        f.save(raw_path)

        # Save immediately with status=processing so dashboard shows it right away
        _save_product(pid, {
            "id":        pid,
            "type":      field,
            "label":     label,
            "name":      f.filename,
            "processed": raw_path,   # temporary — will be replaced by bg job
            "qr":        "",
            "tryon_url": f"{base_url}/tryon/{pid}",
            "status":    "processing",
            "created":   datetime.now().isoformat(),
        })

        # Kick off background processing
        threading.Thread(
            target=_process_in_background,
            args=(pid, raw_path, prod_dir, base_url, field, label, f.filename),
            daemon=True
        ).start()

        created += 1

    if created == 0:
        return "No files uploaded", 400
    return redirect(url_for("dashboard"))


@app.route("/seller/dashboard")
def dashboard():
    return render_template("dashboard.html", products=_all_products())


# ---------------------------------------------------------------------------
# API — poll product status (used by dashboard JS)
# ---------------------------------------------------------------------------

@app.route("/api/product-status/<pid>")
def product_status(pid):
    product = _load_product(pid)
    if not product:
        return jsonify(status="not_found"), 404
    return jsonify(
        status=product.get("status", "ready"),
        tryon_url=product.get("tryon_url", ""),
        qr_url=f"/qr/{pid}",
        product_image_url=f"/product-image/{pid}",
    )


# ---------------------------------------------------------------------------
# Routes — Customer
# ---------------------------------------------------------------------------

@app.route("/tryon/<pid>")
def tryon(pid):
    product = _load_product(pid)
    if not product:
        return "Session not found", 404
    if product.get("status") == "processing":
        return render_template("processing.html", product=product)
    return render_template("tryon.html", product=product)


@app.route("/stream/<pid>")
def stream(pid):
    product = _load_product(pid)
    if not product:
        return "Not found", 404
    return Response(_gen_frames(product),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/tryon-image/<pid>", methods=["POST"])
def tryon_image(pid):
    product = _load_product(pid)
    if not product:
        return jsonify(error="Not found"), 404
    if product.get("status") == "processing":
        return jsonify(error="Product still processing, please wait"), 400
    f = request.files.get("face")
    if not f:
        return jsonify(error="No face image provided"), 400

    buf   = np.frombuffer(f.read(), np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(error="Cannot decode image"), 400

    data = get_face_landmarks(frame)
    if data:
        _apply_overlay(frame, product, data)

    result_path = os.path.join(config.PROCESSED_DIR, pid, "result.jpg")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    cv2.imwrite(result_path, frame)
    return jsonify(result_url=f"/preview/{pid}")


@app.route("/preview/<pid>")
def preview(pid):
    result_path = os.path.join(config.PROCESSED_DIR, pid, "result.jpg")
    if not os.path.exists(result_path):
        return "No result yet", 404
    return send_file(result_path, mimetype="image/jpeg")


@app.route("/download/<pid>")
def download(pid):
    result_path = os.path.join(config.PROCESSED_DIR, pid, "result.jpg")
    if not os.path.exists(result_path):
        return "No result yet", 404
    return send_file(result_path, as_attachment=True,
                     download_name="tryon_result.jpg")


@app.route("/product-image/<pid>")
def product_image(pid):
    product = _load_product(pid)
    if not product:
        return "Not found", 404
    path = product["processed"]
    if _is_cloudinary_url(path):
        return redirect(path)
    if os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return "Not ready yet", 404


@app.route("/qr/<pid>")
def qr_image(pid):
    product = _load_product(pid)
    if not product:
        return "Not found", 404
    path = product.get("qr", "")
    if not path:
        return "QR not ready yet", 404
    if _is_cloudinary_url(path):
        return redirect(path)
    if os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return "Not ready yet", 404


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    ip = _local_ip()
    print(f"\n  Seller → http://{ip}:{port}/seller")
    print(f"  Dashboard → http://{ip}:{port}/seller/dashboard\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
