"""
Offline demo test — runs inference on saved test images and pushes
to Firebase, exactly like main_demo.py would during a live capture.

Use this to verify the dashboard works without the microscope.

Usage:
    python scripts/test_demo_offline.py                       # uses data/test_images/
    python scripts/test_demo_offline.py path/to/folder        # custom folder
    python scripts/test_demo_offline.py path/to/folder --no-push   # skip Firebase
"""

import cv2
import numpy as np
import os
import sys
import time
import glob
from datetime import datetime, timedelta

# ---------- Firebase ----------
import firebase_admin
from firebase_admin import credentials, db, storage

# ---------- TFLite ----------
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow.lite.python import interpreter as tflite


# ============================================================
# CONFIG — keep in sync with main_demo.py
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "demo_polymer_pe_ps_int8.tflite")
FIREBASE_KEY = os.path.join(PROJECT_ROOT, "models", "firebase_key.json")

FIREBASE_DB_URL   = "https://aquatrace-63edb-default-rtdb.asia-southeast1.firebasedatabase.app/"
FIREBASE_BUCKET   = "aquatrace-63edb.firebasestorage.app"
UPLOAD_TO_STORAGE = False        # keep False for offline test — no image upload

IMG_SIZE   = 224
CLASSES    = ["PE", "PS"]
DEVICE_ID  = "OFFLINE_TEST_01"

BINARY_THRESHOLD  = 60
MIN_PARTICLE_AREA = 5

DEFAULT_TEST_DIR = os.path.join(PROJECT_ROOT, "data", "dataset_polymer/test")

_bucket = None


# ============================================================
# INIT
# ============================================================
def init_firebase():
    global _bucket
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_KEY)
        firebase_admin.initialize_app(cred, {
            "databaseURL": FIREBASE_DB_URL,
            "storageBucket": FIREBASE_BUCKET,
        })
    print("✓ Firebase connected")
    if UPLOAD_TO_STORAGE:
        try:
            _bucket = storage.bucket()
            _bucket.exists()
            print(f"✓ Firebase Storage connected ({FIREBASE_BUCKET})")
        except Exception as e:
            print(f"⚠ Storage unavailable ({e})")
            _bucket = None


def init_model():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    in_d  = interpreter.get_input_details()
    out_d = interpreter.get_output_details()
    print(f"✓ TFLite model loaded ({os.path.getsize(MODEL_PATH)/1024:.0f} KB)")
    print(f"  Classes: {CLASSES}")
    return interpreter, in_d, out_d


# ============================================================
# INFERENCE (same as main_demo.py)
# ============================================================
def preprocess(frame):
    h, w = frame.shape[:2]
    s = min(h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    frame = frame[top:top+s, left:left+s]
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = frame_rgb.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0)


def run_inference_single(interpreter, in_d, out_d, frame):
    x = preprocess(frame)
    if in_d[0]["dtype"] == np.int8:
        scale, zero = in_d[0]["quantization"]
        x = (x / scale + zero).astype(np.int8)
    interpreter.set_tensor(in_d[0]["index"], x)
    interpreter.invoke()
    out = interpreter.get_tensor(out_d[0]["index"])[0]
    if out_d[0]["dtype"] == np.int8:
        scale, zero = out_d[0]["quantization"]
        out = (out.astype(np.float32) - zero) * scale
    if out.max() > 1 or out.min() < 0:
        exp = np.exp(out - out.max())
        probs = exp / exp.sum()
    else:
        probs = out
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), probs


def count_particles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= MIN_PARTICLE_AREA]
    return len(valid), valid


# ============================================================
# FIREBASE PUSH
# ============================================================
def push_to_firebase(session_id, polymer, confidence, probs, count, frame_shape):
    ref = db.reference("sensor_readings")
    reading = {
        "device_id": DEVICE_ID,
        "session_id": session_id,
        "timestamp": int(time.time()),
        "timestamp_human": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "wavelengths_captured": [460],
        "primary_wavelength_nm": 460,

        "polymer": polymer,
        "confidence": round(confidence, 4),
        "class_probabilities": {c: round(float(p), 4)
                                for c, p in zip(CLASSES, probs)},
        "per_wavelength": {"460": {"polymer": polymer,
                                    "confidence": round(confidence, 4)}},

        "particle_count": count,
        "image_urls": {},
    }
    key = ref.push(reading)
    print(f"  → Pushed: sensor_readings/{key.key}")
    return key.key


# ============================================================
# COLLECT TEST IMAGES
# ============================================================
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def collect_images(folder):
    """Return list of image paths. Handles flat folder OR PE/ PS subfolders."""
    imgs = []

    # flat
    for f in os.listdir(folder):
        if f.lower().endswith(EXTS):
            imgs.append(os.path.join(folder, f))

    # subfolders
    for cls in CLASSES:
        sub = os.path.join(folder, cls)
        if os.path.isdir(sub):
            for f in os.listdir(sub):
                if f.lower().endswith(EXTS):
                    imgs.append(os.path.join(sub, f))

    return sorted(imgs)


# ============================================================
# MAIN
# ============================================================
def main():
    args = sys.argv[1:]
    push = "--no-push" not in args
    args = [a for a in args if not a.startswith("--")]

    test_dir = args[0] if args else DEFAULT_TEST_DIR

    print("=" * 55)
    print("  AquaTrace — Offline Test (no camera)")
    print("=" * 55)
    print(f"  Test folder : {test_dir}")
    print(f"  Push to FB  : {push}")
    print("=" * 55)

    if not os.path.isdir(test_dir):
        print(f"✗ Folder not found: {test_dir}")
        print(f"  Create it and drop your test images in.")
        return

    images = collect_images(test_dir)
    if not images:
        print(f"✗ No images found in {test_dir}")
        print(f"  Supported extensions: {EXTS}")
        return

    print(f"\n✓ Found {len(images)} images\n")

    if push:
        init_firebase()
    interpreter, in_d, out_d = init_model()

    results = []
    for i, img_path in enumerate(images, 1):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[{i}/{len(images)}] ✗ cannot read {os.path.basename(img_path)}")
            continue

        polymer, conf, probs = run_inference_single(interpreter, in_d, out_d, frame)
        count, _ = count_particles(frame)

        session_id = f"offline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}"
        print(f"[{i}/{len(images)}] {os.path.basename(img_path)}")
        print(f"    → {polymer}  ({conf*100:.1f}%)   particles={count}")

        if push:
            push_to_firebase(session_id, polymer, conf, probs, count, frame.shape)

        results.append((os.path.basename(img_path), polymer, conf, count))

    # summary
    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    for fname, poly, conf, cnt in results:
        print(f"  {fname:40s}  {poly:4s}  {conf*100:5.1f}%  {cnt:4d} particles")

    print(f"\nTotal: {len(results)} images processed")
    if push:
        print("Check your dashboard — it should have updated.")


if __name__ == "__main__":
    main()