"""
AquaTrace — Unified Microplastic Detection Demo
Two modes at startup:
  1) LIVE  — USB microscope, capture 1/2/3 wavelength frames
  2) FOLDER — run inference on a folder of saved images

Both modes share inference, particle counting, and Firebase push.
"""
import os as _os
_os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"     # hide INFO + WARNING + ERROR
_os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"     # silence oneDNN notice
_os.environ["ABSL_MIN_LOG_LEVEL"]    = "3"
_os.environ["KERAS_BACKEND"]         = "tensorflow"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", module="keras")
import cv2
import numpy as np
import os
import sys
import time
import glob
from datetime import datetime, timedelta

import firebase_admin
from firebase_admin import credentials, db, storage

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow.lite.python import interpreter as tflite


# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "demo_polymer_pe_ps_int8.tflite")
FIREBASE_KEY = os.path.join(PROJECT_ROOT, "models", "firebase_key.json")

FIREBASE_DB_URL   = "https://aquatrace-63edb-default-rtdb.asia-southeast1.firebasedatabase.app/"
FIREBASE_BUCKET   = "aquatrace-63edb.appspot.com"
UPLOAD_TO_STORAGE = False         # set False if you don't want image uploads

CAMERA_INDEX = 1                  # USB microscope
IMG_SIZE     = 224
CLASSES      = ["PE", "PS"]       # matches your trained demo model
DEVICE_ID    = "DEMO_LAPTOP_01"

# Wavelengths used in LIVE mode (keyboard 1/2/3)
WAVELENGTHS     = [405, 460, 520]
WAVELENGTH_KEYS = {ord("1"): 405, ord("2"): 460, ord("3"): 520}

# particle detection
BINARY_THRESHOLD  = 60
MIN_PARTICLE_AREA = 5

# Folders
RAW_DIR   = os.path.join(PROJECT_ROOT, "data", "dataset_polymer", "raw")
TEST_DIR  = os.path.join(PROJECT_ROOT, "data", "dataset_polymer", "test")

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
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
    print("✓ Firebase connected (RTDB)")

    if UPLOAD_TO_STORAGE:
        try:
            _bucket = storage.bucket()
            _bucket.exists()
            print(f"✓ Firebase Storage connected ({FIREBASE_BUCKET})")
        except Exception as e:
            print(f"⚠ Firebase Storage unavailable ({e}) — continuing without uploads")
            _bucket = None


def init_model():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    in_d  = interpreter.get_input_details()
    out_d = interpreter.get_output_details()
    print(f"✓ TFLite model loaded ({os.path.getsize(MODEL_PATH)/1024:.0f} KB)")
    print(f"  Classes: {CLASSES}")
    return interpreter, in_d, out_d


def init_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)   # Windows
    # cap = cv2.VideoCapture(CAMERA_INDEX)                # Mac/Linux
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {CAMERA_INDEX}. "
            "Close Windows Camera app and try index 0/1/2."
        )
    print(f"✓ Camera connected (index {CAMERA_INDEX})")
    return cap


# ============================================================
# INFERENCE
# ============================================================
def preprocess(frame):
    h, w = frame.shape[:2]
    s = min(h, w)
    top  = (h - s) // 2
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


def run_inference_ensemble(interpreter, in_d, out_d, frames_by_wl):
    """frames_by_wl: {wl: bgr}  → ensemble average across available wavelengths."""
    all_probs, per_wl = [], {}
    for wl, frame in frames_by_wl.items():
        poly, conf, probs = run_inference_single(interpreter, in_d, out_d, frame)
        all_probs.append(probs)
        per_wl[wl] = {"polymer": poly, "confidence": round(conf, 4)}
    avg = np.mean(all_probs, axis=0)
    idx = int(np.argmax(avg))
    return CLASSES[idx], float(avg[idx]), avg, per_wl


# ============================================================
# PARTICLE COUNT
# ============================================================
def count_particles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= MIN_PARTICLE_AREA]
    return len(valid), valid


def count_particles_multi(frames_by_wl):
    best_wl, best_score, best_count, best_contours = None, -1.0, 0, []
    for wl, frame in frames_by_wl.items():
        count, contours = count_particles(frame)
        score = sum(cv2.contourArea(c) for c in contours)
        if score > best_score:
            best_wl, best_score, best_count, best_contours = wl, score, count, contours
    return best_wl, best_count, best_contours


# ============================================================
# STORAGE
# ============================================================
def upload_thumbnail(local_path, session_id, wavelength):
    if _bucket is None:
        return None
    try:
        img = cv2.imread(local_path)
        h, w = img.shape[:2]
        scale = 640.0 / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        tmp = local_path.replace(".jpg", "_thumb.jpg")
        cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, 75])

        blob_path = f"captures/{session_id}/{wavelength}nm.jpg"
        blob = _bucket.blob(blob_path)
        blob.upload_from_filename(tmp, content_type="image/jpeg")
        os.remove(tmp)
        return blob.generate_signed_url(expiration=timedelta(days=7), method="GET")
    except Exception as e:
        # one-line message; first line of the exception is enough
        msg = str(e).split("\n")[0][:120]
        print(f"  ⚠ Storage upload skipped ({wavelength}nm): {msg}")
        return None


# ============================================================
# FIREBASE PUSH (used by both modes)
# ============================================================
def push_to_firebase(session_id, polymer, confidence, probs, count,
                     best_wl, frames_by_wl, image_urls, per_wl, mode):
    ref = db.reference("sensor_readings")
    reading = {
        "device_id": DEVICE_ID,
        "session_id": session_id,
        "mode": mode,                      # "live" or "folder"
        "timestamp": int(time.time()),
        "timestamp_human": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "wavelengths_captured": sorted(frames_by_wl.keys()),
        "primary_wavelength_nm": best_wl,

        "polymer": polymer,
        "confidence": round(confidence, 4),
        "class_probabilities": {c: round(float(p), 4)
                                for c, p in zip(CLASSES, probs)},
        "per_wavelength": {str(k): v for k, v in per_wl.items()},

        "particle_count": count,
        "image_urls": {str(k): v for k, v in image_urls.items() if v},
    }
    key = ref.push(reading)
    print(f"  → Pushed: sensor_readings/{key.key}")
    return key.key


# ============================================================
# LIVE MODE — capture session
# ============================================================
def capture_session(cap):
    """Press 1/2/3 for wavelengths, ENTER to finish."""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    captured = {}
    win = "Capture — 1=405nm  2=460nm  3=520nm  ENTER=done"

    print("\n--- Capture session ---")
    print("  1 = 405nm,  2 = 460nm,  3 = 520nm   (any subset, ENTER when done)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_180)   # USB microscope upside-down fix

        display = frame.copy()
        y = 30
        for wl in WAVELENGTHS:
            done = wl in captured
            mark = "✓" if done else "·"
            color = (0, 255, 0) if done else (180, 180, 180)
            cv2.putText(display, f"{mark} {wl} nm", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            y += 30
        cv2.putText(display, "ENTER = done", (10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key in WAVELENGTH_KEYS:
            wl = WAVELENGTH_KEYS[key]
            captured[wl] = frame.copy()
            print(f"  ✓ Captured {wl} nm")
        elif key in (13, 10):
            if not captured:
                print("  ✗ Nothing captured — press 1/2/3 first.")
                continue
            break
        elif key == 27:
            cv2.destroyWindow(win)
            return None, {}

    cv2.destroyWindow(win)
    return session_id, captured


def run_live_mode(interpreter, in_d, out_d):
    cap = init_camera()
    print("\nReady. SPACE = capture session, Q = quit.\n")
    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.rotate(frame, cv2.ROTATE_180)

        display = frame.copy()
        if last_result:
            txt = (f"{last_result[0]}  {last_result[1]*100:.1f}%  |  "
                   f"{last_result[2]} particles  |  λ={last_result[3]}nm")
            cv2.putText(display, txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("LIVE — SPACE to capture, Q to quit", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            t0 = time.time()
            session_id, frames_by_wl = capture_session(cap)
            if not frames_by_wl:
                print("Capture cancelled.\n")
                continue

            polymer, conf, probs, per_wl = run_inference_ensemble(
                interpreter, in_d, out_d, frames_by_wl)
            best_wl, count, contours = count_particles_multi(frames_by_wl)

            # save to data/dataset_polymer/raw/<session_id>/
            folder = os.path.join(RAW_DIR, session_id)
            os.makedirs(folder, exist_ok=True)
            local_paths = {}
            for wl, frm in frames_by_wl.items():
                p = os.path.join(folder, f"{wl}nm.jpg")
                cv2.imwrite(p, frm)
                local_paths[wl] = p

            annotated = frames_by_wl[best_wl].copy()
            cv2.drawContours(annotated, contours, -1, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(folder, "annotated.jpg"), annotated)

            image_urls = {}
            for wl, path in local_paths.items():
                url = upload_thumbnail(path, session_id, wl)
                if url:
                    image_urls[wl] = url

            push_to_firebase(
                session_id, polymer, conf, probs, count,
                best_wl, frames_by_wl, image_urls, per_wl, mode="live")

            elapsed = (time.time() - t0) * 1000
            print(f"\n→ {polymer} ({conf*100:.1f}%)  particles={count}  λ={best_wl}nm")
            print(f"→ Pipeline: {elapsed:.0f} ms\n")
            last_result = (polymer, conf, count, best_wl)

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# FOLDER MODE
# ============================================================
def collect_images(folder):
    """Flat folder OR PE/ PS subfolders, both handled."""
    imgs = []
    for f in os.listdir(folder):
        if f.lower().endswith(EXTS):
            imgs.append(os.path.join(folder, f))
    for cls in CLASSES:
        sub = os.path.join(folder, cls)
        if os.path.isdir(sub):
            for f in os.listdir(sub):
                if f.lower().endswith(EXTS):
                    imgs.append(os.path.join(sub, f))
    return sorted(imgs)


def run_folder_mode(interpreter, in_d, out_d, folder):
    if not os.path.isdir(folder):
        print(f"✗ Folder not found: {folder}")
        return

    images = collect_images(folder)
    if not images:
        print(f"✗ No images in {folder}")
        print(f"  Supported: {EXTS}")
        return

    print(f"\n✓ Found {len(images)} images in {folder}\n")

    for i, img_path in enumerate(images, 1):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[{i}/{len(images)}] ✗ cannot read {os.path.basename(img_path)}")
            continue

        polymer, conf, probs = run_inference_single(interpreter, in_d, out_d, frame)
        count, contours = count_particles(frame)

        session_id = f"folder_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}"

        # save annotated locally
        folder_out = os.path.join(RAW_DIR, session_id)
        os.makedirs(folder_out, exist_ok=True)
        annotated = frame.copy()
        cv2.drawContours(annotated, contours, -1, (0, 255, 0), 1)
        local = os.path.join(folder_out, "annotated.jpg")
        cv2.imwrite(local, annotated)

        image_urls = {}
        url = upload_thumbnail(local, session_id, 460)
        if url:
            image_urls[460] = url

        print(f"[{i}/{len(images)}] {os.path.basename(img_path)}")
        print(f"    → {polymer} ({conf*100:.1f}%)  particles={count}")

        push_to_firebase(
            session_id, polymer, conf, probs, count,
            460, {460: frame}, image_urls,
            {460: {"polymer": polymer, "confidence": round(conf, 4)}},
            mode="folder")


# ============================================================
# MAIN
# ============================================================
def main():
    print()
    print("╔═══════════════════════════════════════════════════════╗")
    print("║           AquaTrace — Microplastic Detection          ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print("║                                                       ║")
    print("║   How do you want to analyze the sample?              ║")
    print("║                                                       ║")
    print("║     [1]  LIVE  — connect USB microscope               ║")
    print("║            (real-time capture, press SPACE per shot)  ║")
    print("║                                                       ║")
    print("║     [2]  LOCAL — analyze saved images from a folder   ║")
    print("║            (offline testing / batch processing)       ║")
    print("║                                                       ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()
    mode = input("  Your choice [1/2]: ").strip()
    print()

    init_firebase()
    interpreter, in_d, out_d = init_model()

    if mode == "1":
        run_live_mode(interpreter, in_d, out_d)
    elif mode == "2":
        default = TEST_DIR
        print(f"  Enter folder path containing test images")
        print(f"  (press ENTER to use default: {default})")
        user = input("  Path: ").strip()
        folder = user if user else default
        run_folder_mode(interpreter, in_d, out_d, folder)
    else:
        print("✗ Invalid mode. Enter 1 or 2.")
        return

    print("\nDone.")


if __name__ == "__main__":
    main()