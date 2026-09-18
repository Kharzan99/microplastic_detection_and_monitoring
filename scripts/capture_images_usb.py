"""
Dataset collection utility for USB microscope.
Saves frames into data/dataset_polymer/raw/<CLASS>/.

Controls:
    1..6  → switch class (PE, PP, PS, PET, PVC, ABS)
    SPACE → save one frame
    A     → auto-capture 30 frames, 0.3 s apart
    Q     → quit
"""

import cv2
import os
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- CHANGE THESE TWO LINES PER COLLECTION SESSION ----
SPLIT       = "train"    # "train" / "valid" / "test"
WAVELENGTH  = "460nm"    # "405nm" / "460nm" / "520nm"
# -------------------------------------------------------

RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "dataset_polymer",
                            SPLIT, WAVELENGTH)
CAMERA_INDEX = 1
CLASSES      = ["PE", "PP", "PS", "PET", "PVC", "ABS"]
KEY_TO_CLASS = {ord(str(i + 1)): c for i, c in enumerate(CLASSES)}

WINDOW_NAME = "Dataset capture — SPACE=save, 1..6=class, A=auto30, Q=quit"
DISPLAY_W, DISPLAY_H = 1280, 720   # window display size (image is upscaled for display only)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(RAW_DIR, c), exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {CAMERA_INDEX}.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera opened — actual capture resolution: {actual_w}x{actual_h}")

    # resizable window so maximize/fullscreen works
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)

    current = CLASSES[0]
    print("=== Dataset capture ===")
    print("  1..6 = switch class, SPACE = save, A = auto-save 30, Q = quit")
    print(f"  Saving to: {RAW_DIR}\n")

    saved_count = {c: len(os.listdir(os.path.join(RAW_DIR, c))) for c in CLASSES}
    flash_until = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        display = frame.copy()

        # overlay labels (drawn at native resolution, then upscaled together)
        for i, c in enumerate(CLASSES):
            color = (0, 255, 0) if c == current else (180, 180, 180)
            cv2.putText(display, f"{i+1}: {c} [{saved_count[c]}]", (10, 30 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        if time.time() < flash_until:
            cv2.rectangle(display, (0, 0), (display.shape[1] - 1, display.shape[0] - 1),
                          (255, 255, 255), 8)

        cv2.putText(display, f"CURRENT: {current}", (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # upscale for display ONLY — the saved image stays at native resolution
        display = cv2.resize(display, (DISPLAY_W, DISPLAY_H),
                             interpolation=cv2.INTER_LINEAR)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key in KEY_TO_CLASS:
            current = KEY_TO_CLASS[key]
            print(f"→ Class: {current}")
        elif key == ord(" "):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = os.path.join(RAW_DIR, current, f"{ts}.jpg")
            cv2.imwrite(path, frame)          # save at NATIVE resolution
            saved_count[current] += 1
            flash_until = time.time() + 0.08
            print(f"  saved {path}")
        elif key == ord("a"):
            print(f"→ Auto-capturing 30 frames for {current}…")
            for _ in range(30):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.rotate(frame, cv2.ROTATE_180) 
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                path = os.path.join(RAW_DIR, current, f"{ts}.jpg")
                cv2.imwrite(path, frame)
                saved_count[current] += 1
                time.sleep(0.3)
            print(f"  done. {current} total = {saved_count[current]}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nFinal counts:")
    for c in CLASSES:
        print(f"  {c}: {saved_count[c]}")


if __name__ == "__main__":
    main()