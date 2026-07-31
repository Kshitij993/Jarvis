"""
Face Detection
Uses the MediaPipe FaceDetector (Tasks API) for fast, lightweight face detection.
Draws bounding boxes, confidence scores, and 6 key facial points per face.

Model (~1 MB) is downloaded automatically on first run.
Press 'q' to quit.
"""

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_DIR  = Path(__file__).parent / "models"
MODEL_FILE = MODEL_DIR / "blaze_face_short_range.tflite"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def ensure_model() -> None:
    if MODEL_FILE.exists():
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for url in (MODEL_URL, MODEL_URL.replace("/float16/1/", "/float16/latest/")):
        try:
            print(f"Downloading {MODEL_FILE.name} …")
            urllib.request.urlretrieve(url, MODEL_FILE)
            if MODEL_FILE.stat().st_size > 1000:
                print("Done.")
                return
        except Exception:
            if MODEL_FILE.exists():
                MODEL_FILE.unlink(missing_ok=True)
    print(f"\n[ERROR] Could not download {MODEL_FILE.name}.")
    print(f"Download it manually and place it at:\n  {MODEL_FILE}")
    print("See: https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector")
    raise SystemExit(1)


def main() -> None:
    ensure_model()
    options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_FILE)),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,
    )
    cap = cv2.VideoCapture(0)
    t0      = time.monotonic()
    last_ts = -1
    with mp_vision.FaceDetector.create_from_options(options) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms = max(int((time.monotonic() - t0) * 1000), last_ts + 1)
            last_ts = ts_ms
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
            )
            count = 0
            if result.detections:
                count = len(result.detections)
                for i, detection in enumerate(result.detections):
                    bb = detection.bounding_box
                    x1, y1 = bb.origin_x, bb.origin_y
                    x2, y2 = x1 + bb.width, y1 + bb.height
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 100), 2)
                    score = detection.categories[0].score if detection.categories else 0
                    cv2.putText(frame, f"Face {i+1}  {score:.0%}",
                                (x1, max(y1 - 10, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 100), 2)
                    for kp in detection.keypoints:
                        cv2.circle(frame, (int(kp.x * w), int(kp.y * h)),
                                   4, (255, 180, 0), -1)
            cv2.putText(frame, f"Faces: {count}", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Face Detection  (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


def main() -> None:
    cap = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(
        model_selection=0,              # 0 = short range (< 2 m), 1 = full range
        min_detection_confidence=0.5,
    ) as detector:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            count = 0
            if results.detections:
                count = len(results.detections)
                for i, detection in enumerate(results.detections):
                    # ── Bounding box ──────────────────────────
                    bb = detection.location_data.relative_bounding_box
                    x  = max(0, int(bb.xmin * w))
                    y  = max(0, int(bb.ymin * h))
                    bw = int(bb.width  * w)
                    bh = int(bb.height * h)
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 220, 100), 2)

                    # Confidence label
                    cv2.putText(frame, f"Face {i+1}  {detection.score[0]:.0%}",
                                (x, max(y - 10, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 100), 2)

                    # ── 6 key points ──────────────────────────
                    for kp in detection.location_data.relative_keypoints:
                        cv2.circle(frame,
                                   (int(kp.x * w), int(kp.y * h)),
                                   4, (255, 180, 0), -1)

            cv2.putText(frame, f"Faces: {count}",
                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Face Detection  —  q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
