"""
Hand Tracking
Uses the MediaPipe HandLandmarker (Tasks API) to draw the full 21-point
hand skeleton for up to 2 hands in real-time.

Model (~22 MB) is downloaded automatically on first run.
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
MODEL_FILE = MODEL_DIR / "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]
COLORS = [(0, 220, 100), (0, 150, 255)]


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
    print("See: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker")
    raise SystemExit(1)


def draw_hand(frame, landmarks, color: tuple) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 4, color, 1)


def main() -> None:
    ensure_model()
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_FILE)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(0)
    t0      = time.monotonic()
    last_ts = -1
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms = max(int((time.monotonic() - t0) * 1000), last_ts + 1)
            last_ts = ts_ms
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
            )
            if result.hand_landmarks:
                for i, (hand_lm, handedness) in enumerate(
                    zip(result.hand_landmarks, result.handedness)
                ):
                    color = COLORS[i % len(COLORS)]
                    draw_hand(frame, hand_lm, color)
                    label = handedness[0].display_name
                    score = handedness[0].score
                    wrist = hand_lm[0]
                    cx, cy = int(wrist.x * w), int(wrist.y * h)
                    cv2.putText(frame, f"{label}  {score:.0%}",
                                (cx - 30, cy - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Hand Tracking  (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
