"""
Gesture Recognition
Uses the MediaPipe GestureRecognizer (Tasks API) for real-time gesture
classification, supplemented with a custom OK sign detector.

Built-in gestures: Open_Palm, Closed_Fist, Pointing_Up, Thumb_Up,
                   Thumb_Down, Victory, ILoveYou
Custom gestures:   OK

Model (~21 MB) is downloaded automatically on first run.
Press 'q' to quit.
"""

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_DIR  = Path(__file__).parent / "models"
MODEL_FILE = MODEL_DIR / "gesture_recognizer.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

THUMB_TIP = 4; INDEX_TIP  = 8
MIDDLE_TIP = 12; MIDDLE_PIP = 10
RING_TIP   = 16; RING_PIP   = 14
PINKY_TIP  = 20; PINKY_PIP  = 18

_DISPLAY = {
    "None":        "None",
    "Closed_Fist": "Fist",
    "Open_Palm":   "Open Palm",
    "Pointing_Up": "Pointing",
    "Thumb_Down":  "Thumbs Down",
    "Thumb_Up":    "Thumbs Up",
    "Victory":     "Peace / Victory",
    "ILoveYou":    "I Love You",
}


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
    print("See: https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer")
    raise SystemExit(1)


def draw_hand(frame, landmarks, color: tuple) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 4, color, 1)


def check_ok(lm) -> bool:
    dist = np.hypot(lm[THUMB_TIP].x - lm[INDEX_TIP].x,
                    lm[THUMB_TIP].y - lm[INDEX_TIP].y)
    return (dist < 0.07 and
            lm[MIDDLE_TIP].y < lm[MIDDLE_PIP].y and
            lm[RING_TIP].y   < lm[RING_PIP].y   and
            lm[PINKY_TIP].y  < lm[PINKY_PIP].y)


def main() -> None:
    ensure_model()
    options = mp_vision.GestureRecognizerOptions(
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
    with mp_vision.GestureRecognizer.create_from_options(options) as recognizer:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms = max(int((time.monotonic() - t0) * 1000), last_ts + 1)
            last_ts = ts_ms
            result = recognizer.recognize_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
            )
            if result.hand_landmarks:
                for i, (hand_lm, gestures) in enumerate(
                    zip(result.hand_landmarks, result.gestures)
                ):
                    color = (0, 220, 100) if i == 0 else (0, 150, 255)
                    draw_hand(frame, hand_lm, color)
                    if check_ok(hand_lm):
                        name = "OK"
                    else:
                        raw  = gestures[0].category_name if gestures else "None"
                        name = _DISPLAY.get(raw, raw)
                    wrist = hand_lm[0]
                    cx, cy = int(wrist.x * w), int(wrist.y * h)
                    (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(frame, (cx-8, cy-44), (cx+tw+8, cy-12), (30,30,30), -1)
                    cv2.putText(frame, name, (cx, cy-18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow("Gesture Recognition  (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ── Landmark indices ──────────────────────────────────────────────────────────
WRIST       = 0
THUMB_IP    = 3;  THUMB_TIP   = 4
INDEX_PIP   = 6;  INDEX_TIP   = 8
MIDDLE_PIP  = 10; MIDDLE_TIP  = 12
RING_PIP    = 14; RING_TIP    = 16
PINKY_PIP   = 18; PINKY_TIP   = 20


def finger_states(lm) -> dict[str, bool]:
    """Return which fingers are extended."""
    return {
        # Thumb: tip further from wrist than IP joint (works for mirrored feed)
        "thumb":  abs(lm[THUMB_TIP].x - lm[WRIST].x) > abs(lm[THUMB_IP].x  - lm[WRIST].x),
        # Other fingers: tip sits higher on screen (smaller y) than PIP joint
        "index":  lm[INDEX_TIP].y  < lm[INDEX_PIP].y,
        "middle": lm[MIDDLE_TIP].y < lm[MIDDLE_PIP].y,
        "ring":   lm[RING_TIP].y   < lm[RING_PIP].y,
        "pinky":  lm[PINKY_TIP].y  < lm[PINKY_PIP].y,
    }


def classify_gesture(lm) -> str:
    f = finger_states(lm)
    thumb, index, middle, ring, pinky = (
        f["thumb"], f["index"], f["middle"], f["ring"], f["pinky"]
    )

    if not any(f.values()):
        return "Fist"
    if all(f.values()):
        return "Open Palm"
    if index and middle and not ring and not pinky:
        return "Peace / Victory"
    if index and not middle and not ring and not pinky:
        return "Pointing"
    if thumb and not index and not middle and not ring and not pinky:
        return "Thumbs Up" if lm[THUMB_TIP].y < lm[WRIST].y else "Thumbs Down"

    # OK: thumb and index tips touching, other three fingers extended
    dist = np.hypot(lm[THUMB_TIP].x - lm[INDEX_TIP].x,
                    lm[THUMB_TIP].y - lm[INDEX_TIP].y)
    if dist < 0.07 and middle and ring and pinky:
        return "OK"

    return "Unknown"


def main() -> None:
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    lm      = hand_lm.landmark
                    gesture = classify_gesture(lm)
                    cx = int(lm[WRIST].x * w)
                    cy = int(lm[WRIST].y * h)

                    # Dark pill background for readability
                    (tw, th), _ = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(frame, (cx - 8, cy - 44), (cx + tw + 8, cy - 12),
                                  (30, 30, 30), -1)
                    cv2.putText(frame, gesture, (cx, cy - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 128), 2)

            cv2.imshow("Gesture Recognition  —  q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
