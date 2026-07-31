"""
Face Emotion Detection
Uses MediaPipe FaceLandmarker (Tasks API) with face blendshapes to classify
emotions in real-time. Blendshapes are a trained 52-coefficient model output,
giving much better accuracy than the previous geometric heuristic.

Supports up to 4 faces simultaneously.
Emotions: Neutral, Happy, Sad, Surprised, Angry

Model (~29 MB) is downloaded automatically on first run.
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
MODEL_FILE = MODEL_DIR / "face_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

EMOTION_COLORS = {
    "Happy":     (0, 220, 0),
    "Sad":       (200, 80, 0),
    "Surprised": (0, 200, 255),
    "Angry":     (0, 0, 220),
    "Neutral":   (180, 180, 180),
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
    print("See: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker")
    raise SystemExit(1)


def classify_emotion(blendshapes) -> str:
    bs       = {b.category_name: b.score for b in blendshapes}
    smile    = (bs.get("mouthSmileLeft",  0) + bs.get("mouthSmileRight", 0)) * 0.5
    frown    = (bs.get("mouthFrownLeft",  0) + bs.get("mouthFrownRight", 0)) * 0.5
    brow_up  =  bs.get("browInnerUp",     0)
    brow_dn  = (bs.get("browDownLeft",    0) + bs.get("browDownRight",   0)) * 0.5
    jaw      =  bs.get("jawOpen",         0)
    sneer    = (bs.get("noseSneerLeft",   0) + bs.get("noseSneerRight",  0)) * 0.5
    if brow_up > 0.4  and jaw   > 0.35: return "Surprised"
    if smile   > 0.35:                  return "Happy"
    if brow_dn > 0.25 and sneer > 0.1:  return "Angry"
    if frown   > 0.15 or (brow_up > 0.25 and smile < 0.1): return "Sad"
    return "Neutral"


def main() -> None:
    ensure_model()
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_FILE)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=4,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
    )
    cap = cv2.VideoCapture(0)
    t0      = time.monotonic()
    last_ts = -1
    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
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
            if result.face_landmarks:
                blendshapes_list = result.face_blendshapes or [[] for _ in result.face_landmarks]
                for face_lm, bs in zip(result.face_landmarks, blendshapes_list):
                    emotion = classify_emotion(bs) if bs else "Neutral"
                    color   = EMOTION_COLORS[emotion]
                    xs = [lm.x * w for lm in face_lm]
                    ys = [lm.y * h for lm in face_lm]
                    x1, y1 = int(min(xs)), int(min(ys))
                    x2, y2 = int(max(xs)), int(max(ys))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, emotion, (x1, max(y1 - 10, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    for lm in face_lm:
                        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, color, -1)
            cv2.imshow("Face Emotion Detection  (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ── Landmark indices (MediaPipe 468-point model) ──────────────────────────────
# Mouth
UPPER_LIP   = 13
LOWER_LIP   = 14
MOUTH_LEFT  = 61
MOUTH_RIGHT = 291

# Eyebrows
LEFT_BROW_INNER  = 107
RIGHT_BROW_INNER = 336
LEFT_BROW_OUTER  = 70
RIGHT_BROW_OUTER = 300

# Eyes
LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP   = 386
RIGHT_EYE_BOTTOM= 374

# Nose bridge
NOSE_TIP   = 1
NOSE_BRIDGE= 6
# ─────────────────────────────────────────────────────────────────────────────


def _dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def classify_emotion(landmarks, image_width: int, image_height: int) -> str:
    """
    Heuristic emotion classifier based on facial geometry ratios.
    Returns one of: Neutral, Happy, Sad, Surprised, Angry
    """
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * image_width, lm.y * image_height])

    # Mouth openness (vertical / horizontal ratio)
    mouth_open  = _dist(pt(UPPER_LIP), pt(LOWER_LIP))
    mouth_width = _dist(pt(MOUTH_LEFT), pt(MOUTH_RIGHT))
    mouth_ratio = mouth_open / (mouth_width + 1e-6)

    # Mouth corners vs nose tip (smile indicator)
    nose_y          = pt(NOSE_TIP)[1]
    left_corner_y   = pt(MOUTH_LEFT)[1]
    right_corner_y  = pt(MOUTH_RIGHT)[1]
    avg_corner_y    = (left_corner_y + right_corner_y) / 2
    smile_score     = nose_y - avg_corner_y       # positive → corners below nose (frown), negative → smile

    # Eyebrow raise (distance from eye top to brow inner)
    left_brow_raise  = _dist(pt(LEFT_BROW_INNER),  pt(LEFT_EYE_TOP))
    right_brow_raise = _dist(pt(RIGHT_BROW_INNER), pt(RIGHT_EYE_TOP))
    avg_brow_raise   = (left_brow_raise + right_brow_raise) / 2

    # Eye openness
    left_eye_open   = _dist(pt(LEFT_EYE_TOP),  pt(LEFT_EYE_BOTTOM))
    right_eye_open  = _dist(pt(RIGHT_EYE_TOP), pt(RIGHT_EYE_BOTTOM))
    avg_eye_open    = (left_eye_open + right_eye_open) / 2

    # ── Decision rules ───────────────────────────────────────────────────────
    if mouth_ratio > 0.35 and avg_brow_raise > 25:
        return "Surprised"
    if mouth_ratio > 0.2 and smile_score < -5:
        return "Happy"
    if smile_score > 10 and avg_brow_raise < 18:
        return "Sad"
    if avg_brow_raise < 15 and mouth_ratio < 0.1:
        return "Angry"
    return "Neutral"


EMOTION_COLORS = {
    "Happy":     (0, 220, 0),
    "Sad":       (200, 100, 0),
    "Surprised": (0, 200, 255),
    "Angry":     (0, 0, 220),
    "Neutral":   (200, 200, 200),
}

