"""
Face Landmarks
Uses MediaPipe FaceLandmarker (Tasks API) to draw all 478 facial landmarks
(468 face + 10 iris points) in real-time.

Toggles while running:
    c  \u2014  contours (eyes, lips, brows, face oval)
    i  \u2014  landmark index numbers (first 50)

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

_FACE_OVAL  = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,
               378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,
               21,54,103,67,109,10]
_LEFT_EYE   = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33]
_RIGHT_EYE  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]
_LEFT_BROW  = [46,53,52,65,55,70,63,105,66,107]
_RIGHT_BROW = [276,283,282,295,285,300,293,334,296,336]
_OUTER_LIPS = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,61]
_INNER_LIPS = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191,78]
_NOSE       = [168,6,197,195,5,4,1,19,94]

_CONTOURS = [
    (_FACE_OVAL,  (100, 220, 100)),
    (_LEFT_EYE,   (0, 200, 255)),
    (_RIGHT_EYE,  (0, 200, 255)),
    (_LEFT_BROW,  (255, 180, 0)),
    (_RIGHT_BROW, (255, 180, 0)),
    (_OUTER_LIPS, (0, 140, 255)),
    (_INNER_LIPS, (0, 80, 200)),
    (_NOSE,       (200, 200, 200)),
]


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


def draw_contours(frame, lm) -> None:
    h, w = frame.shape[:2]
    for seq, color in _CONTOURS:
        pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in seq if i < len(lm)]
        for j in range(len(pts) - 1):
            cv2.line(frame, pts[j], pts[j+1], color, 1)


def main() -> None:
    ensure_model()
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_FILE)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=2,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
    )
    cap           = cv2.VideoCapture(0)
    t0            = time.monotonic()
    last_ts       = -1
    show_contours = True
    show_idx      = False
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
                for face_lm in result.face_landmarks:
                    for idx, lm in enumerate(face_lm):
                        px, py = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (px, py), 1, (0, 200, 255), -1)
                        if show_idx and idx < 50:
                            cv2.putText(frame, str(idx), (px+2, py-2),
                                        cv2.FONT_HERSHEY_PLAIN, 0.6, (200,200,200), 1)
                    if show_contours:
                        draw_contours(frame, face_lm)
            hud = [
                f"c \u2014 contours  ({'on' if show_contours else 'off'})",
                f"i \u2014 indices   ({'on' if show_idx      else 'off'})",
            ]
            for j, line in enumerate(hud):
                cv2.putText(frame, line, (12, 28 + j * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow("Face Landmarks  (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if   key == ord("q"): break
            elif key == ord("c"): show_contours = not show_contours
            elif key == ord("i"): show_idx      = not show_idx
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
