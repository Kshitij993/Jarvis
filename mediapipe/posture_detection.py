"""
Posture Detection
Uses the MediaPipe PoseLandmarker (Tasks API) for real-time posture analysis.
Detects shoulder tilt, forward head position, and spine angle.

Model (~5 MB) is downloaded automatically on first run.
Press 'q' to quit.
"""

import math
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_DIR  = Path(__file__).parent / "models"
MODEL_FILE = MODEL_DIR / "pose_landmarker_lite.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

L_SHOULDER = 11; R_SHOULDER = 12
L_EAR = 7;  R_EAR = 8
L_HIP = 23; R_HIP = 24

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(17,19),(19,15),(15,21),
    (12,14),(14,16),(16,18),(18,20),(20,16),(16,22),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(29,31),(31,27),
    (24,26),(26,28),(28,30),(30,32),(32,28),
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
    print("See: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker")
    raise SystemExit(1)


def _angle(a, b, c) -> float:
    ax, ay = a.x - b.x, a.y - b.y
    cx, cy = c.x - b.x, c.y - b.y
    return abs(math.degrees(math.atan2(abs(ax*cy - ay*cx), ax*cx + ay*cy)))


def draw_skeleton(frame, lm) -> None:
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in POSE_CONNECTIONS:
        if lm[a].visibility > 0.5 and lm[b].visibility > 0.5:
            cv2.line(frame, pts[a], pts[b], (100, 200, 100), 2)
    for i, (x, y) in enumerate(pts):
        if lm[i].visibility > 0.5:
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), 4, (100, 200, 100), 1)


def analyse_posture(lm) -> tuple[str, list[str]]:
    issues = []
    tilt = abs(lm[L_SHOULDER].y - lm[R_SHOULDER].y) * 100
    if tilt > 3:
        issues.append(f"Shoulder tilt  ({tilt:.1f}%)")
    side_l = min(lm[L_EAR].visibility, lm[L_SHOULDER].visibility, lm[L_HIP].visibility)
    side_r = min(lm[R_EAR].visibility, lm[R_SHOULDER].visibility, lm[R_HIP].visibility)
    ear, shoulder, hip = (
        (lm[L_EAR], lm[L_SHOULDER], lm[L_HIP]) if side_l >= side_r
        else (lm[R_EAR], lm[R_SHOULDER], lm[R_HIP])
    )
    neck_fwd = (ear.x - shoulder.x) * 100
    if neck_fwd > 4:
        issues.append(f"Head forward  ({neck_fwd:.1f}%)")
    spine = _angle(ear, shoulder, hip)
    if spine < 155:
        issues.append(f"Spine angle  ({spine:.0f}\u00b0)")
    return ("Good Posture" if not issues else "Poor Posture"), issues


def main() -> None:
    ensure_model()
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_FILE)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(0)
    t0      = time.monotonic()
    last_ts = -1
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms = max(int((time.monotonic() - t0) * 1000), last_ts + 1)
            last_ts = ts_ms
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
            )
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                draw_skeleton(frame, lm)
                status, issues = analyse_posture(lm)
                colour = (100, 220, 100) if status == "Good Posture" else (50, 50, 255)
                cv2.putText(frame, status, (16, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
                for i, issue in enumerate(issues):
                    cv2.putText(frame, f"  {issue}", (16, 72 + i * 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 150, 255), 1)
            cv2.imshow("Posture Detection  (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
