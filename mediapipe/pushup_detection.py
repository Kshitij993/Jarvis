"""
Push-Up Detection & Counter
Uses the MediaPipe PoseLandmarker (Tasks API) to count push-up reps
by tracking the elbow angle in real-time.

Model (~5 MB) is downloaded automatically on first run.
Press 'r' to reset the counter, 'q' to quit.
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

ELBOW_DOWN = 90
ELBOW_UP   = 160

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(17,19),(19,15),(15,21),
    (12,14),(14,16),(16,18),(18,20),(20,16),(16,22),
    (11,23),(12,24),(23,24),
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
        if a < len(lm) and b < len(lm):
            if lm[a].visibility > 0.5 and lm[b].visibility > 0.5:
                cv2.line(frame, pts[a], pts[b], (100, 200, 100), 2)
    for i, (x, y) in enumerate(pts[:25]):
        if lm[i].visibility > 0.5:
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), 4, (100, 200, 100), 1)


def draw_panel(frame, count: int, stage: str, angle: float) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 120), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, "REPS",        (10,  30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)
    cv2.putText(frame, str(count),    (10,  80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,220,0),     3)
    cv2.putText(frame, "STAGE",       (130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)
    cv2.putText(frame, stage.upper(), (130, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,255),   2)
    cv2.putText(frame, f"Elbow: {angle:.1f} deg", (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)


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
    cap   = cv2.VideoCapture(0)
    t0      = time.monotonic()
    last_ts = -1
    count = 0
    stage = "up"
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
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
            angle = 0.0
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                draw_skeleton(frame, lm)
                if lm[11].visibility >= lm[12].visibility:
                    shoulder, elbow, wrist = lm[11], lm[13], lm[15]
                else:
                    shoulder, elbow, wrist = lm[12], lm[14], lm[16]
                angle = _angle(shoulder, elbow, wrist)
                if angle < ELBOW_DOWN:
                    stage = "down"
                if angle > ELBOW_UP and stage == "down":
                    stage = "up"; count += 1
                ex, ey = int(elbow.x * w), int(elbow.y * h)
                cv2.circle(frame, (ex, ey), 12, (0, 255, 255), -1)
                cv2.putText(frame, f"{angle:.0f}\u00b0", (ex + 15, ey),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            draw_panel(frame, count, stage, angle)
            cv2.imshow("Push-Up Detection  (q/r to quit/reset)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("r"): count = 0; stage = "up"
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


def calculate_angle(a, b, c) -> float:
    """Return the angle (degrees) at joint b formed by points a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


# Thresholds
ELBOW_DOWN_ANGLE = 90    # arm is bent (bottom of push-up)
ELBOW_UP_ANGLE   = 160   # arm is extended (top of push-up)


def draw_counter_panel(frame, count: int, stage: str, angle: float):
    """Overlay a semi-transparent panel showing rep count and stage."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 120), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, "REPS",  (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(frame, str(count), (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 220, 0), 3)

    cv2.putText(frame, "STAGE", (130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(frame, stage.upper(), (130, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

    cv2.putText(frame, f"Elbow: {angle:.1f} deg", (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def run(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}.")
        return

    count = 0
    stage = "up"   # start assuming arms are extended

    print("[INFO] Starting push-up detection. Press 'q' to quit, 'r' to reset counter.")

    with mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            angle = 0.0

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                def coords(part):
                    p = lm[part]
                    return [p.x * w, p.y * h]

                # Use the side with higher visibility for angle calculation
                left_vis  = lm[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility
                right_vis = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility

                if left_vis >= right_vis:
                    shoulder = coords(mp_pose.PoseLandmark.LEFT_SHOULDER)
                    elbow    = coords(mp_pose.PoseLandmark.LEFT_ELBOW)
                    wrist    = coords(mp_pose.PoseLandmark.LEFT_WRIST)
                else:
                    shoulder = coords(mp_pose.PoseLandmark.RIGHT_SHOULDER)
                    elbow    = coords(mp_pose.PoseLandmark.RIGHT_ELBOW)
                    wrist    = coords(mp_pose.PoseLandmark.RIGHT_WRIST)

                angle = calculate_angle(shoulder, elbow, wrist)

                # ── State machine ─────────────────────────────────────────────
                if angle < ELBOW_DOWN_ANGLE:
                    stage = "down"
                if angle > ELBOW_UP_ANGLE and stage == "down":
                    stage = "up"
                    count += 1
                    print(f"[REP] Count: {count}")

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                # Highlight elbow joint
                ex, ey = int(elbow[0]), int(elbow[1])
                cv2.circle(frame, (ex, ey), 12, (0, 255, 255), -1)
                cv2.putText(frame, f"{angle:.0f}°", (ex + 15, ey),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            draw_counter_panel(frame, count, stage, angle)

            cv2.imshow("Push-Up Detection", frame)
            key = cv2.waitKey(5) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                count = 0
                stage = "up"
                print("[INFO] Counter reset.")

