"""
Face Recognition — Real-time Webcam
Reads embeddings.json (built by train.py) and recognises faces from the webcam.

  Green box + name  → known person
  Red box           → unknown face

Run train.py first, then:
    python recognition/recognize.py

Options:
    --camera 0          webcam index (default 0)
    --threshold 0.45    cosine-distance cut-off; lower = stricter (default 0.45)
    --every 3           run detection every N frames for smoother display
"""

import argparse
import json
import sys
import warnings
import cv2
import numpy as np
from pathlib import Path

# insightface/face_align.py calls the deprecated skimage tform.estimate().
# This is a bug in the insightface library, not our code.  The correct fix is
# to upgrade:  pip install --upgrade insightface
# Until the installed version patches it, suppress to keep output clean.
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("[ERROR] insightface not installed.  Run:  pip install insightface onnxruntime")
    sys.exit(1)

EMBED_FILE = Path(__file__).parent / "embeddings.json"
MODELS_DIR = Path(__file__).parent / "models"


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Lower = more similar.  0 = identical,  2 = opposite."""
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def draw_face(frame, x1: int, y1: int, x2: int, y2: int,
              name: str, confidence: float) -> None:
    known   = name != "Unknown"
    colour  = (0, 220, 80) if known else (0, 60, 220)
    label   = f"{name}  {confidence:.0%}" if known else "Unknown"

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    # Filled label background for readability
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), colour, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera",    type=int,   default=0)
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="Cosine distance threshold (lower = stricter)")
    ap.add_argument("--every",     type=int,   default=3,
                    help="Run detection every N frames")
    args = ap.parse_args()

    if not EMBED_FILE.exists():
        print(f"[ERROR] No embeddings found at {EMBED_FILE}")
        print("Run train.py first.")
        sys.exit(1)

    known: dict[str, np.ndarray] = {
        name: np.array(emb)
        for name, emb in json.loads(EMBED_FILE.read_text()).items()
    }
    print(f"Loaded {len(known)} person(s): {', '.join(sorted(known))}")

    print("Loading InsightFace model (buffalo_sc, CPU-optimised)...")
    app = FaceAnalysis(
        name="buffalo_sc",
        root=str(MODELS_DIR),                    # models in recognition/models/
        allowed_modules=["detection", "recognition"],  # skip landmark + age/gender
        providers=["CPUExecutionProvider"],
    )
    # det_size=(160,160) for real-time speed; raise to (320,320) for small/far faces
    app.prepare(ctx_id=0, det_size=(160, 160))

    cap     = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # DirectShow avoids MSMF async stderr warning
    frame_n = 0
    cached: list[tuple] = []   # (x1, y1, x2, y2, name, confidence)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_n += 1

        if frame_n % args.every == 0:
            faces  = app.get(frame)
            cached = []

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-9)

                best_name = "Unknown"
                best_dist = args.threshold

                for name, known_emb in known.items():
                    d = cosine_distance(emb, known_emb)
                    if d < best_dist:
                        best_dist = d
                        best_name = name

                confidence = 1.0 - best_dist  # higher = more certain
                cached.append((x1, y1, x2, y2, best_name, confidence))

        # Always draw from cache so the box stays visible between detection frames
        for x1, y1, x2, y2, name, conf in cached:
            draw_face(frame, x1, y1, x2, y2, name, conf)

        # HUD
        cv2.putText(frame, f"People: {len(cached)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Face Recognition  (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
