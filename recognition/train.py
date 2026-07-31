"""
Face Recognition — Train
Scans known_faces/ for images, extracts face embeddings using InsightFace,
and saves them to embeddings.json.

Naming convention for images:
    john_1.jpg, john_2.jpg   → person "john"
    alice_1.jpg, alice_2.jpg → person "alice"
    (anything before the last underscore+number is the person's name)

Run:
    python recognition/train.py

Requirements:
    pip install insightface onnxruntime
"""

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

KNOWN_DIR  = Path(__file__).parent / "known_faces"
EMBED_FILE = Path(__file__).parent / "embeddings.json"
MODELS_DIR = Path(__file__).parent / "models"   # models stored here, not in home dir


def main() -> None:
    KNOWN_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    images = sorted(
        list(KNOWN_DIR.glob("*.jpg")) +
        list(KNOWN_DIR.glob("*.jpeg")) +
        list(KNOWN_DIR.glob("*.png"))
    )

    if not images:
        print(f"No images found in  {KNOWN_DIR}")
        print("Add images like:  john_1.jpg  john_2.jpg  alice_1.jpg  ...")
        return

    print("Loading InsightFace model (buffalo_sc, first run downloads ~100 MB)...")
    app = FaceAnalysis(
        name="buffalo_sc",
        root=str(MODELS_DIR),                    # store models here
        allowed_modules=["detection", "recognition"],  # skip landmark + age/gender
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(320, 320))

    # person_name → list of embedding vectors
    person_embeddings: dict[str, list[np.ndarray]] = {}

    for img_path in images:
        # "john_doe_2" → "john_doe",  "alice_1" → "alice"
        stem  = img_path.stem
        parts = stem.rsplit("_", 1)
        name  = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ✗  {img_path.name}  (cannot read file)")
            continue

        faces = app.get(img)
        if not faces:
            print(f"  ✗  {img_path.name}  (no face detected — try a clearer photo)")
            continue

        # Use the largest detected face if multiple appear in one photo
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        person_embeddings.setdefault(name, []).append(face.embedding)
        print(f"  ✓  {img_path.name}  → {name}")

    if not person_embeddings:
        print("\nNo embeddings created.")
        return

    # Average all embeddings per person then L2-normalise (needed for cosine similarity)
    averaged: dict[str, list[float]] = {}
    for name, embs in person_embeddings.items():
        avg = np.mean(embs, axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-9)
        averaged[name] = avg.tolist()

    EMBED_FILE.write_text(json.dumps(averaged, indent=2))
    print(f"\nSaved {len(averaged)} person(s) to {EMBED_FILE.name}")
    for name in sorted(averaged):
        n = len(person_embeddings[name])
        print(f"  {name}  ({n} image{'s' if n > 1 else ''})")


if __name__ == "__main__":
    main()
