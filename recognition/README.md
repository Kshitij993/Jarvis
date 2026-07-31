# recognition/

Real-time face recognition using [InsightFace](https://github.com/deepinsight/insightface) — the same library used in production systems. No cloud API, runs entirely on CPU.

---

## Quick start

### 1. Install

```bash
pip install insightface onnxruntime
```

### 2. Add training photos

Put photos in `recognition/known_faces/`. Name them with the person's name followed by `_N`:

```
known_faces/
    john_1.jpg
    john_2.jpg
    alice_1.jpg
    alice_2.jpg
```

More photos per person = more accurate. 3–5 clear, well-lit photos is enough.

### 3. Train (build embeddings)

```bash
python recognition/train.py
```

Creates `recognition/embeddings.json` with one averaged face vector per person. Only needs to run when you add/change photos.

### 4. Run recognition

```bash
python recognition/recognize.py
```

---

## How it works

| Step | What happens |
|---|---|
| `train.py` | Reads each photo → detects the largest face → extracts a 512-dim embedding vector → averages all vectors per person → saves to JSON |
| `recognize.py` | Detects faces every 3 frames → extracts embedding → computes cosine distance against every known person → draws green box + name if distance < threshold |

**Threshold** (default `0.45`): lower = stricter. Adjust with `--threshold 0.4` if you get false positives, or `--threshold 0.5` if known people aren't being detected.

---

## Options

```bash
python recognition/recognize.py --camera 1          # use camera index 1
python recognition/recognize.py --threshold 0.40    # stricter matching
python recognition/recognize.py --every 5           # run detection every 5 frames
```
