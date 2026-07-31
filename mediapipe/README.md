# mediapipe/

Real-time computer vision scripts using MediaPipe and OpenCV. Each one opens your webcam and runs until you press `q`.

---

### `posture_detection.py`
Watches your posture via webcam. Tracks shoulder alignment, neck tilt, and the ear→shoulder→hip angle to decide if you're sitting straight or slouching. Shows a live **Good / Poor Posture** label on the frame.

```bash
python mediapipe/posture_detection.py
```

---

### `face_emotion_detection.py`
Detects emotions on up to 4 faces at once using the 468-point Face Mesh. Classifies each face as Neutral, Happy, Sad, Surprised, or Angry based on geometric ratios (mouth openness, smile corners, eyebrow height).

> The classifier is heuristic-based. For better accuracy swap `classify_emotion()` with a trained model that takes the full landmark vector.

```bash
python mediapipe/face_emotion_detection.py
```

---

### `face_detection.py`
Fast, lightweight face detection using MediaPipe's bounding-box detector. Draws a box and the 6 key facial points (eyes, nose, mouth, ears) for every face it finds, plus a confidence score. Much faster than Face Mesh — use this when you just need to know where faces are.

```bash
python mediapipe/face_detection.py
```

---

### `face_landmarks.py`
Draws all 468 Face Mesh landmarks (478 with iris tracking enabled). Everything is toggleable while the script is running:

| Key | Toggle |
|-----|--------|
| `m` | Iris mesh |
| `c` | Contours (eyes, lips, face oval) |
| `t` | Full tesselation |
| `i` | Landmark index numbers (first 50) |

```bash
python mediapipe/face_landmarks.py
```

---

### `hand_tracking.py`
Tracks up to 2 hands simultaneously. Draws the full 21-point skeleton (landmarks + connections) and labels each hand as Left or Right with a confidence score.

```bash
python mediapipe/hand_tracking.py
```

---

### `gesture_recognition.py`
Builds on hand tracking to classify gestures in real-time. Recognised gestures: **Open Palm, Fist, Thumbs Up, Thumbs Down, Peace / Victory, Pointing, OK**. Works by checking which fingers are extended and comparing thumb/index tip distance for the OK sign.

```bash
python mediapipe/gesture_recognition.py
```

---

### `pushup_detection.py`
Counts push-up reps by tracking your elbow angle. Picks whichever arm is more visible and uses an up/down state machine (down < 90°, up > 160°) to count each full rep. Press `r` to reset the counter.

```bash
python mediapipe/pushup_detection.py
```

---

`q` quits any script — `r` resets the counter in pushup_detection.
