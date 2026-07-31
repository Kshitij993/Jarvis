# Robotics Project

Real-time computer vision, speech, and AI assistant scripts using MediaPipe, OpenCV, PyAudio, Vosk, and InsightFace.

```
Robotics/
├── jarvis/         AI personal assistant — camera + face recognition + STT + LLM chat
├── mediapipe/      face detection, emotion, landmarks, hand tracking, gesture, posture, push-ups
├── audio/          microphone selector, record/play, speech-to-text, text-to-speech, live STT UI
├── recognition/    face recognition (train on photos, identify people in real-time)
├── llm/            LLM chat UI — OpenBridge AI personal server
├── voice_output/   standalone text-to-speech UI
├── ui/             tkinter window starter scripts
└── utilities/      camera, microphone, system info helpers
```

## Setup

Run the installer for your platform — it handles Python, the virtual environment, and all packages automatically.

**Windows**
```bat
install.bat
```

**Linux**
```bash
chmod +x install.sh && ./install.sh
```

## Activating the environment

You need to activate the venv every time you open a new terminal.

**Windows — Command Prompt**
```bat
venv\Scripts\activate.bat
```

**Windows — PowerShell**
```powershell
.\venv\Scripts\Activate.ps1
```
> If you get a "scripts disabled" error, run this once first:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Linux**
```bash
source venv/bin/activate
```

You'll know it's active when you see `(venv)` at the start of your prompt.

## Running scripts

With the venv active, just run the script directly:

```bash
python mediapipe/face_emotion_detection.py
python ui/camera_feed.py
python utilities/camera_detector.py
```

## Using IDLE

Don't open IDLE from the Start Menu — it uses the system Python which doesn't have the packages. Launch it from inside the activated venv instead:

```bash
python -m idlelib
```

IDLE opens as a separate window (the terminal just returns to the prompt — that's normal). From there, **File → Open** your script, then **F5** to run it.

## What's in each folder

**jarvis/** — `jarvis.py`
AI personal assistant combining camera, face recognition, emotion detection, speech-to-text, and LLM chat.

**recognition/** — `train.py`, `recognize.py`
See [recognition/README.md](recognition/README.md)

**llm/** — `llm_ui.py`
Chat UI for OpenBridge AI (or any OpenAI-compatible server). Config stored in `llm/.llm_config.json`.
See [llm/README.md](llm/README.md)

**voice_output/** — `tts_ui.py`
Standalone text-to-speech UI with voice/rate/volume controls and history.

**audio/** — `audio_input_selector.py`, `record_and_play.py`, `speech_to_text.py`, `text_to_speech.py`, `live_speech_to_text_ui.py`, `live_speech_to_text_offline_ui.py`
See [audio/README.md](audio/README.md)

**mediapipe/** — `face_emotion_detection.py`, `face_detection.py`, `face_landmarks.py`, `posture_detection.py`, `pushup_detection.py`, `hand_tracking.py`, `gesture_recognition.py`
See [mediapipe/README.md](mediapipe/README.md)

**ui/** — `basic_window.py`, `camera_feed.py`, `audio_input.py`
See [ui/README.md](ui/README.md)

**utilities/** — `camera_detector.py`, `microphone_detector.py`, `system_info.py`
See [utilities/README.md](utilities/README.md)

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Camera capture and image processing |
| `mediapipe` | Pose, face mesh, and landmark detection |
| `numpy` | Numerical operations on landmark arrays |
| `pyaudio` | Microphone device enumeration |
| `psutil` | System hardware information |
