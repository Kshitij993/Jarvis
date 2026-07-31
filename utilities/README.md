# utilities/

Helper modules for detecting hardware and reading system info. Each one works standalone (just run it directly) or can be imported into other scripts.

---

### `camera_detector.py`
Scans for available cameras by probing OpenCV indices and returns a list of `CameraInfo` objects with the index and backend name.

```bash
python utilities/camera_detector.py
```

```python
from utilities.camera_detector import list_cameras
cameras = list_cameras()
```

---

### `microphone_detector.py`
Lists all audio input devices using PyAudio. Returns `MicrophoneInfo` objects with the device index, name, and sample rate.

```bash
python utilities/microphone_detector.py
```

```python
from utilities.microphone_detector import list_microphones
mics = list_microphones()
```

---

### `system_info.py`
Prints a summary of your system: OS, CPU, RAM, Python version, and OpenCV build info. Useful for debugging environment issues.

```bash
python utilities/system_info.py
```

```python
from utilities.system_info import get_system_info
info = get_system_info()
```
