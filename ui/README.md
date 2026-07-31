# ui/

tkinter starter scripts for building Python desktop windows. No third-party UI framework needed — everything here uses the standard library plus packages already in `requirements.txt`.

---

### `basic_window.py`
The foundation. Shows how to set up a window properly: centring it on screen, adding a menu bar, a status bar at the bottom, a custom close dialog, and a `Ctrl+Q` shortcut. Good starting point if you're building something new.

```bash
python ui/basic_window.py
```

---

### `camera_feed.py`
Live webcam feed inside a tkinter window. Automatically detects available cameras and lets you pick one from a dropdown. The frame is scaled to fit the window and the UI stays responsive because the camera is polled with `root.after()` rather than a blocking loop.

```bash
python ui/camera_feed.py
```

---

### `audio_input.py`
Microphone level monitor. Shows a colour-coded volume bar (green → yellow → red) with a peak-hold marker and a live dBFS readout. Audio capture runs in a background thread so the window never freezes.

```bash
python ui/audio_input.py
```

---

All three scripts follow the same structure: `_configure_root`, `_build_*` methods for each UI section, and `_on_close` for cleanup. Copy any of them as a base for your own windows.
