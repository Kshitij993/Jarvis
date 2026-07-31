"""
Live Speech to Text — Offline UI
Works entirely on your device. No internet connection needed.

Uses Vosk for recognition, which gives true real-time partial results —
words appear on screen as you speak, not after you finish.

How silence detection works:
  The "Pause to finalise" slider sets how long audio must stay below the
  energy threshold before the current phrase is committed to the transcript.
  Raise it to avoid mid-sentence cuts. Lower it for snappier response.

First run:
  Click "Download Model" (~50 MB, one-time). It saves to audio/models/ and
  is reused automatically on every future run.

Run:
    python audio/live_speech_to_text_offline_ui.py
"""

import numpy as _np
import json
import math
import queue as _queue
import struct
import sys
import threading
import time
import tkinter as tk
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import pyaudio

try:
    import pyaudiowpatch as _pa_lib
except ImportError:
    import pyaudio as _pa_lib          # type: ignore[no-redef]

try:
    from vosk import KaldiRecognizer, Model, SetLogLevel
    SetLogLevel(-1)   # suppress Vosk's verbose console output
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utilities.microphone_detector import (
    MicrophoneInfo, list_microphones, list_system_audio_devices,
    get_best_system_audio_device,
)

# ── Model catalogue ──────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / "models"

VOSK_MODELS: dict[str, dict] = {
    "Small EN  ~40 MB   (fast)": {
        "id":   "vosk-model-small-en-us-0.15",
        "url":  "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "size": "~40 MB",
        "note": "Fast and lightweight. Great for real-time transcription on any CPU.",
    },
    "Medium EN  ~128 MB  (recommended)": {
        "id":   "vosk-model-en-us-0.22-lgraph",
        "url":  "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip",
        "size": "~128 MB",
        "note": "Same acoustic engine as the Large model, compressed language graph. "
                "Noticeably better than Small for everyday speech; 3\u00d7 faster than Large. "
                "May struggle with rare words or names not in its fixed vocabulary.",
    },
    "Large EN  ~1.8 GB  (best accuracy)": {
        "id":   "vosk-model-en-us-0.22",
        "url":  "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
        "size": "~1.8 GB",
        "note": "Highest accuracy English model. Needs more RAM and a fast CPU.",
    },
    "GigaSpeech EN  ~2.3 GB  (meetings/video)": {
        "id":   "vosk-model-en-us-0.42-gigaspeech",
        "url":  "https://alphacephei.com/vosk/models/vosk-model-en-us-0.42-gigaspeech.zip",
        "size": "~2.3 GB",
        "note": "Trained on diverse audio (meetings, YouTube, podcasts). Best for system audio.",
    },
    "Indian EN  ~36 MB  (IN accent)": {
        "id":   "vosk-model-small-en-in-0.4",
        "url":  "https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip",
        "size": "~36 MB",
        "note": "Optimised for Indian English accent.",
    },
    "Indian EN  ~1 GB  (IN accent, high accuracy)": {
        "id":   "vosk-model-en-in-0.5",
        "url":  "https://alphacephei.com/vosk/models/vosk-model-en-in-0.5.zip",
        "size": "~1 GB",
        "note": "High-accuracy Indian English model. Better than the small version for complex vocabulary.",
    },
}

# ── Audio constants ───────────────────────────────────────────────────────────
CHUNK    = 2048
FORMAT   = pyaudio.paInt16
CHANNELS = 1
RATE     = 16000   # Vosk small model expects 16 kHz


# ── Audio conversion helpers (replaces removed audioop module) ────────────────

def _to_mono(data: bytes, channels: int) -> bytes:
    """
    Convert multi-channel int16 PCM to mono.

    Uses channel 0 only (NOT an average).  Averaging can silently cancel the
    signal on USB headsets (e.g. Jabra) where ch1 is a reference / feedback
    signal that is phase-inverted relative to ch0.
    """
    if channels == 1:
        return data
    arr = _np.frombuffer(data, dtype=_np.int16).reshape(-1, channels)
    return _np.ascontiguousarray(arr[:, 0]).tobytes()


def _float32_to_int16(data: bytes) -> bytes:
    """Convert float32 PCM to int16 (WASAPI loopback returns float32)."""
    arr = _np.frombuffer(data, dtype=_np.float32)
    return (arr * 32767).clip(-32768, 32767).astype(_np.int16).tobytes()


def _find_mme_device(pa, device_index: int | None) -> int | None:
    """
    Return the MME device index for the same physical hardware as device_index.

    MME routes through the Windows Audio Engine which performs high-quality
    sample-rate conversion automatically (48000→16000 Hz etc.) so no manual
    resampling is needed and Vosk receives clean 16 kHz audio directly.
    Returns None if no matching MME device is found.
    """
    if device_index is None:
        return None
    try:
        target_name = pa.get_device_info_by_index(device_index).get("name", "")
        for h in range(pa.get_host_api_count()):
            if "MME" not in pa.get_host_api_info_by_index(h).get("name", ""):
                continue
            for i in range(pa.get_device_count()):
                dev = pa.get_device_info_by_index(i)
                if (int(dev.get("hostApi", -1)) == h
                        and dev.get("name", "") == target_name
                        and int(dev.get("maxInputChannels", 0)) > 0):
                    return i
    except Exception:
        pass
    return None


def _open_input(pa, device_index, channels: int, rate: int):
    """
    Open an input stream.  Returns (stream, is_float32, actual_channels, actual_rate).

    Strategy:
      1. MME version of the device at the requested rate.
         MME routes through Windows Audio Engine which converts sample rates
         automatically (e.g. Jabra 48 kHz → 16 kHz).  No manual resampling needed.
      2. Original device at the requested rate.
      3. Original device at its native rate (caller must resample).
    """
    mme_idx = _find_mme_device(pa, device_index)

    # Native rate for fallback
    native = rate
    if device_index is not None:
        try:
            native = int(pa.get_device_info_by_index(device_index)
                         .get("defaultSampleRate", rate))
        except Exception:
            pass

    ch_list  = list(dict.fromkeys([channels, 2] if channels == 1 else [channels]))
    fmt_list = [
        (pyaudio.paInt16,   False, CHUNK),
        (pyaudio.paFloat32, True,  CHUNK),
        (pyaudio.paFloat32, True,  0),
        (pyaudio.paInt16,   False, 0),
    ]

    last_err = None

    # Pass 1: MME device at the exact requested rate (Windows SRC, no resampling)
    if mme_idx is not None:
        for ch in ch_list:
            for fmt, as_float, buf in fmt_list:
                try:
                    kw = dict(format=fmt, channels=ch, rate=rate,
                              input=True, input_device_index=mme_idx)
                    if buf > 0: kw["frames_per_buffer"] = buf
                    return pa.open(**kw), as_float, ch, rate
                except Exception as e:
                    last_err = e

    # Pass 2: original device at requested rate, then native rate (caller resamples)
    for r in list(dict.fromkeys([rate, native, 48000, 44100, 16000])):
        for ch in ch_list:
            for fmt, as_float, buf in fmt_list:
                try:
                    kw = dict(format=fmt, channels=ch, rate=r,
                              input=True, input_device_index=device_index)
                    if buf > 0: kw["frames_per_buffer"] = buf
                    return pa.open(**kw), as_float, ch, r
                except Exception as e:
                    last_err = e

    raise OSError(f"Device {device_index}: no supported format — {last_err}")


def _mix_pcm(a: bytes, b: bytes) -> bytes:
    """Average two mono int16 PCM buffers (equal mix of mic + system audio)."""
    n = min(len(a), len(b)) // 2
    if n == 0:
        return a or b
    fa = _np.frombuffer(a[:n * 2], dtype=_np.int16).astype(_np.float32)
    fb = _np.frombuffer(b[:n * 2], dtype=_np.int16).astype(_np.float32)
    return ((fa + fb) / 2).clip(-32768, 32767).astype(_np.int16).tobytes()


def _resample(data: bytes, in_rate: int, out_rate: int) -> bytes:
    """
    Resample mono int16 PCM.  For integer ratios (e.g. 48000→16000 = 3×)
    a box-filter (local average) is applied before decimation to prevent
    aliasing — this is meaningfully better than bare linear interpolation
    for speech fed to Vosk.
    """
    if in_rate == out_rate or len(data) == 0:
        return data
    arr = _np.frombuffer(data, dtype=_np.int16).astype(_np.float32)
    ratio = in_rate / out_rate
    int_ratio = round(ratio)
    if abs(ratio - int_ratio) < 0.05 and int_ratio > 1:
        # Integer downsampling: pad to multiple of int_ratio, average each block
        pad = (-len(arr)) % int_ratio
        if pad:
            arr = _np.concatenate([arr, _np.zeros(pad, dtype=_np.float32)])
        decimated = arr.reshape(-1, int_ratio).mean(axis=1)
        return decimated.clip(-32768, 32767).astype(_np.int16).tobytes()
    # Non-integer ratio: linear interpolation
    out_len = max(1, int(len(arr) * out_rate / in_rate))
    resampled = _np.interp(
        _np.linspace(0, 1, out_len),
        _np.linspace(0, 1, len(arr)),
        arr,
    )
    return resampled.astype(_np.int16).tobytes()


# ── Indicator colours ─────────────────────────────────────────────────────────
_PULSE_LISTEN = ("#3d7a4f", "#a6e3a1")
_PULSE_PROC   = ("#7a6a1e", "#f9e2af")
_IDLE_COL     = "#45475a"


class OfflineSTTWindow:
    BG      = "#1e1e2e"
    SURFACE = "#2a2a3e"
    ACCENT  = "#7c6af7"
    FG      = "#cdd6f4"
    FG_DIM  = "#6c7086"
    RED     = "#f38ba8"
    GREEN   = "#a6e3a1"

    def __init__(self, root: tk.Tk) -> None:
        self.root         = root
        self._running     = False
        self._model: Model | None = None
        self._thread: threading.Thread | None = None

        # Settings
        self._model_key    = tk.StringVar(value=next(iter(VOSK_MODELS)))
        self._pause_var    = tk.DoubleVar(value=1.2)
        self._energy_var   = tk.IntVar(value=50)
        self._auto_energy  = tk.BooleanVar(value=True)
        self._sys_audio_var = tk.BooleanVar(value=False)
        self._mix_mic_var   = tk.BooleanVar(value=False)
        self._sys_audio_dev: MicrophoneInfo | None = None
        self._session_id    = 0  # incremented each Start; old threads exit when they see a mismatch

        # Pulse state
        self._pulse_after  = None
        self._pulse_colors = _PULSE_LISTEN

        self._configure_root()
        self._build_controls()
        self._build_model_bar()
        self._build_settings()
        self._build_indicator()
        self._build_live_preview()
        self._build_transcript()
        self._build_statusbar()
        self._populate_devices()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._check_vosk_installed)

    # ── Root ──────────────────────────────────────────────────

    def _configure_root(self) -> None:
        self.root.title("Live Speech to Text  —  Offline")
        self.root.geometry("740x700")
        self.root.minsize(580, 540)
        self.root.configure(bg=self.BG)
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"740x700+{(sw - 740) // 2}+{(sh - 700) // 2}")

    # ── Controls bar ─────────────────────────────────────────

    def _build_controls(self) -> None:
        ctrl = tk.Frame(self.root, bg=self.SURFACE, pady=10)
        ctrl.pack(fill="x")

        tk.Label(ctrl, text="Microphone:", bg=self.SURFACE, fg=self.FG,
                 font=("Segoe UI", 10)).pack(side="left", padx=(16, 4))

        self.device_var  = tk.StringVar(value="System Default")
        self.device_menu = tk.OptionMenu(ctrl, self.device_var, "System Default")
        self.device_menu.config(
            bg=self.SURFACE, fg=self.FG, font=("Segoe UI", 10),
            highlightthickness=0, relief="flat",
            activebackground=self.ACCENT, activeforeground="#fff",
        )
        self.device_menu["menu"].config(
            bg=self.SURFACE, fg=self.FG,
            activebackground=self.ACCENT, activeforeground="#fff",
        )
        self.device_menu.pack(side="left", padx=(0, 4))

        tk.Button(
            ctrl, text="\u21bb",  # ↻ refresh symbol
            command=self._populate_devices,
            bg=self.SURFACE, fg=self.FG_DIM,
            font=("Segoe UI", 10), relief="flat",
            padx=4, pady=3, cursor="hand2",
            activebackground=self.ACCENT, activeforeground="#fff",
        ).pack(side="left", padx=(0, 8))

        self.toggle_btn = tk.Button(
            ctrl, text="▶  Start",
            command=self._toggle,
            bg=self.ACCENT, fg="#fff",
            font=("Segoe UI", 10, "bold"),
            relief="flat", padx=16, pady=6, cursor="hand2",
            activebackground="#6c5ce7", activeforeground="#fff",
            state="disabled",
        )
        self.toggle_btn.pack(side="left", padx=4)

        tk.Button(
            ctrl, text="\U0001f3a4 Test Mic",
            command=self._test_mic,
            bg=self.SURFACE, fg=self.FG_DIM,
            font=("Segoe UI", 9), relief="flat",
            padx=8, pady=3, cursor="hand2",
            activebackground=self.ACCENT, activeforeground="#fff",
        ).pack(side="left", padx=(0, 8))
        # System audio toggle
        self._sys_btn = tk.Checkbutton(
            ctrl, text="\u266a  System Audio",
            variable=self._sys_audio_var,
            command=self._on_sys_audio_toggle,
            bg=self.SURFACE, fg=self.FG_DIM,
            selectcolor=self.SURFACE,
            activebackground=self.SURFACE, activeforeground=self.FG,
            font=("Segoe UI", 9),
        )
        self._sys_btn.pack(side="left", padx=(8, 0))
        # Mix-mic checkbox (only visible when System Audio is ON)
        self._mix_btn = tk.Checkbutton(
            ctrl, text="+ Mix mic",
            variable=self._mix_mic_var,
            bg=self.SURFACE, fg=self.FG_DIM,
            selectcolor=self.SURFACE,
            activebackground=self.SURFACE, activeforeground=self.FG,
            font=("Segoe UI", 9),
        )
        # packed/unpacked dynamically in _on_sys_audio_toggle
        tk.Button(ctrl, text="Copy", command=self._copy,
                  bg=self.SURFACE, fg=self.FG, font=("Segoe UI", 9),
                  relief="flat", padx=10, pady=6, cursor="hand2",
                  activebackground=self.ACCENT, activeforeground="#fff",
                  ).pack(side="right", padx=4)
        tk.Button(ctrl, text="Clear", command=self._clear,
                  bg=self.SURFACE, fg=self.FG, font=("Segoe UI", 9),
                  relief="flat", padx=10, pady=6, cursor="hand2",
                  activebackground=self.RED, activeforeground="#fff",
                  ).pack(side="right", padx=4)

    # ── Model bar ─────────────────────────────────────────────

    def _build_model_bar(self) -> None:
        bar = tk.Frame(self.root, bg=self.BG, pady=6)
        bar.pack(fill="x", padx=16)

        tk.Label(bar, text="Model:", bg=self.BG, fg=self.FG_DIM,
                 font=("Segoe UI", 9)).pack(side="left")

        self._model_menu_w = tk.OptionMenu(
            bar, self._model_key, *VOSK_MODELS.keys(),
            command=self._on_model_select,
        )
        self._model_menu_w.config(
            bg=self.SURFACE, fg=self.FG, font=("Segoe UI", 9),
            highlightthickness=0, relief="flat", width=34,
            activebackground=self.ACCENT, activeforeground="#fff",
        )
        self._model_menu_w["menu"].config(
            bg=self.SURFACE, fg=self.FG,
            activebackground=self.ACCENT, activeforeground="#fff",
        )
        self._model_menu_w.pack(side="left", padx=(6, 6))

        self._load_btn = tk.Button(
            bar, text="Load", command=self._load_selected_model,
            bg=self.SURFACE, fg=self.FG, font=("Segoe UI", 9),
            relief="flat", padx=10, pady=4, cursor="hand2",
            activebackground=self.ACCENT, activeforeground="#fff",
        )
        self._load_btn.pack(side="left", padx=(0, 4))

        self._dl_btn = tk.Button(
            bar, text="Download", command=self._start_download,
            bg=self.ACCENT, fg="#fff",
            font=("Segoe UI", 9, "bold"),
            relief="flat", padx=10, pady=4, cursor="hand2",
            activebackground="#6c5ce7", activeforeground="#fff",
        )
        self._dl_btn.pack(side="left", padx=(0, 4))

        tk.Button(
            bar, text="Browse", command=self._browse_model,
            bg=self.SURFACE, fg=self.FG, font=("Segoe UI", 9),
            relief="flat", padx=8, pady=4, cursor="hand2",
            activebackground=self.ACCENT, activeforeground="#fff",
        ).pack(side="left")

        # Download progress bar (hidden until download starts)
        self._dl_frame = tk.Frame(self.root, bg=self.BG)
        self._dl_canvas = tk.Canvas(self._dl_frame, bg=self.SURFACE,
                                    height=6, highlightthickness=0)
        self._dl_canvas.pack(fill="x", padx=16, pady=(0, 4))
        self._dl_label = tk.Label(self._dl_frame, text="", bg=self.BG,
                                  fg=self.FG_DIM, font=("Segoe UI", 8))
        self._dl_label.pack(anchor="w", padx=16)

        # Model note label
        self._model_note = tk.Label(
            self.root, text="", bg=self.BG, fg=self.FG_DIM,
            font=("Segoe UI", 8), anchor="w",
        )
        self._model_note.pack(fill="x", padx=16)

        # Separator
        tk.Frame(self.root, bg=self.SURFACE, height=1).pack(fill="x", padx=16, pady=(4, 4))

        # Refresh button state for the default selection
        self._on_model_select(self._model_key.get())

        # Separator
        tk.Frame(self.root, bg=self.SURFACE, height=1).pack(fill="x", padx=16, pady=(0, 4))

    # ── Settings ─────────────────────────────────────────────

    def _build_settings(self) -> None:
        panel = tk.Frame(self.root, bg=self.BG, pady=4)
        panel.pack(fill="x", padx=16)

        row1 = tk.Frame(panel, bg=self.BG)
        row1.pack(fill="x", pady=(0, 4))

        tk.Label(row1, text="Pause to finalise:", bg=self.BG, fg=self.FG_DIM,
                 font=("Segoe UI", 9)).pack(side="left")
        self._pause_lbl = tk.Label(row1, text="1.2 s", bg=self.BG, fg=self.FG,
                                   font=("Segoe UI", 9, "bold"), width=5)
        self._pause_lbl.pack(side="left", padx=(4, 0))
        tk.Scale(row1, variable=self._pause_var, from_=0.3, to=3.0, resolution=0.1,
                 orient="horizontal", length=180, bg=self.BG, fg=self.FG,
                 troughcolor=self.SURFACE, highlightthickness=0, showvalue=False,
                 command=lambda _: self._pause_lbl.config(
                     text=f"{self._pause_var.get():.1f} s")
                 ).pack(side="left", padx=(0, 24))

        row2 = tk.Frame(panel, bg=self.BG)
        row2.pack(fill="x")

        tk.Checkbutton(row2, text="Auto sensitivity", variable=self._auto_energy,
                       bg=self.BG, fg=self.FG_DIM, selectcolor=self.SURFACE,
                       activebackground=self.BG, font=("Segoe UI", 9),
                       command=self._on_auto_toggle,
                       ).pack(side="left")

        tk.Label(row2, text="  Manual threshold:", bg=self.BG, fg=self.FG_DIM,
                 font=("Segoe UI", 9)).pack(side="left", padx=(8, 0))

        self._energy_lbl = tk.Label(row2, text="300", bg=self.BG, fg=self.FG_DIM,
                                    font=("Segoe UI", 9, "bold"), width=5)
        self._energy_lbl.pack(side="left", padx=(0, 4))

        self._energy_scale = tk.Scale(
            row2, variable=self._energy_var, from_=50, to=3000, resolution=50,
            orient="horizontal", length=180, bg=self.BG, fg=self.FG_DIM,
            troughcolor=self.SURFACE, highlightthickness=0, showvalue=False,
            state="disabled",
            command=lambda _: self._energy_lbl.config(text=str(self._energy_var.get())),
        )
        self._energy_scale.pack(side="left")

        tk.Frame(panel, bg=self.SURFACE, height=1).pack(fill="x", pady=(6, 0))

    def _on_auto_toggle(self) -> None:
        auto = self._auto_energy.get()
        self._energy_scale.config(state="disabled" if auto else "normal")
        self._energy_lbl.config(fg=self.FG_DIM if auto else self.FG)

    # ── Indicator ─────────────────────────────────────────────

    def _build_indicator(self) -> None:
        row = tk.Frame(self.root, bg=self.BG, pady=5)
        row.pack(fill="x", padx=16)
        self._ind_canvas = tk.Canvas(row, width=14, height=14,
                                     bg=self.BG, highlightthickness=0)
        self._ind_canvas.pack(side="left")
        self._dot = self._ind_canvas.create_oval(2, 2, 12, 12,
                                                 fill=_IDLE_COL, outline="")
        self._ind_label = tk.Label(row, text="Idle", bg=self.BG, fg=self.FG_DIM,
                                   font=("Segoe UI", 9))
        self._ind_label.pack(side="left", padx=(6, 0))
        # Volume meter: level bar + threshold marker
        self._vol_canvas = tk.Canvas(row, height=10, bg=self.SURFACE,
                                     highlightthickness=0)
        self._vol_canvas.pack(side="left", fill="x", expand=True, padx=(12, 0))

    def _draw_vol_meter(self, rms: float, thresh: float, has_speech: bool) -> None:
        c = self._vol_canvas
        w = c.winfo_width() or 200
        h = 10
        c.delete("all")
        # Scale: threshold sits at 40% of bar width
        scale = max(thresh * 2.5, 1.0)
        frac  = min(1.0, rms / scale)
        th_x  = int(w * thresh / scale)
        color = self.GREEN if has_speech else self.FG_DIM
        c.create_rectangle(0, 1, int(w * frac), h - 1, fill=color, outline="")
        c.create_line(th_x, 0, th_x, h, fill=self.RED, width=2)

    # ── Live preview (partial results) ───────────────────────

    def _build_live_preview(self) -> None:
        outer = tk.Frame(self.root, bg=self.BG)
        outer.pack(fill="x", padx=16, pady=(0, 6))

        tk.Label(outer, text="Live", bg=self.BG, fg=self.FG_DIM,
                 font=("Segoe UI", 9)).pack(anchor="w")

        self._live_text = tk.Text(outer, bg=self.SURFACE, fg=self.GREEN,
                                  font=("Segoe UI", 10, "italic"),
                                  height=2, relief="flat", wrap="word",
                                  padx=10, pady=6, state="disabled",
                                  insertbackground=self.FG)
        self._live_text.pack(fill="x")

    # ── Transcript ────────────────────────────────────────────

    def _build_transcript(self) -> None:
        outer = tk.Frame(self.root, bg=self.BG)
        outer.pack(fill="both", expand=True, padx=16, pady=(0, 6))

        tk.Label(outer, text="Transcript", bg=self.BG, fg=self.FG_DIM,
                 font=("Segoe UI", 9)).pack(anchor="w")

        box = tk.Frame(outer, bg=self.SURFACE)
        box.pack(fill="both", expand=True, pady=(4, 0))

        self.transcript = tk.Text(box, bg=self.SURFACE, fg=self.FG,
                                  font=("Segoe UI", 11), relief="flat",
                                  wrap="word", padx=12, pady=10,
                                  insertbackground=self.FG, state="disabled")
        sb = tk.Scrollbar(box, command=self.transcript.yview,
                          bg=self.SURFACE, troughcolor=self.BG)
        self.transcript.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.transcript.pack(side="left", fill="both", expand=True)

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self.root, bg=self.SURFACE, height=26)
        bar.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Load a model to begin.")
        tk.Label(bar, textvariable=self.status_var, bg=self.SURFACE, fg=self.FG_DIM,
                 font=("Segoe UI", 9), anchor="w", padx=12).pack(side="left")
        self._words_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self._words_var, bg=self.SURFACE, fg=self.FG_DIM,
                 font=("Segoe UI", 9), anchor="e", padx=12).pack(side="right")

    # ── Devices ───────────────────────────────────────────────

    def _populate_devices(self) -> None:
        """Scan for working microphones and repopulate the dropdown."""
        prev = self.device_var.get()
        mics = list_microphones(test_devices=True, include_system_audio=False)
        # Sort WASAPI devices first — they work better with USB/BT headsets
        mics.sort(key=lambda m: (
            0 if "wasapi" in m.host_api_name.lower() else 1,
            0 if m.category == "microphone"           else 1,
        ))
        menu = self.device_menu["menu"]
        menu.delete(0, "end")
        menu.add_command(label="System Default",
                         command=lambda: self.device_var.set("System Default"))
        labels = []
        for mic in mics:
            sr  = f" {int(mic.default_sample_rate // 1000)}kHz"
            api = (mic.host_api_name
                   .replace("Windows WASAPI",      "WASAPI")
                   .replace("Windows DirectSound", "DS")
                   .replace("Windows ",             ""))
            tag = f"  [{mic.category}]" if mic.category != "unknown" else ""
            label = f"{mic.name[:22]}{sr} [{api}]{tag}  [idx {mic.index}]"
            labels.append(label)
            menu.add_command(label=label,
                             command=lambda v=label: self.device_var.set(v))
        if prev in labels:
            self.device_var.set(prev)
        else:
            self.device_var.set("System Default")
        if not mics:
            self.status_var.set("No devices found — connect a mic and click \u21bb.")
        else:
            self.status_var.set(
                f"{len(mics)} device(s). "
                "Pick [WASAPI] for USB/BT headsets.")

    def _test_mic(self) -> None:
        """Record 3 seconds from the selected device, save to a WAV file,
        and report the peak RMS so the user can confirm audio is arriving."""
        import wave, tempfile, os
        dev_idx = self._get_device_index()
        self.status_var.set("Recording 3 s test clip — speak now...")
        self.root.update()

        def _do_record():
            pa = _pa_lib.PyAudio()
            try:
                stream, as_float, actual_ch, actual_rate = _open_input(
                    pa, dev_idx, 1, RATE)
                frames, rms_vals = [], []
                n_chunks = max(1, int(actual_rate / CHUNK * 3))
                for _ in range(n_chunks):
                    raw = stream.read(CHUNK, exception_on_overflow=False)
                    if as_float:  raw = _float32_to_int16(raw)
                    d = _to_mono(raw, actual_ch)
                    if actual_rate != RATE: d = _resample(d, actual_rate, RATE)
                    frames.append(d)
                    n = len(d) // 2
                    if n:
                        s = struct.unpack(f"{n}h", d)
                        rms_vals.append(math.sqrt(sum(v*v for v in s) / n))
                stream.stop_stream(); stream.close()
                peak_rms = max(rms_vals) if rms_vals else 0
                avg_rms  = sum(rms_vals) / len(rms_vals) if rms_vals else 0
                # Save WAV — use a plain path (not NamedTemporaryFile which
                # keeps a file handle open and causes PermissionError on unlink)
                import tempfile
                tmp_path = Path(tempfile.gettempdir()) / "stt_test_mic.wav"
                with wave.open(str(tmp_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(RATE)
                    wf.writeframes(b"".join(frames))
                msg = (f"Peak RMS: {peak_rms:.0f}  Avg RMS: {avg_rms:.0f}\n"
                       f"Rate opened: {actual_rate} Hz  channels: {actual_ch}")
                if peak_rms < 50:
                    msg += "\n\nWARNING: audio is nearly silent.\nCheck: Windows Sound > Input > mic volume/mute."
                elif peak_rms > 50:
                    msg += "\n\nAudio level looks good. If transcription still fails,\ntry selecting the [MME] device from the dropdown."
                self.root.after(0, lambda m=msg, p=str(tmp_path): (
                    self.status_var.set(f"Test done — peak RMS: {peak_rms:.0f}"),
                    messagebox.showinfo("Mic Test", m, parent=self.root),
                    os.unlink(p) if os.path.exists(p) else None,
                ))
            except Exception as exc:
                self.root.after(0, lambda e=exc: (
                    self.status_var.set(f"Test failed: {e}"),
                    messagebox.showerror("Mic Test Failed", str(e), parent=self.root),
                ))
            finally:
                pa.terminate()

        threading.Thread(target=_do_record, daemon=True).start()

    def _on_sys_audio_toggle(self) -> None:
        if self._sys_audio_var.get():
            dev = get_best_system_audio_device()
            if dev is None:
                self._sys_audio_var.set(False)
                messagebox.showinfo(
                    "System Audio Not Available",
                    "No loopback device found for the current output.\n\n"
                    "Best fix: pip install pyaudiowpatch\n"
                    "(makes any headset/speaker capturable automatically)\n\n"
                    "Without it: enable 'Stereo Mix' in Sound \u2192 Recording tab.",
                    parent=self.root,
                )
                return
            self._sys_audio_dev = dev
            self.device_menu.config(state="disabled")
            self._mix_btn.pack(side="left", padx=(4, 0))
            self.status_var.set(
                f"System audio: {dev.name}  ({dev.default_sample_rate:.0f} Hz)"
            )
        else:
            self._sys_audio_dev = None
            self._mix_mic_var.set(False)
            self._mix_btn.pack_forget()
            if not self._running:
                self.device_menu.config(state="normal")
            self.status_var.set("Ready.")

    def _get_device_index(self) -> int | None:
        val = self.device_var.get()
        if val == "System Default":
            # Return None — PyAudio uses the MME default device which routes
            # through Windows Audio Engine for automatic sample-rate conversion.
            return None
        return int(val.rsplit("idx ", 1)[-1].rstrip("]"))

    # ── Vosk / model checks ───────────────────────────────────

    def _check_vosk_installed(self) -> None:
        if not VOSK_AVAILABLE:
            messagebox.showerror(
                "vosk not installed",
                "Run:  pip install vosk\n\nthen restart this script.",
                parent=self.root,
            )
            return
        # Auto-load the first model that is already downloaded
        for key, info in VOSK_MODELS.items():
            model_dir = MODELS_DIR / info["id"]
            if model_dir.exists() and (model_dir / "am" / "final.mdl").exists():
                self._model_key.set(key)
                self._on_model_select(key)
                self._try_load_model(str(model_dir), silent=True)
                return
        self.status_var.set("No model loaded — select a model and click Download.")

    def _on_model_select(self, key: str | None = None) -> None:
        """Refresh Load/Download button labels when a model is selected."""
        if key is None:
            key = self._model_key.get()
        info      = VOSK_MODELS.get(key, {})
        model_dir = MODELS_DIR / info.get("id", "")
        downloaded = (model_dir.exists() and
                      (model_dir / "am" / "final.mdl").exists())
        self._load_btn.config(
            text="Load  \u2713" if downloaded else "Load",
            bg="#3d7a4f"    if downloaded else self.SURFACE,
            fg="#fff"       if downloaded else self.FG,
        )
        self._dl_btn.config(
            text=f"Re-download  {info.get('size','')}" if downloaded
                 else f"Download  {info.get('size','')}"
        )
        self._model_note.config(text=info.get("note", ""))

    def _load_selected_model(self) -> None:
        key  = self._model_key.get()
        info = VOSK_MODELS[key]
        self._try_load_model(str(MODELS_DIR / info["id"]))

    def _try_load_model(self, path: str, silent: bool = False) -> bool:
        p = Path(path)
        if not p.exists() or not (p / "am" / "final.mdl").exists():
            if not silent:
                messagebox.showwarning(
                    "Model not found",
                    f"No Vosk model found at:\n{path}\n\n"
                    "Select a model from the dropdown and click Download.",
                    parent=self.root,
                )
            self.status_var.set("No model loaded — select a model and click Download.")
            return False
        try:
            self.status_var.set(f"Loading model from {p.name} ...")
            self.root.update()
            self._model = Model(str(p))
            self.toggle_btn.config(state="normal")
            self.status_var.set(f"Model ready: {p.name}")
            self._on_model_select()  # refresh button colours
            return True
        except Exception as exc:
            messagebox.showerror("Model error", str(exc), parent=self.root)
            return False

    def _browse_model(self) -> None:
        d = filedialog.askdirectory(title="Select Vosk model folder",
                                    initialdir=str(MODELS_DIR))
        if d:
            self._try_load_model(d)

    # ── Model download ────────────────────────────────────────

    def _start_download(self) -> None:
        if not VOSK_AVAILABLE:
            messagebox.showerror("vosk not installed",
                                 "Run:  pip install vosk  first.",
                                 parent=self.root)
            return
        key      = self._model_key.get()
        info     = VOSK_MODELS[key]
        model_id = info["id"]
        zip_path = MODELS_DIR / (model_id + ".zip")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        self._dl_btn.config(state="disabled", text="Downloading...")
        self._load_btn.config(state="disabled")
        self._dl_frame.pack(fill="x", after=self._dl_btn.master)

        threading.Thread(
            target=self._download_worker,
            args=(zip_path, info["url"], model_id),
            daemon=True,
        ).start()

    def _download_worker(self, zip_path: Path, url: str, model_id: str) -> None:
        try:
            self.root.after(0, lambda: self._dl_label.config(
                text=f"Downloading {model_id}.zip ..."))
            with urllib.request.urlopen(url) as resp:
                total   = int(resp.headers.get("Content-Length", 0))
                done    = 0
                buf     = bytearray()
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    buf  += chunk
                    done += len(chunk)
                    if total:
                        pct = done / total
                        self.root.after(0, lambda p=pct: self._update_dl_bar(p))
            zip_path.write_bytes(buf)

            self.root.after(0, lambda: self._dl_label.config(text="Extracting..."))
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(MODELS_DIR)
            zip_path.unlink()

            self.root.after(0, lambda mid=model_id: self._on_download_done(mid))

        except Exception as exc:
            self.root.after(0, lambda e=exc: (
                messagebox.showerror("Download failed", str(e), parent=self.root),
                self._reset_dl_ui(),
            ))

    def _update_dl_bar(self, fraction: float) -> None:
        w = self._dl_canvas.winfo_width() or 600
        self._dl_canvas.delete("all")
        self._dl_canvas.create_rectangle(0, 0, w, 6, fill=self.SURFACE, outline="")
        self._dl_canvas.create_rectangle(0, 0, int(w * fraction), 6,
                                          fill=self.ACCENT, outline="")
        self._dl_label.config(text=f"Downloading...  {int(fraction * 100)}%")

    def _on_download_done(self, model_id: str) -> None:
        self._reset_dl_ui()
        self._on_model_select()   # refresh Load/Download button labels
        self._try_load_model(str(MODELS_DIR / model_id))

    def _reset_dl_ui(self) -> None:
        self._dl_btn.config(state="normal")
        self._load_btn.config(state="normal")
        self._on_model_select()   # restore correct button text
        self._dl_frame.pack_forget()

    # ── Start / Stop ─────────────────────────────────────────

    def _toggle(self) -> None:
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        if self._model is None:
            messagebox.showwarning("No model", "Load a model first.", parent=self.root)
            return
        # If system audio is on, re-scan for the loopback device now.
        # The output device may have changed since the checkbox was ticked
        # (e.g. headset plugged in) — always use the freshest device.
        if self._sys_audio_var.get():
            dev = get_best_system_audio_device()
            if dev is None:
                messagebox.showwarning(
                    "System Audio Lost",
                    "The system-audio loopback device is no longer available.\n"
                    "Your audio output device may have changed.\n\n"
                    "Disable '\u266a System Audio', reconnect your device, then\n"
                    "re-enable it to pick up the new loopback.\n\n"
                    "For best results: pip install pyaudiowpatch",
                    parent=self.root,
                )
                return
            self._sys_audio_dev = dev
            self.status_var.set(
                f"System audio: {dev.name}  ({dev.default_sample_rate:.0f} Hz)"
            )
        # Kill any previous session before starting a new one.
        # Without this, the old thread wakes up when _running goes True again
        # and fights the new thread for the same audio device → crash → auto-stop.
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        time.sleep(0.05)          # let the OS fully release audio device handles
        self._session_id += 1
        self._running = True
        self.toggle_btn.config(text="\u25a0  Stop", bg=self.RED)
        self.device_menu.config(state="disabled")
        self._sys_btn.config(state="disabled")
        self._set_indicator("pulse", _PULSE_LISTEN, "Listening")
        if self._sys_audio_dev:
            self.status_var.set(
                f"Capturing system audio: {self._sys_audio_dev.name}"
            )
        else:
            self.status_var.set("Listening \u2014 speak whenever you're ready")
        self._thread = threading.Thread(
            target=self._listen_loop,
            args=(self._sys_audio_dev, self._mix_mic_var.get(), self._session_id),
            daemon=True
        )
        self._thread.start()

    def _stop(self) -> None:
        self._running = False
        self.toggle_btn.config(text="\u25b6  Start", bg=self.ACCENT)
        self.device_menu.config(state="normal")
        self._sys_btn.config(state="normal")
        self._set_indicator("idle")
        self._set_live("")
        self.status_var.set("Stopped.")

    # ── Core listen loop ──────────────────────────────────────

    def _listen_loop(self, sys_dev: MicrophoneInfo | None = None,
                     mix_mic: bool = False,
                     session_id: int = 0) -> None:
        """
        Feeds raw 16 kHz mono PCM into Vosk.  session_id is checked on every
        iteration so that a stale thread from a previous Start/Stop cycle exits
        immediately instead of fighting with the new capture thread.
        """
        pa = _pa_lib.PyAudio()
        rec = KaldiRecognizer(self._model, RATE)
        rec.SetWords(True)
        is_speaking = False; silence_start = None
        current_partial = ""; auto_threshold = self._energy_var.get()

        sys_ch = sys_rate = None
        if sys_dev:
            sys_ch   = min(sys_dev.max_input_channels, 2)
            sys_rate = int(sys_dev.default_sample_rate)
        mic_idx = self._get_device_index()
        _rms_tick   = [0]   # mutable counter for periodic RMS display
        _zero_since = [0.0] # track when sustained silence started (for warning)

        # ── Vosk processing (shared by all modes) ─────────────────────────
        def _feed(data: bytes) -> None:
            nonlocal is_speaking, silence_start, current_partial, rec, auto_threshold
            n = len(data) // 2
            if n == 0:
                return
            samps = struct.unpack(f"{n}h", data)
            rms = math.sqrt(sum(s * s for s in samps) / n)
            thresh = (auto_threshold if self._auto_energy.get()
                      else self._energy_var.get())
            has_speech = rms > thresh

            # Show live RMS in indicator every ~5 chunks ≈ 0.6 s
            _rms_tick[0] = (_rms_tick[0] + 1) % 5
            if _rms_tick[0] == 0:
                lbl = (f"SPEECH  rms:{int(rms)}" if has_speech
                       else f"Listening  {int(rms)}/{int(thresh)}")
                self.root.after(0, lambda l=lbl: self._ind_label.config(text=l))
                self.root.after(0, lambda r=rms, t=thresh, h=has_speech:
                                self._draw_vol_meter(r, t, h))

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result()).get("text", "").strip()
                if result:
                    ts = datetime.now().strftime("%H:%M:%S")
                    self.root.after(0, lambda t=result, s=ts: self._commit(t, s))
                is_speaking = False; silence_start = None; current_partial = ""
                self.root.after(0, lambda: self._set_live(""))
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "").strip()
                if partial != current_partial:
                    current_partial = partial
                    self.root.after(0, lambda p=partial: self._set_live(p))

            if has_speech:
                is_speaking = True; silence_start = None
                self.root.after(0, lambda: self._set_indicator(
                    "pulse", _PULSE_LISTEN, "Listening"))
            elif is_speaking:
                if silence_start is None:
                    silence_start = time.monotonic()
                elif (time.monotonic() - silence_start) >= self._pause_var.get():
                    final = json.loads(rec.FinalResult()).get("text", "").strip()
                    if final:
                        ts = datetime.now().strftime("%H:%M:%S")
                        self.root.after(0, lambda t=final, s=ts: self._commit(t, s))
                    rec = KaldiRecognizer(self._model, RATE)
                    rec.SetWords(True)
                    is_speaking = False; silence_start = None; current_partial = ""
                    self.root.after(0, lambda: self._set_live(""))

        try:
            if sys_dev and mix_mic:
                # ── Mixed mode: mic + system audio blended ────────────────
                SILENCE = bytes(CHUNK * 2)
                mic_q = _queue.Queue(maxsize=30)
                sys_q = _queue.Queue(maxsize=30)

                def _cap_mic() -> None:
                    pa_m = _pa_lib.PyAudio()
                    try:
                        s, as_float, mic_ch, mic_rate = _open_input(pa_m, mic_idx, 1, RATE)
                        buf = bytearray()
                        TARGET = CHUNK * 2
                        while self._running and self._session_id == session_id:
                            raw = s.read(CHUNK, exception_on_overflow=False)
                            d = _float32_to_int16(raw) if as_float else raw
                            if mic_ch   > 1:    d = _to_mono(d, mic_ch)
                            if mic_rate != RATE: d = _resample(d, mic_rate, RATE)
                            buf.extend(d)
                            while len(buf) >= TARGET:
                                if not mic_q.full(): mic_q.put(bytes(buf[:TARGET]))
                                del buf[:TARGET]
                        s.stop_stream(); s.close()
                    except Exception as e:
                        self.root.after(0, lambda err=e: self.status_var.set(
                            f"Mic capture failed: {err}"))
                    finally:
                        pa_m.terminate()

                def _cap_sys() -> None:
                    pa_s = _pa_lib.PyAudio()
                    try:
                        s, as_float, actual_ch, actual_rate = _open_input(
                            pa_s, sys_dev.index, sys_ch, sys_rate)
                        buf = bytearray()
                        TARGET = CHUNK * 2
                        while self._running and self._session_id == session_id:
                            raw = s.read(CHUNK, exception_on_overflow=False)
                            if as_float: raw = _float32_to_int16(raw)
                            d = _to_mono(raw, actual_ch)
                            if actual_rate != RATE: d = _resample(d, actual_rate, RATE)
                            buf.extend(d)
                            while len(buf) >= TARGET:
                                if not sys_q.full(): sys_q.put(bytes(buf[:TARGET]))
                                del buf[:TARGET]
                        s.stop_stream(); s.close()
                    except Exception: pass
                    finally:
                        pa_s.terminate()

                threading.Thread(target=_cap_mic, daemon=True).start()
                threading.Thread(target=_cap_sys, daemon=True).start()

                while self._running and self._session_id == session_id:
                    try: mic_d = mic_q.get(timeout=0.5)
                    except _queue.Empty: continue
                    sys_d = sys_q.get_nowait() if not sys_q.empty() else SILENCE
                    _feed(_mix_pcm(mic_d, sys_d))

            else:
                # ── Single stream mode ────────────────────────────────────
                if sys_dev:
                    dev_idx, dev_ch, dev_rate = sys_dev.index, sys_ch, sys_rate
                else:
                    dev_idx, dev_ch, dev_rate = mic_idx, CHANNELS, RATE

                # _open_input tries int16 then float32 then driver-chosen buffer —
                # this handles regular mics AND WASAPI/loopback devices uniformly.
                stream, _use_float, dev_ch, dev_rate = _open_input(pa, dev_idx, dev_ch, dev_rate)

                fmt_name = "float32→int16" if _use_float else "int16"
                note = f" → resample to 16kHz" if dev_rate != RATE else ""
                self.root.after(0, lambda r=dev_rate, c=dev_ch, f=fmt_name, n=note:
                    self.status_var.set(
                        f"Stream: {r} Hz/{c}ch/{f}{n} — calibrating..."))

                # Warm-up calibration (mic only).
                # Apply the SAME conversions as the main loop so that RMS values
                # are comparable and the threshold is set correctly even when the
                # device uses float32 or a non-16-kHz native rate.
                if not sys_dev and self._auto_energy.get():
                    rms_samples = []
                    for _ in range(max(1, int(RATE / CHUNK))):
                        raw = stream.read(CHUNK, exception_on_overflow=False)
                        if _use_float: raw = _float32_to_int16(raw)
                        raw = _to_mono(raw, dev_ch)
                        if dev_rate != RATE: raw = _resample(raw, dev_rate, RATE)
                        n = len(raw) // 2
                        if n:
                            s = struct.unpack(f"{n}h", raw)
                            rms_samples.append(
                                math.sqrt(sum(v * v for v in s) / n))
                        rec.AcceptWaveform(raw)
                    if rms_samples:
                        avg = sum(rms_samples) / len(rms_samples)
                        # Use 1.1× ambient (was 1.5×) and lower the minimum
                        # so headsets with quiet output still trigger recognition
                        auto_threshold = max(30, int(avg * 1.1))
                    self.root.after(0, lambda thr=auto_threshold:
                        self.status_var.set(
                            f"Calibrated — threshold: {thr:.0f}  "
                            f"(bar must exceed red line to trigger Vosk)"))

                while self._running and self._session_id == session_id:
                    raw  = stream.read(CHUNK, exception_on_overflow=False)
                    if _use_float: raw = _float32_to_int16(raw)
                    data = _to_mono(raw, dev_ch)
                    if dev_rate != RATE: data = _resample(data, dev_rate, RATE)
                    _feed(data)

                stream.stop_stream()
                stream.close()

        except Exception as exc:
            if self._running:
                self.root.after(0, lambda e=exc: (
                    messagebox.showerror("Error", str(e), parent=self.root),
                    self._stop(),
                ))
        finally:
            pa.terminate()

    # ── UI helpers ────────────────────────────────────────────

    def _set_live(self, text: str) -> None:
        self._live_text.config(state="normal")
        self._live_text.delete("1.0", "end")
        if text:
            self._live_text.insert("end", text)
        self._live_text.config(state="disabled")

    def _commit(self, text: str, ts: str) -> None:
        self.transcript.config(state="normal")
        if self.transcript.index("end-1c") != "1.0":
            self.transcript.insert("end", "\n")
        self.transcript.insert("end", f"[{ts}]  {text}")
        self.transcript.see("end")
        self.transcript.config(state="disabled")
        self._set_live("")
        wc = len(self.transcript.get("1.0", "end-1c").split())
        self._words_var.set(f"{wc} word{'s' if wc != 1 else ''}")

    def _copy(self) -> None:
        content = self.transcript.get("1.0", "end-1c")
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.status_var.set("Transcript copied to clipboard.")

    def _clear(self) -> None:
        self.transcript.config(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.config(state="disabled")
        self._words_var.set("")
        self.status_var.set("Transcript cleared.")

    # ── Pulse indicator ───────────────────────────────────────

    def _set_indicator(self, mode: str,
                       colors: tuple[str, str] | None = None,
                       label: str = "") -> None:
        if self._pulse_after:
            self.root.after_cancel(self._pulse_after)
            self._pulse_after = None
        if mode == "idle":
            self._ind_canvas.itemconfig(self._dot, fill=_IDLE_COL)
            self._ind_label.config(text="Idle", fg=self.FG_DIM)
        elif mode == "pulse" and colors:
            self._pulse_colors = colors
            self._pulse_phase  = 0
            self._ind_label.config(text=label, fg=self.FG)
            self._animate_pulse()

    def _animate_pulse(self) -> None:
        if not self._running:
            return
        col = self._pulse_colors[self._pulse_phase % 2]
        self._ind_canvas.itemconfig(self._dot, fill=col)
        self._pulse_phase += 1
        self._pulse_after = self.root.after(600, self._animate_pulse)

    # ── Close ─────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    OfflineSTTWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
