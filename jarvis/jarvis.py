"""
Jarvis — AI Personal Assistant

Combines:
  - Live camera feed with face recognition (InsightFace, recognition/embeddings.json)
  - Facial emotion detection (MediaPipe FaceLandmarker blendshapes)
  - Offline speech-to-text (Vosk, audio/models/)
  - LLM conversation (OpenBridge AI — reads llm/.llm_config.json)

Features:
  - Greets recognised people by name with time-appropriate greeting
  - Comments on detected emotion
  - Microphone + optional system audio capture
  - Offline fallback when LLM is not configured or unreachable

Run:
    python jarvis/jarvis.py
"""

import json
import math
import os
import queue
import random
import re
import struct
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

# ── Project root paths ────────────────────────────────────────────────────────
_ROOT        = Path(__file__).resolve().parent.parent
_EMBED_FILE  = _ROOT / "recognition" / "embeddings.json"
_REC_MODELS  = _ROOT / "recognition" / "models"
_MP_MODELS   = _ROOT / "mediapipe"   / "models"
_AUDIO_MODELS= _ROOT / "audio"       / "models"
_LLM_CONFIG  = _ROOT / "llm"         / ".llm_config.json"

sys.path.insert(0, str(_ROOT))
from utilities.microphone_detector import list_microphones, get_best_system_audio_device

# ── Optional heavy dependencies ───────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

try:
    import pyaudiowpatch as _pa_lib
except ImportError:
    import pyaudio as _pa_lib          # type: ignore[no-redef]

import pyaudio

try:
    import pyttsx3 as _pyttsx3
    TTS_OK = True
except ImportError:
    TTS_OK = False

try:
    from vosk import KaldiRecognizer, Model, SetLogLevel
    SetLogLevel(-1)
    VOSK_OK = True
except ImportError:
    VOSK_OK = False

try:
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        from insightface.app import FaceAnalysis as _FaceAnalysis
    FACE_OK = True
except ImportError:
    FACE_OK = False

try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    MP_OK = True
except ImportError:
    MP_OK = False

import urllib.request
import urllib.error

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK          = 2048
RATE           = 16000
DETECT_EVERY   = 15          # run face/emotion detection every N camera frames
FACE_THRESHOLD = 0.45        # cosine distance — lower = stricter

# ── Colour palette ────────────────────────────────────────────────────────────
BG      = "#1e1e2e"
SURFACE = "#2a2a3e"
ACCENT  = "#7c6af7"
FG      = "#cdd6f4"
FG_DIM  = "#6c7086"
RED     = "#f38ba8"
GREEN   = "#a6e3a1"
YELLOW  = "#f9e2af"


# ─────────────────────────────────────────────────────────────────────────────
# TTS — spawn a fresh Python process per utterance (fixes pyttsx3 singleton bug
#         where engine.runAndWait() silently breaks after the first call on Windows)
# ─────────────────────────────────────────────────────────────────────────────

# Persistent TTS subprocess: Python starts once, but pyttsx3 engine is refreshed
# per utterance by clearing its internal singleton cache before each init() call.
# This avoids both the ~2 s Python startup lag AND the SAPI5 reuse breakage.
_PERSISTENT_TTS_SCRIPT = r"""
import sys, json, pyttsx3

sys.stdout.write('ready\n')
sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
        if msg.get('cmd') == 'quit':
            break
        text = msg.get('text', '')
        if text:
            # Clear pyttsx3 singleton cache: fresh engine every utterance
            # (no Python startup cost since the subprocess stays alive)
            try:
                pyttsx3._activeEngines.clear()
            except Exception:
                pass
            e = pyttsx3.init()
            e.setProperty('rate', 165)
            e.setProperty('volume', 0.95)
            e.say(text)
            e.runAndWait()
    except Exception:
        pass
    sys.stdout.write('ok\n')
    sys.stdout.flush()
"""


class _TTSWorker(threading.Thread):
    """Persistent TTS subprocess — pyttsx3 initialised once, no per-utterance startup lag."""

    def __init__(self, on_start=None, on_done=None) -> None:
        super().__init__(daemon=True)
        self._q        = queue.Queue()
        self._on_start = on_start
        self._on_done  = on_done
        self._proc: subprocess.Popen | None = None

    def _launch_engine(self) -> bool:
        """Start (or restart) the persistent subprocess. Returns True when ready."""
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-c", _PERSISTENT_TTS_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            reply = self._proc.stdout.readline().decode().strip()
            return reply == "ready"
        except Exception:
            return False

    def run(self) -> None:
        if not TTS_OK:
            return
        self._launch_engine()    # warm up once at start — first speak() has no lag
        while True:
            text = self._q.get()
            if text is None:
                self._quit_engine()
                break
            try:
                # Restart if subprocess died (e.g. after stop())
                if self._proc is None or self._proc.poll() is not None:
                    if not self._launch_engine():
                        continue
                self._proc.stdin.write((json.dumps({"text": text}) + "\n").encode())
                self._proc.stdin.flush()
                self._proc.stdout.readline()    # block until subprocess writes "ok"
            except Exception:
                try:
                    self._launch_engine()
                except Exception:
                    pass
            finally:
                if self._on_done:
                    self._on_done()

    def _quit_engine(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.write(b'{"cmd":"quit"}\n')
                self._proc.stdin.flush()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.terminate()
                except Exception:
                    pass

    def speak(self, text: str) -> None:
        """Set SPEAKING state immediately, then queue the utterance."""
        if self._on_start:
            self._on_start()   # mutes STT before subprocess receives the text
        self._q.put(text)

    def stop(self) -> None:
        """Interrupt current speech — subprocess restarts automatically on next speak()."""
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def shutdown(self) -> None:
        self._q.put(None)




def time_greeting() -> str:
    h = datetime.now().hour
    if 5  <= h < 12: return "Good morning"
    if 12 <= h < 17: return "Good afternoon"
    if 17 <= h < 21: return "Good evening"
    return "Good night"


# Words that are meaningless on their own (Vosk sometimes commits a single
# filler word mid-sentence when the user pauses briefly)
_STT_STOPWORDS = frozenset({
    "the", "a", "an", "in", "and", "or", "but", "to", "of", "at",
    "i", "it", "is", "be", "as", "for", "on", "are", "by", "with",
    "he", "she", "they", "we", "you", "do", "did", "not", "no",
    "up", "so", "if", "was", "that", "this", "just", "ok", "okay",
})


def _is_meaningful(text: str) -> bool:
    """Return True only if the utterance has enough content to send to the LLM."""
    words = [w for w in text.lower().split() if w.isalpha()]
    if not words:
        return False
    if len(words) == 1:
        # Single word: accept only if it’s not a stopword AND long enough
        return words[0] not in _STT_STOPWORDS and len(words[0]) > 3
    if len(words) == 2:
        # Two words: require at least one substantive (non-stop) word
        substantive = [w for w in words if w not in _STT_STOPWORDS]
        return bool(substantive)
    # Three or more words — always worth sending
    return True


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def classify_emotion_blendshapes(blendshapes) -> str:
    """Same logic as mediapipe/face_emotion_detection.py."""
    bs = {b.category_name: b.score for b in blendshapes}
    smile   = (bs.get("mouthSmileLeft",  0) + bs.get("mouthSmileRight", 0)) * .5
    frown   = (bs.get("mouthFrownLeft",  0) + bs.get("mouthFrownRight", 0)) * .5
    brow_up =  bs.get("browInnerUp",     0)
    brow_dn = (bs.get("browDownLeft",    0) + bs.get("browDownRight",   0)) * .5
    jaw     =  bs.get("jawOpen",         0)
    sneer   = (bs.get("noseSneerLeft",   0) + bs.get("noseSneerRight",  0)) * .5
    if brow_up > 0.4  and jaw   > 0.35: return "Surprised"
    if smile   > 0.35:                  return "Happy"
    if brow_dn > 0.25 and sneer > 0.10: return "Angry"
    if frown   > 0.15 or (brow_up > 0.25 and smile < 0.1): return "Sad"
    return "Neutral"


def load_llm_config() -> dict:
    if _LLM_CONFIG.exists():
        try:
            return json.loads(_LLM_CONFIG.read_text())
        except Exception:
            pass
    return {"provider": "ollama"}


def _is_llm_configured(cfg: dict) -> bool:
    """Return False when no API key is set."""
    if cfg.get("provider", "") == "none":
        return False
    return bool(cfg.get("api_key", "").strip())


def _local_respond(user_text: str = "", ctx: dict | None = None,
                   notified: bool = False) -> str:
    """Minimal fallback when no LLM is connected."""
    t   = user_text.lower().strip()
    now = datetime.now()
    if re.search(r'\b(time|clock)\b', t):
        return f"The current time is {now.strftime('%I:%M %p')}."
    if re.search(r'\b(date|today)\b', t):
        return f"Today is {now.strftime('%A, %B %d, %Y')}."
    if notified:
        return "I'm in offline mode — no LLM is connected."
    return ("No LLM connected. Start Ollama, add a Groq API key, "
            "or configure Phi Silica / Edge AI.")


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers (same as live_speech_to_text_offline_ui.py)
# ─────────────────────────────────────────────────────────────────────────────

def _float32_to_int16(data: bytes) -> bytes:
    arr = np.frombuffer(data, dtype=np.float32)
    return (arr * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


def _to_mono(data: bytes, channels: int) -> bytes:
    if channels == 1:
        return data
    arr = np.frombuffer(data, dtype=np.int16).reshape(-1, channels)
    return np.ascontiguousarray(arr[:, 0]).tobytes()


def _resample(data: bytes, in_rate: int, out_rate: int) -> bytes:
    if in_rate == out_rate or not data:
        return data
    arr   = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    ratio = in_rate / out_rate
    n     = round(ratio)
    if abs(ratio - n) < 0.05 and n > 1:
        pad = (-len(arr)) % n
        if pad:
            arr = np.concatenate([arr, np.zeros(pad, dtype=np.float32)])
        return arr.reshape(-1, n).mean(axis=1).clip(-32768, 32767).astype(np.int16).tobytes()
    out_len = max(1, int(len(arr) * out_rate / in_rate))
    return np.interp(np.linspace(0, 1, out_len),
                     np.linspace(0, 1, len(arr)), arr
                     ).astype(np.int16).tobytes()


def _find_mme_device(pa, device_index):
    if device_index is None:
        return None
    try:
        name = pa.get_device_info_by_index(device_index).get("name", "")
        for h in range(pa.get_host_api_count()):
            if "MME" not in pa.get_host_api_info_by_index(h).get("name", ""):
                continue
            for i in range(pa.get_device_count()):
                dev = pa.get_device_info_by_index(i)
                if (int(dev.get("hostApi", -1)) == h
                        and dev.get("name", "") == name
                        and int(dev.get("maxInputChannels", 0)) > 0):
                    return i
    except Exception:
        pass
    return None


def _open_input(pa, device_index, channels: int, rate: int):
    mme = _find_mme_device(pa, device_index)
    native = rate
    if device_index is not None:
        try:
            native = int(pa.get_device_info_by_index(device_index)
                         .get("defaultSampleRate", rate))
        except Exception:
            pass
    ch_list  = list(dict.fromkeys([channels, 2] if channels == 1 else [channels]))
    fmt_list = [(pyaudio.paInt16, False, CHUNK),
                (pyaudio.paFloat32, True,  CHUNK),
                (pyaudio.paFloat32, True,  0),
                (pyaudio.paInt16,   False, 0)]
    last_err = None
    if mme is not None:
        for ch in ch_list:
            for fmt, as_float, buf in fmt_list:
                try:
                    kw = dict(format=fmt, channels=ch, rate=rate,
                              input=True, input_device_index=mme)
                    if buf > 0: kw["frames_per_buffer"] = buf
                    return pa.open(**kw), as_float, ch, rate
                except Exception as e:
                    last_err = e
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


# Same friendly names as the audio STT UI so the model list matches
_VOSK_FRIENDLY = {
    "vosk-model-small-en-us-0.15":          "Small EN   ~40 MB   (fast)",
    "vosk-model-en-us-0.22-lgraph":         "Medium EN  ~128 MB  (recommended)",
    "vosk-model-en-us-0.22":                "Large EN   ~1.8 GB  (best accuracy)",
    "vosk-model-en-us-0.42-gigaspeech":     "GigaSpeech ~2.3 GB  (meetings/video)",
    "vosk-model-small-en-in-0.4":          "Indian EN  ~36 MB   (IN accent)",
    "vosk-model-en-in-0.5":                "Indian EN  ~1 GB    (IN accent, HQ)",
}



class JarvisWindow:

    def __init__(self, root: tk.Tk) -> None:
        self.root = root

        # State
        self._running       = False
        self._session_id    = 0
        self._cam_thread    = None
        self._audio_thread  = None
        self._greeted       = set()         # names already greeted this session
        self._last_emotion  = {}            # name → last emotion comment sent
        self._messages: list[dict] = []     # LLM conversation history
        self._current_context = {}          # {name, emotion, confidence}
        self._llm_unreachable = False       # set True after first connection failure
        self._offline_notified = False       # set True after first offline message shown
        self._llm_busy         = False       # True while an LLM response is in flight
        self._pending_llm      = False       # user spoke while LLM was busy
        self._stt_buffer       = ""          # accumulates text across AcceptWaveform resets
        self._detect_q         = queue.Queue(maxsize=1)  # camera → detect thread
        self._last_detections: list[dict] = []           # drawn by cam loop

        # Face recognition
        self._known: dict[str, np.ndarray] = {}
        self._face_app = None
        self._mp_landmarker = None

        # Vosk model
        self._vosk_model = None

        # Audio settings
        self._mic_var     = tk.StringVar(value="System Default")
        self._sys_var     = tk.BooleanVar(value=False)
        self._mix_var     = tk.BooleanVar(value=False)
        self._sys_dev     = None
        self._speaking    = False   # True while TTS is playing (mutes STT)

        # TTS
        self._tts_enabled = tk.BooleanVar(value=True)
        self._tts = _TTSWorker(
            on_start=lambda: self._set_tts_speaking(True),
            on_done =lambda: self._set_tts_speaking(False),
        )
        if TTS_OK:
            self._tts.start()

        # Vosk model selector
        self._vosk_model_var = tk.StringVar()
        self._vosk_options: list[tuple[str, str, Path]] = []

        # LLM
        self._llm_cfg     = load_llm_config()

        self._build_ui()
        self._load_models()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.title("Jarvis")
        self.root.geometry("1080x700")
        self.root.minsize(860, 560)
        self.root.configure(bg=BG)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"1080x700+{(sw-1080)//2}+{(sh-700)//2}")

        # Header
        hdr = tk.Frame(self.root, bg=SURFACE)
        hdr.pack(fill="x")
        self._status_dot = tk.Label(hdr, text="●", bg=SURFACE, fg=RED,
                                    font=("Segoe UI", 12))
        self._status_dot.pack(side="left", padx=(12, 4), pady=8)
        tk.Label(hdr, text="JARVIS", bg=SURFACE, fg=ACCENT,
                 font=("Segoe UI", 14, "bold")).pack(side="left")
        provider_name = self._llm_cfg.get("provider", "ollama").upper()
        self._llm_lbl = tk.Label(hdr, text=f"  LLM: {provider_name}",
                                 bg=SURFACE, fg=FG_DIM, font=("Segoe UI", 9))
        self._llm_lbl.pack(side="left", padx=8)

        self._start_btn = tk.Button(
            hdr, text="▶  Start", command=self._toggle,
            bg=ACCENT, fg="#fff", font=("Segoe UI", 10, "bold"),
            relief="flat", padx=16, pady=4, cursor="hand2")
        self._start_btn.pack(side="right", padx=4)
        self._tts_btn = tk.Checkbutton(
            hdr, text="🔊", variable=self._tts_enabled,
            bg=SURFACE, fg=GREEN, selectcolor=SURFACE,
            activebackground=SURFACE, font=("Segoe UI", 13), cursor="hand2")
        self._tts_btn.pack(side="right", padx=(0, 4))
        if not TTS_OK:
            self._tts_btn.config(state="disabled", fg=FG_DIM)

        # Main area — use grid so camera stays fixed and right panel fills rest
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=8, pady=(4, 0))
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Left: camera — fixed 400×300 container that never collapses
        left = tk.Frame(main, bg=BG)
        left.grid(row=0, column=0, sticky="n", padx=(0, 8), pady=4)
        tk.Label(left, text="CAMERA", bg=BG, fg=FG_DIM,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        cam_box = tk.Frame(left, bg="#000", width=400, height=300)
        cam_box.pack()
        cam_box.pack_propagate(False)   # stays 400×300 regardless of image
        self._cam_label = tk.Label(cam_box, bg="#000", text="Not started",
                                   fg=FG_DIM, font=("Segoe UI", 10))
        self._cam_label.pack(expand=True)

        # Detection status panel — below camera, both run in parallel
        det_panel = tk.Frame(left, bg=SURFACE)
        det_panel.pack(fill="x", pady=(4, 0))
        face_side = tk.Frame(det_panel, bg=SURFACE)
        face_side.pack(side="left", fill="both", expand=True, padx=(10, 4), pady=6)
        tk.Label(face_side, text="FACE RECOGNITION", bg=SURFACE, fg=FG_DIM,
                 font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self._det_face_lbl = tk.Label(face_side, text="—", bg=SURFACE, fg=FG_DIM,
                                      font=("Segoe UI", 11, "bold"))
        self._det_face_lbl.pack(anchor="w")
        tk.Label(det_panel, text="│", bg=SURFACE, fg="#313244",
                 font=("Segoe UI", 18)).pack(side="left")
        emot_side = tk.Frame(det_panel, bg=SURFACE)
        emot_side.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=6)
        tk.Label(emot_side, text="EMOTION", bg=SURFACE, fg=FG_DIM,
                 font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self._det_emot_lbl = tk.Label(emot_side, text="—", bg=SURFACE, fg=FG_DIM,
                                      font=("Segoe UI", 11, "bold"))
        self._det_emot_lbl.pack(anchor="w")

        # STT panel — below detection, mirroring the audio STT UI
        stt_panel = tk.Frame(left, bg=SURFACE)
        stt_panel.pack(fill="x", pady=(4, 0))
        stt_hdr = tk.Frame(stt_panel, bg=SURFACE)
        stt_hdr.pack(fill="x", padx=10, pady=(6, 0))
        tk.Label(stt_hdr, text="SPEECH TO TEXT", bg=SURFACE, fg=FG_DIM,
                 font=("Segoe UI", 7, "bold")).pack(side="left")
        stt_mid = tk.Frame(stt_panel, bg=SURFACE)
        stt_mid.pack(fill="x", padx=10, pady=(3, 0))
        self._stt_dot = tk.Label(stt_mid, text="●", bg=SURFACE, fg=FG_DIM,
                                  font=("Segoe UI", 11))
        self._stt_dot.pack(side="left")
        self._stt_status_lbl = tk.Label(stt_mid, text="  Not started", bg=SURFACE,
                                         fg=FG_DIM, font=("Segoe UI", 9))
        self._stt_status_lbl.pack(side="left")
        stt_bot = tk.Frame(stt_panel, bg=SURFACE)
        stt_bot.pack(fill="x", padx=10, pady=(2, 7))
        self._stt_partial_lbl = tk.Label(stt_bot, text="", bg=SURFACE, fg=FG_DIM,
                                          font=("Segoe UI", 9, "italic"),
                                          wraplength=370, justify="left", anchor="w")
        self._stt_partial_lbl.pack(fill="x")

        # Right: chat fills remaining width (detection info is below camera)
        right = tk.Frame(main, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", pady=4)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        conv_hdr = tk.Frame(right, bg=BG)
        conv_hdr.grid(row=0, column=0, sticky="nsew")
        conv_hdr.rowconfigure(1, weight=1)
        conv_hdr.columnconfigure(0, weight=1)
        tk.Label(conv_hdr, text="CONVERSATION", bg=BG, fg=FG_DIM,
                 font=("Segoe UI", 8, "bold")).grid(row=0, column=0, sticky="w")
        chat_box = tk.Frame(conv_hdr, bg=SURFACE)
        chat_box.grid(row=1, column=0, sticky="nsew", pady=(2, 0))
        chat_box.rowconfigure(0, weight=1)
        chat_box.columnconfigure(0, weight=1)
        self._chat = tk.Text(chat_box, bg=SURFACE, fg=FG, font=("Segoe UI", 10),
                             relief="flat", wrap="word", padx=8, pady=6, state="disabled")
        self._chat.tag_config("jarvis", foreground=ACCENT, font=("Segoe UI", 10, "bold"))
        self._chat.tag_config("you",   foreground=GREEN,  font=("Segoe UI", 10, "bold"))
        self._chat.tag_config("info",  foreground=FG_DIM, font=("Segoe UI", 9, "italic"))
        sb = tk.Scrollbar(chat_box, command=self._chat.yview, bg=SURFACE, troughcolor=BG)
        self._chat.config(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky="ns")
        self._chat.grid(row=0, column=0, sticky="nsew")

        # Bottom controls — 3 rows
        ctrl = tk.Frame(self.root, bg=SURFACE)
        ctrl.pack(fill="x", side="bottom")

        # Row 1: Microphone
        r1 = tk.Frame(ctrl, bg=SURFACE)
        r1.pack(fill="x", padx=12, pady=(6, 2))
        tk.Label(r1, text="Microphone:", bg=SURFACE, fg=FG_DIM,
                 font=("Segoe UI", 9)).pack(side="left")
        self._mic_menu = tk.OptionMenu(r1, self._mic_var, "System Default")
        self._mic_menu.config(bg=SURFACE, fg=FG, highlightthickness=0,
                              relief="flat", font=("Segoe UI", 9),
                              activebackground=ACCENT, activeforeground="#fff")
        self._mic_menu["menu"].config(bg=SURFACE, fg=FG,
                                      activebackground=ACCENT, activeforeground="#fff")
        self._mic_menu.pack(side="left", padx=(6, 4))
        tk.Button(r1, text="↻", command=self._populate_mics,
                  bg=SURFACE, fg=FG_DIM, font=("Segoe UI", 10),
                  relief="flat", padx=4, pady=1, cursor="hand2"
                  ).pack(side="left", padx=(0, 16))
        self._audio_status = tk.Label(r1, text="", bg=SURFACE, fg=FG_DIM,
                                      font=("Segoe UI", 8))
        self._audio_status.pack(side="right", padx=4)

        # Row 2: Audio source + voice output
        r2 = tk.Frame(ctrl, bg=SURFACE)
        r2.pack(fill="x", padx=12, pady=(0, 2))
        tk.Checkbutton(r2, text="♪  System Audio", variable=self._sys_var,
                       command=self._on_sys_toggle,
                       bg=SURFACE, fg=FG_DIM, selectcolor=SURFACE,
                       activebackground=SURFACE, font=("Segoe UI", 9)
                       ).pack(side="left", padx=(0, 4))
        self._mix_btn = tk.Checkbutton(r2, text="+ Mix mic", variable=self._mix_var,
                                       bg=SURFACE, fg=FG_DIM, selectcolor=SURFACE,
                                       activebackground=SURFACE, font=("Segoe UI", 9))
        # shown/hidden by _on_sys_toggle
        tk.Label(r2, text="  │", bg=SURFACE, fg="#313244",
                 font=("Segoe UI", 9)).pack(side="left", padx=(10, 10))
        tk.Checkbutton(r2, text="🔊  Voice output", variable=self._tts_enabled,
                       bg=SURFACE, fg=FG_DIM, selectcolor=SURFACE,
                       activebackground=SURFACE, font=("Segoe UI", 9),
                       state="normal" if TTS_OK else "disabled"
                       ).pack(side="left")

        # Row 3: STT model
        r3 = tk.Frame(ctrl, bg=SURFACE)
        r3.pack(fill="x", padx=12, pady=(0, 6))
        tk.Label(r3, text="STT Model:", bg=SURFACE, fg=FG_DIM,
                 font=("Segoe UI", 9)).pack(side="left")
        self._vosk_menu = tk.OptionMenu(r3, self._vosk_model_var, "— no models —")
        self._vosk_menu.config(bg=SURFACE, fg=FG, highlightthickness=0,
                               relief="flat", font=("Segoe UI", 9), width=34,
                               activebackground=ACCENT, activeforeground="#fff")
        self._vosk_menu["menu"].config(bg=SURFACE, fg=FG,
                                       activebackground=ACCENT, activeforeground="#fff")
        self._vosk_menu.pack(side="left", padx=(6, 4))
        tk.Button(r3, text="Load", command=self._load_vosk_model,
                  bg=ACCENT, fg="#fff", font=("Segoe UI", 9, "bold"),
                  relief="flat", padx=10, pady=2, cursor="hand2"
                  ).pack(side="left")
        self._model_status = tk.Label(r3, text="", bg=SURFACE, fg=FG_DIM,
                                      font=("Segoe UI", 8))
        self._model_status.pack(side="left", padx=(8, 0))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._populate_mics()
        self._scan_vosk_models()
        if not TTS_OK:
            self._chat_info("pyttsx3 not installed — no voice.  Run: pip install pyttsx3")

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        def _load():
            # Face recognition
            if FACE_OK and _EMBED_FILE.exists():
                try:
                    app = _FaceAnalysis(
                        name="buffalo_sc", root=str(_REC_MODELS),
                        allowed_modules=["detection", "recognition"],
                        providers=["CPUExecutionProvider"])
                    app.prepare(ctx_id=0, det_size=(160, 160))
                    known = {
                        n: np.array(e)
                        for n, e in json.loads(_EMBED_FILE.read_text()).items()
                    }
                    self._face_app = app
                    self._known    = known
                except Exception as e:
                    self.root.after(0, lambda err=str(e): self._chat_info(
                        f"✗ Face recognition: {err}"))

            # MediaPipe emotion
            if MP_OK:
                model_file = _MP_MODELS / "face_landmarker.task"
                if not model_file.exists():
                    _MP_MODELS.mkdir(parents=True, exist_ok=True)
                    _url = ("https://storage.googleapis.com/mediapipe-models/"
                            "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
                    try:
                        urllib.request.urlretrieve(_url, model_file)
                    except Exception as dl_err:
                        self.root.after(0, lambda e=str(dl_err): self._chat_info(
                            f"— Emotion model download failed: {e}"))
                if model_file.exists():
                    try:
                        try:
                            base_opts = _mp_python.BaseOptions(
                                model_asset_path=str(model_file),
                                enable_model_metadata_logging=False)
                        except TypeError:
                            base_opts = _mp_python.BaseOptions(
                                model_asset_path=str(model_file))
                        opts = _mp_vision.FaceLandmarkerOptions(
                            base_options=base_opts,
                            running_mode=_mp_vision.RunningMode.IMAGE,
                            num_faces=4,
                            output_face_blendshapes=True,
                            min_face_detection_confidence=0.5)
                        self._mp_landmarker = _mp_vision.FaceLandmarker.create_from_options(opts)
                    except Exception as e:
                        self.root.after(0, lambda err=str(e): self._chat_info(
                            f"✗ Emotion detection: {err}"))

            # Vosk — load first available model
            if VOSK_OK:
                vosk_dirs = sorted(_AUDIO_MODELS.glob("vosk-model-*")) if _AUDIO_MODELS.exists() else []
                for d in vosk_dirs:
                    if (d / "am" / "final.mdl").exists():
                        try:
                            self._vosk_model = Model(str(d))
                            break
                        except Exception:
                            pass

        threading.Thread(target=_load, daemon=True).start()

    # ── Device management ─────────────────────────────────────────────────────

    def _populate_mics(self) -> None:
        prev = self._mic_var.get()
        mics = list_microphones(test_devices=True, include_system_audio=False)
        mics.sort(key=lambda m: (0 if "wasapi" in m.host_api_name.lower() else 1))
        menu = self._mic_menu["menu"]
        menu.delete(0, "end")
        labels = ["System Default"]
        menu.add_command(label="System Default",
                         command=lambda: self._mic_var.set("System Default"))
        for mic in mics:
            sr  = f" {int(mic.default_sample_rate//1000)}kHz"
            api = mic.host_api_name.replace("Windows WASAPI","WASAPI").replace("Windows ","")
            lbl = f"{mic.name[:20]}{sr} [{api}]  [idx {mic.index}]"
            labels.append(lbl)
            menu.add_command(label=lbl, command=lambda v=lbl: self._mic_var.set(v))
        self._mic_var.set(prev if prev in labels else "System Default")

    def _on_sys_toggle(self) -> None:
        if self._sys_var.get():
            dev = get_best_system_audio_device()
            if dev is None:
                self._sys_var.set(False)
                messagebox.showinfo("System Audio",
                                    "No loopback device found.\npip install pyaudiowpatch",
                                    parent=self.root)
                return
            self._sys_dev = dev
            self._mix_btn.pack(side="left", padx=(0, 4))
        else:
            self._sys_dev = None
            self._mix_var.set(False)
            self._mix_btn.pack_forget()

    def _get_mic_index(self):
        val = self._mic_var.get()
        if val == "System Default":
            return None
        return int(val.rsplit("idx ", 1)[-1].rstrip("]"))

    def _scan_vosk_models(self) -> None:
        """Populate the STT model dropdown — friendly names matching the audio STT UI."""
        options = []
        if _AUDIO_MODELS.exists():
            for d in sorted(_AUDIO_MODELS.iterdir()):
                if d.is_dir() and (d / "am" / "final.mdl").exists():
                    friendly = _VOSK_FRIENDLY.get(d.name, d.name)
                    options.append((friendly, d.name, d))
        self._vosk_options = options
        menu = self._vosk_menu["menu"]
        menu.delete(0, "end")
        if options:
            for friendly, _, _ in options:
                menu.add_command(label=friendly,
                                 command=lambda v=friendly: self._vosk_model_var.set(v))
            self._vosk_model_var.set(options[0][0])
            if self._vosk_model is None:
                threading.Thread(target=self._load_vosk_model, daemon=True).start()
        else:
            self._vosk_model_var.set("— download in audio STT UI —")

    def _load_vosk_model(self) -> None:
        """Load the selected Vosk model in a background thread."""
        selected_friendly = self._vosk_model_var.get()
        if not selected_friendly or selected_friendly.startswith("—"):
            return
        path = None
        for friendly, name, p in self._vosk_options:
            if friendly == selected_friendly:
                path = p
                break
        if path is None:
            return
        self.root.after(0, lambda: self._model_status.config(
            text=f"Loading…"))
        try:
            model = Model(str(path))
            self._vosk_model = model
            self.root.after(0, lambda sf=selected_friendly:
                self._model_status.config(text=f"✓ {sf[:30]}"))
        except Exception as e:
            self.root.after(0, lambda err=str(e): (
                self._model_status.config(text=f"✗ {err[:40]}"),
                self._chat_info(f"Failed to load model: {err}"),
            ))

    # ── Start / Stop ──────────────────────────────────────────────────────────

    def _toggle(self) -> None:
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        self._running = True
        self._session_id += 1
        self._start_btn.config(text="■  Stop", bg=RED)
        self._status_dot.config(fg=GREEN)
        self._greeted.clear()
        self._last_emotion.clear()
        self._messages.clear()
        self._stt_buffer       = ""
        self._llm_unreachable  = False   # reset so the API is retried
        self._offline_notified = False
        self._llm_busy         = False
        self._pending_llm      = False

        sid = self._session_id
        self._last_detections = []
        self._detect_thread = threading.Thread(target=self._detect_loop, args=(sid,), daemon=True)
        self._cam_thread    = threading.Thread(target=self._cam_loop,   args=(sid,), daemon=True)
        self._audio_thread  = threading.Thread(target=self._audio_loop, args=(sid,), daemon=True)
        self._detect_thread.start()
        self._cam_thread.start()
        self._audio_thread.start()

    def _stop(self) -> None:
        self._running = False
        self._start_btn.config(text="▶  Start", bg=ACCENT)
        self._status_dot.config(fg=RED)
        self._cam_label.config(image="", text="Camera not started")
        self._det_face_lbl.config(text="—", fg=FG_DIM)
        self._det_emot_lbl.config(text="—", fg=FG_DIM)
        self._stt_dot.config(fg=FG_DIM)
        self._stt_status_lbl.config(text="  Not started", fg=FG_DIM)
        self._stt_partial_lbl.config(text="")
        self._audio_status.config(text="")

    # ── Camera loop ───────────────────────────────────────────────────────────

    def _cam_loop(self, sid: int) -> None:
        cap     = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        frame_n = 0

        while self._running and self._session_id == sid:
            ret, frame = cap.read()
            if not ret:
                break

            frame_n += 1
            frame = cv2.flip(frame, 1)

            # Queue a copy for the detect thread — non-blocking, drop if busy
            if frame_n % DETECT_EVERY == 0:
                try:
                    self._detect_q.put_nowait(frame.copy())
                except queue.Full:
                    pass

            # Draw using last known detections (updated by detect thread)
            annotated = frame.copy()
            for d in self._last_detections:
                x1, y1, x2, y2 = d["bbox"]
                name  = d["name"]
                conf  = d["confidence"]
                emot  = d.get("emotion", "")
                color = (0, 220, 80) if name != "Unknown" else (0, 60, 220)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{name} {conf:.0%}" if name != "Unknown" else "Unknown"
                if emot and emot != "Neutral":
                    label += f"  {emot}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Resize to fixed display size (400×300) before converting
            annotated = cv2.resize(annotated, (400, 300), interpolation=cv2.INTER_LINEAR)

            # Convert to tkinter image
            h, w = annotated.shape[:2]
            rgb   = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img   = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.root.after(0, lambda i=img: self._update_cam(i))

        cap.release()

    # ── Detection loop ───────────────────────────────────────────────────

    def _detect_loop(self, sid: int) -> None:
        """Face + emotion detection runs here so the camera loop never stutters."""
        while self._running and self._session_id == sid:
            try:
                frame = self._detect_q.get(timeout=0.5)
            except queue.Empty:
                continue
            detections = self._detect(frame)
            self._last_detections = detections   # atomic list replacement
            for d in detections:
                self.root.after(0, lambda det=d: self._on_detection(det))

    def _detect(self, frame: np.ndarray) -> list[dict]:
        results = []
        h, w = frame.shape[:2]

        if self._face_app is None:
            return results

        faces = self._face_app.get(frame)

        # Emotion detection via MediaPipe
        emotions_map: dict[tuple, str] = {}
        if self._mp_landmarker is not None:
            try:
                ts_ms  = int(time.monotonic() * 1000)
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                mp_res = self._mp_landmarker.detect(mp_img)
                if mp_res.face_landmarks and mp_res.face_blendshapes:
                    for lm_list, bs in zip(mp_res.face_landmarks, mp_res.face_blendshapes):
                        if not lm_list:
                            continue
                        cx = int(sum(lm.x for lm in lm_list) / len(lm_list) * w)
                        cy = int(sum(lm.y for lm in lm_list) / len(lm_list) * h)
                        em = classify_emotion_blendshapes(bs)
                        emotions_map[(cx, cy)] = em
            except Exception:
                pass

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-9)

            best_name, best_dist = "Unknown", FACE_THRESHOLD
            for name, known_emb in self._known.items():
                d = cosine_distance(emb, known_emb)
                if d < best_dist:
                    best_dist = d
                    best_name = name

            # Match nearest emotion centroid
            fc_x = (x1 + x2) // 2
            fc_y = (y1 + y2) // 2
            emotion = "Neutral"
            if emotions_map:
                nearest = min(emotions_map.keys(),
                              key=lambda p: abs(p[0]-fc_x) + abs(p[1]-fc_y))
                if abs(nearest[0]-fc_x) + abs(nearest[1]-fc_y) < 150:
                    emotion = emotions_map[nearest]

            results.append({
                "bbox":       (x1, y1, x2, y2),
                "name":       best_name,
                "confidence": 1.0 - best_dist,
                "emotion":    emotion,
            })
        return results

    def _update_cam(self, img) -> None:
        self._cam_label.config(image=img, text="")
        self._cam_label.image = img  # prevent GC

    # ── Detection event → greet / comment ─────────────────────────────────────

    def _on_detection(self, det: dict) -> None:
        name    = det["name"]
        emotion = det.get("emotion", "Neutral")
        conf    = det.get("confidence", 0.0)

        # Update detection panel below camera
        if name != "Unknown":
            self._det_face_lbl.config(text=f"👤  {name}  {conf:.0%}", fg=GREEN)
        else:
            self._det_face_lbl.config(text="Unknown", fg=FG_DIM)
        _emot_icons = {"Happy": "😊", "Sad": "😔", "Surprised": "😲", "Angry": "😠"}
        if emotion != "Neutral":
            _icon = _emot_icons.get(emotion, "")
            self._det_emot_lbl.config(
                text=f"{_icon}  {emotion}" if _icon else emotion,
                fg=YELLOW if emotion == "Surprised" else
                   GREEN  if emotion == "Happy" else
                   RED    if emotion in ("Angry", "Sad") else FG)
        else:
            self._det_emot_lbl.config(text="Neutral", fg=FG_DIM)

        self._current_context = {"name": name, "emotion": emotion, "confidence": conf}

        # Greeting: once per person per session
        if name != "Unknown" and name not in self._greeted:
            self._greeted.add(name)
            greeting = time_greeting()
            trigger  = f"{greeting}, {name}! Great to see you."
            if emotion == "Happy":
                trigger += " You look wonderful today!"
            elif emotion in ("Sad", "Angry"):
                trigger += f" You seem a bit {emotion.lower()} — is everything okay?"
            elif emotion == "Surprised":
                trigger += " You look surprised — everything alright?"
            self._trigger_llm(trigger, auto=True)

        # Emotion comment: once per emotion change per person
        elif name != "Unknown" and emotion != "Neutral":
            key = (name, emotion)
            if key not in self._last_emotion:
                self._last_emotion[key] = True
                if emotion in ("Sad", "Angry"):
                    self._trigger_llm(
                        f"[Jarvis notices {name} looks {emotion.lower()}]", auto=True)

    # ── Audio loop ────────────────────────────────────────────────────────────

    def _audio_loop(self, sid: int) -> None:
        if self._vosk_model is None:
            self.root.after(0, lambda: (
                self._stt_dot.config(fg=RED),
                self._stt_status_lbl.config(
                    text="  No model loaded — select one below", fg=RED),
            ))
            return

        pa  = _pa_lib.PyAudio()
        rec = KaldiRecognizer(self._vosk_model, RATE)
        rec.SetWords(True)
        is_speaking = False
        silence_start = None
        auto_threshold = 150
        PAUSE = 2.0   # seconds of silence before committing final result

        sys_dev  = self._sys_dev
        mix_mic  = self._mix_var.get()
        mic_idx  = self._get_mic_index()

        self.root.after(0, lambda: (
            self._stt_dot.config(fg=GREEN),
            self._stt_status_lbl.config(text="  Listening…", fg=GREEN),
        ))

        try:
            if sys_dev and mix_mic:
                # Mixed mode: two threads feeding a shared queue
                mic_q  = queue.Queue(maxsize=30)
                sys_q  = queue.Queue(maxsize=30)
                SILENCE_BUF = bytes(CHUNK * 2)
                sys_ch   = min(sys_dev.max_input_channels, 2)
                sys_rate = int(sys_dev.default_sample_rate)

                def _cap_mic():
                    pa_m = _pa_lib.PyAudio()
                    try:
                        s, af, ch, r = _open_input(pa_m, mic_idx, 1, RATE)
                        buf = bytearray()
                        T   = CHUNK * 2
                        while self._running and self._session_id == sid:
                            raw = s.read(CHUNK, exception_on_overflow=False)
                            d   = _float32_to_int16(raw) if af else raw
                            if ch > 1:     d = _to_mono(d, ch)
                            if r  != RATE: d = _resample(d, r, RATE)
                            buf.extend(d)
                            while len(buf) >= T:
                                if not mic_q.full(): mic_q.put(bytes(buf[:T]))
                                del buf[:T]
                        s.stop_stream(); s.close()
                    except Exception: pass
                    finally: pa_m.terminate()

                def _cap_sys():
                    pa_s = _pa_lib.PyAudio()
                    try:
                        s, af, ch, r = _open_input(pa_s, sys_dev.index, sys_ch, sys_rate)
                        buf = bytearray()
                        T   = CHUNK * 2
                        while self._running and self._session_id == sid:
                            raw = s.read(CHUNK, exception_on_overflow=False)
                            if af: raw = _float32_to_int16(raw)
                            d = _to_mono(raw, ch)
                            if r != RATE: d = _resample(d, r, RATE)
                            buf.extend(d)
                            while len(buf) >= T:
                                if not sys_q.full(): sys_q.put(bytes(buf[:T]))
                                del buf[:T]
                        s.stop_stream(); s.close()
                    except Exception: pass
                    finally: pa_s.terminate()

                threading.Thread(target=_cap_mic, daemon=True).start()
                threading.Thread(target=_cap_sys, daemon=True).start()

                while self._running and self._session_id == sid:
                    try: mic_d = mic_q.get(timeout=0.5)
                    except queue.Empty: continue
                    sys_d = sys_q.get_nowait() if not sys_q.empty() else SILENCE_BUF
                    a = np.frombuffer(mic_d[:len(sys_d)], np.int16).astype(np.float32)
                    b = np.frombuffer(sys_d[:len(mic_d)], np.int16).astype(np.float32)
                    data = ((a + b) / 2).clip(-32768, 32767).astype(np.int16).tobytes()
                    is_speaking, silence_start = self._feed_vosk(
                        rec, data, auto_threshold, PAUSE, is_speaking, silence_start)
            else:
                dev_idx  = sys_dev.index if sys_dev else mic_idx
                dev_ch   = min(sys_dev.max_input_channels, 2) if sys_dev else 1
                dev_rate = int(sys_dev.default_sample_rate) if sys_dev else RATE

                stream, _use_float, dev_ch, dev_rate = _open_input(
                    pa, dev_idx, dev_ch, dev_rate)

                # Calibrate threshold
                rms_s = []
                for _ in range(max(1, int(RATE / CHUNK))):
                    raw = stream.read(CHUNK, exception_on_overflow=False)
                    if _use_float: raw = _float32_to_int16(raw)
                    raw = _to_mono(raw, dev_ch)
                    if dev_rate != RATE: raw = _resample(raw, dev_rate, RATE)
                    n = len(raw) // 2
                    if n:
                        s_vals = struct.unpack(f"{n}h", raw)
                        rms_s.append(math.sqrt(sum(v*v for v in s_vals)/n))
                    rec.AcceptWaveform(raw)
                if rms_s:
                    auto_threshold = max(30, int(sum(rms_s)/len(rms_s)*1.1))

                while self._running and self._session_id == sid:
                    raw = stream.read(CHUNK, exception_on_overflow=False)
                    if _use_float: raw = _float32_to_int16(raw)
                    data = _to_mono(raw, dev_ch)
                    if dev_rate != RATE: data = _resample(data, dev_rate, RATE)
                    is_speaking, silence_start = self._feed_vosk(
                        rec, data, auto_threshold, PAUSE, is_speaking, silence_start)

                stream.stop_stream(); stream.close()

        except Exception as exc:
            self.root.after(0, lambda e=exc: (
                self._stt_dot.config(fg=RED),
                self._stt_status_lbl.config(text=f"  Error: {e}", fg=RED),
                self._audio_status.config(text=f"Audio error: {e}"),
            ))
        finally:
            pa.terminate()
            self.root.after(0, lambda: (
                self._stt_dot.config(fg=FG_DIM),
                self._stt_status_lbl.config(text="  Stopped", fg=FG_DIM),
                self._stt_partial_lbl.config(text=""),
                self._audio_status.config(text=""),
            ))

    def _feed_vosk(self, rec, data, threshold, pause, is_speaking, silence_start):
        n = len(data) // 2
        if n == 0:
            return is_speaking, silence_start
        # Mute STT while Jarvis is speaking — also flush Vosk's buffer so its
        # own voice doesn’t contaminate the next utterance
        if self._speaking:
            # Flush Vosk only on first chunk (when user was mid-utterance)
            if is_speaking or self._stt_buffer:
                rec.FinalResult()        # discard buffered audio
                self._stt_buffer = ""
            # Return clean state so silence timer never fires on resume
            return False, None

        samps = struct.unpack(f"{n}h", data)
        rms   = math.sqrt(sum(s*s for s in samps) / n)
        has_speech = rms > threshold

        if rec.AcceptWaveform(data):
            # Vosk flushed its internal buffer — accumulate the chunk instead of
            # immediately sending it (the user may not have finished speaking)
            chunk = json.loads(rec.Result()).get("text", "").strip()
            if chunk:
                self._stt_buffer = (self._stt_buffer + " " + chunk).strip()
            display = self._stt_buffer
            if display:
                self.root.after(0, lambda p=display: (
                    self._stt_dot.config(fg=YELLOW),
                    self._stt_status_lbl.config(text="  Hearing…", fg=YELLOW),
                    self._stt_partial_lbl.config(text=p[:80]),
                ))
        else:
            partial = json.loads(rec.PartialResult()).get("partial", "").strip()
            if partial:
                # Show buffer + live partial together
                display = (self._stt_buffer + " " + partial).strip()
                self.root.after(0, lambda p=display: (
                    self._stt_dot.config(fg=YELLOW),
                    self._stt_status_lbl.config(text="  Hearing…", fg=YELLOW),
                    self._stt_partial_lbl.config(text=p[:80]),
                ))

        if has_speech:
            is_speaking = True
            silence_start = None
        elif is_speaking:
            if silence_start is None:
                silence_start = time.monotonic()
            elif time.monotonic() - silence_start >= pause:
                # Silence threshold reached — commit everything
                tail  = json.loads(rec.FinalResult()).get("text", "").strip()
                final = (self._stt_buffer + " " + tail).strip()
                self._stt_buffer = ""
                if final and _is_meaningful(final):
                    self.root.after(0, lambda t=final: self._on_speech(t))
                else:
                    # Ignored (too short / filler) — reset panel
                    self.root.after(0, lambda: (
                        self._stt_dot.config(fg=GREEN),
                        self._stt_status_lbl.config(text="  Listening…", fg=GREEN),
                        self._stt_partial_lbl.config(text=""),
                    ))
                is_speaking   = False
                silence_start = None

        return is_speaking, silence_start

    def _set_tts_speaking(self, speaking: bool) -> None:
        """Called from TTS worker / LLM thread — safe to call from any thread."""
        self._speaking = speaking
        if not hasattr(self, "_stt_dot"):
            return  # UI not yet built
        if speaking:
            self.root.after(0, lambda: (
                self._stt_dot.config(fg=ACCENT),
                self._stt_status_lbl.config(text="  Speaking…", fg=ACCENT),
                self._stt_partial_lbl.config(text=""),
            ))
        elif self._running:
            self.root.after(0, lambda: (
                self._stt_dot.config(fg=GREEN),
                self._stt_status_lbl.config(text="  Listening…", fg=GREEN),
            ))

    def _on_speech(self, text: str) -> None:
        self._audio_status.config(text="")
        self._stt_dot.config(fg=GREEN)
        self._stt_status_lbl.config(text="  Listening…", fg=GREEN)
        self._stt_partial_lbl.config(text="")
        self._chat_append("You", text, "you")
        self._messages.append({"role": "user", "content": text})
        if self._llm_busy:
            self._pending_llm = True   # will be answered after current response finishes
        else:
            threading.Thread(target=self._llm_respond, daemon=True).start()

    # ── LLM ───────────────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        ctx  = self._current_context
        name = ctx.get("name", "")
        emot = ctx.get("emotion", "Neutral")
        conf = ctx.get("confidence", 0.0)
        now  = datetime.now()

        lines = [
            "You are Jarvis, an AI personal assistant with live camera vision and voice.",
            "Be warm, natural, and concise (2-3 sentences unless more detail is requested).",
            "",
        ]

        # Vision context
        if name and name != "Unknown":
            lines.append(f"PERSON IN VIEW: {name}  (recognition confidence {conf:.0%})")
            if emot and emot != "Neutral":
                lines.append(f"DETECTED EMOTION: {emot} — respond with appropriate empathy.")
        else:
            lines.append("PERSON IN VIEW: not recognised")

        # Time context
        lines.append(f"TIME: {time_greeting()}, {now.strftime('%A %B %d, %Y at %I:%M %p')}.")

        # Conversation context
        n_turns = sum(1 for m in self._messages if m["role"] == "user")
        if n_turns > 1:
            lines.append(
                f"CONVERSATION: turn {n_turns} of the current session — "
                "continue naturally without re-introducing yourself.")

        return "\n".join(lines)

    def _trigger_llm(self, message: str, auto: bool = False) -> None:
        """Auto-triggered messages (greetings/emotion comments) go straight to LLM."""
        if self._llm_busy:
            return  # never stack on top of an active response
        self._messages.append({"role": "user", "content": message})
        threading.Thread(target=self._llm_respond, args=(auto,), daemon=True).start()

    def _llm_respond(self, is_auto: bool = False) -> None:
        """Thin controller: sets busy flag, runs impl, clears flag, processes queue."""
        self._llm_busy = True
        try:
            self._llm_respond_impl(is_auto)
        finally:
            self._llm_busy = False
            if self._pending_llm:
                self._pending_llm = False
                threading.Thread(target=self._llm_respond, daemon=True).start()

    def _llm_respond_impl(self, is_auto: bool = False) -> None:
        cfg  = self._llm_cfg
        sys_msg  = self._build_system_prompt()
        msgs     = [{"role": "system", "content": sys_msg}] + self._messages[-20:]

        if not _is_llm_configured(cfg) or self._llm_unreachable:
            ctx   = self._current_context
            user  = self._messages[-1]["content"] if self._messages else ""
            reply = _local_respond(user, ctx, notified=self._offline_notified)
            self._offline_notified = True
            self.root.after(0, lambda: self._chat_append("Jarvis", "", "jarvis"))
            self.root.after(0, lambda r=reply: self._stream_chat(r))
            self.root.after(0, lambda: (
                self._chat.config(state="normal"),
                self._chat.insert("end", "\n"),
                self._chat.config(state="disabled"),
            ))
            self._messages.append({"role": "assistant", "content": reply})
            self.root.after(0, lambda: self._llm_lbl.config(
                text="  LLM: Offline", fg="#f59e0b"))
            if reply and TTS_OK and self._tts_enabled.get():
                self._tts.speak(reply)
            return

        full = ""
        try:
            api_key = cfg.get("api_key", "")
            if not api_key:
                raise RuntimeError(
                    "No API key — add  api_key  to llm/.llm_config.json")
            api_url = cfg.get(
                "api_url",
                "https://openbridgeai.kshitijks.com/api/v1/chat/completions")
            payload = json.dumps({
                "messages": msgs,
                "save_conversation": False,
            }).encode()
            req = urllib.request.Request(
                api_url, data=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data   = json.loads(resp.read())
                full   = data["choices"][0]["message"]["content"]

        except (ConnectionRefusedError, OSError,
                urllib.error.URLError):
            # Network / connection failure → latch into offline mode silently
            self._llm_unreachable = True
            ctx   = self._current_context
            user  = (self._messages[-1]["content"]
                     if self._messages else "")
            reply = _local_respond(user, ctx, notified=self._offline_notified)
            self._offline_notified = True
            self.root.after(0, lambda: self._chat_append("Jarvis", "", "jarvis"))
            self.root.after(0, lambda r=reply: self._stream_chat(r))
            self.root.after(0, lambda: self._llm_lbl.config(
                text="  LLM: Offline", fg="#f59e0b"))
            self.root.after(0, lambda: (
                self._chat.config(state="normal"),
                self._chat.insert("end", "\n"),
                self._chat.config(state="disabled"),
            ))
            self._messages.append({"role": "assistant", "content": reply})
            if reply and TTS_OK and self._tts_enabled.get():
                self._tts.speak(reply)
            return
        except Exception as e:
            self.root.after(0, lambda err=str(e): self._stream_chat(
                f"\n[Error: {err}]"))
            return

        if not full:
            return  # empty response — no header, no entry in history

        self.root.after(0, lambda: self._chat_append("Jarvis", "", "jarvis"))
        self.root.after(0, lambda t=full: self._stream_chat(t))
        self.root.after(0, lambda: (
            self._chat.config(state="normal"),
            self._chat.insert("end", "\n"),
            self._chat.config(state="disabled"),
        ))
        self._messages.append({"role": "assistant", "content": full})
        if TTS_OK and self._tts_enabled.get():
            self._tts.speak(full)

    # ── Chat text helpers ─────────────────────────────────────────────────────

    def _chat_append(self, speaker: str, text: str, tag: str) -> None:
        self._chat.config(state="normal")
        if self._chat.index("end-1c") != "1.0":
            self._chat.insert("end", "\n")
        self._chat.insert("end", f"{speaker}:\n", tag)
        if text:
            self._chat.insert("end", text + "\n")
        self._chat.see("end")
        self._chat.config(state="disabled")

    def _chat_info(self, text: str) -> None:
        self._chat.config(state="normal")
        self._chat.insert("end", f"{text}\n", "info")
        self._chat.see("end")
        self._chat.config(state="disabled")

    def _stream_chat(self, chunk: str) -> None:
        self._chat.config(state="normal")
        self._chat.insert("end", chunk)
        self._chat.see("end")
        self._chat.config(state="disabled")

    def _on_close(self) -> None:
        self._stop()
        if TTS_OK:
            self._tts.shutdown()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    JarvisWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
