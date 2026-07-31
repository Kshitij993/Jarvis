"""
Voice Output — Text-to-Speech UI
Type text, press Speak (or Ctrl+Enter) and hear it through your speakers.

Features:
  - All voices installed on your system
  - Speech rate and volume sliders
  - Stop mid-speech at any time
  - Spoken phrase history (click to re-speak)
  - Fully offline — pyttsx3 only

Run:
    python voice_output/tts_ui.py
"""

import json
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk

import pyttsx3

# ── Colour palette (matches the rest of the project) ─────────────────────────
BG      = "#1e1e2e"
SURFACE = "#2a2a3e"
ACCENT  = "#7c6af7"
FG      = "#cdd6f4"
FG_DIM  = "#6c7086"
RED     = "#f38ba8"
GREEN   = "#a6e3a1"
YELLOW  = "#f9e2af"


# ─────────────────────────────────────────────────────────────────────────────
# TTS worker — pyttsx3 must run on its own thread (not the UI thread)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Speech subprocess helper
# pyttsx3.init() returns a cached singleton — calling it again after the first
# runAndWait() gives back the same broken engine.  Running each request in a
# fresh Python subprocess guarantees a clean engine every time.
# ─────────────────────────────────────────────────────────────────────────────

_SPEAK_SCRIPT = r"""
import sys, json, pyttsx3
p = json.loads(sys.argv[1])
e = pyttsx3.init()
e.setProperty("rate",   p["rate"])
e.setProperty("volume", p["vol"])
if p["vid"]:
    e.setProperty("voice", p["vid"])
e.say(p["text"])
e.runAndWait()
"""


def _spawn_speech(text: str, rate: int, volume: float,
                  voice_id: str | None) -> subprocess.Popen:
    args = json.dumps({"text": text, "rate": rate,
                       "vol": volume, "vid": voice_id})
    return subprocess.Popen(
        [sys.executable, "-c", _SPEAK_SCRIPT, args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TTS worker thread
# ─────────────────────────────────────────────────────────────────────────────
class _TTSWorker(threading.Thread):
    """Queues speak requests; each request runs in its own subprocess."""

    def __init__(self, on_start=None, on_done=None) -> None:
        super().__init__(daemon=True)
        self._q        = queue.Queue()
        self._on_start = on_start
        self._on_done  = on_done
        self._rate     = 165
        self._volume   = 0.95
        self._voice_id: str | None = None
        self._proc: subprocess.Popen | None = None

    def run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            kind, value = item
            if kind != "speak":
                continue
            try:
                self._proc = _spawn_speech(
                    value, self._rate, self._volume, self._voice_id)
                if self._on_start:
                    self._on_start()
                self._proc.wait()          # block until subprocess finishes
            except Exception:
                pass
            finally:
                self._proc = None
                if self._on_done:
                    self._on_done()

    def speak(self, text: str) -> None:
        self._q.put(("speak", text))

    def stop(self) -> None:
        """Terminate the current speech subprocess immediately."""
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

    def set_rate(self, rate: int) -> None:
        self._rate = rate

    def set_volume(self, vol: float) -> None:
        self._volume = vol

    def set_voice(self, voice_id: str) -> None:
        self._voice_id = voice_id

    def shutdown(self) -> None:
        self._q.put(None)

    def get_voices(self):
        try:
            eng = pyttsx3.init()
            voices = eng.getProperty("voices") or []
            eng.stop()
            return voices
        except Exception:
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────
class TTSWindow:

    MAX_HISTORY = 30

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self._speaking = False
        self._history: list[str] = []

        self._rate_var   = tk.IntVar(value=165)
        self._vol_var    = tk.DoubleVar(value=95)
        self._voice_var  = tk.StringVar()

        self._worker = _TTSWorker(
            on_start=self._on_speak_start,
            on_done =self._on_speak_done,
        )
        self._worker.start()
        self._voices: list = self._worker.get_voices()

        self._build_ui()
        # Sync initial slider values into worker settings
        self._worker.set_rate(self._rate_var.get())
        self._worker.set_volume(self._vol_var.get() / 100.0)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.title("Voice Output")
        self.root.geometry("560x540")
        self.root.minsize(440, 420)
        self.root.configure(bg=BG)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"560x540+{(sw-560)//2}+{(sh-540)//2}")

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=SURFACE)
        hdr.pack(fill="x")
        self._dot = tk.Label(hdr, text="●", bg=SURFACE, fg=FG_DIM,
                             font=("Segoe UI", 11))
        self._dot.pack(side="left", padx=(12, 4), pady=8)
        tk.Label(hdr, text="VOICE OUTPUT", bg=SURFACE, fg=ACCENT,
                 font=("Segoe UI", 13, "bold")).pack(side="left")
        self._status_lbl = tk.Label(hdr, text="Ready", bg=SURFACE, fg=FG_DIM,
                                    font=("Segoe UI", 9))
        self._status_lbl.pack(side="left", padx=12)

        # ── Settings row ─────────────────────────────────────────────────────
        settings = tk.Frame(self.root, bg=SURFACE)
        settings.pack(fill="x", padx=0)
        inner = tk.Frame(settings, bg=SURFACE)
        inner.pack(fill="x", padx=12, pady=8)

        # Voice picker
        tk.Label(inner, text="Voice:", bg=SURFACE, fg=FG_DIM,
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        voice_names = [v.name for v in self._voices] if self._voices else ["(default)"]
        self._voice_menu = ttk.Combobox(inner, textvariable=self._voice_var,
                                        values=voice_names, state="readonly",
                                        width=28, font=("Segoe UI", 9))
        if voice_names:
            self._voice_var.set(voice_names[0])
        self._voice_menu.grid(row=0, column=1, sticky="w", padx=(6, 20))
        self._voice_menu.bind("<<ComboboxSelected>>", self._on_voice_change)

        # Rate slider
        tk.Label(inner, text="Rate:", bg=SURFACE, fg=FG_DIM,
                 font=("Segoe UI", 9)).grid(row=0, column=2, sticky="w")
        self._rate_lbl = tk.Label(inner, text=f"{self._rate_var.get()} wpm",
                                  bg=SURFACE, fg=FG, font=("Segoe UI", 9), width=7)
        tk.Scale(inner, variable=self._rate_var, from_=80, to=350,
                 orient="horizontal", bg=SURFACE, fg=FG, troughcolor=BG,
                 highlightthickness=0, showvalue=False, length=100,
                 command=self._on_rate_change
                 ).grid(row=0, column=3, padx=(4, 4))
        self._rate_lbl.grid(row=0, column=4, sticky="w")

        # Volume slider
        tk.Label(inner, text="Vol:", bg=SURFACE, fg=FG_DIM,
                 font=("Segoe UI", 9)).grid(row=1, column=2, sticky="w", pady=(4, 0))
        self._vol_lbl = tk.Label(inner, text=f"{int(self._vol_var.get())}%",
                                 bg=SURFACE, fg=FG, font=("Segoe UI", 9), width=7)
        tk.Scale(inner, variable=self._vol_var, from_=0, to=100,
                 orient="horizontal", bg=SURFACE, fg=FG, troughcolor=BG,
                 highlightthickness=0, showvalue=False, length=100,
                 command=self._on_vol_change
                 ).grid(row=1, column=3, padx=(4, 4), pady=(4, 0))
        self._vol_lbl.grid(row=1, column=4, sticky="w", pady=(4, 0))

        # ── Separator ─────────────────────────────────────────────────────────
        tk.Frame(self.root, bg="#313244", height=1).pack(fill="x")

        # ── Text input ────────────────────────────────────────────────────────
        input_frame = tk.Frame(self.root, bg=BG)
        input_frame.pack(fill="both", expand=True, padx=12, pady=(10, 0))
        input_frame.rowconfigure(0, weight=1)
        input_frame.columnconfigure(0, weight=1)

        tk.Label(input_frame, text="TYPE YOUR TEXT", bg=BG, fg=FG_DIM,
                 font=("Segoe UI", 7, "bold")).grid(row=0, column=0,
                                                    columnspan=2, sticky="w")
        self._text = tk.Text(input_frame, bg=SURFACE, fg=FG,
                             font=("Segoe UI", 12), relief="flat",
                             wrap="word", padx=10, pady=10,
                             insertbackground=ACCENT,
                             selectbackground=ACCENT, selectforeground="#fff",
                             height=6)
        self._text.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        sb = tk.Scrollbar(input_frame, command=self._text.yview,
                          bg=SURFACE, troughcolor=BG)
        self._text.config(yscrollcommand=sb.set)
        sb.grid(row=1, column=1, sticky="ns", pady=(4, 0))
        input_frame.rowconfigure(1, weight=1)

        self._text.focus()
        self._text.bind("<Control-Return>", lambda e: self._speak())

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = tk.Frame(self.root, bg=BG)
        btn_row.pack(fill="x", padx=12, pady=8)

        self._speak_btn = tk.Button(
            btn_row, text="▶  Speak",
            command=self._speak,
            bg=ACCENT, fg="#fff", font=("Segoe UI", 10, "bold"),
            relief="flat", padx=20, pady=6, cursor="hand2")
        self._speak_btn.pack(side="left")

        tk.Label(btn_row, text="  Ctrl+Enter",
                 bg=BG, fg=FG_DIM, font=("Segoe UI", 8)).pack(side="left")

        self._stop_btn = tk.Button(
            btn_row, text="■  Stop",
            command=self._stop,
            bg=SURFACE, fg=RED, font=("Segoe UI", 10, "bold"),
            relief="flat", padx=14, pady=6, cursor="hand2",
            state="disabled")
        self._stop_btn.pack(side="left", padx=(12, 0))

        tk.Button(btn_row, text="Clear",
                  command=lambda: self._text.delete("1.0", "end"),
                  bg=SURFACE, fg=FG_DIM, font=("Segoe UI", 9),
                  relief="flat", padx=10, pady=6, cursor="hand2"
                  ).pack(side="right")

        # ── History ───────────────────────────────────────────────────────────
        tk.Frame(self.root, bg="#313244", height=1).pack(fill="x")
        hist_hdr = tk.Frame(self.root, bg=BG)
        hist_hdr.pack(fill="x", padx=12, pady=(6, 2))
        tk.Label(hist_hdr, text="HISTORY", bg=BG, fg=FG_DIM,
                 font=("Segoe UI", 7, "bold")).pack(side="left")
        tk.Button(hist_hdr, text="Clear history",
                  command=self._clear_history,
                  bg=BG, fg=FG_DIM, font=("Segoe UI", 7),
                  relief="flat", cursor="hand2").pack(side="right")

        hist_box = tk.Frame(self.root, bg=SURFACE)
        hist_box.pack(fill="x", expand=False, padx=12, pady=(0, 10))
        hist_box.columnconfigure(0, weight=1)

        self._hist_list = tk.Listbox(
            hist_box, bg=SURFACE, fg=FG_DIM, font=("Segoe UI", 9),
            relief="flat", selectbackground=ACCENT, selectforeground="#fff",
            activestyle="none", height=5, bd=0,
            highlightthickness=0)
        self._hist_list.pack(side="left", fill="x", expand=True, padx=(6, 0), pady=4)
        hist_sb = tk.Scrollbar(hist_box, command=self._hist_list.yview,
                               bg=SURFACE, troughcolor=BG)
        self._hist_list.config(yscrollcommand=hist_sb.set)
        hist_sb.pack(side="right", fill="y", pady=4)
        self._hist_list.bind("<Double-Button-1>", self._on_history_dclick)
        tk.Label(self.root, text="Double-click a phrase to speak it again",
                 bg=BG, fg=FG_DIM, font=("Segoe UI", 7)).pack(pady=(0, 6))

    # ── Settings callbacks ────────────────────────────────────────────────────

    def _on_voice_change(self, _=None) -> None:
        name = self._voice_var.get()
        voice = next((v for v in self._voices if v.name == name), None)
        if voice:
            self._worker.set_voice(voice.id)

    def _on_rate_change(self, val) -> None:
        rate = int(float(val))
        self._rate_lbl.config(text=f"{rate} wpm")
        self._worker.set_rate(rate)

    def _on_vol_change(self, val) -> None:
        vol = int(float(val))
        self._vol_lbl.config(text=f"{vol}%")
        self._worker.set_volume(vol / 100.0)

    # ── Speak / Stop ──────────────────────────────────────────────────────────

    def _speak(self) -> None:
        text = self._text.get("1.0", "end").strip()
        if not text:
            return
        self._worker.speak(text)
        # Add to history (most recent at top, no duplicates)
        if text in self._history:
            self._history.remove(text)
        self._history.insert(0, text)
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[:self.MAX_HISTORY]
        self._refresh_history()

    def _stop(self) -> None:
        self._worker.stop()

    # ── TTS callbacks (called from worker thread → schedule on UI thread) ─────

    def _on_speak_start(self) -> None:
        self._speaking = True
        self.root.after(0, self._set_speaking_ui, True)

    def _on_speak_done(self) -> None:
        self._speaking = False
        self.root.after(0, self._set_speaking_ui, False)

    def _set_speaking_ui(self, speaking: bool) -> None:
        if speaking:
            self._dot.config(fg=GREEN)
            self._status_lbl.config(text="Speaking…", fg=GREEN)
            self._speak_btn.config(state="disabled")
            self._stop_btn.config(state="normal")
        else:
            self._dot.config(fg=FG_DIM)
            self._status_lbl.config(text="Ready", fg=FG_DIM)
            self._speak_btn.config(state="normal")
            self._stop_btn.config(state="disabled")

    # ── History ───────────────────────────────────────────────────────────────

    def _refresh_history(self) -> None:
        self._hist_list.delete(0, "end")
        for phrase in self._history:
            display = phrase[:80] + ("…" if len(phrase) > 80 else "")
            self._hist_list.insert("end", display)

    def _on_history_dclick(self, _=None) -> None:
        sel = self._hist_list.curselection()
        if not sel:
            return
        full_text = self._history[sel[0]]
        self._text.delete("1.0", "end")
        self._text.insert("1.0", full_text)
        self._speak()

    def _clear_history(self) -> None:
        self._history.clear()
        self._refresh_history()

    def _on_close(self) -> None:
        self._worker.shutdown()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    TTSWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
