"""
Live Speech to Text — Advanced UI

Improvements over a basic listener:
  - Mic stays open continuously (no gaps or cut-offs between phrases)
  - "Pause to finalise" is fully configurable — raise it so a brief mid-
    sentence pause doesn't end the phrase prematurely
  - Max phrase length cap (stops very long recordings)
  - Auto or manual microphone sensitivity
  - All settings take effect immediately without restarting
  - Live animated indicator  (green pulse = listening, yellow = processing)

Requires internet — uses Google Web Speech API (no key needed for basic use).

Run:
    python audio/live_speech_to_text_ui.py
"""

import sys
import threading
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from pathlib import Path

import speech_recognition as sr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utilities.microphone_detector import list_microphones


# ── Indicator pulse colours ───────────────────────────────────────────────────
_PULSE_LISTEN = ("#3d7a4f", "#a6e3a1")   # dark / light green  (listening)
_PULSE_PROC   = ("#7a6a1e", "#f9e2af")   # dark / light yellow (processing)
_IDLE_COL     = "#45475a"                # grey  (stopped)


class LiveSTTWindow:
    BG      = "#1e1e2e"
    SURFACE = "#2a2a3e"
    ACCENT  = "#7c6af7"
    FG      = "#cdd6f4"
    FG_DIM  = "#6c7086"
    RED     = "#f38ba8"

    def __init__(self, root: tk.Tk) -> None:
        self.root     = root
        self._running = False
        self._r       = sr.Recognizer()
        self._thread: threading.Thread | None = None

        # Settings variables (set before _build_settings so widgets can bind them)
        self._pause_var      = tk.DoubleVar(value=1.5)   # seconds of silence → phrase end
        self._max_phrase_var = tk.IntVar(value=15)       # hard cap on phrase length (s)
        self._auto_energy    = tk.BooleanVar(value=True) # auto sensitivity
        self._energy_var     = tk.IntVar(value=300)      # manual energy threshold

        # Pulse animation state
        self._pulse_colors  = _PULSE_LISTEN
        self._pulse_phase   = 0
        self._pulse_after   = None

        self._configure_root()
        self._build_controls()
        self._build_settings()
        self._build_indicator()
        self._build_transcript()
        self._build_statusbar()
        self._populate_devices()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Root ──────────────────────────────────────────────────

    def _configure_root(self) -> None:
        self.root.title("Live Speech to Text")
        self.root.geometry("720x620")
        self.root.minsize(560, 480)
        self.root.configure(bg=self.BG)
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"720x620+{(sw - 720) // 2}+{(sh - 620) // 2}")

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
        self.device_menu.pack(side="left", padx=(0, 12))

        self.toggle_btn = tk.Button(
            ctrl, text="▶  Start Listening",
            command=self._toggle,
            bg=self.ACCENT, fg="#fff",
            font=("Segoe UI", 10, "bold"),
            relief="flat", padx=16, pady=6, cursor="hand2",
            activebackground="#6c5ce7", activeforeground="#fff",
        )
        self.toggle_btn.pack(side="left", padx=4)

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

    # ── Settings panel ────────────────────────────────────────

    def _build_settings(self) -> None:
        panel = tk.Frame(self.root, bg=self.BG, pady=6)
        panel.pack(fill="x", padx=16)

        # ── Row 1: pause + max phrase ─────────────────────────
        row1 = tk.Frame(panel, bg=self.BG)
        row1.pack(fill="x", pady=(0, 4))

        # Pause to finalise
        tk.Label(row1, text="Pause to finalise:", bg=self.BG, fg=self.FG_DIM,
                 font=("Segoe UI", 9)).pack(side="left")

        self._pause_label = tk.Label(row1, text="1.5 s", bg=self.BG, fg=self.FG,
                                     font=("Segoe UI", 9, "bold"), width=5)
        self._pause_label.pack(side="left", padx=(0, 4))

        tk.Scale(
            row1, variable=self._pause_var,
            from_=0.5, to=3.0, resolution=0.1,
            orient="horizontal", length=160,
            bg=self.BG, fg=self.FG, troughcolor=self.SURFACE,
            highlightthickness=0, showvalue=False,
            command=self._on_pause_change,
        ).pack(side="left", padx=(0, 24))

        # Max phrase length
        tk.Label(row1, text="Max phrase:", bg=self.BG, fg=self.FG_DIM,
                 font=("Segoe UI", 9)).pack(side="left")

        self._phrase_label = tk.Label(row1, text="15 s", bg=self.BG, fg=self.FG,
                                      font=("Segoe UI", 9, "bold"), width=5)
        self._phrase_label.pack(side="left", padx=(0, 4))

        tk.Scale(
            row1, variable=self._max_phrase_var,
            from_=5, to=60, resolution=1,
            orient="horizontal", length=160,
            bg=self.BG, fg=self.FG, troughcolor=self.SURFACE,
            highlightthickness=0, showvalue=False,
            command=self._on_phrase_change,
        ).pack(side="left")

        # ── Row 2: sensitivity ────────────────────────────────
        row2 = tk.Frame(panel, bg=self.BG)
        row2.pack(fill="x")

        tk.Checkbutton(
            row2, text="Auto sensitivity", variable=self._auto_energy,
            bg=self.BG, fg=self.FG_DIM, selectcolor=self.SURFACE,
            activebackground=self.BG, activeforeground=self.FG,
            font=("Segoe UI", 9),
            command=self._on_auto_energy_toggle,
        ).pack(side="left")

        tk.Label(row2, text="  Manual threshold:", bg=self.BG, fg=self.FG_DIM,
                 font=("Segoe UI", 9)).pack(side="left", padx=(8, 0))

        self._energy_label = tk.Label(row2, text="300", bg=self.BG, fg=self.FG_DIM,
                                      font=("Segoe UI", 9, "bold"), width=5)
        self._energy_label.pack(side="left", padx=(0, 4))

        self._energy_scale = tk.Scale(
            row2, variable=self._energy_var,
            from_=50, to=3000, resolution=50,
            orient="horizontal", length=180,
            bg=self.BG, fg=self.FG_DIM, troughcolor=self.SURFACE,
            highlightthickness=0, showvalue=False,
            state="disabled",
            command=self._on_energy_change,
        )
        self._energy_scale.pack(side="left")

        # Separator
        tk.Frame(panel, bg=self.SURFACE, height=1).pack(fill="x", pady=(6, 0))

    # ── Live indicator ────────────────────────────────────────

    def _build_indicator(self) -> None:
        row = tk.Frame(self.root, bg=self.BG, pady=6)
        row.pack(fill="x", padx=16)

        self._indicator = tk.Canvas(row, width=14, height=14,
                                    bg=self.BG, highlightthickness=0)
        self._indicator.pack(side="left")
        self._dot = self._indicator.create_oval(2, 2, 12, 12, fill=_IDLE_COL, outline="")

        self._indicator_label = tk.Label(row, text="Idle", bg=self.BG, fg=self.FG_DIM,
                                         font=("Segoe UI", 9))
        self._indicator_label.pack(side="left", padx=(6, 0))

    # ── Transcript ────────────────────────────────────────────

    def _build_transcript(self) -> None:
        outer = tk.Frame(self.root, bg=self.BG)
        outer.pack(fill="both", expand=True, padx=16, pady=(0, 6))

        box = tk.Frame(outer, bg=self.SURFACE)
        box.pack(fill="both", expand=True)

        self.transcript = tk.Text(
            box, bg=self.SURFACE, fg=self.FG,
            font=("Segoe UI", 11), relief="flat",
            wrap="word", padx=12, pady=10,
            insertbackground=self.FG, state="disabled",
        )
        sb = tk.Scrollbar(box, command=self.transcript.yview,
                          bg=self.SURFACE, troughcolor=self.BG)
        self.transcript.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.transcript.pack(side="left", fill="both", expand=True)

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self.root, bg=self.SURFACE, height=26)
        bar.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Ready — select a mic and press Start.")
        tk.Label(bar, textvariable=self.status_var, bg=self.SURFACE, fg=self.FG_DIM,
                 font=("Segoe UI", 9), anchor="w", padx=12).pack(side="left")
        self._words_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self._words_var, bg=self.SURFACE, fg=self.FG_DIM,
                 font=("Segoe UI", 9), anchor="e", padx=12).pack(side="right")

    # ── Device helpers ────────────────────────────────────────

    def _populate_devices(self) -> None:
        mics = list_microphones()
        if not mics:
            return
        menu = self.device_menu["menu"]
        menu.delete(0, "end")
        menu.add_command(label="System Default",
                         command=lambda: self.device_var.set("System Default"))
        for mic in mics:
            label = f"{mic.name[:38]}  [idx {mic.index}]"
            menu.add_command(label=label,
                             command=lambda v=label: self.device_var.set(v))

    def _selected_mic(self) -> sr.Microphone:
        val = self.device_var.get()
        if val == "System Default":
            return sr.Microphone()
        idx = int(val.rsplit("idx ", 1)[-1].rstrip("]"))
        return sr.Microphone(device_index=idx)

    # ── Settings callbacks ────────────────────────────────────

    def _on_pause_change(self, _=None) -> None:
        v = self._pause_var.get()
        self._pause_label.config(text=f"{v:.1f} s")
        # Apply immediately if running
        self._r.pause_threshold = v
        self._r.non_speaking_duration = min(0.3, v * 0.35)

    def _on_phrase_change(self, _=None) -> None:
        self._phrase_label.config(text=f"{self._max_phrase_var.get()} s")

    def _on_energy_change(self, _=None) -> None:
        v = self._energy_var.get()
        self._energy_label.config(text=str(v))
        if not self._auto_energy.get():
            self._r.energy_threshold = v
            self._r.dynamic_energy_threshold = False

    def _on_auto_energy_toggle(self) -> None:
        auto = self._auto_energy.get()
        self._energy_scale.config(state="disabled" if auto else "normal")
        self._energy_label.config(fg=self.FG_DIM if auto else self.FG)
        self._r.dynamic_energy_threshold = auto
        if not auto:
            self._r.energy_threshold = self._energy_var.get()

    # ── Start / Stop ─────────────────────────────────────────

    def _toggle(self) -> None:
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        self._running = True
        self.toggle_btn.config(text="■  Stop Listening", bg=self.RED)
        self.device_menu.config(state="disabled")
        self.status_var.set("Calibrating for background noise...")
        self._set_indicator("pulse", _PULSE_LISTEN, "Calibrating...")
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def _stop(self) -> None:
        self._running = False
        self.toggle_btn.config(text="▶  Start Listening", bg=self.ACCENT)
        self.device_menu.config(state="normal")
        self.status_var.set("Stopped.")
        self._set_indicator("idle")

    # ── Core listen loop (background thread) ─────────────────

    def _listen_loop(self) -> None:
        """
        Keeps the mic stream open and calls r.listen() in a tight loop.

        pause_threshold  — seconds of silence that end a phrase (key setting)
        timeout          — how long to wait for speech to START before looping
                           back to check self._running (not the silence gap)
        phrase_time_limit — hard cap on a single phrase
        """
        try:
            mic = self._selected_mic()
            with mic as source:
                # One-time calibration
                self._r.adjust_for_ambient_noise(source, duration=1)
                self._r.dynamic_energy_threshold = self._auto_energy.get()
                if not self._auto_energy.get():
                    self._r.energy_threshold = self._energy_var.get()

                self.root.after(0, lambda: (
                    self.status_var.set("Listening — speak whenever you're ready"),
                    self._set_indicator("pulse", _PULSE_LISTEN, "Listening"),
                ))

                while self._running:
                    # Re-read slider values on every iteration (user may adjust them)
                    self._r.pause_threshold = self._pause_var.get()
                    self._r.non_speaking_duration = min(0.3, self._pause_var.get() * 0.35)

                    try:
                        # timeout=1 → if no speech starts within 1 s, loop back
                        # (lets us check self._running without a long block)
                        audio = self._r.listen(
                            source,
                            timeout=1,
                            phrase_time_limit=self._max_phrase_var.get(),
                        )
                    except sr.WaitTimeoutError:
                        continue  # no speech, check _running and try again

                    if not self._running:
                        break

                    # Switch indicator to "processing" while we await the API
                    self.root.after(0, lambda: (
                        self._set_indicator("pulse", _PULSE_PROC, "Processing..."),
                        self.status_var.set("Processing..."),
                    ))

                    # Transcribe in yet another thread so we keep listening
                    threading.Thread(
                        target=self._transcribe, args=(audio,), daemon=True
                    ).start()

        except Exception as exc:
            if self._running:
                self.root.after(0, lambda e=exc: (
                    messagebox.showerror("Mic error", str(e), parent=self.root),
                    self._stop(),
                ))

    # ── Transcription (background thread) ────────────────────

    def _transcribe(self, audio: sr.AudioData) -> None:
        try:
            text = self._r.recognize_google(audio)
        except sr.UnknownValueError:
            self.root.after(0, lambda: (
                self._set_indicator("pulse", _PULSE_LISTEN, "Listening"),
                self.status_var.set("Couldn't understand that — listening again..."),
            ))
            return
        except sr.RequestError as exc:
            self.root.after(0, lambda e=exc: self.status_var.set(f"API error: {e}"))
            return

        ts = datetime.now().strftime("%H:%M:%S")
        self.root.after(0, lambda t=text, s=ts: (
            self._append(t, s),
            self._set_indicator("pulse", _PULSE_LISTEN, "Listening"),
            self.status_var.set("Listening — speak whenever you're ready"),
        ))

    # ── Transcript helpers ────────────────────────────────────

    def _append(self, text: str, ts: str) -> None:
        self.transcript.config(state="normal")
        if self.transcript.index("end-1c") != "1.0":
            self.transcript.insert("end", "\n")
        self.transcript.insert("end", f"[{ts}]  {text}")
        self.transcript.see("end")
        self.transcript.config(state="disabled")
        # Update word count in status bar
        content = self.transcript.get("1.0", "end-1c")
        wc = len(content.split())
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

    # ── Pulse indicator animation ─────────────────────────────

    def _set_indicator(self, mode: str,
                       colors: tuple[str, str] | None = None,
                       label: str = "") -> None:
        if self._pulse_after:
            self.root.after_cancel(self._pulse_after)
            self._pulse_after = None

        if mode == "idle":
            self._indicator.itemconfig(self._dot, fill=_IDLE_COL)
            self._indicator_label.config(text="Idle", fg=self.FG_DIM)
        elif mode == "pulse" and colors:
            self._pulse_colors = colors
            self._pulse_phase  = 0
            self._indicator_label.config(text=label, fg=self.FG)
            self._animate_pulse()

    def _animate_pulse(self) -> None:
        if not self._running:
            return
        col = self._pulse_colors[self._pulse_phase % 2]
        self._indicator.itemconfig(self._dot, fill=col)
        self._pulse_phase += 1
        self._pulse_after = self.root.after(600, self._animate_pulse)

    # ── Close ─────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    LiveSTTWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()


import sys
import threading
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from pathlib import Path

import speech_recognition as sr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utilities.microphone_detector import list_microphones


class LiveSTTWindow:
    BG      = "#1e1e2e"
    SURFACE = "#2a2a3e"
    ACCENT  = "#7c6af7"
    FG      = "#cdd6f4"
    FG_DIM  = "#6c7086"
    RED     = "#f38ba8"

    def __init__(self, root: tk.Tk) -> None:
        self.root      = root
        self._running  = False
        self._r        = sr.Recognizer()
        self._stop_fn  = None   # returned by listen_in_background

        self._configure_root()
        self._build_controls()
        self._build_transcript()
        self._build_statusbar()
        self._populate_devices()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Layout ────────────────────────────────────────────────

    def _configure_root(self) -> None:
        self.root.title("Live Speech to Text")
        self.root.geometry("700x520")
        self.root.minsize(500, 380)
        self.root.configure(bg=self.BG)
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"700x520+{(sw - 700) // 2}+{(sh - 520) // 2}")

    def _build_controls(self) -> None:
        ctrl = tk.Frame(self.root, bg=self.SURFACE, pady=10)
        ctrl.pack(fill="x")

        tk.Label(ctrl, text="Microphone:", bg=self.SURFACE, fg=self.FG,
                 font=("Segoe UI", 10)).pack(side="left", padx=(16, 4))

        self.device_var = tk.StringVar(value="System Default")
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
        self.device_menu.pack(side="left", padx=(0, 12))

        self.toggle_btn = tk.Button(
            ctrl, text="▶  Start Listening",
            command=self._toggle,
            bg=self.ACCENT, fg="#fff",
            font=("Segoe UI", 10, "bold"),
            relief="flat", padx=16, pady=6, cursor="hand2",
            activebackground="#6c5ce7", activeforeground="#fff",
        )
        self.toggle_btn.pack(side="left", padx=4)

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

    def _build_transcript(self) -> None:
        outer = tk.Frame(self.root, bg=self.BG)
        outer.pack(fill="both", expand=True, padx=16, pady=(10, 6))

        tk.Label(outer, text="Transcript", bg=self.BG, fg=self.FG_DIM,
                 font=("Segoe UI", 9)).pack(anchor="w")

        box = tk.Frame(outer, bg=self.SURFACE)
        box.pack(fill="both", expand=True, pady=(4, 0))

        self.transcript = tk.Text(
            box, bg=self.SURFACE, fg=self.FG,
            font=("Segoe UI", 11), relief="flat",
            wrap="word", padx=12, pady=10,
            insertbackground=self.FG,
            state="disabled",
        )
        sb = tk.Scrollbar(box, command=self.transcript.yview,
                          bg=self.SURFACE, troughcolor=self.BG)
        self.transcript.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.transcript.pack(side="left", fill="both", expand=True)

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self.root, bg=self.SURFACE, height=26)
        bar.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Ready — select a mic and press Start.")
        tk.Label(bar, textvariable=self.status_var, bg=self.SURFACE, fg=self.FG_DIM,
                 font=("Segoe UI", 9), anchor="w", padx=12).pack(side="left")

    # ── Devices ───────────────────────────────────────────────

    def _populate_devices(self) -> None:
        mics = list_microphones()
        if not mics:
            return
        menu = self.device_menu["menu"]
        menu.delete(0, "end")
        menu.add_command(label="System Default",
                         command=lambda: self.device_var.set("System Default"))
        for mic in mics:
            label = f"{mic.name[:38]}  [idx {mic.index}]"
            menu.add_command(label=label,
                             command=lambda v=label: self.device_var.set(v))

    def _selected_mic(self) -> sr.Microphone:
        val = self.device_var.get()
        if val == "System Default":
            return sr.Microphone()
        idx = int(val.rsplit("idx ", 1)[-1].rstrip("]"))
        return sr.Microphone(device_index=idx)

    # ── Listening ─────────────────────────────────────────────

    def _toggle(self) -> None:
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        mic = self._selected_mic()
        self.status_var.set("Calibrating for background noise...")
        self.toggle_btn.config(state="disabled")

        def _calibrate():
            try:
                with mic as source:
                    self._r.adjust_for_ambient_noise(source, duration=1)
                self._stop_fn = self._r.listen_in_background(
                    mic, self._on_audio, phrase_time_limit=10
                )
                self._running = True
                self.root.after(0, lambda: (
                    self.toggle_btn.config(
                        text="■  Stop Listening", bg=self.RED, state="normal"),
                    self.device_menu.config(state="disabled"),
                    self.status_var.set("Listening...  speak whenever you're ready."),
                ))
            except Exception as exc:
                self.root.after(0, lambda: (
                    messagebox.showerror("Error",
                                         f"Could not open microphone:\n{exc}",
                                         parent=self.root),
                    self.toggle_btn.config(state="normal"),
                    self.status_var.set("Failed to start."),
                ))

        threading.Thread(target=_calibrate, daemon=True).start()

    def _stop(self) -> None:
        self._running = False
        if self._stop_fn:
            self._stop_fn(wait_for_stop=False)
            self._stop_fn = None
        self.toggle_btn.config(text="▶  Start Listening", bg=self.ACCENT)
        self.device_menu.config(state="normal")
        self.status_var.set("Stopped.")

    def _on_audio(self, recognizer: sr.Recognizer, audio: sr.AudioData) -> None:
        """Background thread — called once per captured phrase."""
        try:
            text = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return
        except sr.RequestError as exc:
            self.root.after(0, lambda: self.status_var.set(f"API error: {exc}"))
            return
        ts = datetime.now().strftime("%H:%M:%S")
        self.root.after(0, lambda t=text, s=ts: self._append(t, s))

    def _append(self, text: str, ts: str) -> None:
        self.transcript.config(state="normal")
        if self.transcript.index("end-1c") != "1.0":
            self.transcript.insert("end", "\n")
        self.transcript.insert("end", f"[{ts}]  {text}")
        self.transcript.see("end")
        self.transcript.config(state="disabled")
        preview = text[:60] + ("..." if len(text) > 60 else "")
        self.status_var.set(f"Last:  {preview}")

    # ── Toolbar ───────────────────────────────────────────────

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
        self.status_var.set("Transcript cleared.")

    # ── Close ─────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    LiveSTTWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
