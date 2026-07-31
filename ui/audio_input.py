"""
Audio Input Monitor Window
Visualises live microphone input levels inside a tkinter window.

Features:
  - Microphone device selector (auto-detected via utilities/microphone_detector)
  - Colour-coded volume bar:  green (quiet) → yellow (medium) → red (loud)
  - Peak-hold indicator (white marker that decays slowly)
  - Live dBFS readout (decibels relative to full scale)
  - dB scale labels beneath the bar
  - Start / Stop monitoring toggle
  - Audio capture runs in a background thread so the UI stays responsive

Run:
    python ui/audio_input.py        (from project root)
    python audio_input.py           (from inside ui/)

Extra dependency: pyaudio — already listed in requirements.txt.
"""

import sys
import math
import struct
import threading
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

import pyaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utilities.microphone_detector import list_microphones


# ── Audio constants ───────────────────────────────────────────
CHUNK            = 1024   # samples per read
RATE             = 44100  # Hz
CHANNELS         = 1
FORMAT           = pyaudio.paInt16
MAX_INT16        = 32768.0
PEAK_HOLD_FRAMES = 40     # frames before peak marker starts falling


class AudioInputWindow:
    """Tkinter window that visualises live microphone levels."""

    BG      = "#1e1e2e"
    SURFACE = "#2a2a3e"
    ACCENT  = "#7c6af7"
    FG      = "#cdd6f4"
    FG_DIM  = "#6c7086"
    GREEN   = "#a6e3a1"
    YELLOW  = "#f9e2af"
    RED     = "#f38ba8"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self._pa: pyaudio.PyAudio | None     = None
        self._stream: pyaudio.Stream | None  = None
        self._running   = False
        self._thread: threading.Thread | None = None

        # Meter state
        self._bar_frac  = 0.0   # 0.0–1.0
        self._peak_frac = 0.0
        self._peak_timer = 0

        self._configure_root()
        self._build_controls()
        self._build_meter()
        self._build_statusbar()
        self._populate_devices()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Layout ────────────────────────────────────────────────

    def _configure_root(self) -> None:
        self.root.title("Robotics — Audio Input Monitor")
        self.root.geometry("640x400")
        self.root.resizable(False, False)
        self.root.configure(bg=self.BG)
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"640x400+{(sw - 640) // 2}+{(sh - 400) // 2}")

    def _build_controls(self) -> None:
        ctrl = tk.Frame(self.root, bg=self.SURFACE, pady=10)
        ctrl.pack(fill="x")

        tk.Label(
            ctrl, text="Microphone:", bg=self.SURFACE, fg=self.FG,
            font=("Segoe UI", 10),
        ).pack(side="left", padx=(16, 4))

        self.device_var = tk.StringVar(value="— no devices —")
        self.device_menu = tk.OptionMenu(ctrl, self.device_var, "")
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
            ctrl, text="▶  Start Monitoring",
            command=self._toggle,
            bg=self.ACCENT, fg="#fff",
            font=("Segoe UI", 10, "bold"),
            relief="flat", padx=16, pady=6, cursor="hand2",
            activebackground="#6c5ce7", activeforeground="#fff",
        )
        self.toggle_btn.pack(side="left", padx=4)

    def _build_meter(self) -> None:
        outer = tk.Frame(self.root, bg=self.BG)
        outer.pack(fill="both", expand=True, padx=30, pady=20)

        # Large dBFS readout
        self.db_var = tk.StringVar(value="— dBFS")
        tk.Label(
            outer, textvariable=self.db_var,
            bg=self.BG, fg=self.FG,
            font=("Segoe UI", 30, "bold"),
        ).pack()

        # Volume bar
        self.canvas = tk.Canvas(outer, bg=self.SURFACE, height=52,
                                highlightthickness=0)
        self.canvas.pack(fill="x", pady=(12, 4))
        self.canvas.bind("<Configure>", lambda _e: self._redraw())

        # dB scale labels
        scale = tk.Frame(outer, bg=self.BG)
        scale.pack(fill="x")
        for lbl in ("-60 dB", "-40 dB", "-20 dB", "-10 dB", "0 dB"):
            tk.Label(scale, text=lbl, bg=self.BG, fg=self.FG_DIM,
                     font=("Segoe UI", 8)).pack(side="left", expand=True)

        # Legend
        legend = tk.Frame(outer, bg=self.BG)
        legend.pack(fill="x", pady=(10, 0))
        for colour, label in [
            (self.GREEN,  "Quiet  (< −18 dB)"),
            (self.YELLOW, "Medium  (−18 to −6 dB)"),
            (self.RED,    "Loud  (> −6 dB)"),
        ]:
            dot = tk.Label(legend, text="●", bg=self.BG, fg=colour,
                           font=("Segoe UI", 11))
            dot.pack(side="left", padx=(12, 2))
            tk.Label(legend, text=label, bg=self.BG, fg=self.FG_DIM,
                     font=("Segoe UI", 9)).pack(side="left", padx=(0, 6))

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self.root, bg=self.SURFACE, height=26)
        bar.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Ready — select a microphone to begin.")
        tk.Label(
            bar, textvariable=self.status_var,
            bg=self.SURFACE, fg=self.FG_DIM,
            font=("Segoe UI", 9), anchor="w", padx=12,
        ).pack(side="left")

    # ── Device management ────────────────────────────────────

    def _populate_devices(self) -> None:
        mics = list_microphones()
        if not mics:
            self.status_var.set("No microphone devices found.")
            self.toggle_btn.config(state="disabled")
            return

        menu = self.device_menu["menu"]
        menu.delete(0, "end")
        for mic in mics:
            label = f"{mic.name[:38]}  [idx {mic.index}]"
            menu.add_command(
                label=label,
                command=lambda v=label: self.device_var.set(v),
            )
        first = mics[0]
        self.device_var.set(f"{first.name[:38]}  [idx {first.index}]")

    def _selected_device_index(self) -> int:
        # Label format: "name  [idx N]"
        val = self.device_var.get()
        return int(val.rsplit("idx ", 1)[-1].rstrip("]"))

    # ── Start / Stop ─────────────────────────────────────────

    def _toggle(self) -> None:
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        idx = self._selected_device_index()
        self._pa = pyaudio.PyAudio()
        try:
            self._stream = self._pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=idx,
                frames_per_buffer=CHUNK,
            )
        except Exception as exc:
            messagebox.showerror(
                "Error", f"Could not open microphone:\n{exc}", parent=self.root
            )
            self._pa.terminate()
            self._pa = None
            return

        self._running    = True
        self._bar_frac   = 0.0
        self._peak_frac  = 0.0
        self._peak_timer = 0
        self.toggle_btn.config(text="■  Stop Monitoring", bg=self.RED)
        self.device_menu.config(state="disabled")
        self.status_var.set(f"Monitoring microphone index {idx} ...")

        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()

    def _stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None
        self.toggle_btn.config(text="▶  Start Monitoring", bg=self.ACCENT)
        self.device_menu.config(state="normal")
        self.db_var.set("— dBFS")
        self._bar_frac  = 0.0
        self._peak_frac = 0.0
        self._redraw()
        self.status_var.set("Monitoring stopped.")

    # ── Audio capture thread ──────────────────────────────────

    def _audio_loop(self) -> None:
        """Background thread: reads audio chunks and schedules UI updates."""
        while self._running and self._stream:
            try:
                data = self._stream.read(CHUNK, exception_on_overflow=False)
            except OSError:
                break

            samples = struct.unpack(f"{CHUNK}h", data)
            rms = math.sqrt(sum(s * s for s in samples) / CHUNK)
            db  = 20 * math.log10(rms / MAX_INT16) if rms > 0 else -60.0
            db  = max(-60.0, min(0.0, db))
            frac = (db + 60.0) / 60.0   # −60 dBFS → 0.0,  0 dBFS → 1.0

            self.root.after(0, self._update_meter, db, frac)

    # ── Meter update (main thread) ────────────────────────────

    def _update_meter(self, db: float, frac: float) -> None:
        if not self._running:
            return
        self.db_var.set(f"{db:+.1f} dBFS")
        self._bar_frac = frac

        # Peak hold
        if frac >= self._peak_frac:
            self._peak_frac  = frac
            self._peak_timer = 0
        else:
            self._peak_timer += 1
            if self._peak_timer > PEAK_HOLD_FRAMES:
                self._peak_frac = max(0.0, self._peak_frac - 0.01)

        self._redraw()

    def _redraw(self) -> None:
        c = self.canvas
        w = c.winfo_width()  or 580
        h = c.winfo_height() or 52
        c.delete("all")

        bw = self._bar_frac  * w
        pw = self._peak_frac * w

        # Background
        c.create_rectangle(0, 0, w, h, fill=self.SURFACE, outline="")

        # Colour zones:  green 0–70 %,  yellow 70–90 %,  red 90–100 %
        for x0, x1, colour in [
            (0,        w * 0.70, self.GREEN),
            (w * 0.70, w * 0.90, self.YELLOW),
            (w * 0.90, w,        self.RED),
        ]:
            fill_x1 = min(bw, x1)
            if fill_x1 > x0:
                c.create_rectangle(x0, 5, fill_x1, h - 5,
                                   fill=colour, outline="")

        # Peak-hold marker
        if pw > 3:
            c.create_rectangle(pw - 3, 2, pw + 3, h - 2,
                               fill="#ffffff", outline="")

        # Zone divider lines
        for pct in (0.70, 0.90):
            xp = int(w * pct)
            c.create_line(xp, 0, xp, h, fill=self.BG, width=2)

    # ── Close ─────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    AudioInputWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
