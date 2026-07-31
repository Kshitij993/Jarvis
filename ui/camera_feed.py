"""
Camera Feed Window
Displays a live webcam feed inside a tkinter window.

Features:
  - Camera selection drop-down (auto-detected via utilities/camera_detector)
  - Start / Stop stream toggle button
  - Live FPS counter and frame-resolution display
  - Feed scales to fit the window while keeping aspect ratio
  - Graceful resource cleanup on window close

Run:
    python ui/camera_feed.py        (from project root)
    python camera_feed.py           (from inside ui/)

Extra dependency (beyond requirements.txt): Pillow  — already listed.
"""

import sys
import time
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

import cv2
from PIL import Image, ImageTk

# Allow running from inside ui/ as well as from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utilities.camera_detector import list_cameras


class CameraFeedWindow:
    """Tkinter window that shows a live OpenCV camera feed."""

    BG      = "#1e1e2e"
    SURFACE = "#2a2a3e"
    ACCENT  = "#7c6af7"
    FG      = "#cdd6f4"
    FG_DIM  = "#6c7086"
    RED     = "#e06c75"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self._cap: cv2.VideoCapture | None = None
        self._running   = False
        self._after_id: str | None = None
        self._fps_times: list[float] = []
        self._camera_indices: list[int] = []

        self._configure_root()
        self._build_controls()
        self._build_feed_area()
        self._build_statusbar()
        self._populate_cameras()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Layout ────────────────────────────────────────────────

    def _configure_root(self) -> None:
        self.root.title("Robotics — Camera Feed")
        self.root.geometry("900x620")
        self.root.minsize(640, 480)
        self.root.configure(bg=self.BG)
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"900x620+{(sw - 900) // 2}+{(sh - 620) // 2}")

    def _build_controls(self) -> None:
        ctrl = tk.Frame(self.root, bg=self.SURFACE, pady=10)
        ctrl.pack(fill="x")

        tk.Label(
            ctrl, text="Camera:", bg=self.SURFACE, fg=self.FG,
            font=("Segoe UI", 10),
        ).pack(side="left", padx=(16, 4))

        self.camera_var = tk.StringVar(value="— no cameras detected —")
        self.camera_menu = tk.OptionMenu(ctrl, self.camera_var, "")
        self.camera_menu.config(
            bg=self.SURFACE, fg=self.FG, font=("Segoe UI", 10),
            highlightthickness=0, relief="flat",
            activebackground=self.ACCENT, activeforeground="#fff",
        )
        self.camera_menu["menu"].config(
            bg=self.SURFACE, fg=self.FG,
            activebackground=self.ACCENT, activeforeground="#fff",
        )
        self.camera_menu.pack(side="left", padx=(0, 12))

        self.toggle_btn = tk.Button(
            ctrl, text="▶  Start",
            command=self._toggle_stream,
            bg=self.ACCENT, fg="#fff",
            font=("Segoe UI", 10, "bold"),
            relief="flat", padx=16, pady=6, cursor="hand2",
            activebackground="#6c5ce7", activeforeground="#fff",
        )
        self.toggle_btn.pack(side="left", padx=4)

        # Stats on the right
        self.fps_var = tk.StringVar(value="FPS: —")
        self.res_var = tk.StringVar(value="Resolution: —")
        tk.Label(ctrl, textvariable=self.res_var, bg=self.SURFACE, fg=self.FG_DIM,
                 font=("Segoe UI", 9)).pack(side="right", padx=12)
        tk.Label(ctrl, textvariable=self.fps_var, bg=self.SURFACE, fg=self.FG_DIM,
                 font=("Segoe UI", 9)).pack(side="right", padx=4)

    def _build_feed_area(self) -> None:
        self.feed_frame = tk.Frame(self.root, bg="#000000")
        self.feed_frame.pack(fill="both", expand=True, padx=16, pady=(10, 6))

        self.feed_label = tk.Label(
            self.feed_frame, bg="#000000",
            text="No camera stream.\nSelect a camera and press  ▶ Start.",
            fg=self.FG_DIM, font=("Segoe UI", 12),
        )
        self.feed_label.pack(fill="both", expand=True)

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self.root, bg=self.SURFACE, height=26)
        bar.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Ready — select a camera to begin.")
        tk.Label(
            bar, textvariable=self.status_var,
            bg=self.SURFACE, fg=self.FG_DIM,
            font=("Segoe UI", 9), anchor="w", padx=12,
        ).pack(side="left")

    # ── Camera management ────────────────────────────────────

    def _populate_cameras(self) -> None:
        cameras = list_cameras()
        if not cameras:
            self.status_var.set("No cameras detected.")
            self.toggle_btn.config(state="disabled")
            return

        self._camera_indices = [c.index for c in cameras]
        menu = self.camera_menu["menu"]
        menu.delete(0, "end")
        for cam in cameras:
            label = f"Camera {cam.index}  ({cam.backend})"
            menu.add_command(
                label=label,
                command=lambda v=label: self.camera_var.set(v),
            )
        self.camera_var.set(f"Camera {cameras[0].index}  ({cameras[0].backend})")

    def _selected_index(self) -> int:
        val = self.camera_var.get()
        # Extract the index number from "Camera N  (...)"
        return int(val.split()[1])

    def _toggle_stream(self) -> None:
        if self._running:
            self._stop_stream()
        else:
            self._start_stream()

    def _start_stream(self) -> None:
        idx = self._selected_index()
        cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open camera {idx}.", parent=self.root)
            return
        self._cap = cap
        self._running = True
        self._fps_times.clear()
        self.toggle_btn.config(text="■  Stop", bg=self.RED)
        self.camera_menu.config(state="disabled")
        self.status_var.set(f"Streaming from Camera {idx} ...")
        self._schedule_frame()

    def _stop_stream(self) -> None:
        self._running = False
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        if self._cap:
            self._cap.release()
            self._cap = None
        self.feed_label.config(
            image="",
            text="Stream stopped.\nSelect a camera and press  ▶ Start.",
            fg=self.FG_DIM,
        )
        self.toggle_btn.config(text="▶  Start", bg=self.ACCENT)
        self.camera_menu.config(state="normal")
        self.fps_var.set("FPS: —")
        self.res_var.set("Resolution: —")
        self.status_var.set("Stream stopped.")

    # ── Frame loop ────────────────────────────────────────────

    def _schedule_frame(self) -> None:
        if self._running:
            self._after_id = self.root.after(15, self._update_frame)

    def _update_frame(self) -> None:
        if not self._running or self._cap is None:
            return

        ret, frame = self._cap.read()
        if not ret:
            self._stop_stream()
            self.status_var.set("Camera disconnected.")
            return

        # FPS calculation over a rolling 1-second window
        now = time.time()
        self._fps_times.append(now)
        self._fps_times = [t for t in self._fps_times if now - t <= 1.0]
        self.fps_var.set(f"FPS: {len(self._fps_times)}")

        h, w = frame.shape[:2]
        self.res_var.set(f"Resolution: {w}×{h}")

        # Scale to fit the label while preserving aspect ratio
        lw = self.feed_label.winfo_width()
        lh = self.feed_label.winfo_height()
        if lw > 1 and lh > 1:
            scale = min(lw / w, lh / h)
            frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )

        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.feed_label.config(image=img, text="")
        self.feed_label.image = img   # prevent garbage collection

        self._schedule_frame()

    # ── Close ─────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._stop_stream()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    CameraFeedWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
