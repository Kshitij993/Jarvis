"""
Basic Window — UI Starter Template
Demonstrates how to build a clean, well-structured application window
using Python's built-in tkinter library.  No extra packages required.

Features:
  - Dark-themed styling with plain tkinter (no third-party theme needed)
  - Menu bar with File and Help menus
  - Status bar at the bottom
  - Custom close confirmation dialog
  - Keyboard shortcut: Ctrl+Q to quit
  - Open additional windows from the menu / button

Run:
    python ui/basic_window.py        (from project root)
    python basic_window.py           (from inside ui/)
"""

import tkinter as tk
from tkinter import messagebox


class BasicWindow:
    """A clean, minimal application window template."""

    # ── Colour palette ────────────────────────────────────────
    BG      = "#1e1e2e"   # main background
    SURFACE = "#2a2a3e"   # panel / card background
    ACCENT  = "#7c6af7"   # purple accent
    FG      = "#cdd6f4"   # primary text
    FG_DIM  = "#6c7086"   # muted / secondary text
    RED     = "#e06c75"   # destructive actions

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self._configure_root()
        self._build_menu()
        self._build_body()
        self._build_statusbar()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Control-q>", lambda _e: self._on_close())

    # ── Setup ─────────────────────────────────────────────────

    def _configure_root(self) -> None:
        self.root.title("Robotics — Basic Window")
        self.root.geometry("700x450")
        self.root.minsize(500, 350)
        self.root.configure(bg=self.BG)
        # Centre on screen
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"700x450+{(sw - 700) // 2}+{(sh - 450) // 2}")

    def _build_menu(self) -> None:
        menubar = tk.Menu(
            self.root, bg=self.SURFACE, fg=self.FG,
            activebackground=self.ACCENT, activeforeground="#fff",
            bd=0, relief="flat",
        )

        file_menu = tk.Menu(
            menubar, tearoff=0, bg=self.SURFACE, fg=self.FG,
            activebackground=self.ACCENT, activeforeground="#fff",
        )
        file_menu.add_command(label="New Window",    command=self._new_window)
        file_menu.add_separator()
        file_menu.add_command(label="Quit  Ctrl+Q",  command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(
            menubar, tearoff=0, bg=self.SURFACE, fg=self.FG,
            activebackground=self.ACCENT, activeforeground="#fff",
        )
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _build_body(self) -> None:
        outer = tk.Frame(self.root, bg=self.BG)
        outer.pack(fill="both", expand=True, padx=24, pady=(18, 6))

        card = tk.Frame(outer, bg=self.SURFACE)
        card.pack(fill="both", expand=True)
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)

        tk.Label(
            card, text="Robotics  UI  Toolkit",
            font=("Segoe UI", 22, "bold"),
            bg=self.SURFACE, fg=self.ACCENT,
        ).grid(row=0, column=0, pady=(36, 4))

        tk.Label(
            card,
            text=(
                "This is the basic window template.\n\n"
                "Open  camera_feed.py  to display a live webcam feed.\n"
                "Open  audio_input.py  to monitor microphone levels."
            ),
            font=("Segoe UI", 11), bg=self.SURFACE, fg=self.FG,
            justify="center",
        ).grid(row=1, column=0)

        btn_frame = tk.Frame(card, bg=self.SURFACE)
        btn_frame.grid(row=2, column=0, pady=(20, 36))

        self._btn(btn_frame, "New Window", self._new_window).pack(side="left", padx=8)
        self._btn(btn_frame, "About",      self._show_about).pack(side="left", padx=8)
        self._btn(btn_frame, "Quit",       self._on_close, colour=self.RED).pack(side="left", padx=8)

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self.root, bg=self.SURFACE, height=26)
        bar.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            bar, textvariable=self.status_var,
            bg=self.SURFACE, fg=self.FG_DIM,
            font=("Segoe UI", 9), anchor="w", padx=12,
        ).pack(side="left", fill="y")
        tk.Label(
            bar, text="Robotics Project  •  Python / tkinter",
            bg=self.SURFACE, fg=self.FG_DIM,
            font=("Segoe UI", 9), anchor="e", padx=12,
        ).pack(side="right", fill="y")

    # ── Widget helpers ────────────────────────────────────────

    def _btn(self, parent: tk.Widget, text: str, command, colour: str = "") -> tk.Button:
        c = colour or self.ACCENT
        btn = tk.Button(
            parent, text=text, command=command,
            bg=c, fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0,
            padx=18, pady=8, cursor="hand2",
            activebackground="#6c5ce7", activeforeground="#ffffff",
        )
        btn.bind("<Enter>", lambda _e, b=btn: b.config(bg="#6c5ce7"))
        btn.bind("<Leave>", lambda _e, b=btn, co=c: b.config(bg=co))
        return btn

    # ── Actions ───────────────────────────────────────────────

    def _new_window(self) -> None:
        BasicWindow(tk.Toplevel(self.root))
        self.status_var.set("Opened a new window.")

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About",
            "Robotics UI Toolkit\n\n"
            "Basic window template built with Python's built-in tkinter.\n"
            "Part of the Robotics project.",
            parent=self.root,
        )

    def _on_close(self) -> None:
        if messagebox.askokcancel("Quit", "Close this window?", parent=self.root):
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    BasicWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
