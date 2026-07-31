"""
LLM Chat UI — OpenBridge AI (personal server)

Reads / writes the same llm/.llm_config.json that Jarvis uses.
No extra libraries needed beyond the standard library.

Run:
    python llm/llm_ui.py
"""

import json
import threading
import tkinter as tk
from pathlib import Path
import urllib.request
import urllib.error

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).parent
CONFIG_FILE = _HERE / ".llm_config.json"

_DEFAULT_URL = "https://openbridgeai.kshitijks.com/api/v1/chat/completions"

# ── Colour palette (matches Jarvis) ──────────────────────────────────────────
BG      = "#1e1e2e"
SURFACE = "#2a2a3e"
ACCENT  = "#7c6af7"
FG      = "#cdd6f4"
FG_DIM  = "#6c7086"
RED     = "#f38ba8"
GREEN   = "#a6e3a1"


class LLMChatWindow:

    def __init__(self, root: tk.Tk) -> None:
        self.root      = root
        self._messages: list[dict] = []
        self._sending  = False

        self._api_url_var  = tk.StringVar(value=_DEFAULT_URL)
        self._api_key_var  = tk.StringVar()
        self._system_var   = tk.StringVar(
            value="You are Jarvis, a smart and friendly AI assistant. Be concise and helpful.")
        self._show_key     = False

        self._load_config()
        self._build_ui()

    # ── Config ────────────────────────────────────────────────────────────────

    def _load_config(self) -> None:
        if CONFIG_FILE.exists():
            try:
                cfg = json.loads(CONFIG_FILE.read_text())
                self._api_url_var.set(cfg.get("api_url", _DEFAULT_URL))
                self._api_key_var.set(cfg.get("api_key", ""))
                self._system_var.set(cfg.get("system_prompt", self._system_var.get()))
            except Exception:
                pass

    def _save_config(self) -> None:
        cfg = {}
        if CONFIG_FILE.exists():
            try:
                cfg = json.loads(CONFIG_FILE.read_text())
            except Exception:
                pass
        cfg.update({
            "provider":      "custom",
            "api_key":       self._api_key_var.get().strip(),
            "api_url":       self._api_url_var.get().strip(),
            "system_prompt": self._system_var.get().strip(),
        })
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
        self._set_status("Config saved.")

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _btn(self, parent, text, cmd, bg=None, fg="#fff",
             font=("Segoe UI", 9, "bold"), **kw) -> tk.Button:
        bg = bg or ACCENT
        b  = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                       font=font, relief="flat", cursor="hand2",
                       activebackground="#6c5ce7", activeforeground="#fff", **kw)
        b.bind("<Enter>", lambda _: b.config(bg="#6c5ce7" if bg == ACCENT else "#3a3a5e"))
        b.bind("<Leave>", lambda _: b.config(bg=bg))
        return b

    def _lbl(self, parent, text, dim=False, bg=None, **kw) -> tk.Label:
        return tk.Label(parent, text=text, bg=bg or BG,
                        fg=FG_DIM if dim else FG,
                        font=("Segoe UI", 9), **kw)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.title("LLM Chat — OpenBridge AI")
        self.root.geometry("860x720")
        self.root.minsize(640, 500)
        self.root.configure(bg=BG)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"860x720+{(sw-860)//2}+{(sh-720)//2}")

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=SURFACE)
        hdr.pack(fill="x")
        tk.Label(hdr, text="LLM Chat", bg=SURFACE, fg=ACCENT,
                 font=("Segoe UI", 14, "bold"), pady=10).pack(side="left", padx=16)
        tk.Label(hdr, text="OpenBridge AI", bg=SURFACE, fg=FG_DIM,
                 font=("Segoe UI", 9)).pack(side="left")

        # ── Settings panel ────────────────────────────────────────────────────
        cfg_frame = tk.Frame(self.root, bg=SURFACE, pady=8)
        cfg_frame.pack(fill="x", padx=0, pady=(0, 2))

        # API URL row
        r1 = tk.Frame(cfg_frame, bg=SURFACE)
        r1.pack(fill="x", padx=14, pady=(2, 4))
        self._lbl(r1, "API URL:", dim=True, bg=SURFACE).pack(side="left")
        tk.Entry(r1, textvariable=self._api_url_var,
                 bg="#313244", fg=FG, insertbackground=FG,
                 relief="flat", font=("Segoe UI", 9), width=60
                 ).pack(side="left", padx=(6, 0), fill="x", expand=True)

        # API Key row
        r2 = tk.Frame(cfg_frame, bg=SURFACE)
        r2.pack(fill="x", padx=14, pady=(0, 4))
        self._lbl(r2, "API Key:", dim=True, bg=SURFACE).pack(side="left")
        self._key_entry = tk.Entry(
            r2, textvariable=self._api_key_var, show="u25cf",
            bg="#313244", fg=FG, insertbackground=FG,
            relief="flat", font=("Segoe UI", 9), width=44)
        self._key_entry.pack(side="left", padx=(6, 6))
        self._btn(r2, "Show", self._toggle_key,
                  bg=SURFACE, fg=FG_DIM, padx=8, pady=3).pack(side="left", padx=(0, 6))
        self._btn(r2, "Save", self._save_config,
                  padx=10, pady=3).pack(side="left")

        # System prompt row
        r3 = tk.Frame(cfg_frame, bg=SURFACE)
        r3.pack(fill="x", padx=14, pady=(0, 6))
        self._lbl(r3, "System:", dim=True, bg=SURFACE).pack(side="left")
        tk.Entry(r3, textvariable=self._system_var,
                 bg="#313244", fg=FG_DIM, insertbackground=FG,
                 relief="flat", font=("Segoe UI", 9)
                 ).pack(side="left", fill="x", expand=True, padx=(6, 0))

        # ── Chat area ─────────────────────────────────────────────────────────
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=14, pady=(6, 0))
        box = tk.Frame(outer, bg=SURFACE)
        box.pack(fill="both", expand=True)
        self._chat = tk.Text(
            box, bg=SURFACE, fg=FG, font=("Segoe UI", 11),
            relief="flat", wrap="word", padx=12, pady=8, state="disabled")
        self._chat.tag_config("you",  foreground=ACCENT, font=("Segoe UI", 11, "bold"))
        self._chat.tag_config("asst", foreground=GREEN,  font=("Segoe UI", 11, "bold"))
        self._chat.tag_config("err",  foreground=RED)
        sb = tk.Scrollbar(box, command=self._chat.yview, bg=SURFACE, troughcolor=BG)
        self._chat.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._chat.pack(side="left", fill="both", expand=True)

        # ── Input row ─────────────────────────────────────────────────────────
        row = tk.Frame(self.root, bg=BG, pady=8)
        row.pack(fill="x", padx=14)
        self._input = tk.Text(
            row, bg=SURFACE, fg=FG, insertbackground=FG,
            relief="flat", font=("Segoe UI", 11),
            height=2, wrap="word", padx=8, pady=6)
        self._input.pack(side="left", fill="x", expand=True)
        self._input.bind("<Return>", self._on_enter)
        btns = tk.Frame(row, bg=BG)
        btns.pack(side="left", padx=(8, 0))
        self._send_btn = self._btn(btns, "Send", self._send, padx=14, pady=7)
        self._send_btn.pack(fill="x", pady=(0, 4))
        self._btn(btns, "New chat", self._new_chat,
                  bg=SURFACE, fg=FG_DIM, padx=10, pady=5).pack(fill="x")

        # ── Status bar ────────────────────────────────────────────────────────
        bar = tk.Frame(self.root, bg=SURFACE, height=22)
        bar.pack(fill="x", side="bottom")
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(bar, textvariable=self._status_var, bg=SURFACE, fg=FG_DIM,
                 font=("Segoe UI", 8), anchor="w", padx=10).pack(side="left")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Key show/hide ─────────────────────────────────────────────────────────

    def _toggle_key(self) -> None:
        self._show_key = not self._show_key
        self._key_entry.config(show="" if self._show_key else "u25cf")

    # ── Chat ──────────────────────────────────────────────────────────────────

    def _on_enter(self, event) -> str:
        if not (event.state & 1):
            self._send()
            return "break"
        return ""

    def _send(self) -> None:
        if self._sending:
            return
        text = self._input.get("1.0", "end-1c").strip()
        if not text:
            return
        self._input.delete("1.0", "end")
        self._append_msg("You", text, "you")
        self._messages.append({"role": "user", "content": text})
        self._sending = True
        self._send_btn.config(state="disabled", text="...")
        self._append_msg("Assistant", "", "asst")
        self._set_status("Thinking...")
        threading.Thread(target=self._run_chat, daemon=True).start()

    def _run_chat(self) -> None:
        api_key = self._api_key_var.get().strip()
        if not api_key:
            self.root.after(0, lambda: self._done_reply(
                "", "No API key — enter it above and click Save."))
            return

        api_url = self._api_url_var.get().strip() or _DEFAULT_URL
        msgs    = ([{"role": "system", "content": self._system_var.get()}]
                   + self._messages)
        payload = json.dumps({
            "messages":          msgs,
            "save_conversation": False,
        }).encode()
        req = urllib.request.Request(
            api_url, data=payload,
            headers={"Content-Type":  "application/json",
                     "Authorization": f"Bearer {api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                full = data["choices"][0]["message"]["content"]
                self.root.after(0, lambda t=full: self._stream(t))
                self.root.after(0, lambda: self._done_reply(full))
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            self.root.after(0, lambda b=body: self._done_reply(
                "", f"HTTP {e.code}: {b[:300]}"))
        except Exception as e:
            self.root.after(0, lambda err=str(e): self._done_reply("", err))

    # ── Chat helpers ──────────────────────────────────────────────────────────

    def _append_msg(self, speaker: str, text: str, tag: str) -> None:
        self._chat.config(state="normal")
        if self._chat.index("end-1c") != "1.0":
            self._chat.insert("end", "\n")
        self._chat.insert("end", f"{speaker}:\n", tag)
        if text:
            self._chat.insert("end", text + "\n")
        self._chat.see("end")
        self._chat.config(state="disabled")

    def _stream(self, chunk: str) -> None:
        self._chat.config(state="normal")
        self._chat.insert("end", chunk)
        self._chat.see("end")
        self._chat.config(state="disabled")

    def _done_reply(self, full: str, error: str = "") -> None:
        self._sending = False
        self._send_btn.config(state="normal", text="Send")
        if error:
            self._chat.config(state="normal")
            self._chat.insert("end", f"\n[Error: {error}]\n", "err")
            self._chat.config(state="disabled")
            self._set_status(f"Error: {error[:80]}")
        else:
            self._chat.config(state="normal")
            self._chat.insert("end", "\n")
            self._chat.config(state="disabled")
            self._messages.append({"role": "assistant", "content": full})
            self._set_status("Ready")

    def _new_chat(self) -> None:
        self._messages.clear()
        self._chat.config(state="normal")
        self._chat.delete("1.0", "end")
        self._chat.config(state="disabled")
        self._set_status("New chat started.")

    def _set_status(self, text: str) -> None:
        self._status_var.set(text)

    def _on_close(self) -> None:
        self._save_config()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    LLMChatWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
