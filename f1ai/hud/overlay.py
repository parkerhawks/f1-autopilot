"""A click-through, always-on-top window that draws the HUD over F1 25.

Runs as its own process, subscribing to the telemetry bus. Nothing here can
block the control loop.

Three Win32 extended styles make it behave like an overlay rather than an
application window:

  WS_EX_TRANSPARENT   mouse clicks pass straight through to the game
  WS_EX_NOACTIVATE    the window never takes focus -- essential, because F1 25
                      stops accepting gamepad input the moment it loses focus,
                      which would silently kill the agent's control
  WS_EX_TOOLWINDOW    keeps it out of the alt-tab list

IMPORTANT: F1 25 must run in BORDERLESS WINDOWED mode. Exclusive fullscreen
takes ownership of the display surface and no overlay will be visible over it.
That is the same setting the capture path needs, so it is not an extra
constraint -- just one more reason it is non-negotiable.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import tkinter as tk
from ctypes import wintypes

from PIL import ImageTk

from .bus import HUD_PORT, HudSnapshot, HudSubscriber
from .panel import KEY, HudPanel

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080

REDRAW_MS = 50          # 20 Hz is plenty for a display; the loop runs at 30
STALE_AFTER = 60        # redraws with no data before showing "waiting"


def _key_hex() -> str:
    return "#%02x%02x%02x" % KEY


def _make_click_through(root: tk.Tk) -> bool:
    """Apply the overlay extended styles. False if not on Windows."""
    if sys.platform != "win32":
        return False
    try:
        hwnd = wintypes.HWND(int(root.frame(), 16))
        user32 = ctypes.windll.user32
        # 64-bit safe variants; the *W suffix keeps the string handling wide.
        get_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        get_long.restype = ctypes.c_ssize_t
        set_long.restype = ctypes.c_ssize_t
        styles = get_long(hwnd, GWL_EXSTYLE)
        set_long(hwnd, GWL_EXSTYLE,
                 styles | WS_EX_LAYERED | WS_EX_TRANSPARENT
                 | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
        return True
    except Exception:
        return False


class Overlay:
    def __init__(self, x: int = 24, y: int = 24, opacity: float = 0.92,
                 port: int = HUD_PORT, click_through: bool = True):
        self.panel = HudPanel()
        self.sub = HudSubscriber(port=port)
        self.snap = HudSnapshot()
        self.stale = 0
        self._photo = None

        self.root = tk.Tk()
        self.root.title("f1-autopilot HUD")
        self.root.overrideredirect(True)          # no title bar or chrome
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", opacity)
        self.root.attributes("-transparentcolor", _key_hex())
        self.root.geometry(f"+{x}+{y}")
        self.root.configure(bg=_key_hex())

        self.label = tk.Label(self.root, bd=0, highlightthickness=0,
                              bg=_key_hex())
        self.label.pack()

        self.root.update_idletasks()
        self.styled = _make_click_through(self.root) if click_through else False

        # With click-through on, the window cannot be closed by mouse, so bind
        # a key and keep a console message as the escape hatch.
        self.root.bind("<Escape>", lambda _e: self.stop())
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self._running = True

    def _tick(self) -> None:
        if not self._running:
            return
        # The subscriber carries the camera frame forward across the snapshots
        # that omit it, so this only has to worry about liveness.
        snap = self.sub.latest()
        if snap is not None:
            self.snap = snap
            self.stale = 0
        else:
            self.stale += 1

        if self.stale > STALE_AFTER:
            self.snap = HudSnapshot(mode="DEMO")

        img = self.panel.render(self.snap)
        # Hold a reference: Tk does not own PhotoImage memory and will show a
        # blank window if this is garbage collected.
        self._photo = ImageTk.PhotoImage(img)
        self.label.configure(image=self._photo)
        self.root.after(REDRAW_MS, self._tick)

    def run(self) -> None:
        print(f"overlay listening on udp://127.0.0.1:{HUD_PORT}")
        print(f"  click-through: {'on' if self.styled else 'OFF (not applied)'}")
        print("  F1 25 must be in BORDERLESS WINDOWED mode to see this.")
        print("  Ctrl-C here to close.")
        self.root.after(REDRAW_MS, self._tick)
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        self._running = False
        try:
            self.sub.close()
        finally:
            self.root.quit()
            self.root.destroy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-x", type=int, default=24, help="window x position")
    ap.add_argument("-y", type=int, default=24, help="window y position")
    ap.add_argument("--opacity", type=float, default=0.92)
    ap.add_argument("--port", type=int, default=HUD_PORT)
    ap.add_argument("--no-click-through", action="store_true",
                    help="leave the window interactive (useful for debugging)")
    a = ap.parse_args()
    Overlay(a.x, a.y, a.opacity, a.port,
            click_through=not a.no_click_through).run()


if __name__ == "__main__":
    main()
