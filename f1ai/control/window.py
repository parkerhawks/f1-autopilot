"""Locate the F1 25 window, so capture grabs the game and nothing else.

Capturing the whole screen assumes the game owns the whole screen. When it does
not -- a borderless window that is not maximised, a second monitor, a launcher
sitting alongside -- the frames fed to the CNN contain desktop, and the network
spends its capacity modelling a file browser.

That is not hypothetical: the first in-game capture on this project had roughly
half the frame taken up by another application's UI, and the policy could not
possibly have learned to drive from it.

Finding the window by title and capturing its CLIENT area (which excludes any
border and title bar) removes the whole class of problem, and keeps working
when the window is moved or resized between sessions.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

# Substrings tried in order. F1 titles have varied across releases, so several
# are checked rather than assuming one exact string.
DEFAULT_TITLES = ("F1 25", "F1® 25", "F1 24", "F1")


class _RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


class _POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


def _enum_windows() -> list[tuple[int, str]]:
    """Every visible top-level window, as (hwnd, title)."""
    user32 = ctypes.windll.user32
    out: list[tuple[int, str]] = []

    proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND,
                                   wintypes.LPARAM)

    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        out.append((hwnd, buf.value))
        return True

    user32.EnumWindows(proc_type(callback), 0)
    return out


def list_windows() -> list[str]:
    """Visible window titles, for when auto-detection picks the wrong one."""
    return [t for _, t in _enum_windows() if t.strip()]


def find_game_region(titles: tuple[str, ...] = DEFAULT_TITLES
                     ) -> tuple[int, int, int, int] | None:
    """(left, top, right, bottom) of the game's client area in screen pixels.

    Returns None if no matching window is found, so the caller can fall back to
    full-screen capture rather than crashing -- but it should say so loudly,
    because full-screen capture is what produced the desktop-in-frame bug.
    """
    if sys.platform != "win32":
        return None

    user32 = ctypes.windll.user32
    windows = _enum_windows()

    hwnd = None
    for needle in titles:
        for h, title in windows:
            if needle.lower() in title.lower():
                hwnd = h
                break
        if hwnd:
            break
    if hwnd is None:
        return None

    # Client area only: GetWindowRect would include the border and title bar,
    # putting a strip of window chrome into every training frame.
    rect = _RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None

    origin = _POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None

    left, top = origin.x, origin.y
    right, bottom = left + rect.right, top + rect.bottom
    if right - left < 320 or bottom - top < 240:
        return None            # minimised or a stray match
    return (left, top, right, bottom)


def describe() -> str:
    region = find_game_region()
    if region is None:
        return "no F1 window found"
    l, t, r, b = region
    return f"F1 window client area {r - l}x{b - t} at ({l}, {t})"
