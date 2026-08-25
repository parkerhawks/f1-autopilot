"""Screen capture that avoids dxcam's region parameter.

`dxcam.grab(region=...)` hard-crashes this machine -- exit code 9, an access
violation inside the native layer, with no Python traceback. It fails for any
region at all, including small interior ones, while `grab()` with no region
works perfectly. That is a library bug, not a configuration mistake, and it is
particularly nasty because the process dies without printing anything, so it
looks like the tool never ran.

So the region is never passed down. The full frame is captured and sliced in
numpy instead. Full-screen grab measures 2.47 ms and the slice is a view, so
the workaround is free.

Every capture in the project goes through here, so the quirk is documented and
worked around exactly once.
"""

from __future__ import annotations

import numpy as np


class ScreenCapture:
    """Grabs the screen, optionally cropped to a region, as BGR uint8."""

    def __init__(self, region: tuple[int, int, int, int] | None = None,
                 output_idx: int = 0):
        import dxcam

        self.camera = dxcam.create(output_idx=output_idx, output_color="BGR")
        if self.camera is None:
            raise RuntimeError(
                "dxcam could not open the display. On hybrid-graphics laptops "
                "try a different output_idx."
            )
        self.region = self._sanitise(region)
        self._last: np.ndarray | None = None

    @staticmethod
    def _sanitise(region):
        if region is None:
            return None
        left, top, right, bottom = (int(v) for v in region)
        left, top = max(0, left), max(0, top)
        if right <= left or bottom <= top:
            return None
        return (left, top, right, bottom)

    def grab(self) -> np.ndarray | None:
        """Newest frame, or None if the screen has not changed since the last
        call. Desktop Duplication only produces a frame on change."""
        frame = self.camera.grab()          # never pass region: see module docstring
        if frame is None:
            return None
        if self.region is not None:
            left, top, right, bottom = self.region
            # Clamp to the frame: the window can be moved or the resolution
            # changed between construction and now, and an out-of-range slice
            # would silently return an empty array rather than raising.
            h, w = frame.shape[:2]
            frame = frame[max(0, top):min(h, bottom),
                          max(0, left):min(w, right)]
            if frame.size == 0:
                return None
        self._last = frame
        return frame

    def grab_or_last(self) -> np.ndarray | None:
        """Newest frame, falling back to the previous one.

        The control loop needs a frame every tick; a screen that has not
        changed is not an error, and reusing beats blocking.
        """
        frame = self.grab()
        return frame if frame is not None else self._last

    def close(self) -> None:
        self.camera = None
