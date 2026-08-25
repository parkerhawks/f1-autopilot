"""Virtual Xbox 360 pad, so the policy's outputs reach F1 25 as analog input.

F1 25 takes no programmatic input.  vgamepad drives the ViGEm bus driver, which
presents a synthetic controller at the HID layer -- indistinguishable to the
game from a real pad, and crucially *analog*, which keyboard emulation is not.
Analog matters: steering is a continuous action and quantising it to
left/right/none makes the imitation problem unlearnable.

Requires the ViGEmBus driver (bundled with the vgamepad installer).
"""

from __future__ import annotations

import time

try:
    import vgamepad as vg
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "vgamepad is not installed.  pip install vgamepad\n"
        "The installer also sets up the ViGEmBus driver; accept the prompt."
    ) from e

# Xbox stick is int16; triggers are uint8.
_STICK_MAX = 32767
_TRIGGER_MAX = 255


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class VirtualPad:
    """Analog control surface matching the F1 telemetry convention.

    steer:    -1.0 full left .. +1.0 full right
    throttle:  0.0 .. 1.0
    brake:     0.0 .. 1.0

    These are deliberately the same ranges the game *reports* in
    CarTelemetryData, so recorded human input can be replayed through this
    class unchanged -- which is what makes the Phase 2 sanity check possible.
    """

    def __init__(self) -> None:
        self._pad = vg.VX360Gamepad()
        self._last = (0.0, 0.0, 0.0)
        # Give the driver a moment to enumerate before the first write.
        time.sleep(0.5)
        self.neutral()

    def set(self, steer: float, throttle: float, brake: float) -> None:
        steer = _clamp(steer, -1.0, 1.0)
        throttle = _clamp(throttle, 0.0, 1.0)
        brake = _clamp(brake, 0.0, 1.0)

        self._pad.left_joystick(
            x_value=int(steer * _STICK_MAX), y_value=0
        )
        self._pad.right_trigger(value=int(throttle * _TRIGGER_MAX))
        self._pad.left_trigger(value=int(brake * _TRIGGER_MAX))
        self._pad.update()
        self._last = (steer, throttle, brake)

    def neutral(self) -> None:
        """Centre steering, release both pedals.  Call this on any exit path."""
        self.set(0.0, 0.0, 0.0)

    def press_button(self, button, hold: float = 0.08) -> None:
        """Tap a face button -- used later for menu navigation on reset."""
        self._pad.press_button(button=button)
        self._pad.update()
        time.sleep(hold)
        self._pad.release_button(button=button)
        self._pad.update()

    @property
    def last(self) -> tuple[float, float, float]:
        return self._last

    def close(self) -> None:
        # Leaving a virtual pad holding full throttle is a genuinely bad time.
        try:
            self.neutral()
        finally:
            self._pad = None

    def __enter__(self) -> "VirtualPad":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
