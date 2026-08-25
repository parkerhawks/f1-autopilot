"""One-way telemetry bus between the training process and the overlay.

The overlay runs in its own process and receives snapshots over loopback UDP.
That isolation is the whole point: a GUI redraw, a stalled window manager or an
outright crash in the overlay must never add latency to the 30 Hz control loop,
and at 30 Hz a single dropped frame of *display* costs nothing. UDP's
unreliability is a feature here -- the publisher never blocks, never retries,
and never cares whether anyone is listening.

Snapshots are JSON so the schema stays inspectable with any UDP dump tool.
The camera frame is carried as a base64 PNG at a reduced rate, because it is
by far the largest field and the eye cannot use 30 Hz of it anyway.
"""

from __future__ import annotations

import base64
import io
import json
import socket
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

HUD_PORT = 20779
_MAX_DATAGRAM = 60000  # comfortably under the 65507-byte UDP payload ceiling


@dataclass
class HudSnapshot:
    """Everything the overlay can display. All fields optional-by-default so
    the schema can grow without breaking a running overlay."""

    mode: str = "DEMO"              # DEMO | TRAIN | EVAL

    # Driving
    speed_kph: float = 0.0
    gear: int = 0
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0

    # Lap timing
    current_lap_s: float = 0.0
    last_lap_s: float | None = None
    best_lap_s: float | None = None
    lap_times: list[float] = field(default_factory=list)

    # Reward
    step_reward: float = 0.0
    episode_return: float = 0.0

    # Training
    env_steps: int = 0
    grad_steps: int = 0
    episodes: int = 0
    buffer_fill: float = 0.0        # 0..1
    actor_loss: float | None = None
    critic_loss: float | None = None
    alpha: float | None = None

    # What the network sees (base64 PNG); set via attach_frame()
    frame_png: str | None = None

    def attach_frame(self, frame: np.ndarray) -> None:
        """Encode a (H, W) uint8 greyscale frame for transport."""
        buf = io.BytesIO()
        Image.fromarray(frame, mode="L").save(buf, format="PNG", optimize=False)
        self.frame_png = base64.b64encode(buf.getvalue()).decode("ascii")

    def decode_frame(self) -> np.ndarray | None:
        if not self.frame_png:
            return None
        raw = base64.b64decode(self.frame_png)
        return np.array(Image.open(io.BytesIO(raw)).convert("L"))


# Windows returns WSAECONNRESET on a UDP socket after the peer's port replies
# with ICMP Port Unreachable -- which happens constantly here, because training
# normally starts before anyone opens the overlay. Left alone it poisons the
# sending socket and every later datagram fails. This ioctl suppresses it.
_SIO_UDP_CONNRESET = 0x9800000C


def _ignore_icmp_port_unreachable(sock: socket.socket) -> None:
    if hasattr(socket, "SIO_UDP_CONNRESET") or hasattr(sock, "ioctl"):
        try:
            sock.ioctl(_SIO_UDP_CONNRESET, False)
        except (AttributeError, OSError, ValueError):
            pass  # not Windows, or not supported; harmless either way


class HudPublisher:
    """Fire-and-forget sender. Never raises into the caller's control loop."""

    def __init__(self, host: str = "127.0.0.1", port: int = HUD_PORT):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _ignore_icmp_port_unreachable(self._sock)
        self.dropped = 0
        self.sent = 0

    def publish(self, snap: HudSnapshot) -> None:
        try:
            payload = json.dumps(asdict(snap)).encode("utf-8")
            if len(payload) > _MAX_DATAGRAM:
                # Too big almost always means the frame; drop it rather than
                # the numbers, which are what the display is actually for.
                snap.frame_png = None
                payload = json.dumps(asdict(snap)).encode("utf-8")
            self._sock.sendto(payload, self._addr)
            self.sent += 1
        except OSError:
            self.dropped += 1

    def close(self) -> None:
        self._sock.close()


class HudSubscriber:
    """Receives snapshots, keeping only the newest."""

    # Bound on how many datagrams one drain will consume. Without a cap, a
    # publisher that outpaces the reader keeps the drain loop permanently fed
    # and latest() never returns -- which hangs the overlay's Tk callback and
    # freezes the window on its first frame.
    _MAX_DRAIN = 256

    def __init__(self, host: str = "127.0.0.1", port: int = HUD_PORT,
                 timeout: float | None = None):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        _ignore_icmp_port_unreachable(self._sock)
        self._sock.bind((host, port))
        # Non-blocking, always. A timeout here would mean "wait for a gap in
        # the stream", and at 30 Hz there is never a gap.
        self._sock.setblocking(False)
        self._last_frame_png: str | None = None

    def latest(self) -> HudSnapshot | None:
        """Newest snapshot available right now, discarding any backlog.

        Returns immediately -- None if nothing has arrived since the last call.
        """
        snap = None
        for _ in range(self._MAX_DRAIN):
            try:
                data = self._sock.recv(65535)
            except (BlockingIOError, InterruptedError):
                break                      # nothing left queued
            except OSError:
                break
            try:
                snap = HudSnapshot(**json.loads(data.decode("utf-8")))
            except (ValueError, TypeError):
                continue  # schema drift from an older publisher; skip it

        if snap is not None:
            # The camera image rides on only every Nth snapshot to keep
            # datagrams small, so most snapshots arrive frameless. Carrying the
            # last one forward here -- rather than in each consumer -- stops
            # the view strobing between the image and "no signal".
            if snap.frame_png is None:
                snap.frame_png = self._last_frame_png
            else:
                self._last_frame_png = snap.frame_png
        return snap

    def close(self) -> None:
        self._sock.close()
