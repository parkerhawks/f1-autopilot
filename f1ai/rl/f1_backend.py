"""The F1 25 backend: screen capture + UDP telemetry + virtual gamepad.

Implements the same three methods as `MockBackend`, so `RacingEnv`, the replay
buffer, SAC and the overlay all run against the real game unchanged. That is
the whole point of the boundary -- the learning code has no idea which world it
is in.

UNVERIFIED AGAINST THE REAL GAME. Every byte offset here comes from
`packets.py`, which is proven self-consistent but has never seen F1 25. Run
`tools/probe_telemetry.py` first; if packet sizes disagree with EXPECTED_SIZES,
fix the layouts before anything in this file can be trusted.

Two things differ from the simulator, and both are improvements:

**Off-track is the game's verdict, not ours.** `CarTelemetryData.surfaceType`
classifies the surface under each wheel. That is ground truth and needs no
track map, so there is no centreline of ours to disagree with the game's idea
of where the track is.

**Reset is the hard part.** An RL agent crashes constantly early in training.
If recovery needs a human, in-game training is dead regardless of how good the
algorithm is. `reset()` drives F1 25's Time Trial restart through the virtual
pad, and the button sequence below is a first guess that MUST be verified and
timed against the real menus -- see `tools/probe_reset.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..control.gamepad import VirtualPad
from ..telemetry.listener import TelemetryListener
from ..telemetry.packets import (
    DRIVER_FLYING_LAP, DRIVER_IN_GARAGE, PacketId, parse_car_motion,
    parse_car_telemetry, parse_driver_status, parse_lap_distance,
    parse_lap_invalid, parse_pit_status,
)
from .env import FRAME_H, FRAME_W, BackendState

# surfaceType values that count as racing surface. The F1 spec enumerates
# tarmac as 0; kerbs are a separate code and are legitimately part of the
# racing line, so they are NOT counted as off-track.
TARMAC = 0
KERB = 1
ON_SURFACE = frozenset({TARMAC, KERB})


@dataclass
class ResetSequence:
    """Button presses that put the car back on track, with their timings.

    Filled in by measuring the real menus. Every entry is (button, wait_after).
    Kept as data rather than code so it can be re-timed without touching logic
    -- menu animations differ between machines and change with patches.
    """
    steps: tuple = ()
    settle_seconds: float = 2.0
    name: str = "reset"

    @classmethod
    def time_trial_restart(cls) -> "ResetSequence":
        """Restart Lap -- back to the start/finish line. Verified working."""
        import vgamepad as vg
        return cls(
            steps=(
                (vg.XUSB_BUTTON.XUSB_GAMEPAD_START, 0.6),   # pause
                (vg.XUSB_BUTTON.XUSB_GAMEPAD_A, 0.8),       # "Restart Lap"
                (vg.XUSB_BUTTON.XUSB_GAMEPAD_A, 1.2),       # confirm
            ),
            settle_seconds=3.0,
            name="restart lap",
        )

    @classmethod
    def pit_escape(cls) -> "ResetSequence":
        """Get out of the pit lane: B, then A, A.

        The pit lane needs its own recovery, and getting this wrong is what
        killed a run: the pause menu the normal resets rely on is not reachable
        from behind the pit prompt, so firing a Restart Lap there left the game
        in a state it never came out of.

        Everything else about the standard resets was fine and is unchanged.

        Verify with:
          probe_reset.py --test "b:0.5,a:0.8,a:1.0"
        """
        import vgamepad as vg
        return cls(
            steps=(
                (vg.XUSB_BUTTON.XUSB_GAMEPAD_B, 0.5),
                (vg.XUSB_BUTTON.XUSB_GAMEPAD_A, 0.8),
                (vg.XUSB_BUTTON.XUSB_GAMEPAD_A, 1.0),
            ),
            settle_seconds=2.5,
            name="pit escape",
        )

    @classmethod
    def reset_to_track(cls) -> "ResetSequence":
        """Reset to Track -- back on the racing line WHERE the car went off.

        This is the one that fixes the plateau. Restarting every episode at the
        start/finish line means the agent sees the first 2 km on every one of
        hundreds of episodes and the last 3 km almost never, so it stops
        improving exactly where its data runs out. Resuming from the point of
        failure pushes the visited-state distribution forward around the lap.

        The pause menu ordering is: Restart Lap / Instant Replay / Reset to
        Track, so this is two positions down. VERIFY with
        `probe_reset.py --test "start:0.6,down:0.25,down:0.25,a:0.9"` -- menu
        layouts vary by game mode.
        """
        import vgamepad as vg
        return cls(
            steps=(
                (vg.XUSB_BUTTON.XUSB_GAMEPAD_START, 0.6),
                (vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN, 0.25),
                (vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN, 0.25),
                (vg.XUSB_BUTTON.XUSB_GAMEPAD_A, 0.9),
            ),
            settle_seconds=2.5,
            name="reset to track",
        )


class F1Backend:
    """Live F1 25 as a reinforcement-learning environment backend."""

    # Crop calibrated against F1 25 at Monza, 2026-08-17, cockpit camera at
    # 2560x1369. Verified with tools/tune_crop.py across five driving samples:
    # the band holds the road converging to its vanishing point with barriers
    # framing both sides, and excludes the HUD panels above and the tyre
    # temperature display below. The front tyres remain in the bottom corners
    # on purpose -- they are constant, and they give the policy a fixed
    # reference for where the car sits relative to the road.
    #
    # Re-run tune_crop.py if the camera, resolution or window size changes.
    # These are only DEFAULTS -- every tool takes --crop-top / --crop-frac, so
    # a different setup never requires editing source.
    def __init__(self, capture_region: tuple[int, int, int, int] | None = None,
                 crop_frac: float = 0.26, crop_top: float = 0.38,
                 reset_sequence: ResetSequence | None = None,
                 telemetry_port: int = 20777, create_pad: bool = True,
                 track_map=None):
        import cv2

        from ..control.capture import ScreenCapture

        self._cv2 = cv2
        # Auto-locate the game window unless a region was given explicitly.
        # Full-screen capture silently includes whatever else is on screen,
        # and the first run on this project fed the network half a desktop.
        if capture_region is None:
            from ..control.window import find_game_region
            capture_region = find_game_region()
            if capture_region is None:
                print("WARNING: no F1 window found; capturing the whole "
                      "screen.\n  Anything else on screen will end up in the "
                      "training frames.")
            else:
                l, t, r, b = capture_region
                print(f"capturing F1 window: {r - l}x{b - t} at ({l}, {t})")
        self.region = capture_region
        # Cropping happens in numpy inside ScreenCapture; dxcam's own region
        # parameter crashes the process on this machine. See control/capture.py.
        self.camera = ScreenCapture(region=capture_region)
        self.crop_frac = crop_frac
        self.crop_top = crop_top

        self.telemetry = TelemetryListener(port=telemetry_port).start()
        if not self.telemetry.wait_for_data(timeout=30.0):
            raise RuntimeError(
                "No UDP telemetry. Check Settings > Telemetry Settings: "
                "UDP on, 127.0.0.1:20777, format 2025."
            )

        # Recording human laps must NOT create a virtual pad: a second
        # controller present alongside the real one can capture the game's
        # input focus, and the human silently stops being able to drive.
        self.pad = VirtualPad() if create_pad else None
        if create_pad:
            import vgamepad as vg
            self._A = vg.XUSB_BUTTON.XUSB_GAMEPAD_A
        self.reset_sequence = reset_sequence or ResetSequence.time_trial_restart()
        # Mostly resume where the car failed, so experience spreads around the
        # lap; occasionally restart from the line, so full clean laps still get
        # driven and lap times remain measurable. All-resume would never
        # produce a timed lap; all-restart is what caused the plateau.
        self.resume_sequence = ResetSequence.reset_to_track()
        self.pit_sequence = ResetSequence.pit_escape()
        # 1 = always Restart Lap (most reliable, but every episode starts at
        # the line, which is what caused the plateau). Higher values mix in
        # Reset to Track to spread experience around the lap. Start at 1 until
        # resets are proven stable, then raise it.
        self.restart_every = 1
        self._resets = 0
        self._failed_stops = 0
        self._in_pit_recovery = False
        self._pit_escapes = 0
        self._menu_escapes = 0
        self.last_escape = "-"
        self.stop_seconds = 0.0
        self.last_reset_kind = "-"

        self._last_frame = np.zeros((FRAME_H, FRAME_W), np.uint8)
        self._track_length = 5000.0   # replaced by the first LapData reading
        self._last_frame_id = -1
        self._stale_since: float | None = None
        self._prev_lap_distance = 0.0
        self._garage_steps = 0
        # Held here as well as in the env, because the offset has to be
        # computed at observation time from the world position.
        self.track_map = track_map

    # -- Backend protocol --------------------------------------------------

    def reset(self) -> None:
        """Put the car back on track without human help.

        Releases the controls first: leaving full throttle applied while menus
        animate is how a "reset" ends with the car already in a wall.
        """
        if self.pad is None:
            raise RuntimeError("backend was built without a pad (create_pad=False)")
        self.pad.neutral()
        self._resets += 1

        # UNIVERSAL PRELUDE, RUN BEFORE EVERY RESET.
        #
        # Three runs died trying to detect which stopped state the car was in
        # -- pit lane, garage, some menu -- and then firing a matching
        # sequence. Detection kept being wrong (pitStatus never moves in Time
        # Trial) and mis-aimed sequences wedged the game somewhere it never
        # came back from.
        #
        # So stop detecting. Pausing, then pressing A three times, clears the
        # garage screen and dismisses any confirmation dialog, and does nothing
        # harmful when the car is simply stopped on track. Running it
        # unconditionally means the reset works from every state without ever
        # needing to know which one it was in.
        time.sleep(self.PRELUDE_WAIT_S)
        for _ in range(3):
            self.pad.press_button(self._A)
            time.sleep(self.PRELUDE_GAP_S)
        time.sleep(0.4)

        if self._in_pit_recovery:
            self._in_pit_recovery = False
            self._pit_escapes += 1

        use_restart = (self.restart_every > 0
                       and self._resets % self.restart_every == 0)

        seq = self.reset_sequence
        if not use_restart:
            # "Reset to Track" only appears in the pause menu once the car is
            # stationary, so it has to be stopped first. That is cheaper than
            # it sounds: an episode usually ends because the car already hit
            # something, and full braking from 300 km/h takes about 2 s
            # anyway -- less than the track position a Restart Lap throws away.
            if self._brake_to_stop():
                seq = self.resume_sequence
            else:
                # Still rolling: the menu option will not be there, and blindly
                # pressing Down-Down-A would select whatever else is in that
                # slot. Fall back to the restart that always works.
                self._failed_stops += 1

        self.last_reset_kind = seq.name

        for button, wait in seq.steps:
            self.pad.press_button(button)
            time.sleep(wait)
        time.sleep(seq.settle_seconds)
        # Prime the frame stack so the first observation is not stale menu art.
        for _ in range(3):
            self._grab()
            time.sleep(1 / 30)

    def apply(self, steer: float, throttle: float, brake: float) -> None:
        if self.pad is None:
            return          # recording mode: the human is driving
        self.pad.set(steer, throttle, brake)

    # -- stopping the car so Reset to Track becomes available ---------------

    STOPPED_KPH = 3.0
    BRAKE_TIMEOUT_S = 7.0

    def _current_kph(self) -> float | None:
        pkt = self.telemetry.latest(PacketId.CAR_TELEMETRY)
        if pkt is None:
            return None
        return float(parse_car_telemetry(pkt[1],
                                         self.telemetry.player_car_index).speed_kph)

    def _brake_to_stop(self) -> bool:
        """Hold full brake until the car is stationary. False on timeout.

        Steering is held straight rather than left wherever the crash put it,
        so the car does not pirouette while slowing and end up facing the wrong
        way when it is put back on track.
        """
        start = time.perf_counter()
        while time.perf_counter() - start < self.BRAKE_TIMEOUT_S:
            self.pad.set(0.0, 0.0, 1.0)
            time.sleep(0.05)
            kph = self._current_kph()
            if kph is None:
                return False        # no telemetry: cannot confirm, do not guess
            if kph < self.STOPPED_KPH:
                self.pad.neutral()
                self.stop_seconds = time.perf_counter() - start
                return True
        self.pad.neutral()
        self.stop_seconds = time.perf_counter() - start
        return False

    def observe(self) -> BackendState:
        frame = self._grab()

        tel = self.telemetry.latest(PacketId.CAR_TELEMETRY)
        mot = self.telemetry.latest(PacketId.MOTION)
        lap = self.telemetry.latest(PacketId.LAP_DATA)
        idx = self.telemetry.player_car_index

        if tel is None or mot is None:
            # Telemetry gap: report the last frame with zeroed dynamics rather
            # than raising, so a dropped packet does not kill a training run.
            return BackendState(
                frame=frame, speed_mps=0.0, yaw_rate=0.0, g_lat=0.0, g_lon=0.0,
                gear=0, lap_distance=0.0, lateral_offset=0.0, on_track=True,
            )

        t = parse_car_telemetry(tel[1], idx)
        m = parse_car_motion(mot[1], idx)

        wheels_off = sum(1 for s in t.surface_type if s not in ON_SURFACE)

        lap_distance = 0.0
        lap_invalid = False
        pit_status = 0
        if lap is not None:
            try:
                lap_distance = parse_lap_distance(lap[1], idx)
                lap_invalid = parse_lap_invalid(lap[1], idx)
                pit_status = parse_pit_status(lap[1], idx)
            except RuntimeError:
                # Offsets not calibrated yet; probe_telemetry.py --scan-floats
                # fills them in. Progress reward is meaningless until then.
                lap_distance = 0.0

        # Track the stationary-and-frozen condition across calls.
        stopped = t.speed_kph < self.GARAGE_KPH
        frozen = abs(lap_distance - self._prev_lap_distance) < 0.05
        self._prev_lap_distance = lap_distance
        if stopped and frozen:
            self._garage_steps += 1
        else:
            self._garage_steps = 0

        # driverStatus is the primary signal and the only one confirmed to move
        # on a pit entry; the stationary-and-frozen heuristic is a backstop for
        # states the enum does not cover. `status >= 0` guards the case where
        # LapData has not arrived yet, which reads as -1 rather than garage.
        status = self.driver_status()
        in_garage = (status == DRIVER_IN_GARAGE
                     or self._garage_steps >= self.GARAGE_STEPS)

        return BackendState(
            frame=frame,
            speed_mps=t.speed_kph / 3.6,
            # Yaw rate is not transmitted directly; the world angular velocity
            # about the vertical axis is the closest available equivalent.
            yaw_rate=self._yaw_rate(m),
            g_lat=m.g_lat,
            g_lon=m.g_lon,
            gear=t.gear,
            lap_distance=lap_distance,
            on_track=wheels_off == 0,
            wheels_off=wheels_off,
            lap_invalid=lap_invalid,
            pit_status=pit_status,
            in_garage=in_garage,
            # The game reports what it APPLIED, after ~82 ms of steering ramp.
            # Measured on this build, that ramp is four times the transport
            # delay, so treating command and effect as the same thing would
            # hide most of the actuator's state from the policy.
            applied_steer=t.steer,
            applied_throttle=t.throttle,
            applied_brake=t.brake,
            world_x=m.world_x,
            world_z=m.world_z,
            has_world_position=True,
            # Distance from the demonstrated racing line. Zero when the map
            # carries no geometry, in which case the caller falls back to
            # surface type -- which cannot see a tarmac run-off.
            lateral_offset=(
                self.track_map.lateral_offset(m.world_x, m.world_z,
                                              lap_distance)
                if self.track_map is not None and self.track_map.has_geometry
                else 0.0),
        )

    # A car that is stopped with its lap distance frozen, while telemetry is
    # still arriving, is not driving. That is the garage screen -- or any other
    # non-driving state the game has dropped us into.
    GARAGE_KPH = 1.5
    GARAGE_STEPS = 12          # ~0.4 s at 30 Hz

    # The universal prelude: settle, then three A presses to clear the garage
    # screen or any dialog, before the reset sequence proper.
    PRELUDE_WAIT_S = 1.5
    PRELUDE_GAP_S = 0.45

    def looks_like_garage(self) -> bool:
        """Behavioural detection of 'not driving', needing no byte offset.

        Written because the flag-based approach kept aiming at the wrong
        event. In Time Trial, entering the pits does not put the car in a pit
        lane -- it drops straight to the garage screen, so `pitStatus` never
        moves. Rather than guess which flag does move, this watches for the
        thing that is unambiguously true in every such state: the car is
        stationary, lap distance has stopped advancing, and packets are still
        arriving (so the game has not simply crashed).

        Works whatever the game calls the state, and survives a patch that
        renumbers the enums.
        """
        return self._garage_steps >= self.GARAGE_STEPS

    def driver_status(self) -> int:
        """0 garage, 1 flying lap, 2 in lap, 3 out lap, 4 on track, -1 unknown."""
        pkt = self.telemetry.latest(PacketId.LAP_DATA)
        if pkt is None:
            return -1
        try:
            return parse_driver_status(pkt[1], self.telemetry.player_car_index)
        except RuntimeError:
            return -1

    def escape_via_garage(self, budget_seconds: float = 60.0) -> bool:
        """Back out to the garage, then start a flying lap.

        The recovery that actually works, and the reason is that it is driven
        by an observed state rather than a stopwatch:

            press B until the game says we are in the GARAGE
            then press A twice to start a flying lap

        Every earlier attempt fired a fixed number of presses and hoped the
        prompt had appeared. It never reliably had -- the timing varies, so
        the same sequence lands somewhere different each time and eventually
        wedges the game in a submenu. Backing all the way out to a known state
        first removes the guesswork: the garage is unambiguous, telemetry
        reports it, and there is exactly one way forward from there.
        """
        if self.pad is None:
            return False
        import vgamepad as vg

        deadline = time.perf_counter() + budget_seconds
        presses = 0

        # Phase 1: back out until the game reports the garage. driverStatus is
        # confirmed to read 0 there and 1 on a flying lap, so this is an
        # observed state rather than an assumption about menu timing.
        while time.perf_counter() < deadline:
            if self.driver_status() == DRIVER_IN_GARAGE:
                break
            self.pad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            presses += 1
            for _ in range(6):
                time.sleep(0.12)
                if self.driver_status() == DRIVER_IN_GARAGE:
                    break
        else:
            self.last_escape = f"B x{presses} (never reached the garage)"
            return False

        if self.driver_status() != DRIVER_IN_GARAGE:
            self.last_escape = f"B x{presses} (never reached the garage)"
            return False

        # Phase 2: out of the garage and onto a flying lap.
        time.sleep(0.6)
        for _ in range(2):
            self.pad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            time.sleep(0.9)

        # Confirm we are actually on a flying lap rather than in another menu.
        for _ in range(40):
            time.sleep(0.25)
            status = self.driver_status()
            if status == DRIVER_FLYING_LAP and self.stale_seconds() < 2.0:
                self._menu_escapes += 1
                self.last_escape = f"B x{presses} -> garage -> A A"
                return True

        self.last_escape = f"B x{presses} -> garage -> A A (did not resume)"
        return False

    def escape_menus(self, budget_seconds: float = 30.0) -> bool:
        """Get out of whatever menu the game is sitting in, adaptively.

        A FIXED sequence cannot work here, which is what kept wedging the run.
        Firing B,A,A at a fixed moment assumes the prompt has already appeared;
        press too early and the inputs go nowhere, press too late and they land
        somewhere deeper. The prompt's timing varies, so the sequence has to be
        driven by feedback rather than by a stopwatch.

        So this presses one button at a time and CHECKS TELEMETRY after each.
        The moment packets resume, the car is back under our control and it
        stops -- no further presses to wander deeper into a submenu.

        Button order matters: B backs out, A confirms whatever dialog is in the
        way, START closes the pause menu. Cycling all three covers every state
        the game can be sitting in without needing to know which one it is.
        """
        if self.pad is None:
            return False
        import vgamepad as vg

        buttons = (
            (vg.XUSB_BUTTON.XUSB_GAMEPAD_B, "B"),
            (vg.XUSB_BUTTON.XUSB_GAMEPAD_A, "A"),
            (vg.XUSB_BUTTON.XUSB_GAMEPAD_A, "A"),
            (vg.XUSB_BUTTON.XUSB_GAMEPAD_B, "B"),
            (vg.XUSB_BUTTON.XUSB_GAMEPAD_START, "START"),
        )

        deadline = time.perf_counter() + budget_seconds
        pressed: list[str] = []
        i = 0
        while time.perf_counter() < deadline:
            button, label = buttons[i % len(buttons)]
            self.pad.press_button(button)
            pressed.append(label)
            i += 1
            # Give the UI time to react, then look before pressing again.
            for _ in range(8):
                time.sleep(0.12)
                if self.stale_seconds() < 1.5:
                    self._menu_escapes += 1
                    self.last_escape = " ".join(pressed)
                    return True
        self.last_escape = " ".join(pressed)
        return False

    def request_pit_escape(self) -> None:
        """Make the next reset use the pit-lane escape sequence."""
        self._in_pit_recovery = True

    def reset_stats(self) -> dict:
        """How resets are actually going -- worth watching on a long run.

        A high `failed_stops` fraction means the car is rarely coming to rest
        inside the timeout, so most resets are silently falling back to
        Restart Lap and the state distribution is not spreading around the lap
        the way the resume strategy intends.
        """
        return {
            "resets": self._resets,
            "failed_stops": self._failed_stops,
            "failed_frac": self._failed_stops / max(1, self._resets),
            "pit_escapes": self._pit_escapes,
            "menu_escapes": self._menu_escapes,
            "last_stop_s": self.stop_seconds,
            "last_kind": self.last_reset_kind,
        }

    def stale_seconds(self) -> float:
        """How long the telemetry stream has been frozen.

        F1 25 crashes. When it does, the UDP socket stays open and the last
        packet sits in the listener forever, so a training loop happily keeps
        stepping -- writing thousands of identical transitions into the replay
        buffer at whatever the final frozen state was. Nothing raises. An
        overnight run would wake up with a poisoned buffer and a policy trained
        on a screenshot.

        Frame identifier is the right signal rather than packet arrival: it
        increments every physics tick, so it catches a hung game as well as a
        dead socket.
        """
        pkt = self.telemetry.latest(PacketId.CAR_TELEMETRY)
        now = time.perf_counter()
        if pkt is None:
            if self._stale_since is None:
                self._stale_since = now
            return now - self._stale_since

        frame_id = pkt[0].frame_id
        if frame_id != self._last_frame_id:
            self._last_frame_id = frame_id
            self._stale_since = None
            return 0.0
        if self._stale_since is None:
            self._stale_since = now
        return now - self._stale_since

    @property
    def track_length(self) -> float:
        return self._track_length

    def close(self) -> None:
        try:
            if self.pad is not None:
                self.pad.close()
        finally:
            self.telemetry.stop()
            self.camera.close()

    # -- internals ---------------------------------------------------------

    def _grab(self) -> np.ndarray:
        """Capture and preprocess, reusing the last frame if none is ready.

        Desktop Duplication returns None when the screen has not changed since
        the last call. With the game rendering that is rare, but at a frame cap
        below the control rate it does happen, and reusing beats blocking.
        """
        raw = self.camera.grab()
        if raw is None:
            return self._last_frame

        h = raw.shape[0]
        top = int(h * self.crop_top)
        band = raw[top:top + int(h * self.crop_frac)]
        grey = self._cv2.cvtColor(band, self._cv2.COLOR_BGR2GRAY)
        self._last_frame = self._cv2.resize(
            grey, (FRAME_W, FRAME_H), interpolation=self._cv2.INTER_AREA)
        return self._last_frame

    @staticmethod
    def _yaw_rate(m) -> float:
        """Approximate yaw rate from lateral g and speed.

        The Motion packet carries angular velocity in F1 24+, but the field is
        in MotionEx (packet 13) whose layout is not yet pinned here. Until it
        is, a_lat = v * yaw_rate inverts to a usable estimate, and the state
        vector only needs the magnitude to be roughly right.
        """
        speed = (m.vel_x ** 2 + m.vel_y ** 2 + m.vel_z ** 2) ** 0.5
        if speed < 1.0:
            return 0.0
        return float(np.clip(m.g_lat * 9.81 / speed, -1.5, 1.5))
