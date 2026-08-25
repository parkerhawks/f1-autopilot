"""The racing environment: reset, step, reward, termination.

Backend-agnostic. `MockBackend` runs the bicycle-model simulator; the F1 25
backend (Phase 3) will implement the same three methods against screen capture,
UDP telemetry and the virtual pad. Everything above this line -- reward shaping,
episode logic, the learner -- is shared, so the algorithm is developed and
debugged offline and then pointed at the game.

A deliberate asymmetry runs through this file:

    the OBSERVATION contains only what a real car knows about itself
    the REWARD may use privileged track knowledge

The policy sees pixels plus proprioception (speed, yaw rate, g-forces, its own
last action). It does NOT see its lateral offset from the centreline or the
curvature ahead, even though the simulator knows both exactly. Handing those to
the state branch would let the policy ignore the camera entirely and reduce the
whole thing to classical control on a known map -- the CNN would be decorative.
Vision has to answer "where am I on the road and where does it go", because
that is the question an autonomous vehicle actually faces.

Using privileged information in the reward is a different matter and is
standard practice in robotics and AV training: the reward is only evaluated
during training, never at deployment, so it cannot leak into the policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..sim.render import TrackRenderer
from ..sim.track import Track, make_track
from ..sim.vehicle import V_MAX, Vehicle, VehicleState
from .track_map import MAP_FEATURES, TrackMap

# Observation
FRAME_H, FRAME_W = 96, 192
FRAME_STACK = 3          # gives the CNN motion; a single frame has no velocity

# Steps of commanded-action history carried in the observation. Measured on
# F1 25: 18 ms transport + 82 ms steering ramp = ~100 ms, which is 3 control
# steps at 30 Hz. With fewer than that the agent cannot see the actions already
# in flight, the problem stops being Markovian, and the classic symptom is a
# policy that oscillates while over-correcting for inputs it cannot observe.
ACTION_HISTORY = 3

# speed, yaw rate, g_lat, g_lon, gear   (5)
# applied steer/throttle/brake          (3)  -- what the CAR is doing
# commanded action history              (9)  -- what we ASKED for, still landing
# track map preview                     (6)  -- what the road ahead does
#
# The map block carries forward curvature and a reference speed profile, and
# nothing about lateral position: where the car sits ACROSS the road stays the
# camera's job. Handing that over as well would leave the CNN with nothing to
# contribute and turn this into trajectory-following on a known map.
STATE_DIM = 5 + 3 + 3 * ACTION_HISTORY + MAP_FEATURES

# Named slices into the state vector. Anything that needs to reach into a
# specific block uses these rather than a literal index -- a hardcoded `vec[8:]`
# silently started returning the map features too when this layout last grew.
IDX_DYNAMICS = slice(0, 5)
IDX_APPLIED = slice(5, 8)
IDX_HISTORY = slice(8, 8 + 3 * ACTION_HISTORY)
IDX_MAP = slice(8 + 3 * ACTION_HISTORY, STATE_DIM)

# Episode
CONTROL_HZ = 30.0
MAX_EPISODE_SECONDS = 180.0
STUCK_SPEED_KPH = 25.0
STUCK_SECONDS = 3.0

# Reward weights
W_PROGRESS = 1.0         # per metre of centreline progress
W_OFFTRACK = 4.0         # per metre outside the racing surface, per step
W_JERK = 0.9             # discourages the buzzing that plagues naive SAC

# Reward for matching the demonstrated speed at this point on the circuit.
#
# Progress alone rewards speed only indirectly -- a metre is a metre whether it
# took a tenth of a second or a full second -- so with discounting the pull
# toward pace is weak and easily outweighed by the safety of going slowly. This
# pays directly against the human's own speed profile, which turns "go faster"
# into a dense per-step signal instead of something the critic has to infer
# across an eleven-second horizon.
#
# Deliberately NOT capped at the reference: exceeding it pays more, so the
# policy is free to beat the demonstration rather than being anchored to it.
W_SPEED = 0.6
SPEED_TOLERANCE = 0.25   # fraction of reference speed treated as "matching"
W_CRASH = -40.0          # one-off, on leaving the track entirely

# Progress earned off the racing surface earns no credit, fading to zero over
# this distance past the edge. This is the term that makes cutting pointless.
#
# The first version of this reward charged a flat 0.6/m for being off-track
# while progress paid ~2.8/step at racing speed, so running wide through every
# corner was strictly profitable. SAC found that immediately: it learned a
# 27.5s lap that spent 2.8% of its time past the boundary, versus the expert's
# 35.6s -- a training curve that looked like a triumph and was partly a cheat.
# Denying progress credit removes the incentive rather than merely taxing it;
# a penalty alone can always be outrun by going fast enough.
OFFTRACK_FADE = 1.0      # metres past the edge over which credit decays to 0
OFFTRACK_TERMINATE = 2.0  # metres past the edge that ends the episode

# Metres of equivalent excess per wheel off tarmac, used when the backend has
# no track map (i.e. the real game). Calibrated so two wheels off ~= 1 m, the
# point at which progress credit has decayed to zero.
OFFTRACK_PER_WHEEL = 0.5

# Separate thresholds for the GEOMETRIC signal, which is measured in real
# metres from the racing line and therefore needs its own scale.
#
# Reusing the wheels-off numbers here was a mistake that cost a whole run.
# OFFTRACK_TERMINATE = 2.0 makes sense against a severity that maxes at 2.0
# with four wheels off; against metres it ends the episode 12 m from the line,
# and because the reference is a RACING line that hugs the inside of corners,
# a car legitimately using the far side of the road is already close to that.
# The agent was terminated at the Rettifilo on every single lap, could never
# drive through it, and distance collapsed from 8,956 m to 1,116 m.
#
# These give a wide band in which running wide costs progressively more but
# the episode continues, so there is a gradient to learn from -- and a
# termination point far enough out that only the escape road reaches it.
GEO_FADE = 5.0            # metres past half_width over which credit decays
GEO_TERMINATE = 12.0      # metres past half_width that ends the episode

# Speed lost in a single control step that can only mean an impact. Under peak
# braking the car sheds about 45 m/s^2, so at 30 Hz a legitimate step never
# costs more than ~5.4 km/h. A barrier costs far more, instantly.
IMPACT_KPH_DROP = 14.0
# Steps at the start of an episode during which lap invalidation is ignored,
# so a stale flag from before the reset cannot end the episode instantly.
INVALID_GRACE_STEPS = 30
W_IMPACT = -60.0         # heavier than a track-limits excursion, as it should be


class Backend(Protocol):
    """What the environment needs from a world, real or simulated."""

    def reset(self) -> None: ...
    def apply(self, steer: float, throttle: float, brake: float) -> None: ...
    def observe(self) -> "BackendState": ...


@dataclass
class BackendState:
    frame: np.ndarray        # (H, W) uint8 greyscale
    speed_mps: float
    yaw_rate: float
    g_lat: float
    g_lon: float
    gear: int
    lap_distance: float      # privileged: reward only
    lateral_offset: float    # privileged: reward only
    on_track: bool

    # How far off the racing surface, in metres. The simulator knows this
    # exactly from its centreline. F1 25 does not hand out a track map, so the
    # real backend reports `wheels_off` instead -- the game's own per-wheel
    # surface classification, which is ground truth and needs no map at all.
    # Building a centreline by driving a recorded lap would work too, but it
    # would be our estimate of the track edge competing with the game's.
    wheels_off: int = 0      # 0-4, from CarTelemetryData.surfaceType
    lap_invalid: bool = False  # the game's own track-limits verdict
    pit_status: int = 0      # 0 on track, 1 pitting, 2 in the pit area
    # Not driving: the garage screen, a menu, anywhere the car is parked with
    # its lap distance frozen. Detected behaviourally rather than from a flag,
    # because in Time Trial the pit entry drops straight to the garage and
    # `pit_status` never moves.
    in_garage: bool = False

    # World position, used to measure distance from the racing line.
    #
    # Needed because surface type cannot see Monza's tarmac run-offs: cutting
    # the Rettifilo chicane keeps all four wheels on tarmac, so `wheels_off`
    # stays zero and progress pays in full. Geometry is the only signal that
    # distinguishes the track from the road beside it.
    world_x: float = 0.0
    world_z: float = 0.0
    has_world_position: bool = False

    # What the car is ACTUALLY doing, as opposed to what we commanded. F1 25
    # ramps gamepad steering over ~82 ms, so a command and its effect are
    # different things for several control steps. Telemetry reports the applied
    # value, which makes the actuator's state observable rather than hidden --
    # without it the agent has to infer mid-ramp position from pixels alone.
    applied_steer: float = 0.0
    applied_throttle: float = 0.0
    applied_brake: float = 0.0


class MockBackend:
    """Bicycle model + renderer, stepped at the control rate."""

    def __init__(self, track: Track | None = None, seed: int | None = None,
                 actuation_delay_steps: int = 0):
        self.track = track if track is not None else make_track()
        self.renderer = TrackRenderer(self.track, FRAME_W, FRAME_H)
        self.dt = 1.0 / CONTROL_HZ
        self.delay_steps = actuation_delay_steps
        self.rng = np.random.default_rng(seed)
        self.veh = Vehicle(delay_steps=actuation_delay_steps)
        self.reset()

    def reset(self) -> None:
        # Start at a random point on the lap, with small perturbations to
        # position, heading and speed. Always starting from the same clean
        # grid slot would leave the policy with no experience of recovery,
        # which is the failure mode DAgger exists to fix in the imitation
        # setting -- randomised starts buy the same coverage for free here.
        i = int(self.rng.integers(0, len(self.track.x)))
        heading = self.track.heading_at(i)
        offset = float(self.rng.uniform(-0.4, 0.4)) * self.track.width
        self.veh = Vehicle(delay_steps=self.delay_steps)
        self.veh.state = VehicleState(
            x=float(self.track.x[i]) - math.cos(heading) * offset,
            z=float(self.track.z[i]) + math.sin(heading) * offset,
            yaw=heading + float(self.rng.uniform(-0.12, 0.12)),
            v=float(self.rng.uniform(25.0, 65.0)),
        )

    def apply(self, steer: float, throttle: float, brake: float) -> None:
        # The vehicle records what the tyres actually received, which after the
        # actuation delay is not what was just commanded; observe() reads it
        # from there, mirroring how F1 25 reports applied rather than requested
        # controls.
        self.veh.step(steer, throttle, brake, self.dt)

    def observe(self) -> BackendState:
        s = self.veh.state
        _, lap_s, offset = self.track.nearest(s.x, s.z)
        return BackendState(
            frame=self.renderer.render(s.x, s.z, s.yaw),
            speed_mps=s.v,
            yaw_rate=s.yaw_rate,
            g_lat=s.accel_lat / 9.81,
            g_lon=s.accel_lon / 9.81,
            gear=self.veh.gear,
            lap_distance=lap_s,
            lateral_offset=offset,
            on_track=abs(offset) < self.track.width,
            applied_steer=self.veh.last_applied[0],
            applied_throttle=self.veh.last_applied[1],
            applied_brake=self.veh.last_applied[2],
        )

    @property
    def track_length(self) -> float:
        return self.track.length


@dataclass
class StepResult:
    frames: np.ndarray       # (FRAME_STACK, H, W) uint8
    state: np.ndarray        # (STATE_DIM,) float32
    reward: float
    terminated: bool         # crashed or stuck
    truncated: bool          # ran out of time
    info: dict = field(default_factory=dict)


class RacingEnv:
    """Continuous-control racing environment over any Backend."""

    def __init__(self, backend: MockBackend, track_map: TrackMap | None = None):
        self.backend = backend
        # Optional on purpose: the simulator has no recorded map and does not
        # need one, so the features are zero-filled there. Same observation
        # width either way, so a policy trained in one world still loads in the
        # other -- it simply gets no preview.
        self.track_map = track_map
        self.max_steps = int(MAX_EPISODE_SECONDS * CONTROL_HZ)
        self._stack = np.zeros((FRAME_STACK, FRAME_H, FRAME_W), np.uint8)
        self._prev_action = np.zeros(3, np.float32)
        # Newest command last, so history[-1] is the most recent.
        self._action_history = np.zeros((ACTION_HISTORY, 3), np.float32)
        self._prev_s = 0.0
        self._steps = 0
        self._slow_steps = 0
        self._distance = 0.0
        self._lap_start_step = 0
        self._lap_valid = False
        # `_lap_valid` means "a full lap, not a partial one".
        # `_lap_clean` means "the game still counts it" -- different things.
        self._lap_clean = True
        self._offtrack_steps = 0
        # Edge-triggered: the flag stays set for the rest of an invalid lap,
        # so terminating on the level would fire again on the very next step
        # after a reset that lands mid-invalid-lap.
        self._lap_was_invalid = False
        self._was_in_pit = False
        self._prev_speed_kph: float | None = None
        self._last_impact_kph = 0.0
        self.lap_times: list[float] = []
        # Laps completed but not counted by the game. Tracked separately so a
        # run can report how much of its pace was legal.
        self.invalid_lap_times: list[float] = []

    # -- episode -----------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self.backend.reset()
        obs = self.backend.observe()
        # Fill the stack with the same frame so the first transitions are not
        # trained against zeros.
        self._stack[:] = obs.frame
        self._prev_action[:] = 0.0
        self._action_history[:] = 0.0
        self._prev_s = obs.lap_distance
        self._steps = 0
        self._slow_steps = 0
        self._distance = 0.0
        self._lap_start_step = 0
        self._lap_valid = False
        # `_lap_valid` means "a full lap, not a partial one".
        # `_lap_clean` means "the game still counts it" -- different things.
        self._lap_clean = True
        self._offtrack_steps = 0
        # Seed from what the game currently reports, NOT from False.
        #
        # "Reset to Track" invalidates the lap, so an episode resumed that way
        # begins on an already-invalid lap. Re-arming the edge trigger here
        # made the very first step look like a fresh invalidation, which
        # terminated the episode, which triggered another reset -- an infinite
        # loop that burns the whole night while the step counter keeps rising.
        self._lap_was_invalid = obs.lap_invalid
        self._was_in_pit = obs.pit_status != 0 or obs.in_garage
        self._prev_speed_kph: float | None = None

        # Where the car ACTUALLY IS after the reset.
        #
        # The spin-up target used to be taken from where the previous episode
        # ended, which is wrong whenever the reset moves the car -- and with
        # Restart Lap it always does. Crash at the Rettifilo and the next
        # episode would target the chicane's 74 km/h while sitting on the start
        # line, hand over at a crawl, trundle to the chicane and crash again.
        # A self-reinforcing loop that held average speed at 114 km/h and
        # pinned every evaluation at ~1000 m.
        self.reset_lap_distance = obs.lap_distance
        self._last_impact_kph = 0.0
        self.lap_times = []
        self.invalid_lap_times = []
        return self._stack.copy(), self._state_vector(obs)

    def step(self, action: np.ndarray) -> StepResult:
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))

        self.backend.apply(steer, throttle, brake)
        obs = self.backend.observe()
        self._steps += 1

        # -- progress along the centreline, wrapping at the line -----------
        length = self.backend.track_length
        ds = obs.lap_distance - self._prev_s
        if ds < -0.5 * length:          # crossed start/finish forwards
            ds += length
            # Episodes begin at a random point on the lap, so the first
            # crossing ends a PARTIAL lap. Recording it would report lap times
            # far below the truth -- and lap time is this project's headline
            # metric. Use the first crossing only to start the clock.
            if self._lap_valid:
                lap_steps = self._steps - self._lap_start_step
                lap_time = lap_steps / CONTROL_HZ
                if self._lap_clean:
                    self.lap_times.append(lap_time)
                else:
                    # A lap the game will not count is not a lap time. Recording
                    # it anyway reported an 84.77s best for laps that had run
                    # wide -- flattering, and not a number anyone could quote.
                    self.invalid_lap_times.append(lap_time)
            self._lap_valid = True
            self._lap_start_step = self._steps

            # RE-ARM TRACK-LIMITS ENFORCEMENT FOR THE NEW LAP.
            #
            # This is what was missing. "Reset to Track" invalidates the lap,
            # so a resumed episode began with the edge trigger already latched
            # and it never fired again -- five of every six episodes ran with
            # track limits unenforced. Crossing the line starts a genuinely new
            # lap, and its validity has to be judged fresh.
            self._lap_was_invalid = obs.lap_invalid
            self._lap_clean = not obs.lap_invalid
        elif ds > 0.5 * length:         # went backwards over the line
            ds -= length
        self._prev_s = obs.lap_distance
        self._distance += ds

        # -- reward --------------------------------------------------------
        # Progress, not speed. Rewarding raw speed produces a policy that
        # drives fast in the wrong direction; rewarding centreline progress
        # makes going the wrong way actively negative.
        excess, fade_at, terminate_at = self._offtrack_excess(obs)

        # Credit for progress decays to nothing once off the surface, so a
        # shortcut through the scenery advances the lap without advancing the
        # return. Taxing off-track distance is not enough on its own -- a fast
        # enough policy simply pays the tax.
        credit = max(0.0, 1.0 - excess / fade_at)

        # THE PIT LANE EARNS NOTHING.
        #
        # lapDistance keeps climbing while the car drives down the pit lane, so
        # without this the progress reward pays for it at exactly the same rate
        # as the racing line. The agent was not misunderstanding the track --
        # it was told the pit lane IS the track, and drove there accordingly.
        # A terminal penalty alone cannot undo that: the metres banked on the
        # way in are already paid.
        if obs.pit_status != 0 or obs.in_garage:
            credit = 0.0

        reward = W_PROGRESS * ds * credit
        reward -= W_OFFTRACK * excess

        # Pace against the demonstrated profile, but only while on the racing
        # surface -- otherwise a policy could earn speed reward for barrelling
        # through a runoff, which is the reward hacking we already closed once.
        if credit > 0.0 and self.track_map is not None:
            ref = self.arrival_speed_mps(obs.lap_distance)
            if ref > 5.0:
                ratio = obs.speed_mps / ref
                # 1.0 at the reference speed, rising above it, falling away
                # below. Clipped so one slow corner cannot dominate the return.
                pace = np.clip((ratio - (1.0 - SPEED_TOLERANCE))
                               / SPEED_TOLERANCE, -1.0, 1.5)
                reward += W_SPEED * float(pace) * credit
        if excess > 0.0:
            self._offtrack_steps += 1

        action_vec = np.array([steer, throttle, brake], np.float32)
        reward -= W_JERK * float(np.abs(action_vec - self._prev_action).sum())
        self._prev_action = action_vec
        self._action_history = np.roll(self._action_history, -1, axis=0)
        self._action_history[-1] = action_vec

        # -- termination ---------------------------------------------------
        terminated = False
        term_reason = None
        # `>=`, not `>`. With the real game the severity signal is
        # wheels_off * OFFTRACK_PER_WHEEL, which tops out at exactly 2.0 when
        # all four wheels are off tarmac -- so a strict `>` made off-track
        # termination unreachable in-game, and every episode instead ended on
        # the stuck timer three seconds after the car had already crashed.
        if excess >= terminate_at:
            reward += W_CRASH
            terminated = True
            term_reason = "offtrack"

        # The pit lane. Invisible to every other check: it is tarmac so no
        # wheels read off-surface, it does not invalidate the lap, and the pit
        # limiter holds the car above the stuck threshold. Every signal we have
        # says "normal driving".
        #
        # SPEED MATTERS MORE THAN ANYTHING HERE. This used to wait out the
        # 30-step grace period, a full second at 80+ km/h, by which point the
        # car is deep in the pit lane and F1 25 has taken over with its own
        # prompt -- and a reset driven into that prompt wedges the game in a
        # menu it never leaves. Caught on the first flagged step the car is
        # still on the racing surface and the ordinary pause menu still works.
        #
        # Edge-triggered and seeded at reset, like lap invalidation, so an
        # episode that begins in the pit area does not terminate instantly.
        if (obs.pit_status != 0 or obs.in_garage) and not self._was_in_pit:
            reward += W_CRASH
            terminated = True
            term_reason = term_reason or (
                "garage" if obs.in_garage else "pit lane")
        self._was_in_pit = obs.pit_status != 0 or obs.in_garage

        # The game's own track-limits verdict. Better than any geometric rule
        # we could write: it already knows this circuit's kerbs and local
        # exceptions. A lap the game will not count is worthless, so there is
        # nothing to gain by continuing to drive it.
        # A grace period on top of the edge trigger. The invalid flag can lag
        # the reset by a packet or two, so an episode resumed onto an invalid
        # lap could still see a spurious rising edge a few steps in. Belt and
        # braces against a failure mode that costs an entire run.
        if (obs.lap_invalid and not self._lap_was_invalid
                and self._steps > INVALID_GRACE_STEPS):
            reward += W_CRASH
            terminated = True
            term_reason = term_reason or "lap invalidated"
        if obs.lap_invalid:
            self._lap_clean = False
        self._lap_was_invalid = obs.lap_invalid

        # Barrier impact. Detected from the speed trace rather than a damage
        # packet, because a wall costs far more speed in one step than the
        # tyres can physically shed.
        speed_kph = obs.speed_mps * 3.6
        if self._prev_speed_kph is not None:
            drop = self._prev_speed_kph - speed_kph
            if drop > IMPACT_KPH_DROP:
                reward += W_IMPACT
                terminated = True
                term_reason = term_reason or "impact"
                self._last_impact_kph = drop
        self._prev_speed_kph = speed_kph

        if speed_kph < STUCK_SPEED_KPH:
            self._slow_steps += 1
            if self._slow_steps > STUCK_SECONDS * CONTROL_HZ:
                terminated = True     # spun, beached, or refusing to move
                term_reason = term_reason or "stuck"
        else:
            self._slow_steps = 0

        truncated = self._steps >= self.max_steps

        self._stack = np.roll(self._stack, -1, axis=0)
        self._stack[-1] = obs.frame

        return StepResult(
            frames=self._stack.copy(),
            state=self._state_vector(obs),
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info={
                "distance": self._distance,
                "speed_kph": obs.speed_mps * 3.6,
                "offset": obs.lateral_offset,
                "laps": len(self.lap_times),
                "invalid_laps": len(self.invalid_lap_times),
                "last_lap": self.lap_times[-1] if self.lap_times else None,
                # Surfaced so training can be monitored for reward hacking
                # rather than discovering it only at evaluation time.
                "offtrack_frac": self._offtrack_steps / max(1, self._steps),
                # "offtrack" and "stuck" are different failures needing
                # different fixes; reporting them as one word cost an entire
                # diagnostic session.
                "term_reason": term_reason,
                "wheels_off": obs.wheels_off,
                "pit_status": obs.pit_status,
                "excess": excess,
                # Needed by the post-reset spin-up to look up how fast the car
                # should be travelling at this point on the circuit.
                "lap_distance": obs.lap_distance,
            },
        )

    # -- post-reset spin-up ------------------------------------------------

    def arrival_speed_mps(self, lap_distance: float) -> float:
        """How fast the car should be moving at this point on the circuit.

        Resetting mid-lap leaves the car stationary, which corrupts the state
        distribution in a way that is easy to miss: the agent would learn "at
        Roggia, accelerate from zero" when the situation it must actually
        handle is "arrive at Roggia at 300 km/h and brake". Every position it
        resets at would be learned with the wrong speed attached.

        The demonstrated speed profile already says what the right answer is,
        so the car is brought up to it before the policy takes over. Without a
        map there is nothing to aim at, and a modest fixed speed is used purely
        to get the car rolling.
        """
        if self.track_map is None:
            return 40.0
        return float(np.interp(lap_distance % self.track_map.length,
                               self.track_map.s, self.track_map.ref_speed))

    # -- off-track severity ------------------------------------------------

    def _offtrack_excess(self, obs: BackendState) -> float:
        """Metres off the racing surface, from whichever signal exists.

        The simulator measures it against its own centreline. F1 25 has no
        track map to measure against, so the real backend reports how many
        wheels the game itself says are off tarmac, and that count is converted
        to an equivalent excess here. Two wheels off is roughly the car's
        half-width past the line, which is the point where a real steward would
        call it -- so the scale is anchored to something meaningful rather than
        tuned to taste.

        Keeping the conversion in one place means the reward, the termination
        rule and the off-track statistics all agree, whichever backend is
        running underneath.
        """
        track = getattr(self.backend, "track", None)
        if track is not None:
            return (max(0.0, abs(obs.lateral_offset) - track.width),
                    OFFTRACK_FADE, OFFTRACK_TERMINATE)

        # Real game: geometry first, surface type as a floor.
        #
        # Surface type alone cannot see Monza's tarmac run-offs -- cutting the
        # Rettifilo keeps four wheels on tarmac, so wheels_off reads zero and
        # the shortcut pays in full. Distance from the demonstrated racing line
        # catches it.
        #
        # Each signal carries its own fade and termination distances, because
        # they are in different units: one is a 0-2 severity score, the other
        # is metres. Sharing thresholds between them is what broke the last run.
        wheels = OFFTRACK_PER_WHEEL * obs.wheels_off
        # An out-lap reports a NEGATIVE lapDistance, and none of the racing
        # line's assumptions hold there -- it is not a timed lap and the car
        # may legitimately be on the pit exit road. Surface type still applies.
        if (obs.has_world_position and obs.lap_distance >= 0.0
                and self.track_map is not None
                and self.track_map.has_geometry):
            geometric = max(0.0, obs.lateral_offset - self.track_map.half_width)
            # Compare on a normalised scale so the harsher verdict wins even
            # though the two are measured differently.
            if geometric / GEO_TERMINATE >= wheels / OFFTRACK_TERMINATE:
                return geometric, GEO_FADE, GEO_TERMINATE
        return wheels, OFFTRACK_FADE, OFFTRACK_TERMINATE

    # -- observation -------------------------------------------------------

    def _state_vector(self, obs: BackendState) -> np.ndarray:
        """Proprioception only -- nothing that reveals track position.

        Three groups, roughly unit-scaled so the MLP need not learn the
        normalisation itself:

          dynamics   speed, yaw rate, g-forces, gear
          actuator   what the car is currently doing
          in flight  the last ACTION_HISTORY commands, which have not fully
                     taken effect yet given ~100 ms of delay plus ramp

        The last group is what keeps a delayed control problem Markovian.
        """
        if self.track_map is not None:
            preview = self.track_map.features(obs.lap_distance, obs.speed_mps)
        else:
            preview = np.zeros(MAP_FEATURES, np.float32)

        return np.concatenate([
            np.array([
                obs.speed_mps / V_MAX,
                np.clip(obs.yaw_rate, -1.5, 1.5),
                np.clip(obs.g_lat / 5.0, -1.5, 1.5),
                np.clip(obs.g_lon / 5.0, -1.5, 1.5),
                obs.gear / 8.0,
                obs.applied_steer,
                obs.applied_throttle,
                obs.applied_brake,
            ], dtype=np.float32),
            self._action_history.reshape(-1),
            preview,
        ]).astype(np.float32)
