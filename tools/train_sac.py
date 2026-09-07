"""Train SAC on the mock environment, with the diagnostics needed to debug it.

Every metric that distinguishes one failure mode from another is written to a
CSV each step, and the live HUD is fed in parallel. When training disappoints,
the question is never "is it working" but "which of five things broke", and
these columns are what separate them.

  python tools/train_sac.py --steps 150000
  python tools/train_sac.py --steps 150000 --hud     # feed the overlay

Structure note: the actor and learner run in one thread here because the mock
steps far faster than real time, so there is nothing to overlap. Against F1 25
the learner must move to its own thread -- the game does not pause for a
gradient step. `learn_ratio` exists to keep that honest: it is the number of
gradient steps per environment step, and against the real game it becomes a
budget rather than a constant.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.buffer import ReplayBuffer  # noqa: E402
from f1ai.rl.env import (  # noqa: E402
    ACTION_HISTORY, CONTROL_HZ, FRAME_H, FRAME_STACK, FRAME_W, STATE_DIM,
    MockBackend, RacingEnv,
)
from f1ai.rl.sac import SACAgent, SACConfig  # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "runs"

# Seconds of frozen telemetry before we assume the game has died. Generous
# enough to survive a pause menu or a loading screen, short enough that an
# overnight crash costs minutes rather than the whole night.
STALE_ABORT_S = 20.0
# How long to keep waiting for a crashed or paused game before giving up. Long
# enough that a restart at 3 a.m. still salvages the night.
STALE_GIVEUP_MIN = 45.0

# Roughly every 10 minutes of wall clock at 30 Hz.
CHECKPOINT_EVERY = 20_000

# A genuine early episode lasts a second or two even when the policy is random,
# so a run of episodes shorter than this means something structural is wrong.
RESET_LOOP_STEPS = 20
RESET_LOOP_LIMIT = 25


def evaluate(agent: SACAgent, seed: int, max_steps: int = 4200,
             env: RacingEnv | None = None, eval_spinup: bool = False,
             start_obs: tuple | None = None) -> dict:
    """Deterministic rollout: no exploration noise, no learning.

    Training-time lap times are polluted by the entropy noise the policy is
    deliberately injecting, so the honest number comes from a greedy run using
    the mean action.

    MUST evaluate in the same world it trained in. This originally built a
    fresh MockBackend unconditionally, which meant an in-game run was scored
    inside the SIMULATOR on entirely different synthetic visuals -- reporting
    57 m for a policy that was covering 1,260 m in the game. The numbers were
    not noisy, they were measuring a different thing altogether.
    """
    if env is None:
        env = RacingEnv(MockBackend(seed=seed))

    # Reuse the caller's fresh start rather than resetting again.
    #
    # Evaluation used to reset unconditionally, and against the game that
    # meant a SECOND reset moments after the training loop's own -- the menu
    # inputs landed while the first reset was still settling, and the car
    # ended up parked somewhere it could not move. Every evaluation of a
    # policy that was covering nine kilometres reported zero metres.
    if start_obs is not None:
        frames, state = start_obs
    else:
        frames, state = env.reset()

    # Evaluation needs the same spin-up as training. Without it the car sits
    # stationary wherever the reset dropped it and the policy -- which has
    # never seen 0 km/h -- does nothing, scoring 31 m for an agent that covers
    # 4,000 m in training. The number measured the reset, not the policy.
    from f1ai.rl.spinup import SpinUp
    from f1ai.sim.vehicle import V_MAX as _V_MAX
    spin = SpinUp(env, enabled=eval_spinup)
    spin.begin(env.reset_lap_distance)

    total, steps = 0.0, 0
    for _ in range(max_steps):
        a = agent.act(frames, state, deterministic=True)
        action = spin.action(SACAgent.to_env_action(a),
                             float(state[0]) * _V_MAX)
        res = env.step(action)
        total += res.reward
        steps += 1
        frames, state = res.frames, res.state
        if res.terminated:
            break
    laps = [t for t in env.lap_times if t > 0]
    return {
        "eval_return": total,
        "eval_steps": steps,
        "eval_distance": res.info["distance"],
        "eval_laps": len(laps),
        "eval_best_lap": min(laps) if laps else None,
        "eval_survived": not res.terminated,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--warmup", type=int, default=5_000,
                    help="random-action steps before learning starts")
    ap.add_argument("--capacity", type=int, default=150_000)
    ap.add_argument("--learn-ratio", type=float, default=1.0,
                    help="gradient steps per environment step")
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hud", action="store_true", help="publish to the overlay")
    ap.add_argument("--name", default=None)
    ap.add_argument("--backend", choices=("mock", "f1"), default="mock",
                    help="'f1' drives the real game via capture + pad")
    ap.add_argument("--track", dest="map", default="track",
                    help="f1 only: curvature map for the preview features")
    ap.add_argument("--crop-top", type=float, default=0.38,
                    help="top of the crop as a fraction of frame height; "
                         "find it with tools/tune_crop.py")
    ap.add_argument("--crop-frac", type=float, default=0.26,
                    help="crop height as a fraction of frame height")
    ap.add_argument("--region", default=None,
                    help="f1 only: left,top,right,bottom capture region")
    ap.add_argument("--delay-ms", type=float, default=0.0,
                    help="mock only: inject actuation delay, to rehearse the "
                         "real game's measured lag before committing to it")
    ap.add_argument("--restart-every", type=int, default=1,
                    help="1 = every reset is a Restart Lap (most reliable). "
                         "Higher mixes in Reset to Track to spread experience "
                         "around the lap, at the cost of a trickier recovery")
    ap.add_argument("--save-buffer", action="store_true", default=True,
                    help="write the replay buffer at the end, so it can be "
                         "trained on offline at full GPU speed")
    ap.add_argument("--no-save-buffer", dest="save_buffer",
                    action="store_false")
    ap.add_argument("--init-from", default=None,
                    help="warm-start encoder and actor from a behavioural "
                         "cloning checkpoint (runs/bc/bc.pt)")
    ap.add_argument("--target-late", type=float, default=0.02,
                    help="deadline-miss rate the learner backs off to hold; "
                         "the actor's 30 Hz always takes priority")
    ap.add_argument("--batch", type=int, default=None,
                    help="override SAC batch size; 128 roughly halves the "
                         "cost of a gradient step when the GPU is contended")
    ap.add_argument("--async-learner", action="store_true",
                    help="run gradient steps on their own thread (implied "
                         "by --backend f1, which cannot pause the world)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    run = a.name or time.strftime("sac-%Y%m%d-%H%M%S")
    out = RUNS / run
    out.mkdir(parents=True, exist_ok=True)

    if a.backend == "f1":
        from f1ai.rl.f1_backend import F1Backend
        region = (tuple(int(v) for v in a.region.split(","))
                  if a.region else None)
        _tm = None
        from f1ai.rl.track_map import map_path as _mp
        _map_file = _mp(a.map) if a.map else None
        if _map_file and _map_file.exists():
            from f1ai.rl.track_map import TrackMap as _TM
            _tm = _TM.load(_map_file)
        backend = F1Backend(capture_region=region, track_map=_tm,
                            crop_top=a.crop_top, crop_frac=a.crop_frac)
        backend.restart_every = a.restart_every
        print(f"reset policy: "
              + ("always Restart Lap" if a.restart_every <= 1 else
                 f"Reset to Track, with a Restart Lap every "
                 f"{a.restart_every}"))
        # The game runs whether or not the optimiser has finished, so the
        # learner must not share the actor's thread. Not a preference.
        a.async_learner = True
    else:
        delay_steps = int(round(a.delay_ms / 1000.0 * CONTROL_HZ))
        backend = MockBackend(seed=a.seed, actuation_delay_steps=delay_steps)
        if delay_steps:
            print(f"injecting {a.delay_ms:.0f} ms actuation delay "
                  f"({delay_steps} control steps)")

    tmap = None
    if a.backend == "f1" and a.map:
        from f1ai.rl.track_map import TrackMap, map_path
        mp = map_path(a.map)
        if mp.exists():
            tmap = TrackMap.load(mp)
            print(f"track map {mp.name}: {tmap.length:.0f} m")
        else:
            print(f"WARNING: no map at {mp} -- preview features will be zero")

    env = RacingEnv(backend, track_map=tmap)
    sac_cfg = SACConfig()
    if a.batch:
        sac_cfg.batch_size = a.batch
    agent = SACAgent((FRAME_STACK, FRAME_H, FRAME_W), STATE_DIM, 3,
                     sac_cfg, device="cuda" if torch.cuda.is_available()
                     else "cpu")
    if a.init_from:
        sd = torch.load(a.init_from, map_location=agent.device,
                        weights_only=False)
        # Encoder and actor only. The BC checkpoint's critic never saw a
        # reward, so its Q values are meaningless -- loading them would have
        # the policy optimising noise for its first several thousand steps.
        agent.encoder.load_state_dict(sd["encoder"])
        agent.actor.load_state_dict(sd["actor"])
        import copy
        agent.encoder_target = copy.deepcopy(agent.encoder)
        for p in agent.encoder_target.parameters():
            p.requires_grad_(False)
        agent.anchor_to_bc()
        print(f"warm-started encoder + actor from {a.init_from}")
        print("  critic left at random init: it has never seen a reward")
        print(f"  actor frozen for {sac_cfg.critic_warmup:,} gradient steps "
              f"while the critic catches up")
        print(f"  imitation anchor {sac_cfg.bc_anchor} decaying to 0 over "
              f"{sac_cfg.bc_anchor_decay:,} steps\n")

    buf = ReplayBuffer(a.capacity, (FRAME_H, FRAME_W), STATE_DIM, 3,
                       stack=FRAME_STACK, seed=a.seed)

    pub = None
    if a.hud:
        from f1ai.hud.bus import HudPublisher, HudSnapshot
        pub = HudPublisher()

    csv_path = out / "metrics.csv"
    # `episode_distance` is logged separately from return because they stopped
    # meaning the same thing once the speed term was added -- return now mixes
    # metres with a pace bonus, so it can no longer be read as distance.
    cols = ["step", "grad_steps", "episode", "episode_return", "episode_len",
            "episode_distance", "episode_speed_kph",
            "valid_laps", "invalid_laps",
            "buffer_fill", "critic_loss", "actor_loss", "q_mean", "target_q",
            "alpha", "entropy", "train_best_lap", "sps",
            "eval_return", "eval_laps", "eval_best_lap", "eval_distance"]
    fh = csv_path.open("w", newline="")
    writer = csv.DictWriter(fh, fieldnames=cols)
    writer.writeheader()

    # Record everything that changes what the run MEANS, not just how it was
    # sized. A delay-injected run that does not say so is indistinguishable
    # from a clean one months later, and the comparison silently becomes wrong.
    (out / "config.json").write_text(json.dumps({
        "steps": a.steps, "warmup": a.warmup, "capacity": a.capacity,
        "learn_ratio": a.learn_ratio, "seed": a.seed,
        "backend": a.backend,
        "delay_ms": a.delay_ms,
        "async_learner": a.async_learner,
        "state_dim": STATE_DIM,
        "action_history": ACTION_HISTORY,
        "target_late": a.target_late,
        "sac": vars(sac_cfg),
    }, indent=2))

    print(f"run: {out}")
    print(f"device: {agent.device}, {sum(p.numel() for p in agent.actor.parameters()):,} actor params")
    print(f"warmup {a.warmup:,} random steps, then {a.steps:,} total\n")
    print(f"{'step':>8} {'ep':>5} {'return':>9} {'len':>5} {'critic':>8} "
          f"{'q':>8} {'alpha':>6} {'H':>7} {'lap':>7} {'sps':>6}")
    print("-" * 82)

    from f1ai.rl.spinup import SpinUp
    from f1ai.sim.vehicle import V_MAX
    spin = SpinUp(env, enabled=(a.backend == "f1"))

    frames, state = env.reset()
    buf.start_episode()
    spin.begin(env.reset_lap_distance)
    episode, ep_return, ep_len = 0, 0.0, 0
    short_streak = 0
    last_eval_step = 0
    total_valid = 0
    total_invalid = 0
    best_valid_lap: float | None = None
    recent_returns: deque = deque(maxlen=20)
    metrics: dict[str, float] = {}
    pending_eval: dict = {}
    t0 = time.perf_counter()
    last_report = t0
    steps_at_report = 0
    grad_budget = 0.0

    realtime = a.backend == "f1"
    control_dt = 1.0 / CONTROL_HZ
    next_tick = t0
    late_steps = 0
    late_window: deque = deque(maxlen=300)   # last ~10 s at 30 Hz

    learner = None
    if a.async_learner:
        from f1ai.rl.learner import AsyncLearner
        learner = AsyncLearner(agent, buf, min_buffer=a.warmup,
                               max_ratio=a.learn_ratio,
                               target_late_frac=a.target_late)
        learner.start()
        print(f"async learner started (min buffer {a.warmup:,}, "
              f"ratio ceiling {a.learn_ratio})\n")

    for step in range(1, a.steps + 1):
        # -- act -----------------------------------------------------------
        if step <= a.warmup:
            raw = np.random.uniform(-1, 1, 3).astype(np.float32)
        else:
            raw = agent.act(frames, state, deterministic=False)
        env_action = SACAgent.to_env_action(raw)

        # After a mid-lap reset the car is stationary at a point it would
        # normally arrive at flat out. Drive the pedals until it is up to the
        # demonstrated speed for this position, keeping the policy's steering.
        env_action = spin.action(env_action, float(state[0]) * V_MAX)

        # Store the action that was ACTUALLY applied, not the one the policy
        # asked for. During spin-up those differ, and recording the request
        # would teach the critic that the policy's pedal choice produced an
        # outcome it had nothing to do with.
        stored = np.array([env_action[0],
                           env_action[1] * 2.0 - 1.0,
                           env_action[2] * 2.0 - 1.0], np.float32)
        buf.add(frames[-1], state, stored, 0.0, False)
        slot = (buf.ptr - 1) % buf.capacity

        res = env.step(env_action)
        # The reward and terminal flag belong to the transition just stored.
        buf.rewards[slot] = res.reward
        buf.terminals[slot] = res.terminated

        ep_return += res.reward
        ep_len += 1
        frames, state = res.frames, res.state

        if res.terminated or res.truncated:
            buf.add_final(frames[-1], state)
            recent_returns.append(ep_return)
            laps = [t for t in env.lap_times if t > 0]
            row = {
                "step": step, "grad_steps": agent.grad_steps,
                "episode": episode, "episode_return": round(ep_return, 2),
                "episode_len": ep_len, "buffer_fill": round(buf.fill, 4),
                "valid_laps": res.info.get("laps", 0),
                "invalid_laps": res.info.get("invalid_laps", 0),
                "episode_distance": round(res.info["distance"], 1),
                "episode_speed_kph": round(
                    res.info["distance"] / max(1e-9, ep_len / CONTROL_HZ)
                    * 3.6, 1),
                "train_best_lap": min(laps) if laps else None,
                **{k: round(v, 5) for k, v in metrics.items()},
                **pending_eval,
            }
            writer.writerow({c: row.get(c) for c in cols})
            fh.flush()
            pending_eval = {}

            total_valid += res.info.get("laps", 0)
            total_invalid += res.info.get("invalid_laps", 0)
            for t in env.lap_times:
                if best_valid_lap is None or t < best_valid_lap:
                    best_valid_lap = t

            # A reset loop is the nastiest failure this system has: episodes
            # end instantly, resets fire constantly, the step counter climbs
            # and nothing whatsoever is learned. It looks like progress.
            if ep_len < RESET_LOOP_STEPS:
                short_streak += 1
                if short_streak == RESET_LOOP_LIMIT:
                    print(f"\n!! {RESET_LOOP_LIMIT} consecutive episodes under "
                          f"{RESET_LOOP_STEPS} steps at step {step:,}.")
                    print(f"   Last reason: {res.info.get('term_reason')}. "
                          f"This is a reset loop, not training.")
                    print("   Stopping rather than burning the run.")
                    torch.save(agent.state_dict(), out / "latest.pt")
                    break
            else:
                short_streak = 0

            # The pit lane needs its own escape sequence; neither normal reset
            # can reach the pause menu from behind the pit prompt.
            if realtime and res.info.get("term_reason") in ("pit lane", "garage"):
                env.backend.request_pit_escape()
                print(f"    {res.info['term_reason']} at "
                      f"{res.info['distance']:.0f} m "
                      f"(pitStatus={res.info.get('pit_status')}) -- "
                      f"backing out to the garage")

            episode += 1
            ep_return, ep_len = 0.0, 0
            frames, state = env.reset()
            buf.start_episode()
            # Target the speed for where the car ended up, which the env reads
            # after the reset. Using the pre-reset position is wrong whenever
            # the reset moves the car, and with Restart Lap it always does.
            spin.begin(env.reset_lap_distance)

        # -- learn ---------------------------------------------------------
        if learner is not None:
            learner.note_env_step(step)
            metrics.update(learner.metrics)
            # Recent deadline-miss rate over a short window, so the learner
            # reacts to what the loop is doing NOW rather than to a lifetime
            # average that a bad first minute would dominate forever.
            if realtime:
                late_window.append(0)
                learner.note_actor_late(sum(late_window) / len(late_window))
        elif step > a.warmup:
            grad_budget += a.learn_ratio
            while grad_budget >= 1.0:
                # The actor and alpha update on a slower clock than the critic,
                # so their metrics are absent on most steps. Merge rather than
                # replace: reporting a missing key as 0.0 makes entropy look
                # like it has collapsed when it simply was not computed.
                metrics.update(agent.update(buf.sample(agent.cfg.batch_size)))
                grad_budget -= 1.0

        # -- periodic checkpoint -------------------------------------------
        # An overnight run that only saves on evaluation can lose hours to a
        # crash between evals. Saving is cheap; the night is not.
        if step % CHECKPOINT_EVERY == 0:
            torch.save(agent.state_dict(), out / "latest.pt")

        # -- watchdog ------------------------------------------------------
        # A crashed game leaves the socket open and the last packet frozen, so
        # without this the loop keeps "training" on a still image and quietly
        # fills the replay buffer with thousands of identical transitions.
        if realtime and step % 15 == 0:
            stale = env.backend.stale_seconds()
            if stale > STALE_ABORT_S:
                # WAIT, do not exit. A five-hour run should not end because the
                # game was paused, alt-tabbed, or crashed and got restarted.
                # Training is suspended so nothing frozen enters the buffer,
                # and picks up the moment telemetry resumes.
                torch.save(agent.state_dict(), out / "latest.pt")
                print(f"\n!! telemetry frozen at step {step:,} -- game paused, "
                      f"crashed, or in a menu.")

                # A mis-selected menu item looks exactly like a crash from out
                # here. Try backing out before assuming the worst -- it costs
                # ten seconds and can save the 45-minute timeout.
                print("   trying to back out of any menu...")
                if env.backend.escape_menus():
                    print("   recovered from a menu -- resuming.")
                    frames, state = env.reset()
                    buf.start_episode()
                    spin.begin(res.info.get("lap_distance", 0.0))
                    ep_return, ep_len = 0.0, 0
                    next_tick = time.perf_counter()
                    continue
                print("   not a menu -- the game is gone.")
                print(f"   Training suspended. Checkpoint saved to "
                      f"{out / 'latest.pt'}")
                print(f"   Restart the game and drive; this resumes by itself "
                      f"(giving up after {STALE_GIVEUP_MIN:.0f} min).")
                if learner is not None:
                    learner.note_actor_late(1.0)   # make the learner stand down

                waited = 0.0
                while waited < STALE_GIVEUP_MIN * 60.0:
                    time.sleep(5.0)
                    waited += 5.0
                    if env.backend.stale_seconds() < 2.0:
                        break
                    if waited % 60 < 5:
                        print(f"   waiting... {waited / 60:.0f} min",
                              end="\r", flush=True)

                if env.backend.stale_seconds() >= 2.0:
                    print(f"\n   no telemetry after {STALE_GIVEUP_MIN:.0f} min "
                          f"-- stopping.")
                    break

                print(f"\n   telemetry back after {waited / 60:.1f} min -- "
                      f"resuming.")
                # The car is somewhere unknown after a manual restart, and the
                # frame stack still holds pre-freeze images. Start a clean
                # episode rather than splicing across the gap.
                frames, state = env.reset()
                buf.start_episode()
                ep_return, ep_len = 0.0, 0
                next_tick = time.perf_counter()

        # -- hold the control rate -----------------------------------------
        # Against F1 25 the world does not wait. Overrunning the slot means the
        # policy is acting on stale pixels; `late_steps` counts how often.
        if realtime:
            next_tick += control_dt
            slack = next_tick - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                late_steps += 1
                if late_window:
                    late_window[-1] = 1
                next_tick = time.perf_counter()

        # -- report --------------------------------------------------------
        now = time.perf_counter()
        if now - last_report > 10.0:
            sps = (step - steps_at_report) / (now - last_report)
            laps = [t for t in env.lap_times if t > 0]
            if learner is not None:
                metrics.update(learner.metrics)
            # VRAM is logged because gradient steps degraded 57 ms -> 302 ms
            # across a long run, and two runs died at almost exactly 5.7 hours.
            # If free VRAM falls steadily, the game is leaking and squeezing
            # our allocations; if it is flat, the slowdown is something else
            # and this rules it out cheaply.
            if torch.cuda.is_available() and step % 30000 < 300:
                free_b, total_b = torch.cuda.mem_get_info()
                print(f"    [vram] {free_b / 1e9:.2f} GB free of "
                      f"{total_b / 1e9:.1f} GB, torch holds "
                      f"{torch.cuda.memory_reserved() / 1e9:.2f} GB")
            print(f"{step:>8} {episode:>5} "
                  f"{np.mean(recent_returns) if recent_returns else 0:>9.1f} "
                  f"{ep_len:>5} {metrics.get('critic_loss', 0):>8.3f} "
                  f"{metrics.get('q_mean', 0):>8.2f} "
                  f"{metrics.get('alpha', 0):>6.3f} "
                  f"{metrics.get('entropy', 0):>7.2f} "
                  f"{(min(laps) if laps else 0):>7.2f} {sps:>6.1f}")
            last_report, steps_at_report = now, step

        # Only evaluate at an EPISODE BOUNDARY, never mid-episode.
        #
        # Evaluation calls env.reset(), and firing that while the car is doing
        # 300 km/h runs the reset prelude and menu sequence at a moment they
        # were never designed for -- the car frequently ended up parked in the
        # garage, unable to move, and ten of twelve evaluations reported zero
        # metres for a policy that was covering thousands in training. Waiting
        # for the car to have stopped anyway costs at most one episode of
        # delay and removes the whole failure.
        if (a.eval_every and step > a.warmup
                and step - last_eval_step >= a.eval_every
                and (res.terminated or res.truncated)):
            last_eval_step = step
            # Against the game, reuse the live environment: there is only one
            # car and one track, and a second backend would fight this one for
            # the capture device and the pad.
            # The episode-end handler above has just reset, so hand evaluation
            # that fresh start instead of triggering another one.
            ev = evaluate(agent, seed=1000 + a.seed,
                          env=env if a.backend == "f1" else None,
                          eval_spinup=(a.backend == "f1"),
                          start_obs=((frames, state) if a.backend == "f1"
                                     else None))
            # Write the eval as its own row immediately. Deferring it to the
            # next episode boundary silently loses any eval not followed by
            # one -- which always includes the last, and the last is the
            # result you most want to quote.
            writer.writerow({c: {
                "step": step, "grad_steps": agent.grad_steps,
                "episode": episode, "buffer_fill": round(buf.fill, 4),
                **{k: round(v, 5) for k, v in metrics.items()},
                **ev,
            }.get(c) for c in cols})
            fh.flush()
            pending_eval = {}
            torch.save(agent.state_dict(), out / "latest.pt")
            bl = f"{ev['eval_best_lap']:.2f}s" if ev["eval_best_lap"] else "none"
            print(f"  [eval @ {step:,}] return {ev['eval_return']:.0f}  "
                  f"laps {ev['eval_laps']}  best {bl}  "
                  f"dist {ev['eval_distance']:.0f}m")

            # Evaluation drove the car and left the environment mid-rollout.
            # Continuing training from the stale observation would splice a
            # transition across that gap into the buffer.
            frames, state = env.reset()
            buf.start_episode()
            spin.begin(env.reset_lap_distance)
            ep_return, ep_len = 0.0, 0
            next_tick = time.perf_counter()

        if pub is not None and step % 3 == 0:
            laps = [t for t in env.lap_times if t > 0]
            snap = HudSnapshot(
                mode="TRAIN",
                speed_kph=res.info["speed_kph"],
                # Gear comes from the observation, not the backend: F1Backend
                # has no `veh` to reach into.
                gear=int(state[4] * 8),
                steer=float(env_action[0]), throttle=float(env_action[1]),
                brake=float(env_action[2]),
                lap_times=laps,
                best_lap_s=min(laps) if laps else None,
                last_lap_s=laps[-1] if laps else None,
                step_reward=res.reward, episode_return=ep_return,
                env_steps=step, grad_steps=agent.grad_steps, episodes=episode,
                buffer_fill=buf.fill,
                actor_loss=metrics.get("actor_loss"),
                critic_loss=metrics.get("critic_loss"),
                alpha=metrics.get("alpha"),
            )
            snap.attach_frame(frames[-1])
            pub.publish(snap)

    if learner is not None:
        learner.stop()
    torch.save(agent.state_dict(), out / "final.pt")

    if a.save_buffer:
        buf_path = out / "buffer.npz"
        print(f"\nsaving replay buffer ({buf.size:,} transitions, "
              f"{buf.nbytes() / 1e9:.1f} GB)...")
        buf.save(buf_path)
        print(f"wrote {buf_path}")
        print(f"  train offline on it:  python tools/train_offline.py "
              f"{buf_path} --init-from {out / 'final.pt'}")
    fh.close()
    elapsed = time.perf_counter() - t0
    # `step`, not `a.steps`: a run that stops early on a crash or a reset loop
    # was reporting the rate it would have achieved had it finished, which
    # flattered a 5.7-hour run into looking like 43.7 steps/s.
    print(f"\ndone in {elapsed / 60:.1f} min "
          f"({step / elapsed:.1f} steps/s over {step:,} steps), wrote {csv_path}")
    total_laps = total_valid + total_invalid
    if total_laps:
        print(f"laps: {total_valid:,} valid, {total_invalid:,} invalid "
              f"({100 * total_valid / total_laps:.0f}% clean)")
        if best_valid_lap:
            print(f"  best VALID lap: {best_valid_lap:.2f}s")
        else:
            print("  no valid laps -- every completed lap broke track limits")

    if learner is not None:
        print(f"learner: {learner.grad_steps:,} gradient steps, "
              f"{learner.updates_per_env_step:.2f} per env step, "
              f"{learner.update_ms:.0f} ms each, "
              f"backoff settled at {learner.backoff_ms:.0f} ms")
    if realtime:
        pct = 100.0 * late_steps / max(1, a.steps)
        print(f"control loop: {late_steps:,} late steps ({pct:.1f}%)"
              + ("  -- the actor is overrunning its 33 ms slot"
                 if pct > 5 else ""))
        rs = env.backend.reset_stats()
        print(f"resets: {rs['resets']:,}, "
              f"{rs['failed_frac']:.0%} could not stop in time "
              f"(those fell back to Restart Lap)")
        if rs.get("pit_escapes"):
            print(f"pit lane: entered {rs['pit_escapes']} times, "
                  f"each forced a full Restart Lap")
        print(f"spin-up: {spin.mean_steps:.0f} steps mean "
              f"({spin.mean_steps / CONTROL_HZ:.1f}s) over "
              f"{spin.episodes:,} episodes")
        if rs["failed_frac"] > 0.4:
            print("  Most resets are restarting from the line, so experience")
            print("  is not spreading around the lap. Raise BRAKE_TIMEOUT_S or")
            print("  check that Reset to Track is where the menu sequence says.")
    if a.backend == "f1":
        env.backend.close()


if __name__ == "__main__":
    main()
