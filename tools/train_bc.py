"""Behavioural cloning from recorded laps, to warm-start the SAC actor.

    python tools/train_bc.py demos/laps-20260817-2130
    python tools/train_sac.py --backend f1 --init-from runs/bc/bc.pt ...

Learning to drive from scratch by trial and error costs the phase where the
agent discovers roads exist. A few of your laps skip that entirely, and against
the real game -- where every sample costs wall-clock time and a reset -- that
saving is the difference between an evening and a week.

TWO SUBTLETIES that decide whether this works at all:

**1. The labels are shifted forward in time.** Telemetry reports what the game
APPLIED, and F1 25 ramps steering over ~82 ms on top of ~18 ms of transport. So
the control the car is executing right now is the consequence of an input from
roughly three steps ago. Training a policy to output the currently-applied
value teaches it to lag by exactly that much, and it will understeer out of
every corner. The label at time t is therefore the applied control at t+delay:
"what my command now should cause".

**2. That same shift breaks a shortcut.** The observation already contains the
currently-applied controls, so predicting the current applied value is
available as a trivial identity mapping -- copy input to output, achieve near
zero loss, learn nothing about driving. Shifting the labels forward makes the
copy answer wrong, and the network has to look at the road instead.

What this produces is a warm start, not a finished driver. It can only imitate,
so it cannot exceed the laps it was shown; SAC takes it from there.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1ai.rl.env import (  # noqa: E402
    ACTION_HISTORY, CONTROL_HZ, FRAME_H, FRAME_STACK, FRAME_W, STATE_DIM,
)
from f1ai.rl.sac import SACAgent, SACConfig, random_shift  # noqa: E402
from f1ai.rl.track_map import MAP_FEATURES, TrackMap  # noqa: E402
from f1ai.sim.vehicle import V_MAX  # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "runs"


def build_dataset(lap_files: list[Path], delay_steps: int,
                  min_clean: float, track_map=None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frames, states, actions) built from raw recordings.

    Observations are constructed here rather than at record time, so a change
    to the state vector does not invalidate previously recorded laps.
    """
    all_frames, all_states, all_actions = [], [], []
    kept, skipped = 0, 0

    for path in sorted(lap_files):
        # Materialise each array ONCE. Indexing an NpzFile decompresses the
        # whole array on every access, so reading frames inside the per-sample
        # loop re-inflates ~46 MB thousands of times and dies allocating.
        with np.load(path) as npz:
            d = {k: npz[k] for k in npz.files}

        n = len(d["frame"])
        clean = float((d["wheels_off"] == 0).mean())
        if clean < min_clean:
            print(f"  skip {path.name}: only {clean:.0%} clean")
            skipped += 1
            continue
        if n < FRAME_STACK + delay_steps + 10:
            print(f"  skip {path.name}: too short ({n} steps)")
            skipped += 1
            continue

        applied = np.stack([d["steer"], d["throttle"], d["brake"]], axis=1)

        # Valid indices need `FRAME_STACK-1` frames of history behind them and
        # `delay_steps` of future ahead for the shifted label.
        lo, hi = FRAME_STACK - 1, n - delay_steps

        # Skip the out-lap. A recording that starts in the garage reports a
        # NEGATIVE lapDistance until the car crosses the line, and those frames
        # are the pit lane -- training on them teaches the policy to drive down
        # it, which is exactly the behaviour that has to be unlearned later.
        on_track = d["lap_distance"] >= 0.0
        skipped_outlap = 0

        for t in range(lo, hi):
            if not on_track[t]:
                skipped_outlap += 1
                continue
            stack = d["frame"][t - FRAME_STACK + 1:t + 1]

            # Commanded-action history is not recorded -- the human's raw stick
            # input is not transmitted. Applied controls are the best available
            # proxy, and they are what the policy will see at inference anyway.
            hist = np.zeros((ACTION_HISTORY, 3), np.float32)
            for k in range(ACTION_HISTORY):
                idx = t - ACTION_HISTORY + 1 + k
                if idx >= 0:
                    hist[k] = _to_network_action(applied[idx])

            # Must match RacingEnv._state_vector exactly, including the map
            # block. A mismatch here trains the policy against a different
            # observation layout than it will act under, and nothing detects
            # it -- the shapes still line up.
            preview = (track_map.features(float(d["lap_distance"][t]),
                                          float(d["speed"][t]))
                       if track_map is not None
                       else np.zeros(MAP_FEATURES, np.float32))

            state = np.concatenate([
                np.array([
                    d["speed"][t] / V_MAX,
                    np.clip(d["yaw_rate"][t], -1.5, 1.5),
                    np.clip(d["g_lat"][t] / 5.0, -1.5, 1.5),
                    np.clip(d["g_lon"][t] / 5.0, -1.5, 1.5),
                    d["gear"][t] / 8.0,
                    applied[t][0], applied[t][1], applied[t][2],
                ], np.float32),
                hist.reshape(-1),
                preview,
            ])

            all_frames.append(stack)
            all_states.append(state)
            # The shifted label: what this command should cause.
            all_actions.append(_to_network_action(applied[t + delay_steps]))

        kept += 1
        note = (f", {skipped_outlap} out-lap frames dropped"
                if skipped_outlap else "")
        print(f"  keep {path.name}: {hi - lo - skipped_outlap} samples, "
              f"{clean:.0%} clean{note}")

    if not all_frames:
        raise SystemExit("no usable laps -- lower --min-clean or record more")

    print(f"\n{kept} laps kept, {skipped} skipped")
    return (np.asarray(all_frames, np.uint8),
            np.asarray(all_states, np.float32),
            np.asarray(all_actions, np.float32))


def _to_network_action(applied: np.ndarray) -> np.ndarray:
    """Applied (steer -1..1, throttle 0..1, brake 0..1) -> network space.

    The actor emits tanh outputs in [-1,1]^3 and `to_env_action` maps the
    pedals back, so the labels must live in the same space or the policy is
    trained against a different convention than it acts in.
    """
    return np.array([applied[0],
                     applied[1] * 2.0 - 1.0,
                     applied[2] * 2.0 - 1.0], np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", help="a demos/ directory, or a single .npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--delay-ms", type=float, default=100.0,
                    help="measured control latency; labels shift by this much")
    ap.add_argument("--min-clean", type=float, default=0.95,
                    help="reject laps with more off-track time than this")
    ap.add_argument("--val-frac", type=float, default=0.15)
    # Default OFF, which is not the textbook answer and is the right one here.
    #
    # Measured on held-out data at Monza: augmentation improved steering R2 by
    # 0.007 and cost 0.136 on braking, dropping brake output variance from 0.51
    # to 0.38 -- collapsing toward the mean on the channel that was already
    # weakest. Steering survives a pixel shift because it depends on the
    # immediate road direction; braking does not, because it depends on how far
    # away the corner is, and shifting the frame corrupts exactly that cue.
    #
    # Overfitting is also less of a concern than usual: this policy only ever
    # drives one circuit, so memorising Monza is most of the job rather than a
    # failure. Turn it on if training on several tracks.
    ap.add_argument("--shift-pad", type=int, default=0,
                    help="random-shift augmentation in pixels (default off; "
                         "it costs braking accuracy, see the comment above)")
    ap.add_argument("--track", dest="map", default="track",
                    help="track map for the preview features; must match the "
                         "map used at drive time")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from f1ai.rl.track_map import map_path
    tmap = None
    _mf = map_path(a.map) if a.map else None
    if _mf and _mf.exists():
        tmap = TrackMap.load(_mf)
        print(f"using track map {_mf.name} ({tmap.length:.0f} m)")
    else:
        print(f"NO TRACK MAP at {_mf} -- preview features will be zero, and "
              f"the policy will have to infer corners from pixels alone")

    path = Path(a.session)
    files = ([path] if path.suffix == ".npz"
             else sorted(path.glob("*.npz")))
    if not files:
        raise SystemExit(f"no .npz recordings in {path}")

    delay_steps = int(round(a.delay_ms / 1000.0 * CONTROL_HZ))
    print(f"labels shifted forward {delay_steps} steps "
          f"({a.delay_ms:.0f} ms at {CONTROL_HZ:g} Hz)\n")

    frames, states, actions = build_dataset(files, delay_steps, a.min_clean,
                                            track_map=tmap)
    n = len(frames)

    # Split by contiguous block, not at random: neighbouring frames are nearly
    # identical, so a shuffled split leaks the validation set into training and
    # reports a validation loss that means nothing.
    n_val = int(n * a.val_frac)
    if len(files) == 1 and n_val:
        # One lap covers each corner exactly once, so any contiguous holdout is
        # a stretch of track the model never trains on -- it would fail there
        # for want of data, not want of capacity, and the validation number
        # would be measuring that rather than generalisation.
        print("\nsingle lap: disabling the validation split, because holding")
        print("out a contiguous block removes a corner from training entirely.")
        print("Judge this checkpoint by driving it, not by a loss number.")
        n_val = 0

    idx = np.arange(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    print(f"\n{n:,} samples -> {len(train_idx):,} train, {len(val_idx):,} val")
    print(f"frames {frames.nbytes / 1e6:.0f} MB")
    if n < 6000:
        print(f"\nNOTE: {n:,} samples is a thin dataset for a CNN. If the "
              f"policy\ndrives poorly, adding more clean laps is the first "
              f"thing to try --\nand a single trajectory contains no recovery "
              f"data at all, which is\nthe classic weakness of behavioural "
              f"cloning.")
    print()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent((FRAME_STACK, FRAME_H, FRAME_W), STATE_DIM, 3,
                     SACConfig(batch_size=a.batch), device=str(dev))
    # Train encoder and actor together here; the critic has no targets yet and
    # is left to SAC.
    params = list(agent.encoder.parameters()) + list(agent.actor.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)

    def batches(indices: np.ndarray, shuffle: bool):
        order = np.random.permutation(indices) if shuffle else indices
        for i in range(0, len(order) - a.batch + 1, a.batch):
            sel = order[i:i + a.batch]
            yield (torch.as_tensor(frames[sel], device=dev),
                   torch.as_tensor(states[sel], device=dev),
                   torch.as_tensor(actions[sel], device=dev))

    print(f"{'epoch':>6} {'train':>10} {'val':>10} {'steer':>9} {'pedals':>9}")
    print("-" * 50)
    best_val = float("inf")
    out_dir = Path(a.out) if a.out else RUNS / "bc"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(a.epochs):
        agent.encoder.train(); agent.actor.train()
        tot, nb = 0.0, 0
        for f, s, y in batches(train_idx, shuffle=True):
            # Random-shift augmentation, on training batches only.
            #
            # Without it this overfits hard: validation loss bottomed at epoch
            # 4 and the remaining 36 epochs only drove training loss to 1/80th
            # of validation. Padding and re-cropping each frame stops the
            # network memorising exact pixel positions, which is the wrong
            # invariance for a camera that moves anyway.
            #
            # It also removes a train/serve mismatch: SAC augments frames in
            # every update, so an encoder that never saw a shifted frame during
            # BC meets a different input distribution the moment RL starts.
            f_aug = random_shift(f.float(), a.shift_pad)
            mu, _ = agent.actor(agent.encoder(f_aug), s,
                                deterministic=True)
            loss = F.mse_loss(mu, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            tot += float(loss.detach()); nb += 1

        agent.encoder.eval(); agent.actor.eval()
        vtot, vnb, se, pe = 0.0, 0, 0.0, 0.0
        with torch.no_grad():
            for f, s, y in batches(val_idx, shuffle=False):
                mu, _ = agent.actor(agent.encoder(f.float()), s,
                                    deterministic=True)
                vtot += float(F.mse_loss(mu, y)); vnb += 1
                se += float((mu[:, 0] - y[:, 0]).abs().mean())
                pe += float((mu[:, 1:] - y[:, 1:]).abs().mean())

        tr = tot / max(1, nb)
        va = vtot / max(1, vnb)
        print(f"{epoch + 1:>6} {tr:>10.5f} {va:>10.5f} "
              f"{se / max(1, vnb):>9.4f} {pe / max(1, vnb):>9.4f}")

        # With no validation set there is nothing to select on, so keep the
        # latest. Selecting on training loss would just pick the most
        # overfitted epoch.
        if vnb == 0:
            torch.save(agent.state_dict(), out_dir / "bc.pt")
            best_val = tr
        elif va < best_val:
            best_val = va
            torch.save(agent.state_dict(), out_dir / "bc.pt")

    (out_dir / "bc_config.json").write_text(json.dumps({
        "session": str(path), "laps": [f.name for f in files],
        "samples": int(n), "delay_ms": a.delay_ms,
        "delay_steps": delay_steps, "epochs": a.epochs,
        "state_dim": STATE_DIM, "action_history": ACTION_HISTORY,
        "track_map": a.map if tmap is not None else None,
        "best_val_mse": best_val,
    }, indent=2))

    print(f"\nbest val MSE {best_val:.5f}")
    print(f"wrote {out_dir / 'bc.pt'}")
    print("\nEvaluate it before trusting it:")
    print(f"  python tools/eval_policy.py {out_dir / 'bc.pt'}")
    print("Then warm-start RL from it:")
    print(f"  python tools/train_sac.py --backend f1 "
          f"--init-from {out_dir / 'bc.pt'} --steps 200000")


if __name__ == "__main__":
    main()
