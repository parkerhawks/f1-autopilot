"""Export everything on disk as normalised CSVs, ready to load into a database.

    python tools/export_db.py --out dbexport

The project stores data in whatever shape the training loop needed: NumPy
archives for the track model and the recorded laps, CSV per run for episodes,
PyTorch files for weights. That is right for a 30 Hz control loop and useless
for asking questions across runs -- "which reward configuration produced the
fastest legal lap", or "which corners cost the most time against the human
reference", are joins, and there is nothing here to join.

This flattens all of it into eight related tables:

    track          one row per circuit
    track_point    the circuit geometry, one row per ~5 m
    session        one recording session
    lap            one recorded lap
    sample         one telemetry sample, ~30 per second of driving
    run            one training run and its configuration
    episode        one training episode
    evaluation     one greedy evaluation during a run

Keys are stated in schema.sql, which is written alongside the CSVs.

ONE THING THE FILES NEVER RECORDED
----------------------------------
`runs/<name>/config.json` stores nineteen hyperparameters and does not store
which circuit the run was driven on. Neither does a recording session. That
association has only ever lived in directory names and in the memory of the
person who typed them, which is exactly the failure a schema is supposed to
prevent -- and it is not recoverable for a run whose name says nothing.

So `track_id` is NOT NULL on both `run` and `session`, and every row carries
`track_source` saying how the value was obtained:

    recorded    the file itself stated it
    inferred    taken from the directory name (see TRACK_OF below)
    assumed     no evidence either way; the default was applied

Anything not `recorded` is a claim this exporter is making, not a fact the
project preserved. Analyses that depend on the circuit should say which.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

# Directory-name substring -> circuit. Everything else falls back to DEFAULT
# and is marked 'assumed'. Order matters: first match wins.
TRACK_OF = (
    ("vegas", "lasvegas"),
    ("20260907", "lasvegas"),   # the Las Vegas recording session
)
DEFAULT_TRACK = "monza"

# Runs against the simulator, not the game. Their episodes are real data but
# they were never driven on a circuit at all, so labelling them with one would
# be a lie the schema then propagates into every join.
MOCK_RUNS = {"bc", "delay100", "run1", "run2", "smoke"}

# Superseded by a re-recording. Kept, because the reason is instructive: ~43%
# of every frame was the EA launcher window, so the whole session had to be
# thrown away and shot again. Excluded from analysis via session.superseded.
SUPERSEDED = {"OLD-badcapture-151654"}


def track_for(name: str) -> tuple[str, str]:
    """Return (track_id, how_we_know) for a run or session directory name."""
    low = name.lower()
    for needle, track in TRACK_OF:
        if needle in low:
            return track, "inferred"
    return DEFAULT_TRACK, "assumed"


SCHEMA = """-- f1-autopilot export schema
-- Load with, e.g.:  sqlite3 f1.db < schema.sql
--                   .mode csv
--                   .import track.csv track   (etc., skipping header rows)

CREATE TABLE track (
    track_id      TEXT PRIMARY KEY,
    length_m      REAL NOT NULL,
    half_width_m  REAL NOT NULL,
    n_points      INTEGER NOT NULL,
    min_radius_m  REAL,
    min_speed_kph REAL,
    max_speed_kph REAL
);

CREATE TABLE track_point (
    track_id       TEXT NOT NULL REFERENCES track(track_id),
    seq            INTEGER NOT NULL,
    lap_distance_m REAL NOT NULL,
    curvature      REAL,          -- signed 1/radius
    ref_speed_mps  REAL,          -- demonstrated speed here
    world_x        REAL,
    world_z        REAL,
    PRIMARY KEY (track_id, seq)
);

CREATE TABLE session (
    session_id   TEXT PRIMARY KEY,
    track_id     TEXT NOT NULL REFERENCES track(track_id),
    track_source TEXT NOT NULL,   -- recorded | inferred | assumed
    recorded_at  TEXT,
    n_laps       INTEGER,
    superseded   INTEGER NOT NULL DEFAULT 0,
    CHECK (track_source IN ('recorded', 'inferred', 'assumed'))
);

CREATE TABLE lap (
    lap_id      TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES session(session_id),
    lap_index   INTEGER,
    lap_time_s  REAL,
    n_samples   INTEGER,
    clean_frac  REAL,             -- fraction of samples with all wheels on track
    is_outlap   INTEGER           -- 1 if it contains negative lapDistance
);

CREATE TABLE sample (
    lap_id         TEXT NOT NULL REFERENCES lap(lap_id),
    seq            INTEGER NOT NULL,
    lap_distance_m REAL,
    speed_mps      REAL,
    steer          REAL,          -- applied, -1..1
    throttle       REAL,          -- applied, 0..1
    brake          REAL,          -- applied, 0..1
    gear           INTEGER,
    yaw_rate       REAL,
    g_lat          REAL,
    g_lon          REAL,
    world_x        REAL,
    world_z        REAL,
    wheels_off     INTEGER,       -- 0..4
    PRIMARY KEY (lap_id, seq)
);

CREATE TABLE run (
    run_id        TEXT PRIMARY KEY,
    track_id      TEXT REFERENCES track(track_id),  -- NULL for simulator runs
    track_source  TEXT,           -- recorded | inferred | assumed
    backend       TEXT,           -- 'f1' (real game) or 'mock' (simulator)
    steps         INTEGER,        -- steps REQUESTED, not steps reached
    warmup        INTEGER,
    capacity      INTEGER,
    batch_size    INTEGER,
    gamma         REAL,
    tau           REAL,
    lr            REAL,
    init_alpha    REAL,
    bc_anchor     REAL,
    critic_warmup INTEGER,
    shift_pad     INTEGER,
    learn_ratio   REAL,
    delay_ms      REAL,
    CHECK (track_source IS NULL
           OR track_source IN ('recorded', 'inferred', 'assumed'))
);

CREATE TABLE episode (
    run_id         TEXT NOT NULL REFERENCES run(run_id),
    episode_idx    INTEGER NOT NULL,
    step           INTEGER,
    return_total   REAL,
    length_steps   INTEGER,
    distance_m     REAL,
    mean_speed_kph REAL,
    valid_laps     INTEGER,
    invalid_laps   INTEGER,
    best_lap_s     REAL,
    critic_loss    REAL,
    actor_loss     REAL,
    q_mean         REAL,
    alpha          REAL,
    entropy        REAL,
    PRIMARY KEY (run_id, episode_idx)
);

CREATE TABLE evaluation (
    run_id      TEXT NOT NULL REFERENCES run(run_id),
    step        INTEGER NOT NULL,
    distance_m  REAL,
    laps        INTEGER,
    best_lap_s  REAL,
    PRIMARY KEY (run_id, step)
);

CREATE INDEX idx_sample_dist ON sample(lap_id, lap_distance_m);
CREATE INDEX idx_episode_step ON episode(run_id, step);
CREATE INDEX idx_track_point_dist ON track_point(track_id, lap_distance_m);
"""


def _num(v):
    """CSV cell -> float, or None. Training CSVs leave gaps for metrics that
    are only computed on some steps."""
    if v in ("", "None", None):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int_or_none(v):
    """Like _num but integral, and -- importantly -- returns None for a column
    the run never had. See the note where episodes are written."""
    n = _num(v)
    return None if n is None else int(n)


def export(out: Path, sessions: list[str], sample_stride: int) -> dict:
    from f1ai.rl.track_map import TrackMap, map_path

    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    def write(name: str, header: list[str], rows: list) -> None:
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        counts[name] = len(rows)

    # -- track and track_point -------------------------------------------
    # Every map in maps/, not one named circuit: a database that can only hold
    # one track cannot answer whether anything learned at Monza transfers.
    track_rows, point_rows = [], []
    maps: dict[str, TrackMap] = {}
    for mp in sorted((ROOT / "maps").glob("*.npz")):
        name = mp.stem
        m = maps[name] = TrackMap.load(mp)
        radius = 1.0 / np.maximum(np.abs(m.curvature), 1e-9)
        track_rows.append([
            name, round(m.length, 1), m.half_width, len(m.s),
            round(float(radius.min()), 1),
            round(float(m.ref_speed.min() * 3.6), 1),
            round(float(m.ref_speed.max() * 3.6), 1)])
        point_rows += [
            [name, i, round(float(m.s[i]), 2), round(float(m.curvature[i]), 8),
             round(float(m.ref_speed[i]), 3),
             round(float(m.x[i]), 3) if m.has_geometry else "",
             round(float(m.z[i]), 3) if m.has_geometry else ""]
            for i in range(len(m.s))]

    if not track_rows:
        raise SystemExit("no maps in maps/ -- build one with tools/build_map.py")

    write("track",
          ["track_id", "length_m", "half_width_m", "n_points",
           "min_radius_m", "min_speed_kph", "max_speed_kph"], track_rows)
    write("track_point",
          ["track_id", "seq", "lap_distance_m", "curvature", "ref_speed_mps",
           "world_x", "world_z"], point_rows)

    # -- sessions, laps, samples -----------------------------------------
    session_rows, lap_rows, sample_rows = [], [], []
    for sess in sessions:
        sdir = ROOT / "demos" / sess
        laps = sorted(sdir.glob("*.npz"))
        if not laps:
            continue
        track, how = track_for(sess)
        session_rows.append([sess, track, how, sess.replace("laps-", ""),
                             len(laps), int(sess in SUPERSEDED)])

        for idx, lp in enumerate(laps):
            with np.load(lp) as z:
                d = {k: z[k] for k in z.files}
            lap_id = f"{sess}/{lp.stem}"
            n = len(d["frame"])
            dist = d["lap_distance"]
            is_outlap = int(bool((dist < 0).any()))
            lap_rows.append([
                lap_id, sess, idx,
                round(float(d.get("lap_time", 0.0)), 3), n,
                round(float((d["wheels_off"] == 0).mean()), 4), is_outlap])

            has_world = "world_x" in d
            for i in range(0, n, sample_stride):
                sample_rows.append([
                    lap_id, i, round(float(dist[i]), 2),
                    round(float(d["speed"][i]), 3),
                    round(float(d["steer"][i]), 4),
                    round(float(d["throttle"][i]), 4),
                    round(float(d["brake"][i]), 4),
                    int(d["gear"][i]),
                    round(float(d["yaw_rate"][i]), 5),
                    round(float(d["g_lat"][i]), 4),
                    round(float(d["g_lon"][i]), 4),
                    round(float(d["world_x"][i]), 3) if has_world else "",
                    round(float(d["world_z"][i]), 3) if has_world else "",
                    int(d["wheels_off"][i])])

    write("session", ["session_id", "track_id", "track_source", "recorded_at",
                      "n_laps", "superseded"], session_rows)
    write("lap", ["lap_id", "session_id", "lap_index", "lap_time_s",
                  "n_samples", "clean_frac", "is_outlap"], lap_rows)
    write("sample",
          ["lap_id", "seq", "lap_distance_m", "speed_mps", "steer", "throttle",
           "brake", "gear", "yaw_rate", "g_lat", "g_lon", "world_x", "world_z",
           "wheels_off"], sample_rows)

    # -- runs, episodes, evaluations --------------------------------------
    run_rows, ep_rows, ev_rows = [], [], []
    for rdir in sorted((ROOT / "runs").iterdir()):
        csv_path = rdir / "metrics.csv"
        if not csv_path.exists():
            continue
        cfg = {}
        cfg_path = rdir / "config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
            except ValueError:
                cfg = {}
        sac = cfg.get("sac", {})
        # A simulator run has no circuit. Leaving track_id NULL is the honest
        # encoding; naming one would put fiction into every downstream join.
        is_mock = rdir.name in MOCK_RUNS or cfg.get("backend") == "mock"
        track, how = ("", "") if is_mock else track_for(rdir.name)
        run_rows.append([
            rdir.name, track, how, cfg.get("backend", ""), cfg.get("steps", ""),
            cfg.get("warmup", ""), cfg.get("capacity", ""),
            sac.get("batch_size", ""), sac.get("gamma", ""), sac.get("tau", ""),
            sac.get("lr", ""), sac.get("init_alpha", ""),
            sac.get("bc_anchor", ""), sac.get("critic_warmup", ""),
            sac.get("shift_pad", ""), cfg.get("learn_ratio", ""),
            cfg.get("delay_ms", "")])

        with csv_path.open(encoding="utf-8") as fh:
            for i, r in enumerate(csv.DictReader(fh)):
                if _num(r.get("eval_distance")) is not None:
                    ev_rows.append([rdir.name, int(_num(r.get("step")) or 0),
                                    _num(r.get("eval_distance")),
                                    int(_num(r.get("eval_laps")) or 0),
                                    _num(r.get("eval_best_lap"))])
                    continue
                if _num(r.get("episode_return")) is None:
                    continue
                # NULL, not 0, when the run predates the column. The track
                # limits counters were added after f1-night8; writing 0 would
                # make eight runs look like they completed no legal lap, when
                # in truth nobody was counting. NULL propagates through
                # aggregates; 0 quietly corrupts them.
                ep_rows.append([
                    rdir.name, i, int(_num(r.get("step")) or 0),
                    _num(r.get("episode_return")),
                    int(_num(r.get("episode_len")) or 0),
                    _num(r.get("episode_distance")),
                    _num(r.get("episode_speed_kph")),
                    _int_or_none(r.get("valid_laps")),
                    _int_or_none(r.get("invalid_laps")),
                    _num(r.get("train_best_lap")),
                    _num(r.get("critic_loss")), _num(r.get("actor_loss")),
                    _num(r.get("q_mean")), _num(r.get("alpha")),
                    _num(r.get("entropy"))])

    write("run", ["run_id", "track_id", "track_source", "backend", "steps",
                  "warmup", "capacity", "batch_size", "gamma", "tau", "lr",
                  "init_alpha", "bc_anchor", "critic_warmup", "shift_pad",
                  "learn_ratio", "delay_ms"], run_rows)
    write("episode",
          ["run_id", "episode_idx", "step", "return_total", "length_steps",
           "distance_m", "mean_speed_kph", "valid_laps", "invalid_laps",
           "best_lap_s", "critic_loss", "actor_loss", "q_mean", "alpha",
           "entropy"], ep_rows)
    write("evaluation", ["run_id", "step", "distance_m", "laps", "best_lap_s"],
          ev_rows)

    (out / "schema.sql").write_text(SCHEMA, encoding="utf-8")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="demos/ session names; default is all of them")
    ap.add_argument("--out", default="dbexport")
    ap.add_argument("--sample-stride", type=int, default=1,
                    help="keep every Nth telemetry sample; 1 keeps all ~30 Hz")
    a = ap.parse_args()

    sessions = a.sessions
    if sessions is None:
        sessions = [p.name for p in sorted((ROOT / "demos").iterdir())
                    if p.is_dir() and any(p.glob("*.npz"))]

    out = Path(a.out)
    counts = export(out, sessions, a.sample_stride)

    print(f"wrote {out}/\n")
    print(f"{'table':<14} {'rows':>9}   {'size':>9}")
    print("-" * 38)
    total = 0
    for name, n in counts.items():
        mb = (out / f"{name}.csv").stat().st_size / 1e6
        total += n
        print(f"{name:<14} {n:>9,}   {mb:>8.2f} MB")
    print("-" * 38)
    print(f"{'TOTAL':<14} {total:>9,}")
    print(f"\nschema.sql written alongside. To build a SQLite database:")
    print(f"  python tools/load_db.py --export {out} --db f1.db")


if __name__ == "__main__":
    main()
