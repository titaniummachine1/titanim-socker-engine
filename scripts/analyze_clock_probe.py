"""Turn an exported ClockProbe timeplot into the game's actual time constants.

    python scripts\\analyze_clock_probe.py <timeplot.json> [--final-minute 90]

Derives, from the recording alone:
  * ticks per second and seconds per tick (vs the sim's FIXED_DT = 0.019)
  * total match length in ticks and in seconds
  * how many kickoffs/restarts happened

and, if you pass the final on-screen match clock with `--final-minute`:
  * in-game minutes per real second — the number the tournament ruleset needs

The exports use the system locale, which writes decimals with a comma
(`1,10199964`). That is not valid JSON, so it is repaired before parsing —
a plain `json.load` on these files fails.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SIM_FIXED_DT = 0.019


def load_timeplot(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    # Locale decimal comma -> dot, but only between digits, so the JSON
    # separators between array elements survive.
    return json.loads(re.sub(r"(?<=\d),(?=\d)", ".", raw))


def series(doc: dict, name: str):
    for s in doc.get("series", []):
        if s["name"] == name:
            return s
    return None


def linear_fit(xs, ys) -> tuple[float, float]:
    """Least-squares slope/intercept. No numpy dependency."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return float("nan"), float("nan")
    slope = num / den
    return slope, my - slope * mx


def report_game_clock(doc: dict, span_s: float) -> None:
    """Time compression, straight from the game's own clock getters.

    `Current Simulation Time` is logged as a channel and the plot's x axis is
    real seconds, so the slope between them IS how many game-seconds pass per
    real second. No on-screen reading and no `--final-minute` needed.

    Only meaningful for a REAL-GAME export: the offline sim's float catalog is
    misaligned past ~index 37 and returns nonsense for this whole range (it
    reports Pi as 0.019).
    """
    current = series(doc, "P.F.CurrentSimulationTime")
    maximum = series(doc, "P.F.MaxSimulationTime")
    remaining = series(doc, "P.F.SimulationTimeRemaining")
    fixed_dt = series(doc, "P.F.FixedDeltaTime")
    delta = series(doc, "P.F.DeltaTime")
    pi = series(doc, "P.F.Pi")
    if not current:
        return

    print("\n-- game clock (read from the API, not inferred) --")
    sanity = ""
    if pi and pi["y"]:
        ok = abs(pi["y"][-1] - 3.14159) < 0.01
        sanity = f"  Pi reads {pi['y'][-1]:.5f} " + (
            "(sane — catalog is aligned)" if ok
            else "(WRONG — catalog misaligned, distrust this whole group)")
        print(sanity)

    for label, s in (("Current Simulation Time", current),
                     ("Max Simulation Time", maximum),
                     ("Simulation Time Remaining", remaining),
                     ("Delta Time", delta),
                     ("Fixed Delta Time", fixed_dt)):
        if s and s["y"]:
            ys = s["y"]
            print(f"  {label:26s} first {ys[0]:10.4f}   last {ys[-1]:10.4f}")

    if len(current["x"]) > 1 and span_s > 0:
        slope, _ = linear_fit(current["x"], current["y"])
        print(f"\n  game seconds per real second   {slope:.4f}")
        if slope > 0:
            print(f"  real seconds per game minute   {60.0 / slope:.4f}")
        if maximum and maximum["y"]:
            full = maximum["y"][-1]
            print(f"  match length {full:.1f} game seconds "
                  f"= {full / 60.0:.1f} game minutes "
                  f"= {full / slope:.1f} real seconds" if slope > 0 else "")
    print("  (sim exports are meaningless here — real-game recording only)")


def report_presence(doc: dict) -> None:
    """When did each player stop moving — i.e. get taken out of play?

    There is no "is player N present" query, so this works off the raw X/Z
    channels `circle_probe.py` logs. A player still in play jitters; one that
    has been removed has a position that stops changing entirely. The reported
    time is the first sample of the final motionless run, and only a run that
    lasts to the END of the recording counts — a player standing still for a
    while mid-match (the blank side does exactly that) must not be reported as
    removed, so a stationary stretch that later moves again is ignored.
    """
    prefixes = []
    for team in ("T", "O"):
        for slot in (1, 2, 3, 4):
            name = f"{team}{slot}"
            # Transform channels are emitted as P.X.<short>.X (current) and
            # Probe.<name>.X (older exports).
            for pattern in (f"P.X.{name}", f"Probe.{name}"):
                if series(doc, f"{pattern}.X"):
                    prefixes.append(pattern)
                    break
    if not prefixes:
        return

    print("\n-- player presence (last motionless stretch) --")
    print("  a player removed from play stops moving and never moves again")
    for name in prefixes:
        sx, sz = series(doc, f"{name}.X"), series(doc, f"{name}.Z")
        xs, ys, zs = sx["x"], sx["y"], sz["y"]
        n = min(len(ys), len(zs))
        if n < 3:
            continue
        # Walk back from the end while the position is unchanged.
        eps = 1e-4
        i = n - 1
        while i > 0 and abs(ys[i] - ys[i - 1]) < eps and abs(zs[i] - zs[i - 1]) < eps:
            i -= 1
        frozen_from = xs[i] if i < n - 1 else None
        span = xs[n - 1] - xs[i] if i < n - 1 else 0.0
        final = f"({ys[-1]:.2f}, {zs[-1]:.2f})"

        # Teleports are the real evidence. Standing still only means a player
        # was TOLD to stand still — this probe commands exactly that for slots
        # 2-4 and for the whole blank side, so stillness alone proves nothing.
        # A player taken out of play has to get moved somewhere, which shows up
        # as a single-tick jump far larger than a walk can cover (walk speed is
        # ~7 m/s, so ~0.13 m per 0.019 s tick).
        jumps = []
        for k in range(1, n):
            step = ((ys[k] - ys[k - 1]) ** 2 + (zs[k] - zs[k - 1]) ** 2) ** 0.5
            if step > 2.0:
                jumps.append((xs[k], step, ys[k], zs[k]))

        if frozen_from is None:
            print(f"  {name:16s} moving at the whistle          final {final}")
        else:
            print(f"  {name:16s} motionless from {frozen_from:8.3f}s "
                  f"({span:5.1f}s to the end)  final {final}")
        for when, dist, jx, jz in jumps[:4]:
            print(f"       TELEPORT at {when:8.3f}s  {dist:6.1f} m "
                  f"-> ({jx:.2f}, {jz:.2f})")
        if len(jumps) > 4:
            print(f"       ... and {len(jumps) - 4} more jumps "
                  f"(kickoff resets also teleport everyone)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("timeplot", type=Path)
    ap.add_argument("--final-minute", type=float, default=None,
                    help="on-screen match clock when the match ended")
    args = ap.parse_args()

    doc = load_timeplot(args.timeplot)
    all_series = doc.get("series", [])
    if not all_series:
        print("no series in this export", file=sys.stderr)
        return 1

    # Time comes from the x axis, which is the game's own seconds and is present
    # on EVERY channel — not from any counter. One sample is emitted per tick,
    # so the sample spacing IS the tick interval. This works on any export,
    # including recordings made by graphs that were never built for this.
    reference = all_series[0]
    xs = reference["x"]
    sim_time = doc.get("simTime", xs[-1] if xs else 0.0)
    span_s = xs[-1] - xs[0] if len(xs) > 1 else 0.0

    print(f"file            {args.timeplot.name}")
    print(f"channels        {len(all_series)}")
    print(f"samples         {len(xs)}")
    print(f"recording span  {span_s:.3f} s   (simTime {sim_time:.3f})")

    print("\n-- tick interval (from sample spacing) --")
    if len(xs) > 1:
        mean_dt = span_s / (len(xs) - 1)
        steps = [b - a for a, b in zip(xs, xs[1:])]
        steps_sorted = sorted(steps)
        median_dt = steps_sorted[len(steps_sorted) // 2]
        print(f"  mean dt       {mean_dt:.6f} s   ({1.0 / mean_dt:.2f} ticks/s)")
        print(f"  median dt     {median_dt:.6f} s")
        print(f"  min/max dt    {steps_sorted[0]:.6f} / {steps_sorted[-1]:.6f}")
        verdict = "MATCH" if abs(median_dt - SIM_FIXED_DT) < 5e-4 else \
                  "MISMATCH — the sim's FIXED_DT is wrong"
        print(f"  sim FIXED_DT  {SIM_FIXED_DT:.6f}   ({verdict})")

    report_game_clock(doc, span_s)

    # A counter channel is optional; when present it cross-checks the above.
    counter = series(doc, "ClockProbe.Ticks") or series(doc, "Probe.Ramp")
    if counter and len(counter["x"]) > 1:
        slope, _ = linear_fit(counter["x"], counter["y"])
        print(f"\n  counter '{counter['name']}' rises {slope:.4f} /s "
              f"(final {counter['y'][-1]:.0f})")
        print("  NOTE: an increment feeding N consumers executes N times per "
              "tick, so treat this as a ramp, not a tick count.")

    kicks = series(doc, "ClockProbe.Kickoffs") or series(doc, "Probe.Kickoffs")
    if kicks and kicks["y"]:
        print(f"\n  restarts      {kicks['y'][-1]:.0f} kickoffs")

    for leaf in ("TeamScore", "OppScore", "TeamPossessionPct", "OppPossessionPct",
                 "HasBall"):
        found = series(doc, f"ClockProbe.{leaf}") or series(doc, f"Probe.{leaf}")
        if found and found["y"]:
            print(f"  {leaf:20s} final {found['y'][-1]:.2f}")

    report_presence(doc)

    if args.final_minute is None:
        print("\nPass --final-minute <on-screen clock at the whistle> to get "
              "minutes-per-second.")
        return 0

    print("\n-- in-game minutes --")
    if span_s <= 0:
        print("  recording has no span; cannot derive")
        return 1
    per_second = args.final_minute / span_s
    print(f"  on-screen clock at the end   {args.final_minute:g}'")
    print(f"  in-game minutes per second   {per_second:.4f}")
    print(f"  seconds per in-game minute   {1.0 / per_second:.4f}")
    print(f"  => a '90 minute' match is    {90.0 / per_second:.1f} s of sim time")
    if abs(per_second - 1.0) < 0.05:
        print("  (~1 minute per second — the tournament ruleset's second==minute "
              "assumption holds)")
    else:
        print(f"  NOTE: not 1:1. Scale TournamentRules by {1.0 / per_second:.4f} "
              f"s per in-game minute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
