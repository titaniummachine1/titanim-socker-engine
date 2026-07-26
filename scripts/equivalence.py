"""Is this build behaviourally identical to a reference build?

For refactors that are supposed to change nothing — extracting a
`titanium.nodefn` function, swapping a hand-rolled clamp for the native node,
deleting a line that provably cannot fire — this answers in a couple of minutes
what the full gate answers in much longer, and answers it far more sharply.

The simulator is deterministic for a fixed (home, away, opening, seed), so two
graphs that decide identically produce identical match statistics. Possession
is reported to float precision over thousands of ticks, and the mercy rule ends
a blowout on an exact tick, so ANY behavioural divergence shows up here — it
does not average out.

    python scripts/equivalence.py                       # vs the live champion
    python scripts/equivalence.py --against out/Titanium_challenger.txt
    python scripts/equivalence.py --with-anti-tackle --secs 60

Exit code 0 = identical everywhere. Non-zero = diverged, with the differing
fields printed.

This does NOT replace the gate. Identical behaviour means the refactor is safe
to promote without re-gating; a build that CHANGES behaviour still has to win
the roster and the head-to-head (see `titanium/README.md`).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ENGINE_ROOT = SCRIPTS.parent
WORKSPACE = ENGINE_ROOT.parent
SIM_ROOT = WORKSPACE / "aicomp-soccer-sim"
SAVES = (Path.home() / "AppData" / "LocalLow" / "Unicorn One" / "AIComp"
         / "Saves" / "Soccer")

ROSTER = [
    ("Poponeta", SAVES / "Poponeta.txt"),
    ("AIA3", SAVES / "AIA3.txt"),
    ("Haialand-v2", SAVES / "Haialand-v2.txt"),
]

# Fields that name the file rather than describe the match.
IDENTITY_FIELDS = {"home", "away"}


def build_current(out_path: Path, anti_tackle: bool) -> None:
    """Build the working tree to `out_path` without touching any live file."""
    argv = [sys.executable, "-c", _BUILD_SNIPPET, str(SCRIPTS), str(out_path)]
    if anti_tackle:
        argv.append("--with-anti-tackle")
    done = subprocess.run(argv, capture_output=True, text=True)
    if done.returncode != 0:
        print(done.stdout + done.stderr, file=sys.stderr)
        raise SystemExit("build failed")
    print(done.stdout.strip())


_BUILD_SNIPPET = """
import sys
from pathlib import Path
scripts, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, scripts)
from titanium import _env
from titanium._env import graph_data
from titanium import graph
from AIGamePyLibrary import SaveData
graph.build()
SaveData(out, layout="grid", optimize="release", verbose=False)
n = len(graph_data["serializableNodes"])
c = len(graph_data["serializableConnections"])
print(f"built {Path(out).name}: nodes={n} conns={c} "
      f"{Path(out).stat().st_size / 1e6:.2f} MB")
"""


def run_match(home: str, away: str, opening: str, secs: str, seed: int) -> dict:
    cmd = ["cargo", "run", "--release", "--bin", "soccer_headless", "--",
           "--home", home, "--away", away, "--secs", secs,
           "--opening", opening, "--seed", str(seed), "--quiet"]
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SIM_ROOT))
    if done.returncode != 0:
        print(done.stderr[-2000:], file=sys.stderr)
        raise SystemExit("match failed")
    lines = [l for l in done.stdout.strip().splitlines() if l.strip()]
    return json.loads(lines[-1])


def outcome(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in IDENTITY_FIELDS}


def main() -> int:
    argv = sys.argv[1:]
    anti_tackle = "--with-anti-tackle" in argv or "--challenger" in argv
    secs = argv[argv.index("--secs") + 1] if "--secs" in argv else "180"
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 12345
    if "--against" in argv:
        reference = Path(argv[argv.index("--against") + 1]).resolve()
    else:
        reference = SAVES / "Titanium.txt"
    if not reference.is_file():
        raise SystemExit(f"reference build not found: {reference}")

    with tempfile.TemporaryDirectory() as tmp:
        candidate = Path(tmp) / "candidate.txt"
        build_current(candidate, anti_tackle)
        print(f"reference: {reference}  ({reference.stat().st_size / 1e6:.2f} MB)")
        print(f"fixtures:  {len(ROSTER)} opponents x both sides, {secs}s, seed {seed}\n")

        identical = True
        for name, path in ROSTER:
            if not path.is_file():
                print(f"  {name:12s} SKIPPED (no file at {path})")
                continue
            opponent = f"graph:{path}"
            for opening, as_home in (("home", True), ("away", False)):
                results = {}
                for label, graph_path in (("REF", reference), ("NEW", candidate)):
                    side = f"graph:{graph_path}"
                    h, a = (side, opponent) if as_home else (opponent, side)
                    results[label] = outcome(run_match(h, a, opening, secs, seed))
                ref, new = results["REF"], results["NEW"]
                same = ref == new
                identical &= same
                score = f"{ref.get('score_home')}:{ref.get('score_away')}"
                poss = (f"{ref.get('possession_s_home', 0):.3f}/"
                        f"{ref.get('possession_s_away', 0):.3f}")
                print(f"  {name:12s} {'HOME' if as_home else 'AWAY'}  {score:6s} "
                      f"poss {poss:16s} ticks={ref.get('ticks')}  "
                      f"{'IDENTICAL' if same else '*** DIVERGED ***'}")
                if not same:
                    for key in sorted(set(ref) | set(new)):
                        if ref.get(key) != new.get(key):
                            print(f"        {key}: REF={ref.get(key)!r}  "
                                  f"NEW={new.get(key)!r}")

    if identical:
        print("\nIDENTICAL on every fixture — this refactor changed no decision.")
        return 0
    print("\nDIVERGED — this is a behaviour change. It needs the full gate.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
