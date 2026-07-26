"""Candidate/test/promote file I/O — deliberately separate from
`titanium.graph`, which only builds the node graph in memory and knows
nothing about where any of this ends up on disk.

All graph optimisation is delegated to AIGamePyLibrary.SaveData(optimize=...)
— the same path the pylib PR uses (unread-variable prune, DCE, optional
debug-strip fixpoint). Do not duplicate optimiser passes here.
"""
from __future__ import annotations

import sys

from titanium import _env as _env_bootstrap  # noqa: F401 — sys.path for AIGamePyLibrary
from AIGamePyLibrary import SaveData
from titanium._env import graph_data, WITH_ANTI_TACKLE
from titanium.paths import (
    BACKUPS_DIR,
    CANDIDATE_OUT,
    LOCAL_OUT,
    LOCAL_TEST,
    SAVES,
    SAVES_CHALLENGER,
    SAVES_TEST,
)

STRIP_DEBUG = "--strip-debug" in sys.argv  # legacy alias; release is now the default
DEBUG_BUILD = "--debug" in sys.argv or "--normal" in sys.argv
# Default = pylib optimize="release" (DCE + debug-strip). Use --debug only when
# inspecting TimePlot / DebugDraw sinks.
EXPLAIN = "--explain" in sys.argv
CHALLENGER_OUT = CANDIDATE_OUT.parent / "Titanium_challenger.txt"

def deploy_test_copies(text: str, *, also_challenger_saves: bool | None = None) -> None:
    """Always overwrite Titanium_test for the user to watch — every build/test.

    Writes Unity Saves + local sim copy with a fresh mtime (so Explorer
    shows 'just updated'). Optional Titanium_challenger.txt under Saves
    when this is an anti-tackle build.
    """
    import time

    if also_challenger_saves is None:
        also_challenger_saves = WITH_ANTI_TACKLE

    dests = [SAVES_TEST, LOCAL_TEST]
    if also_challenger_saves:
        dests.append(SAVES_CHALLENGER)

    now = time.time()
    for dest in dests:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        # Force a fresh timestamp even if bytes were identical (Explorer).
        try:
            import os

            os.utime(dest, (now, now))
        except OSError:
            pass
        print(f"Auto-deployed TEST: {dest}")

    print(
        "\n>>> YOU CAN TEST: load 'Titanium_test' in Unity, or:\n"
        "    .\\scripts\\watch_bots.ps1 -Home Titanium_test -Away Haialand-v2\n"
    )


def report_distance_from_control(max_depth: int = 6) -> None:
    """BFS outward from SoccerController nodes (--explain debugging aid)."""
    port = {}
    for n in graph_data["serializableNodes"]:
        for p in n.get("serializablePorts", []):
            port[p["sID"]] = (n["sID"], p["polarity"], p["id"])
    src = {}
    for c in graph_data["serializableConnections"]:
        a, b = port.get(c["port0SID"]), port.get(c["port1SID"])
        if not a or not b:
            continue
        (na, pa, ia), (nb, pb, ib) = a, b
        if pa == 1 and pb == 0:
            src.setdefault(nb, {})[ib] = na
        elif pb == 1 and pa == 0:
            src.setdefault(na, {})[ia] = nb

    nodes = {n["sID"]: n for n in graph_data["serializableNodes"]}
    depth = {}
    frontier = [n["sID"] for n in graph_data["serializableNodes"]
                if n["id"].startswith("SoccerController")]
    for sid in frontier:
        depth[sid] = 0
    d = 0
    while frontier:
        nxt = []
        for sid in frontier:
            for producer in src.get(sid, {}).values():
                if producer not in depth:
                    depth[producer] = d + 1
                    nxt.append(producer)
        frontier = nxt
        d += 1

    total = len(nodes)
    print(f"  distance from player control ({total} nodes):")
    buckets = {}
    for sid, dd in depth.items():
        buckets.setdefault(min(dd, max_depth), []).append(sid)
    for k in sorted(buckets):
        label = f"{k}" if k < max_depth else f"{max_depth}+"
        print(f"    depth {label:>3}: {len(buckets[k]):5d} nodes")
    unreached = total - len(depth)
    print(f"    UNREACHED : {unreached:5d} nodes (affect no player decision)")


def write_candidate() -> None:
    """Write build output. Optimisation = AIGamePyLibrary SaveData only."""
    if WITH_ANTI_TACKLE:
        out_path = CHALLENGER_OUT
    else:
        out_path = CANDIDATE_OUT

    out_path.parent.mkdir(parents=True, exist_ok=True)
    before = len(graph_data["serializableNodes"])
    # Release by default; --debug/--normal keeps DebugDraw/TimePlot sinks.
    mode = "normal" if DEBUG_BUILD else "release"
    if EXPLAIN:
        report_distance_from_control()
    SaveData(str(out_path), layout="grid", optimize=mode, verbose=True)
    after = len(graph_data["serializableNodes"])
    if after != before:
        print(f"optimiser[{mode}]: {before} -> {after} nodes")
    print(f"Wrote candidate: {out_path}")
    print(f"nodes={len(graph_data['serializableNodes'])} conns={len(graph_data['serializableConnections'])}")
    if WITH_ANTI_TACKLE:
        print("  (challenger: binary-search anti_tackle module enabled)")

    # Always auto-deploy Titanium_test — never leave the user with a stale file.
    deploy_test_copies(out_path.read_text(encoding="utf-8"))

    if WITH_ANTI_TACKLE:
        _run_promotion_test()


def _run_promotion_test() -> None:
    """Always run the canonical total-goals promotion test after challenger build."""
    import subprocess
    from pathlib import Path

    sim_root = LOCAL_OUT.parents[2]
    script = sim_root / "scripts" / "promote_pipeline.py"
    if not script.is_file():
        print(f"\nWARN: promotion test not found at {script}")
        return
    print("\n== Running promotion test (automatic) ==\n")
    rc = subprocess.call([sys.executable, str(script)], cwd=str(sim_root))
    if rc != 0:
        raise SystemExit(rc)


def promote_if_requested() -> None:
    if WITH_ANTI_TACKLE:
        return
    if "--promote" not in sys.argv:
        print(
            "\nChallenger builds run promote_pipeline.py automatically.\n"
            "Manual --promote on straight-walk builds skips the tournament."
        )
        return

    text = CANDIDATE_OUT.read_text(encoding="utf-8")
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    if LOCAL_OUT.is_file():
        prior = LOCAL_OUT.read_bytes()
        n = 0
        backup_path = BACKUPS_DIR / "Titanium_pre_promote.txt"
        while backup_path.is_file() and backup_path.read_bytes() != prior:
            n += 1
            backup_path = BACKUPS_DIR / f"Titanium_pre_promote_{n}.txt"
        if not backup_path.is_file():
            backup_path.write_bytes(prior)
        print(f"Backed up previously-live build to {backup_path}")

    SAVES.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    SAVES.write_text(text, encoding="utf-8")
    LOCAL_OUT.write_text(text, encoding="utf-8")
    print(f"PROMOTED to live: {SAVES}")
    print(f"PROMOTED to live: {LOCAL_OUT}")
