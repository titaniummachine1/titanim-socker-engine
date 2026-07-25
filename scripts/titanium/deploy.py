"""Candidate/test/promote file I/O — deliberately separate from
`titanium.graph`, which only builds the node graph in memory and knows
nothing about where any of this ends up on disk."""
from __future__ import annotations

import sys

from titanium._env import graph_data
from AIGamePyLibrary import SaveData
STRIP_DEBUG = "--strip-debug" in sys.argv

from titanium.paths import BACKUPS_DIR, CANDIDATE_OUT, LOCAL_OUT, SAVES, SAVES_TEST


def drop_unread_variables() -> int:
    """Delete SetVariable nodes whose value no GetVariable ever reads.

    AIGamePyLibrary's own `removeUnusedNodes` cannot do this: a SetVariable is
    a SINK, so it always looks "used" and everything feeding it is kept alive
    with it. That is how Titanium ended up carrying the entire 405-node
    TrajectorySolve service -- 29 of its 31 SetVariables were written and
    never read, and the whole computation behind them came along.

    Removing the writes turns that computation into genuine dead nodes, which
    the library's pruner then collects. Run this BEFORE SaveData.

    Returns how many writes were dropped, so a build that suddenly starts
    dropping more is visible rather than silent.
    """
    nodes = graph_data["serializableNodes"]
    read = {n["modifier"] for n in nodes if n["id"] == "GetVariable"}
    doomed = {
        n["sID"] for n in nodes
        if n["id"] == "SetVariable" and n["modifier"] not in read
    }
    if not doomed:
        return 0

    dead_ports = {
        p["sID"] for n in nodes if n["sID"] in doomed
        for p in n.get("serializablePorts", [])
    }
    graph_data["serializableConnections"] = [
        c for c in graph_data["serializableConnections"]
        if c["port0SID"] not in dead_ports and c["port1SID"] not in dead_ports
    ]
    graph_data["serializableNodes"] = [n for n in nodes if n["sID"] not in doomed]
    return len(doomed)


def strip_debug_sinks() -> int:
    """Delete DebugDraw/TimePlot sinks, so the pruner can collect everything
    that existed only to feed them.

    29% of the graph (1340 of 4599 nodes) is reachable ONLY from debug sinks
    and contributes nothing to a SoccerController. Tournaments do not render
    debug lines, so in a competition build that is pure weight -- and the
    save format costs ~2.4 KB per node, which is what pushed the file over
    the 10 MB limit.

    Safe by construction: `titanium.debug_viz` only ever calls Debug*/TimePlot
    and never feeds a movement decision, so removing these cannot change play.
    Verify that claim with a gate run anyway, never on the docstring alone.
    """
    nodes = graph_data["serializableNodes"]
    sinks = {"DebugDrawLine", "DebugDrawDisc", "TimePlot", "Debug"}
    doomed = {n["sID"] for n in nodes if n["id"] in sinks}
    if not doomed:
        return 0
    dead_ports = {
        p["sID"] for n in nodes if n["sID"] in doomed
        for p in n.get("serializablePorts", [])
    }
    graph_data["serializableConnections"] = [
        c for c in graph_data["serializableConnections"]
        if c["port0SID"] not in dead_ports and c["port1SID"] not in dead_ports
    ]
    graph_data["serializableNodes"] = [n for n in nodes if n["sID"] not in doomed]
    return len(doomed)


def drop_dangling_reads() -> int:
    """Delete GetVariable nodes whose output feeds nothing.

    Stripping a debug sink can orphan a GetVariable that only ever fed it.
    The orphan still makes its SetVariable look read, so the write and its
    whole producing subgraph survive -- which is why the passes must run to a
    fixpoint rather than once each.
    """
    nodes = graph_data["serializableNodes"]
    wired = set()
    for c in graph_data["serializableConnections"]:
        wired.add(c["port0SID"])
        wired.add(c["port1SID"])
    doomed = {
        n["sID"] for n in nodes if n["id"] == "GetVariable"
        and not any(p["sID"] in wired for p in n.get("serializablePorts", []))
    }
    if not doomed:
        return 0
    graph_data["serializableNodes"] = [n for n in nodes if n["sID"] not in doomed]
    return len(doomed)


def optimise_to_fixpoint() -> tuple[int, int]:
    """Alternate the variable passes until neither finds anything.

    One pass each is not enough: dropping a write can orphan a read, and
    dropping a read can orphan a write. Loop until stable (bounded, because
    every iteration strictly removes nodes).
    """
    writes = reads = 0
    for _ in range(10):
        w = drop_unread_variables()
        r = drop_dangling_reads()
        writes += w
        reads += r
        if not (w or r):
            break
    return writes, reads


def assert_variables_intact() -> None:
    """Every GetVariable must still have a SetVariable writing its name.

    A variable pair is a data edge with NO wire between the two nodes, so the
    library's pruner cannot see it -- it keeps or drops each node purely on
    whether its own ports are connected. That makes variables the one thing an
    optimiser can silently sever: delete the write and the read still looks
    perfectly connected while now reading nothing.

    Both passes here are meant to be safe (writes are only dropped when no
    read exists; only Debug/TimePlot sinks are stripped), but "meant to be" is
    not a guarantee. This turns it into an enforced invariant that fails the
    build instead of shipping a graph whose variables have been cut.
    """
    nodes = graph_data["serializableNodes"]
    written = {n["modifier"] for n in nodes if n["id"] == "SetVariable"}
    read = {n["modifier"] for n in nodes if n["id"] == "GetVariable"}
    orphaned = read - written
    if orphaned:
        raise SystemExit(
            "OPTIMISER BUG: GetVariable with no SetVariable feeding it: "
            + ", ".join(sorted(orphaned))
        )


def write_candidate() -> None:
    CANDIDATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    before = len(graph_data["serializableNodes"])
    stripped = strip_debug_sinks() if STRIP_DEBUG else 0
    dropped, dangling = optimise_to_fixpoint()
    # SaveData prunes unreachable nodes itself (pruneUnusedNodes defaults True);
    # dropping the unread writes first is what lets it reach their producers.
    assert_variables_intact()
    SaveData(str(CANDIDATE_OUT), layout="grid")
    assert_variables_intact()   # again: SaveData prunes too
    if stripped:
        print(f"optimiser: stripped {stripped} debug/TimePlot sink(s) "
              f"(competition build -- no on-screen debug)")
    if dangling:
        print(f"optimiser: dropped {dangling} GetVariable(s) reading into nothing")
    if dropped or stripped or dangling:
        after = len(graph_data["serializableNodes"])
        print(f"optimiser: dropped {dropped} unread SetVariable(s), "
              f"{before} -> {after} nodes")
    print(f"Wrote candidate: {CANDIDATE_OUT}")
    print(f"nodes={len(graph_data['serializableNodes'])} conns={len(graph_data['serializableConnections'])}")

    SAVES_TEST.parent.mkdir(parents=True, exist_ok=True)
    SAVES_TEST.write_text(CANDIDATE_OUT.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote test build (load '{SAVES_TEST.stem}' in-game to try it yourself): {SAVES_TEST}")


def promote_if_requested() -> None:
    if "--promote" not in sys.argv:
        print(
            "\nNOT deployed to live — this only builds a candidate.\n"
            "Gate it first (aicomp-soccer-sim/scripts/gate_round_robin.py) "
            "against the currently-live build, then re-run with --promote "
            "once it's actually won."
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
