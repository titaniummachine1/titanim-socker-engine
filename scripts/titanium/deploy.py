"""Candidate/test/promote file I/O — deliberately separate from
`titanium.graph`, which only builds the node graph in memory and knows
nothing about where any of this ends up on disk."""
from __future__ import annotations

import sys

from titanium._env import graph_data
from AIGamePyLibrary import SaveData
STRIP_DEBUG = "--strip-debug" in sys.argv
EXPLAIN = "--explain" in sys.argv
DEDUPE = "--no-cse" not in sys.argv

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


def _wiring():
    """(producer_of[node][port_name], nodes_by_sid) from the current graph."""
    nodes = {n["sID"]: n for n in graph_data["serializableNodes"]}
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
    return src, nodes


def _is_constant_origin(sid, src, nodes, depth=0):
    """True if this node is a Vector3 built from three constant zeros.

    Statically detectable at build time: a literal (0,0,0) fed to a draw is
    almost always an uninitialised value rather than an intentional target --
    the centre spot is not somewhere anything meaningful points.
    """
    if depth > 6 or sid not in nodes:
        return False
    n = nodes[sid]
    if n["id"] == "Vector3":
        ins = src.get(sid, {})
        if len(ins) != 3:
            return False
        for comp in ins.values():
            c = nodes.get(comp)
            if not c or c["id"] != "Float":
                return False
            try:
                if float(c.get("modifier", "1")) != 0.0:
                    return False
            except ValueError:
                return False
        return True
    if n["id"] == "Relay":
        nxt = src.get(sid, {}).get("Any1")
        return _is_constant_origin(nxt, src, nodes, depth + 1) if nxt else False
    return False


def warn_origin_draws() -> int:
    """Warn about DebugDraw endpoints hard-wired to (0,0,0).

    Drawing from or to world origin is never meaningful -- it is the centre
    spot, not a target -- so it reliably indicates a value that was never set.
    Titanium was drawing its own-goal threat cone from (0,0) for exactly this
    reason. Warn rather than fail: some of these are inherited from shared
    code and blocking the build would be worse than flagging it.
    """
    src, nodes = _wiring()
    draws = {"DebugDrawLine", "DebugDrawDisc"}
    hits = []
    for n in graph_data["serializableNodes"]:
        if n["id"] not in draws:
            continue
        for pname, producer in src.get(n["sID"], {}).items():
            if pname.startswith("Vector3") and _is_constant_origin(producer, src, nodes):
                hits.append((n["id"], pname))
    for kind, pname in hits:
        print(f"  WARNING: {kind}.{pname} is wired to a constant (0,0,0) -- "
              f"drawing from/to the centre spot usually means an unset value")
    return len(hits)


def report_distance_from_control(max_depth: int = 6) -> None:
    """BFS outward from the nodes that actually move players.

    SoccerController1-4 are depth 0. Every producer feeding them is depth 1,
    their producers depth 2, and so on -- so depth is "how far from affecting
    play". Anything unreachable contributes nothing to a decision at all.

    Printed only under --explain, as a debugging aid for finding code that has
    drifted away from execution.
    """
    src, nodes = _wiring()
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


# Nodes that must NEVER be merged, even when structurally identical: each one
# is an effect, not a value. Two identical SetVariable writes are two writes;
# two identical controllers drive two players; two identical draws are two
# lines. Merging any of them silently deletes behaviour.
_EFFECTFUL = {
    "SetVariable", "GetVariable", "DebugDrawLine", "DebugDrawDisc", "TimePlot",
    "Debug", "Stat", "CreateFunction", "Function", "ConstructSoccerProperties",
}


def deduplicate_nodes() -> int:
    """Common-subexpression elimination over the built graph.

    AIGamePyLibrary's @cache dedupes on PYTHON call arguments, so two calls
    passing different-but-equal Node objects miss it and build structurally
    identical subgraphs anyway. This hashes the BUILT graph instead, which is
    why it still finds ~24% duplication after caching.

    Iterative congruence closure: start by (node id, modifier), then refine by
    (class, input classes) until stable -- two nodes are the same computation
    only if their inputs are also the same computation, all the way down.
    Then keep one representative per class and rewire consumers to it.
    """
    nodes = graph_data["serializableNodes"]
    conns = graph_data["serializableConnections"]
    by_sid = {n["sID"]: n for n in nodes}

    port = {}
    for n in nodes:
        for p in n.get("serializablePorts", []):
            port[p["sID"]] = (n["sID"], p["polarity"], p["id"])

    ins = {}
    for c in conns:
        a, b = port.get(c["port0SID"]), port.get(c["port1SID"])
        if not a or not b:
            continue
        (na, pa, ia), (nb, pb, ib) = a, b
        if pa == 1 and pb == 0:
            ins.setdefault(nb, {})[ib] = na
        elif pb == 1 and pa == 0:
            ins.setdefault(na, {})[ia] = nb

    # ownerFunctionSID is part of identity: two structurally identical nodes in
    # DIFFERENT function bodies are not the same computation, because each body
    # binds its own Param1..N per call. Merging across bodies makes one call's
    # arguments leak into another -- it diverged at tick 0, moving a player 30 m.
    cls = {
        n["sID"]: (n["id"], str(n.get("modifier", "")), n.get("ownerFunctionSID", ""))
        for n in nodes
    }
    for _ in range(12):
        sig = {}
        for sid in cls:
            deps = tuple(sorted((k, cls[v]) for k, v in ins.get(sid, {}).items()))
            sig[sid] = (cls[sid], deps)
        renum, nxt = {}, {}
        for sid, v in sig.items():
            renum.setdefault(v, len(renum))
            nxt[sid] = renum[v]
        if nxt == cls:
            break
        cls = nxt

    groups = {}
    for sid, c in cls.items():
        groups.setdefault(c, []).append(sid)

    # sID -> the representative it is being replaced by
    replace = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort()
        keep = next(
            (m for m in members if by_sid[m]["id"] not in _EFFECTFUL), None
        )
        if keep is None:
            continue          # whole class is effectful: never merge
        for m in members:
            if m != keep and by_sid[m]["id"] not in _EFFECTFUL:
                replace[m] = keep
    if not replace:
        return 0

    # Rewire: any connection referencing a dropped node's OUTPUT port moves to
    # the representative's port of the same name.
    out_port = {}
    for n in nodes:
        for p in n.get("serializablePorts", []):
            if p["polarity"] == 1:
                out_port.setdefault(n["sID"], {})[p["id"]] = p["sID"]

    remap = {}
    for dead, keep in replace.items():
        for pid, psid in out_port.get(dead, {}).items():
            tgt = out_port.get(keep, {}).get(pid)
            if tgt:
                remap[psid] = tgt

    kept = []
    for c in conns:
        c["port0SID"] = remap.get(c["port0SID"], c["port0SID"])
        c["port1SID"] = remap.get(c["port1SID"], c["port1SID"])
        if c["port0SID"] != c["port1SID"]:
            kept.append(c)
    graph_data["serializableConnections"] = kept
    graph_data["serializableNodes"] = [n for n in nodes if n["sID"] not in replace]
    return len(replace)


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
    merged = deduplicate_nodes() if DEDUPE else 0
    if merged:
        print(f"optimiser: merged {merged} duplicate computation(s) (CSE)")
    if EXPLAIN:
        report_distance_from_control()
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
