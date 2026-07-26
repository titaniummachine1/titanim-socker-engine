"""`--profile`: where the graph's size and per-tick cost actually come from.

Answers the two questions that otherwise need a manual archeology pass every
time a feature lands:

  * WHICH module/function emitted these nodes — so "the graph grew by 900" has
    an address instead of being a mystery.
  * How many node EVALUATIONS a tick costs, which is not the same number.
    A `titanium.nodefn` graph function stores its body once but evaluates it
    once per call site, so a change can shrink the file while making the engine
    do more work. Reporting only nodes would hide exactly that trade.

Install BEFORE `graph.build()` (it wraps the node constructor), report after.

    python scripts/build_titanium.py --profile
    python scripts/build_titanium.py --profile --baseline   # rewrite the baseline

The baseline lives next to this file and is compared on every profiled build;
a module that grows more than `REGRESSION_FRAC` is called out by name.
"""
from __future__ import annotations

import collections
import json
import traceback
from pathlib import Path

BASELINE_PATH = Path(__file__).resolve().parent / "profile_baseline.json"
REGRESSION_FRAC = 0.10


def _variant() -> str:
    """Baselines are per build variant — the challenger is ~2x the plain build,
    so comparing one against the other reports a 94% 'regression' that is just
    the anti-tackle module existing."""
    from titanium._env import WITH_ANTI_TACKLE

    return "anti_tackle" if WITH_ANTI_TACKLE else "plain"

_owner: dict[str, str] = {}
_installed = False


def install() -> None:
    """Wrap `AddNode` so every node records the titanium frame that built it."""
    global _installed
    if _installed:
        return
    _installed = True

    import AIGamePyLibrary.lib as lib
    import AIGamePyLibrary.nodes as nodes_mod

    original = lib.AddNode

    def traced(*args, **kwargs):
        node = original(*args, **kwargs)
        _owner[node.data["sID"]] = _blame()
        return node

    lib.AddNode = traced
    nodes_mod.AddNode = traced


def _blame() -> str:
    """Innermost titanium (or ball_trajectory_graph) frame on the stack."""
    for frame in reversed(traceback.extract_stack()[:-2]):
        path = frame.filename.replace("\\", "/")
        if "/titanium/" in path:
            return f"titanium/{path.rsplit('/', 1)[1]}:{frame.name}"
        if "ball_trajectory_graph" in path:
            return f"ball_trajectory_graph:{frame.name}"
    return "<unattributed>"


def _eval_cost(nodes, connections) -> tuple[int, dict[str, int]]:
    """Per-tick node evaluations, counting a function body once per call site.

    Global nodes evaluate once. A node owned by a `CreateFunction` evaluates
    once per `Function` node naming that definition — that is what
    `eval_function_call` does (fresh frame per call), so this is the real
    per-tick work, not the file's node count.
    """
    by_sid = {n["sID"]: n for n in nodes}
    # CreateFunction sID -> its name, so call sites can be counted by name.
    definition_name = {n["sID"]: n.get("modifier", "") for n in nodes
                       if n["id"] == "CreateFunction"}
    calls = collections.Counter(n.get("modifier", "") for n in nodes
                                if n["id"] == "Function")

    body_size = collections.Counter()
    for node in nodes:
        owner_sid = node.get("ownerFunctionSID", "")
        if owner_sid and owner_sid in definition_name:
            body_size[definition_name[owner_sid]] += 1

    total = sum(1 for n in nodes if not n.get("ownerFunctionSID"))
    per_function = {}
    for name, size in body_size.items():
        n_calls = calls.get(name, 0)
        per_function[name] = size * n_calls
        total += size * n_calls
    return total, per_function


def _uncalled_bodies(nodes) -> dict[str, int]:
    """Function bodies no `Function` node calls — shipped but never evaluated."""
    definition_name = {n["sID"]: n.get("modifier", "") for n in nodes
                       if n["id"] == "CreateFunction"}
    calls = collections.Counter(n.get("modifier", "") for n in nodes
                                if n["id"] == "Function")
    body_size = collections.Counter()
    for node in nodes:
        owner_sid = node.get("ownerFunctionSID", "")
        if owner_sid in definition_name:
            body_size[definition_name[owner_sid]] += 1
    return {name: size for name, size in body_size.items() if not calls.get(name)}


def report(nodes, connections, *, write_baseline: bool = False) -> None:
    if not _installed:
        print("profile: install() was not called before build — no attribution")
        return

    by_module = collections.Counter()
    by_function = collections.Counter()
    for node in nodes:
        who = _owner.get(node["sID"], "<unattributed>")
        by_module[who.split(":")[0]] += 1
        by_function[who] += 1

    evals, per_function = _eval_cost(nodes, connections)
    approx_mb = (len(nodes) * 576 + len(connections) * 951) / 1e6

    print("\n== profile ==")
    print(f"  nodes {len(nodes)}   connections {len(connections)}   "
          f"~{approx_mb:.2f} MB   per-tick evaluations {evals}")
    if per_function:
        print("\n  graph functions (body x call sites = evals):")
        for name, cost in sorted(per_function.items(), key=lambda kv: -kv[1]):
            print(f"    {cost:6d}  {name}")

    dead = _uncalled_bodies(nodes)
    if dead:
        # DCE keeps a CreateFunction body alive even when no Function node names
        # it, so these ship without ever evaluating: pure file weight.
        print(f"\n  uncalled function bodies ({sum(dead.values())} nodes, "
              f"never evaluated):")
        for name, size in sorted(dead.items(), key=lambda kv: -kv[1]):
            print(f"    {size:6d}  {name}")

    print("\n  nodes by module:")
    for name, count in by_module.most_common():
        print(f"    {count:6d}  {name}")
    print("\n  nodes by function (top 20):")
    for name, count in by_function.most_common(20):
        print(f"    {count:6d}  {name}")

    current = {"nodes": len(nodes), "connections": len(connections),
               "evals": evals, "by_module": dict(by_module)}

    variant = _variant()
    stored = {}
    if BASELINE_PATH.is_file():
        stored = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    if write_baseline:
        stored[variant] = current
        BASELINE_PATH.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
        print(f"\n  baseline written for '{variant}': {BASELINE_PATH.name}")
        return

    if variant not in stored:
        print(f"\n  no '{variant}' baseline yet — run this same command with "
              f"--baseline to create one")
        return

    print(f"\n  vs baseline ({variant}):")
    _diff(stored[variant], current)


def _diff(base: dict, current: dict) -> None:
    for key in ("nodes", "connections", "evals"):
        was, now = base.get(key, 0), current[key]
        delta = now - was
        pct = (delta / was * 100) if was else 0.0
        flag = "  <-- REGRESSION" if was and delta > was * REGRESSION_FRAC else ""
        print(f"    {key:12s} {was:6d} -> {now:6d}  ({delta:+d}, {pct:+.1f}%){flag}")

    base_modules = base.get("by_module", {})
    grew = []
    for name, now in current["by_module"].items():
        was = base_modules.get(name, 0)
        if now - was > max(20, was * REGRESSION_FRAC):
            grew.append((now - was, name, was, now))
    if grew:
        print("\n    modules that grew:")
        for delta, name, was, now in sorted(grew, reverse=True):
            print(f"      {delta:+6d}  {name}  ({was} -> {now})")
