"""Turn a repeated subgraph into ONE reusable graph function.

Why this exists
---------------
Every Python-level call in this package inlines its whole subtree into the
saved graph. `AIGamePyLibrary`'s constructor-level `@cache` already merges any
instance that COULD be shared, so every instance that survives into the file is
one the cache could not merge — N distinct copies of the same shape.

A `CreateFunction` body is emitted once; each call site is a single `Function`
node plus at most 4 parameter connections. The sim's `eval_function_call`
pushes a fresh call frame (with its own memo) per call site, so the body is
evaluated once per call — exactly as many evaluations as the inlined copies
cost. So this trades `N*body` nodes for `N + body` nodes at NEUTRAL per-tick
work, PROVIDED rule 1 below is respected.

Both halves are already proven in shipped graphs, not just here: the live
`Titanium.txt` carries 7 `CreateFunction` / 15 `Function` nodes (via
`ball_trajectory_graph`), and `Haialand-v2.txt` — a real game export — uses 96
`ClampFloat` nodes.

Rules — all three are load-bearing
----------------------------------
1. Wrap an EXISTING function boundary. Do not pull hoisted work inside: values
   currently computed once outside and shared across instances (e.g.
   `opponent_invariants`) get recomputed per call, which IS a per-tick
   regression even though the file gets smaller.

2. At most 4 parameters, exactly one return value. Extra results have to go
   through `SetVariable`, and those share a single slot that the next call
   clobbers before the first is read. That is the real, reproduced failure
   behind `ball_physics._build_event_leg_chain_function`'s warning.

3. NO NESTING — by policy, not by proven limitation. `build_function_nesting_probe.py`
   asserts a body calling another function "reads back null"; running that case
   through the sim (`scripts/fn_contract_probe.py` + `probe_dump`) shows it
   WORKING, so the claim is false for the sim and untested in Unity. Since a
   body is emitted once, inlining shared maths into it costs nothing, so there
   is no reason to spend the risk. `_BUILDING` enforces the policy at build
   time; lift it only after a real-game read, never on the doc alone.

Typing
------
A `Function` node's declared output type is "Any", and `Node.__add__` / `.x` /
etc. dispatch on `Node.type` — so params and call results would break every
`a - b` and `v.x` at the call sites. `graph_function` therefore stamps the
declared `params=` / `returns=` types onto the Node wrappers. Only the Python
wrapper changes; port wiring falls back to the node's real output port either
way (verified: probe T3/T7 feed a call result into typed Vector3/Float ports).
"""
from __future__ import annotations

from titanium._env import *  # noqa: F401,F403
from titanium._env import graph_data
from AIGamePyLibrary.lib import Node

# Constant nodes are left in global scope rather than captured into the body.
# `Float`/`Bool`/... are `@cache`d process-wide, so capturing one that happened
# to be created first inside a body would silently move a node the rest of the
# graph already shares INTO the function, where it only evaluates during a
# call. Leaving constants global is free (they are leaves) and avoids that;
# probe T5 confirms a body reads an uncaptured global correctly.
_CONSTANT_NODES = {"Float", "Bool", "String", "Color", "Country", "Stat"}

# Name of the function body currently being emitted, if any. See rule 3.
_BUILDING: list[str] = []

REGISTRY: dict[str, int] = {}
"""function name -> body node count. `deploy --profile` uses this to report
per-tick evaluations (calls x body) next to file node counts, so a change that
shrinks the file while growing per-tick work cannot hide."""


def _typed(node, graph_type):
    """Re-wrap `node` claiming `graph_type`, so operator overloads dispatch.

    `Node.type` is a plain instance attribute; the underlying node data and
    ports are untouched, so this only affects which Python overload fires.
    """
    if graph_type is None:
        return node
    clone = Node(node.data, node.outputIndex)
    clone.type = graph_type
    return clone


def graph_function(name: str, params: tuple, returns: str):
    """Decorator: emit `body(*params)` once, make every call a `Function` node.

    `params` is a tuple of graph types ("Vector3" / "Float" / "Bool"), one per
    parameter, max 4. `returns` is the body's output type. Both exist only to
    keep operator overloading working at the call sites; see Typing.

    The body is emitted lazily on first call, so a function nothing calls costs
    nothing — which is why release builds, having stripped `debug_viz`, never
    emit the helpers only it used.
    """
    arity = len(params)
    if not 1 <= arity <= 4:
        raise ValueError(f"{name}: graph functions take 1..4 params, got {arity}")

    state = {"built": False}

    def decorate(body):
        def build_once() -> None:
            if state["built"]:
                return
            state["built"] = True  # set first: a self-recursive body would loop
            before = {n["sID"] for n in graph_data["serializableNodes"]}
            _BUILDING.append(name)
            try:
                fn = CreateFunction(name)
                args = [
                    _typed(getattr(fn, f"Param{i}"), params[i - 1])
                    for i in range(1, arity + 1)
                ]
                result = body(*args)
            finally:
                _BUILDING.pop()
            fn_sid = fn.node.data["sID"]
            owned = 0
            for data in graph_data["serializableNodes"]:
                sid = data["sID"]
                if sid in before or sid == fn_sid:
                    continue
                if data["id"] in _CONSTANT_NODES:
                    continue  # see _CONSTANT_NODES
                AssignToFunction(Node(data), fn)
                owned += 1
            SetFunctionReturn(fn, result)
            REGISTRY[name] = owned

        @cache
        def call(*args):
            if len(args) != arity:
                raise TypeError(f"{name} takes {arity} args, got {len(args)}")
            if _BUILDING:
                raise RuntimeError(
                    f"nested graph function: {name} called while building "
                    f"{_BUILDING[-1]!r}. Nesting is disallowed by POLICY (rule "
                    f"3): it works in the sim but is unverified in Unity, and a "
                    f"body is emitted once so inlining is free. Call a plain "
                    f"Python helper that builds the same nodes instead."
                )
            build_once()
            return _typed(CustomFunction(name, *args), returns)

        call.build_once = build_once
        call.function_name = name
        return call

    return decorate
