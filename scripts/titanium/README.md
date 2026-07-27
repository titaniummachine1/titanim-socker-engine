# Titanium — the competition engine

Build from the repo root above:

```
python scripts\build_titanium.py                         # release graph → Titanium_test
python scripts\build_titanium.py --with-anti-tackle      # challenger graph (no sim gate)
python scripts\build_titanium.py --with-anti-tackle --gate  # also run promotion matches
python scripts\build_titanium.py --debug                 # keep DebugDraw/TimePlot
```

`--strip-debug` still works as a no-op alias for release. Use `--debug` / `--normal`
only when inspecting sinks. Promotion/`cargo` matches are **not** part of a normal
build — pass `--gate` (or `--promote`) when you want that.

Graph optimisation is **only** via `AIGamePyLibrary.SaveData(optimize=...)` —
do not add duplicate prune passes in `deploy.py`.

## Keeping the graph from bloating

Every node evaluates every tick, and the save format costs ~576 B per node and
~951 B per connection (that format is the game's — `Haialand-v2.txt` pays the
same rate), so node count IS both file size and per-tick work.

```
python scripts\build_titanium.py --profile            # who emitted what, vs baseline
python scripts\build_titanium.py --profile --baseline # accept current as the new baseline
python scripts\equivalence.py --against <graph.txt>   # did this refactor change any decision?
python scripts\fn_contract_probe.py                   # re-verify the graph-function contract
```

`--profile` reports nodes AND per-tick evaluations separately, because a
`nodefn` function shrinks the file while evaluating its body once per call —
the two numbers can move in opposite directions and only the profile shows it.

`equivalence.py` runs the roster both sides through the deterministic sim and
diffs the match statistics. Identical possession-to-float-precision and tick
counts over 180 s means the change altered no decision, so it can be promoted
without re-gating. A change that DIVERGES is a behaviour change and still owes
the full gate below.

`fn_contract_probe.py` + `aicomp-soccer-sim`'s `probe_dump` empirically check
what graph functions actually do (multi-call independence, params, globals in
bodies, nesting). Run it rather than trusting any doc — including this one.
Note it already contradicts `build_function_nesting_probe.py`, whose docstring
claims nesting reads back null; in the sim it works.

| module | what it owns |
| --- | --- |
| `nodefn.py` | `@graph_function` — emit a repeated subgraph ONCE, call it N times |
| `profile.py` | `--profile`: node/eval attribution + regression vs a baseline |
| `graph.py` | wiring: per-player controllers, what each slot does |
| `constants.py` | measured physics — do not "tidy" these, they are measurements |
| `anti_tackle.py` | **challenger only** — 5-probe binary search on held-ball heading |
| `ball_physics.py` | trajectory, own-goal threat, meet point (approximation, see below) |
| `goalkeeper.py` | GK policy; sprint only to stop a net-bound loose ball walk can't reach |
| `tackle.py` | who presses Interact on a carrier |
| `shot.py` | shot direction / post clearance |
| `carrier.py` | carrying and clearing |
| `geometry.py` | tangents, angles, unit vectors |
| `debug_viz.py` | ALL drawing and TimePlot — no decisions |
| `deploy.py` / `paths.py` | file I/O and locations |
| `meet_point.py` | exact interception solve — **built but NOT wired** |

## Things already tested that LOST — do not retry blind

Each was more principled than what it replaced, and each gated worse against
Poponeta (champion scores 16:4):

| change | result |
| --- | --- |
| exact pursuit on the real trajectory (4 cm accurate) | 6 : 10 |
| earliest-reachable interception (correct formulation) | 12 : 10 |
| tackle duty: duel winners only, no sacrifice | lost 22:0 head-to-head |
| remove sprinting entirely | 4 : 20 |

The live meet point is a friction-bounded straight line clamped to the pitch —
crude, and it beats all of the above. Accuracy and match results have been
moving in opposite directions; treat any "more correct" change as unproven
until it clears the gate.

## Gate before promoting

```
Poponeta   16 : 4    <- hardest, clear this first
AIA3       16 : 12
head-to-head vs the current champion, BOTH sides
```

Roster wins alone are not enough — the tackle change above passed the roster
and still lost 22:0 head-to-head.
