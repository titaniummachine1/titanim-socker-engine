# Titanium — the competition engine

Build from the repo root above:

```
python scripts\build_titanium.py            # candidate only -> Titanium_test
python scripts\build_titanium.py --promote  # ONLY after it wins a gate
```

| module | what it owns |
| --- | --- |
| `graph.py` | wiring: per-player controllers, what each slot does |
| `constants.py` | measured physics — do not "tidy" these, they are measurements |
| `ball_physics.py` | trajectory, own-goal threat, meet point (approximation, see below) |
| `goalkeeper.py` | GK policy; the only place allowed to sprint outside a 50/50 |
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
