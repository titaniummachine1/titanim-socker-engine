# titanim-socker-engine

**Private** Titanium soccer **decision engine** (not the match simulator).

`main` is the **goat**. Branches must **challenge and beat** `main` before
merge — see [docs/CHALLENGE.md](docs/CHALLENGE.md) (Stockfish-style).

Modular source packs into one game `.txt` with **horizontal, non-overlapping**
module columns — see [docs/BUNDLE_LAYOUT.md](docs/BUNDLE_LAYOUT.md).

## Boundary

| Repo | Owns |
| --- | --- |
| **titanim-socker-engine** (this, private) | Policy: predict, intercept math, flick, orbit, GK/attack roles |
| **aicomp-soccer-sim** (harness) | Physics, possession, headless self-play, AIA graphs, viewer |

Pitch: `Vec2(x,y)` = Unity `(X,Z)` — X goals, Y sidelines. Home attacks `+X`.

## Modules

| Rust module | Job |
| --- | --- |
| `types` / `params` / `sensors` | Shared snap + commands |
| `ball` | Prediction-only free-ball step |
| `predict` | Path, intercept, shot/clear/pass search |
| `titanium` | Roles (GK / attack / lackeys), flick + anti-tackle |

## Dev

```bat
cargo test
```

Sim path-dep:

```toml
titanim-socker-engine = { path = "../titanim-socker-engine" }
```

## License

UNLICENSED / all rights reserved. Private repo — do not mirror.
