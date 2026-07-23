# Bundle layout (single `.txt` for game / sim)

Source of truth for policy stays **modular Rust** (`predict`, intercept math,
`titanium` roles, math utils). For the Unicorn One / AIComp graph runtime we
**bundle** into one `.txt` (or one export the Python compiler emits).

## Spatial rule (horizontal modules)

Think of the graph canvas as a strip:

```
Y (down)
│  [ math_utils ] [ predict ] [ intercept ] [ roles/gk ] [ roles/attack ] …
│       col0           col1         col2          col3           col4
└──────────────────────────────────────────────────────────────────► X (right)
```

1. **Modules sit side-by-side** (columns), reading **left → right**, then next
   band **top → bottom** if needed.
2. **No overlapping nodes.** If module A’s nodes would collide with module B’s
   long chains, **shift B (and everything to its right) further right** by a
   gutter (`MODULE_GUTTER_X`, default ~400–800 graph units).
3. **Inside a module:** nodes flow **top → down** along that column (or a
   narrow vertical stack), edges stay inside the module AABB when possible.
4. **Exports** must be deterministic: same module order → same layout.

## Module columns (v0)

| Order | Module | Responsibility |
| ---: | --- | --- |
| 0 | `math_utils` | dots, clamps, side signs, orbit helpers |
| 1 | `predict` | free-ball path samples |
| 2 | `intercept` | earliest catch, truncate horizon, candidates |
| 3 | `shot_search` | flick aim, evade dirs, safe clears, forward pass |
| 4 | `role_gk` | cover cone + clear |
| 5 | `role_attack` | anti-tackle walk + flick release |
| 6 | `role_lackey` | ahead stations / handoff |
| 7 | `bundle_root` | controllers 1–4 wiring |

## Compiler hook (later)

Python (or Rust) packer:

1. Compile each module to a node list with local coords `(0..W, 0..H)`.
2. Place columns: `origin_x[i+1] = origin_x[i] + width[i] + gutter`.
3. Emit one `.txt` for **game** and optionally a twin for **sim** if APIs differ.
4. Sim harness loads the txt via `graph:<path>` / RuntimeBrain for challenge
   matches when testing the bundled artifact (not only the Rust brain).

Until the packer exists, Rust modules in this crate are the editable source;
`docs/CHALLENGE.md` still gates `main`.
