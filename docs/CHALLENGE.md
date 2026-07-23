# Challenge protocol (Stockfish-style)

`main` is the **goat** — the only promoted champion. Feature branches do not
merge on vibes. They **challenge** `main` in the simulator harness and must
**beat** it under a fixed match budget before promotion.

## Roles

| Ref | Meaning |
| --- | --- |
| `main` | Current champion engine (never force-pushed) |
| `challenge/<name>` | Candidate that wants to dethrone `main` |
| `aicomp-soccer-sim` | Offline physics harness (public/separate). Runs both brains. |

## Rules

1. **Branch from `main`.** Name it `challenge/<short-idea>`.
2. **Change one idea** when possible (flick, orbit, GK clear, pass lane…).
3. **Self-check:** `cargo test` green in this repo.
4. **Challenge match** via the sim (path-dep on your checkout):

   ```bat
   :: from aicomp-soccer-sim, with Cargo.toml pointing at this engine path
   scripts\challenge_titanium.ps1 -Games 10 -WinGoals 5
   ```

   Starts at **10** games. If the score gap is within noise, **double**
   (20 → 40 → 80 → 100 → 200, cap **500**). Same rule as overnight Titanium.

5. **Pass criteria (candidate must clear all):**
   - Candidate win rate vs champion ≥ **55%** at the final N, **or**
   - Elo-style score gap clearly outside noise (document in PR).
   - Also run candidate vs AIA (or prior baseline) so we do not overfit self-play.
   - Sanity: candidate vs itself ≈ coin flip / low goal rate (no one-sided lock).

6. **Fail:** close the challenge branch. Do not merge. Optionally keep notes in
   `data/challenge_log.jsonl` (sim side).

7. **Pass:** open PR → `main` with the challenge JSON attached. Squash-merge
   only after review. Tag `champion-YYYYMMDD` on the new `main` tip.

8. **Never** merge broken `cargo test`, layout-breaking bundle exports, or
   “I think it’s better” without numbers.

## Doubling schedule

| Stage | Games | Use when |
| --- | ---: | --- |
| smoke | 10 | first look |
| check | 20 | 10 looked promising / noisy |
| solid | 40–80 | close race |
| gate | 100–200 | pre-merge |
| cap | 500 | overnight max for now |

## What the sim measures

- `soccer_headless --home titanium --away titanium` is **not** a challenge
  (identical code). Challenge needs **two engine checkouts** or a version tag:
  champion = `main` path/git, candidate = branch path.
- Prefer first-to-N (`--win-goals`) for decisive scores; fixed `--secs` as backup.
