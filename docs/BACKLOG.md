# Titanium backlog — biggest expected win → smallest, ties broken by ease of implementation

Core thesis (from session review, 2026-07-25): the AI is already tactically sound
(clear_shot, threat_cover, tackle_duty, stamina-ordered tackling, spread stations
all ship and pass the full gate). The next real gains aren't "smarter" decisions —
they're **spending less stamina for the same result** and **not committing early**.
Full stamina = un-tackleable, and stamina only refills on a full match reset, never
per-goal, so waste compounds across the whole match.

One crude version of "spend less stamina" was tried today (sprint only when an
opponent is within a flat `4×r_eff` margin of the carrier) and **failed hard**
against Poponeta specifically: possession collapsed to 14s/3600s on one side and
it lost 2-3 outright on the other. Root cause: a flat distance margin fires
almost constantly against an aggressive presser, burning exactly the stamina
that makes a carrier immune to tackles. Reverted. The lesson carries into #4
below — the fix is almost certainly "gate on time-to-intercept, not raw
distance," not "abandon the idea."

Every item below gets its own branch off `main`, tested both sides against the
full opponent roster (AIA, AIA3, DefendSimple, Poponeta) **and** against `main`
itself, before it's allowed to fast-forward in.

---

## Tier S — cheap, safe, do first

### 1. Soft shot search (bounded fallback when all 3 direct candidates are blocked)
- **What**: keep the existing center/left-tangent/right-tangent check in
  `clear_shot`; if all three are blocked, add up to 2 more bounded samples
  (halfway between center↔left, center↔right — 5 total, hard cap, no loop).
- **Why it wins**: catches the case where a single defender happens to stand
  exactly on one of the 3 sampled rays while the true legal lane is a few
  degrees to either side — currently that shot is simply not taken.
- **Effort**: low — same `is_legal_direction` check, 2 more literal candidates.
- **Risk**: low — still a fixed, bounded candidate count (keeps your original
  "max 3 iterations" crash-safety intent; this raises it to a still-bounded 5).

### 2. Force weak-side bias in attacking stations
- **What**: `fwd`/`lat` station offsets for P1 (left) / P2 (right) are
  currently fixed at ±7. Weight the side by opponent count/density on each
  flank instead of a fixed symmetric split.
- **Why it wins**: stretches a defense that's already collapsed to one side
  instead of running a station into a crowd.
- **Effort**: low-medium — reuse `nearest_opponent_dist`-style per-flank
  counting, no new architecture.
- **Risk**: low.

---

## Tier A — high impact, needs careful staged rollout

### 3. Rebuild instant pass (prerequisite for #5, #7, #8 below)
- **Status**: tried once this session, reverted. Root cause was specific and
  understood: charge is always held at max, so every release — including a
  forced pass — fires at max power (~29 m/s) and overruns the receiver.
- **What**: release power should scale down for a short in-air pass instead
  of always dumping full charge. This unblocks dynamic pass timing, passing
  chains, and support runners that ask "if I receive, can I pass again."
- **Effort**: medium-high.
- **Risk**: medium — the failure mode is now diagnosed, not a mystery.

### 4. Panic index for the ball carrier
- **What**: replace ad hoc thresholds with one scalar —
  `panic = f(nearest_enemy_time_to_intercept, intercept_probability, local_density, stamina_difference)`.
  Low → wait/dribble/walk. Medium → prepare a pass. High → instant safe
  pass or shot. This is the principled version of today's failed sprint-margin
  attempt: it gates on **time to intercept**, not raw distance, so a fast
  presser is correctly read as urgent while a slow one isn't.
- **Why it wins**: this is the actual lever behind "spend less stamina for the
  same result" — unifies several independent heuristics (sprint gating, shoot
  timing, pass urgency) into one number instead of separately-tuned flags that
  can fight each other.
- **Effort**: high — touches carrier movement, shoot-now, and (once #3 lands)
  pass timing.
- **Risk**: medium-high. **Test against Poponeta first, not last** — it's the
  opponent that exposed the crude version's failure mode, so it's the cheapest
  possible early warning if the time-based gate has the same problem.

### 5. Stamina-aware dribble-length costing
- **What**: before committing to beat a defender, estimate expected stamina
  cost to reach a shot (how many tackles are likely between here and goal) vs.
  passing now.
- **Depends on**: #4 (needs the panic/threat model to estimate tackle
  likelihood) and ideally #3 (passing needs to actually be live to be the
  alternative).
- **Effort**: high. **Impact**: medium-high, but only once its prerequisites
  exist — sequence this after #3/#4, not before.

### 6. Delay commitment ("patience window")
- **What**: when panic is low, hold the current decision for ~0.1-0.3s instead
  of instantly locking into dribble-vs-pass, using `SetVariable`/`GetVariable`
  as a hold-timer.
- **Depends on**: #4 — without a panic gate, "wait passively" is exactly the
  shape of behavior that got punished by Poponeta's press today. With panic
  gating it correctly, only fires when actually safe to wait.
- **Effort**: medium. **Impact**: medium.

---

## Tier B — real but lower distinct payoff right now

### 7. Dynamic pass timing ("the best pass exists half a second later")
- Blocked on #3 shipping. Effort medium once passing exists.

### 8. Passing chain evaluation (A→B→Shot vs A→Shot)
- Blocked on #3. Doesn't need deep search — just one extra "pass then shoot"
  candidate alongside the existing direct-shot check. Effort medium once #3
  is live.

### 9. Better support spacing ("can I receive / shoot / pass again")
- Partially exists today as static ball-relative stations (no receive/shoot/
  pass viability scoring). Most of its value is gated on #3 — without a real
  pass, "can I pass again" has no payoff. Effort medium, sequence after #3.

---

## Tier C — defensive/GK refinements, smaller distinct wins

### 10. Lane-denial dynamic assignment (refine, not build from scratch)
- **Status**: `threat_cover` already seals shot cones per opponent, but
  assignment is static by slot index (P1↔opponents[0], P2↔opponents[1], ...).
- **What**: assign the *closest available* defender to the *most dangerous*
  open lane instead of a fixed index pairing.
- **Effort**: medium (a small greedy assignment, same spirit as `tackle_duty`'s
  cheapest-winner selection). **Impact**: medium — fixes real cases where a
  static pairing double-covers a harmless opponent while leaving a more
  dangerous one open.

### 11. GK second-ball / rebound prediction
- **What**: after a save, start moving toward the statistically likely rebound
  location instead of reacting once it happens.
- **Effort**: medium. **Impact**: low-medium — rebounds are a small fraction
  of total events, real but not the biggest lever.

---

## Explicitly postponed (agreed)

- **Cross-match / adaptive memory** (e.g. "stop holding max charge for the
  rest of the match after being forced to release") — no infrastructure for
  this exists, and it's a correctness risk (persistent state surviving kickoffs
  needs the same care as the stamina-persistence fix took).
- **Post-bounce bank shots** — physics is calibrated (e≈0.2, μ≈0.35, confirmed
  in `SOCCER_GAME_MODEL.md`) but the docs already flag this exact idea as
  considered and shelved "for cost," and it only ever helps in the narrow case
  where the existing tangent search is *just barely* blocked.
- **Continuous/unbounded gradient optimization for aiming** — conflicts
  directly with the explicit "max 3 iterations, don't crash the real game"
  constraint; #1 above is the bounded version of the same idea.
