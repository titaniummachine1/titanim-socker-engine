"""Gameplay tuning constants — geometry, speeds, and kick physics."""
from __future__ import annotations

import math

# MEASURED prefab BallHoldLocation local Z (AIA_UPSTREAM_QUIRKS #21).
HOLD_OFFSET = 1.67
# +-39.75/+-24.75 against corners of +-40/+-25, so the collision radius is
# 0.25 exactly, identical on both axes. 0.4064 is the VISUAL radius from
# soccer_ball_sim_params.json and is not what the engine collides with.
BALL_RADIUS = 0.25
# Upright post world radius. Not exposed by any SoccerGet — hardcoded from
# the same confirmed/measured Unity geometry the offline sim uses (posts at
# world x=+-40.2, world radius 0.3; see aicomp-soccer-sim/docs/SOCCER_GAME_MODEL.md).
POST_RADIUS = 0.3
POST_CONTACT_RADIUS = POST_RADIUS + BALL_RADIUS

# Extra clearance used when AIMING past a post -- deliberately separate from
# the physical contact radius above.
#
# Aiming used to inherit a fatter contact radius purely because BALL_RADIUS was
# the visual 0.4064 instead of the measured collision 0.25, which handed shot
# selection ~0.156 m of accidental margin. Correcting the physics removed that
# margin and shots started grazing the posts. Keep the physics honest and make
# the margin explicit instead of hiding it in a wrong constant.
POST_AIM_MARGIN = 0.156
POST_AIM_RADIUS = POST_CONTACT_RADIUS + POST_AIM_MARGIN
ANGLE_EPS = math.radians(0.5)
WALK_SPEED = 7.0
SPRINT_SPEED = 8.0
# Confirmed sim step (aicomp-soccer-sim FIXED_DT).
FIXED_DT = 0.019
# Opponent carrier speed while running with the ball. Confirmed measurement
# (AIA_UPSTREAM_QUIRKS.md #11: walk ~7 m/s flat, no stamina throttle) — same
# figure as WALK_SPEED, reused directly rather than guessing a separate
# constant. There is no "Opponent Velocity" getter, so heading comes from the
# held ball offset instead (see `titanium.goalkeeper.carrier_heading`).
CARRIER_SPEED = WALK_SPEED

# Kick launch physics. Not exposed by any SoccerGet — hardcoded from the same
# TimePlot-calibrated constants the offline sim uses (aicomp-soccer-sim
# SimParams::fallback). Used only to ESTIMATE the charge that produced an
# already-observed ball speed (and from it, an estimated airborne hang time)
# — never to predict our own kicks, which we already control directly.
KICK_SPEED_BASE = 10.0 / 9.0
KICK_SPEED_PER_CHARGE = 290.0 / 9.0
KICK_LIFT_BASE = -0.323
KICK_LIFT_PER_CHARGE = 6.6667
# MEASURED 2026-07-25: second differences of Ball.Y over 1845 free-flight
# samples give 17.0000 (p25/p75 16.9984/17.0003), flat across every vertical
# velocity band and with zero damping. NOT Unity's 9.81 -- using 9.81 made
# every airborne phase ~75% too long.
GRAVITY = 17.0
# Launch velocity magnitude is clamped to this; the engine scales planar and
# lift together, which is why both showed the same ~0.87 ratio at full charge.
MAX_BALL_SPEED = 30.0
# Sprint speed once stamina hits zero -- faster than a walk, slower than a
# real sprint.
SPRINT_EMPTY_SPEED = 7.65
# Warmup is already done while they hold+interact; remaining to full is
# (1-charge)*shot_charge_time_s once charge has started climbing.
SHOT_CHARGE_TIME_S = 0.38
# "Fully charged" for release purposes. The engine clamps charge at exactly
# 1.0 and holds it there, so this only needs to clear float noise.
FULL_CHARGE = 0.99
HOLD_PROXY = 0.55
# Ball must stay this far (along pitch X) on the pitch side of our goal plane
# after an end-of-tick hold so anti-tackle rotation cannot own-goal.
OWN_GOAL_PLANE_CLEARANCE = HOLD_OFFSET + BALL_RADIUS + 0.5
# Teammate spacing: never closer than full tackle range (r_int).
TEAMMATE_MIN_SEP_FRAC = 1.0
# Support outlets along AT-safe headings: try far→near until clear_pass opens.
SUPPORT_OUTLET_DISTS = (9.0, 7.5, 6.0, 4.5, 3.5)
# Pass-danger: AT tracks opps inside 2×r_int of carrier center. When an opp has
# closed MORE THAN 1/4 of that approach (remaining gap ≤ ¾ of 2×r = 1.5×r_int),
# holding is too late to wait for body-touch (1×r_int) — seek a pass. Keeping
# the ball in that window costs KEEP_BALL_DANGER_PENALTY on eval.
PASS_DANGER_APPROACH_FRAC = 0.25  # "closed more than 1/4 of the way in"
PASS_DANGER_TRACK_MULT = 2.0  # same 2×r_int bubble as anti_tackle tracking
KEEP_BALL_DANGER_PENALTY = 2.0
# Playable AABB — matches aicomp-soccer-sim SimParams fallback / Unity pitch.
# Held-ball offset is projected into this region (sidelines always; endlines
# only outside the goal mouth) exactly like `project_hold_into_playable`.
PITCH_X_MIN = -40.0
PITCH_X_MAX = 40.0
PITCH_Z_MIN = -25.0
PITCH_Z_MAX = 25.0
GOAL_HALF_WIDTH = 5.7
