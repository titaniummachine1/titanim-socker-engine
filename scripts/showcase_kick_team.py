"""Scripted showcase kick-team — no simulator code changes.

Mirrors aicomp-soccer-sim free-ball physics (SimParams::fallback / step_free_ball)
to record: P1 at center, charge=1.0, planar aim 45deg, until rest.

  py -3.12 scripts/showcase_kick_team.py
  py -3.12 scripts/make_titanium_showcase.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "kick_path_center_max_45.json"  # legacy alias
OUT_45 = ROOT / "out" / "kick_path_center_max_45.json"
OUT_GOAL = ROOT / "out" / "kick_path_center_max_goal.json"

# Match aicomp-soccer-sim::world::FIXED_DT and SimParams::fallback
FIXED_DT = 0.019
WALL_SETTLE_INTO = 5.0


@dataclass
class Params:
    ball_rest_height: float = 0.33
    gravity: float = 9.81
    ball_bounce_e: float = 0.23
    ball_bounce_settle: float = 0.5
    slide_accel: float = 5.95
    stop_speed_eps: float = 0.0001
    kick_speed_base: float = 10.0 / 9.0
    kick_speed_per_charge: float = 290.0 / 9.0
    kick_horiz_cap: float = 29.42
    kick_lift_base: float = -0.323
    kick_lift_per_charge: float = 6.6667
    kick_max_speed: float = 29.94
    wall_e: float = 0.2
    wall_mu: float = 0.35
    x_min: float = -39.5
    x_max: float = 39.5
    z_min: float = -24.7
    z_max: float = 24.7
    goal_half_width: float = 6.0
    goal_line_x: float = 39.5
    posts_x: float = 40.2
    post_contact_radius: float = 0.3 + 0.40637236


@dataclass
class Ball:
    x: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vz: float = 0.0
    height: float = 0.33
    vel_y: float = 0.0

    def grounded(self, p: Params) -> bool:
        return self.height <= p.ball_rest_height + 1e-4 and self.vel_y <= 0.05

    def speed(self) -> float:
        return math.hypot(self.vx, self.vz)


def kick_launch_speeds(charge: float, p: Params) -> tuple[float, float]:
    c = max(0.0, min(1.0, charge))
    horiz = min(p.kick_speed_base + p.kick_speed_per_charge * c, p.kick_horiz_cap)
    lift = max(0.0, p.kick_lift_base + p.kick_lift_per_charge * c)
    spd = math.hypot(horiz, lift)
    if spd > p.kick_max_speed and spd > 1e-6:
        s = p.kick_max_speed / spd
        horiz *= s
        lift *= s
    return horiz, lift


def in_goal_mouth(z: float, half: float) -> bool:
    return abs(z) <= half


def goal_at(b: Ball, p: Params) -> str:
    if b.x >= p.goal_line_x and in_goal_mouth(b.z, p.goal_half_width):
        return "goal_away"
    if b.x <= -p.goal_line_x and in_goal_mouth(b.z, p.goal_half_width):
        return "goal_home"
    return "none"


def bounce_axis(vx: float, vz: float, normal_axis: int, e: float, mu: float) -> tuple[float, float]:
    v = [vx, vz]
    into = abs(v[normal_axis])
    v[normal_axis] *= -e
    t = [v[0], v[1]]
    t[normal_axis] = 0.0
    t_speed = math.hypot(t[0], t[1])
    friction = mu * (1.0 + e) * into
    if t_speed > 0.0:
        kill = min(friction, t_speed)
        scale = (t_speed - kill) / t_speed
        t[0] *= scale
        t[1] *= scale
        v[1 - normal_axis] = t[1 - normal_axis]
    return v[0], v[1]


def wall_hit_axis(vx: float, vz: float, normal_axis: int, e: float, mu: float) -> tuple[float, float]:
    into = abs(vx if normal_axis == 0 else vz)
    if into <= WALL_SETTLE_INTO:
        return 0.0, 0.0
    return bounce_axis(vx, vz, normal_axis, e, mu)


def bounce_circle(vx: float, vz: float, nx: float, nz: float, e: float, mu: float) -> tuple[float, float]:
    vn = vx * nx + vz * nz
    if vn >= 0.0:
        return vx, vz
    into = -vn
    if into <= WALL_SETTLE_INTO:
        return 0.0, 0.0
    tx, tz = vx - nx * vn, vz - nz * vn
    t_speed = math.hypot(tx, tz)
    friction = mu * (1.0 + e) * into
    if t_speed > 0.0:
        kill = min(friction, t_speed)
        scale = (t_speed - kill) / t_speed
        tx *= scale
        tz *= scale
    return tx + nx * (into * e), tz + nz * (into * e)


def resolve_posts(b: Ball, p: Params) -> None:
    if goal_at(b, p) != "none":
        return
    for sx, sz in (
        (p.posts_x, p.goal_half_width),
        (p.posts_x, -p.goal_half_width),
        (-p.posts_x, p.goal_half_width),
        (-p.posts_x, -p.goal_half_width),
    ):
        dx, dz = b.x - sx, b.z - sz
        dist = math.hypot(dx, dz)
        if dist >= p.post_contact_radius or dist < 1e-8:
            continue
        nx, nz = dx / dist, dz / dist
        b.x = sx + nx * p.post_contact_radius
        b.z = sz + nz * p.post_contact_radius
        b.vx, b.vz = bounce_circle(b.vx, b.vz, nx, nz, p.wall_e, p.wall_mu)


def resolve_walls(b: Ball, p: Params) -> None:
    if b.z < p.z_min:
        b.z = p.z_min
        if b.vz < 0.0:
            b.vx, b.vz = wall_hit_axis(b.vx, b.vz, 1, p.wall_e, p.wall_mu)
    elif b.z > p.z_max:
        b.z = p.z_max
        if b.vz > 0.0:
            b.vx, b.vz = wall_hit_axis(b.vx, b.vz, 1, p.wall_e, p.wall_mu)

    if b.x < p.x_min:
        if not in_goal_mouth(b.z, p.goal_half_width):
            b.x = p.x_min
            if b.vx < 0.0:
                b.vx, b.vz = wall_hit_axis(b.vx, b.vz, 0, p.wall_e, p.wall_mu)
    elif b.x > p.x_max:
        if not in_goal_mouth(b.z, p.goal_half_width):
            b.x = p.x_max
            if b.vx > 0.0:
                b.vx, b.vz = wall_hit_axis(b.vx, b.vz, 0, p.wall_e, p.wall_mu)


def step_free_ball(b: Ball, p: Params, dt: float) -> str:
    if dt <= 0.0:
        return goal_at(b, p)

    b.height += b.vel_y * dt - 0.5 * p.gravity * dt * dt
    b.vel_y -= p.gravity * dt
    if b.height <= p.ball_rest_height:
        b.height = p.ball_rest_height
        if b.vel_y < 0.0:
            bounce = -b.vel_y * p.ball_bounce_e
            b.vel_y = 0.0 if bounce < p.ball_bounce_settle else bounce

    spd = b.speed()
    if spd <= p.stop_speed_eps:
        b.vx = b.vz = 0.0
        resolve_walls(b, p)
        end = goal_at(b, p)
        if end != "none":
            return end
        resolve_posts(b, p)
        return goal_at(b, p)

    if b.grounded(p):
        t_stop = spd / p.slide_accel
        dt_use = min(dt, t_stop)
        ax = -b.vx / spd * p.slide_accel
        az = -b.vz / spd * p.slide_accel
        b.x += b.vx * dt_use + 0.5 * ax * dt_use * dt_use
        b.z += b.vz * dt_use + 0.5 * az * dt_use * dt_use
        b.vx += ax * dt_use
        b.vz += az * dt_use
        if b.speed() <= p.stop_speed_eps or dt_use >= t_stop:
            b.vx = b.vz = 0.0
    else:
        b.x += b.vx * dt
        b.z += b.vz * dt

    resolve_walls(b, p)
    end = goal_at(b, p)
    if end != "none":
        return end
    resolve_posts(b, p)
    return goal_at(b, p)


def run_scripted_kick(charge: float = 1.0, aim_deg: float = 0.0) -> dict:
    """Scripted team: carrier at center fires max-charge shot at goal middle (+X)."""
    p = Params()
    horiz, lift = kick_launch_speeds(charge, p)
    rad = math.radians(aim_deg)
    b = Ball(
        x=0.0,
        z=0.0,
        vx=math.cos(rad) * horiz,
        vz=math.sin(rad) * horiz,
        height=p.ball_rest_height,
        vel_y=lift,
    )

    samples = []

    def push(t: float, end: str) -> None:
        samples.append(
            {
                "t": t,
                "x": b.x,
                "z": b.z,
                "height": b.height,
                "vx": b.vx,
                "vz": b.vz,
                "vel_y": b.vel_y,
                "grounded": b.grounded(p),
                "end": end,
            }
        )

    push(0.0, "none")
    t = 0.0
    max_ticks = int(math.ceil(20.0 / FIXED_DT))
    for _ in range(max_ticks):
        end = step_free_ball(b, p, FIXED_DT)
        t += FIXED_DT
        push(t, end)
        if end != "none":
            break
        if b.speed() <= p.stop_speed_eps and b.grounded(p):
            break

    return {
        "source": "scripts/showcase_kick_team.py (mirrors aicomp-soccer-sim step_free_ball)",
        "note": "Scripted showcase team: center -> right goal middle, charge=1.0. No sim repo edits.",
        "charge": charge,
        "aim_deg": aim_deg,
        "fixed_dt": FIXED_DT,
        "horiz_launch": horiz,
        "lift_launch": lift,
        "x_min": p.x_min,
        "x_max": p.x_max,
        "z_min": p.z_min,
        "z_max": p.z_max,
        "samples": samples,
    }


OUT_45 = ROOT / "out" / "kick_path_center_max_45.json"
OUT_GOAL = ROOT / "out" / "kick_path_center_max_goal.json"


def main() -> None:
    OUT_45.parent.mkdir(parents=True, exist_ok=True)
    for aim, path, label in (
        (45.0, OUT_45, "45deg up-right (bounce demo)"),
        (0.0, OUT_GOAL, "0deg goal-middle (intercept demo)"),
    ):
        doc = run_scripted_kick(charge=1.0, aim_deg=aim)
        doc["note"] = f"Scripted showcase kick: center, charge=1.0, aim={aim:.0f}deg. {label}"
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        last = doc["samples"][-1]
        print(
            f"wrote {len(doc['samples'])} samples ({last['t']:.2f}s) "
            f"aim={aim:.0f} horiz={doc['horiz_launch']:.2f} lift={doc['lift_launch']:.2f} -> {path}"
        )


if __name__ == "__main__":
    main()
