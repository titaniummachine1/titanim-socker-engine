//! Free-ball XZ kinematics.
//!
//! `Vec2(x, y)` is the pitch plane `(Unity X, Unity Z)`. Height is kept
//! separately because it affects bounce and whether Coulomb slide applies.

use crate::params::EngineParams;
use crate::types::Vec2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ball {
    pub pos: Vec2,
    pub vel: Vec2,
    pub height: f32,
    pub vel_y: f32,
    pub held: bool,
}

impl Default for Ball {
    fn default() -> Self {
        Self {
            pos: Vec2::ZERO,
            vel: Vec2::ZERO,
            height: 0.33,
            vel_y: 0.0,
            held: false,
        }
    }
}

impl Ball {
    pub fn grounded(&self, params: &EngineParams) -> bool {
        self.height <= params.ball_rest_height + 1e-4 && self.vel_y <= 0.05
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    None,
    GoalHome,
    GoalAway,
}

pub fn in_goal_mouth(z: f32, half_width: f32) -> bool {
    z.abs() <= half_width
}

/// Return horizontal and vertical launch speeds for a normalized charge.
pub fn kick_launch_speeds(charge: f32, params: &EngineParams) -> (f32, f32) {
    let c = charge.clamp(0.0, 1.0);
    let mut horiz =
        (params.kick_speed_base + params.kick_speed_per_charge * c).min(params.kick_horiz_cap);
    let mut lift = (params.kick_lift_base + params.kick_lift_per_charge * c).max(0.0);
    let speed = (horiz * horiz + lift * lift).sqrt();
    if speed > params.kick_max_speed && speed > 1e-6 {
        let scale = params.kick_max_speed / speed;
        horiz *= scale;
        lift *= scale;
    }
    (horiz, lift)
}

/// Advance an unheld ball and report a terminal goal, if any.
pub fn step_free_ball(ball: &mut Ball, params: &EngineParams, dt: f32) -> EndReason {
    if ball.held || dt <= 0.0 {
        return goal_at(ball.pos, params);
    }

    ball.height += ball.vel_y * dt - 0.5 * params.gravity * dt * dt;
    ball.vel_y -= params.gravity * dt;
    if ball.height <= params.ball_rest_height {
        ball.height = params.ball_rest_height;
        if ball.vel_y < 0.0 {
            let bounce = -ball.vel_y * params.ball_bounce_e;
            ball.vel_y = if bounce < params.ball_bounce_settle {
                0.0
            } else {
                bounce
            };
        }
    }

    let speed = ball.vel.length();
    if speed <= params.stop_speed_eps {
        ball.vel = Vec2::ZERO;
        resolve_walls(ball, params);
        return goal_at(ball.pos, params);
    }

    if ball.grounded(params) {
        let dt_use = dt.min(speed / params.slide_accel.max(1e-6));
        let dir = ball.vel / speed;
        let accel = -dir * params.slide_accel;
        ball.pos += ball.vel * dt_use + 0.5 * accel * dt_use * dt_use;
        ball.vel += accel * dt_use;
        if ball.vel.length() <= params.stop_speed_eps || dt_use >= speed / params.slide_accel.max(1e-6)
        {
            ball.vel = Vec2::ZERO;
        }
    } else {
        ball.pos += ball.vel * dt;
    }

    resolve_walls(ball, params);
    let goal = goal_at(ball.pos, params);
    if goal != EndReason::None {
        return goal;
    }
    resolve_posts(ball, params);
    goal_at(ball.pos, params)
}

/// True when the ball center is past a goal line within the mouth.
pub fn goal_at(pos: Vec2, params: &EngineParams) -> EndReason {
    if pos.x >= params.goal_line_x && in_goal_mouth(pos.y, params.goal_half_width) {
        EndReason::GoalAway
    } else if pos.x <= -params.goal_line_x && in_goal_mouth(pos.y, params.goal_half_width) {
        EndReason::GoalHome
    } else {
        EndReason::None
    }
}

fn resolve_walls(ball: &mut Ball, params: &EngineParams) {
    bounce_axis_if_needed(&mut ball.vel, &mut ball.pos.y, params.z_min, params.z_max, 1, params);

    // The endline is open in the goal mouth, closed elsewhere.
    if ball.pos.x < params.x_min && !in_goal_mouth(ball.pos.y, params.goal_half_width) {
        ball.pos.x = params.x_min;
        if ball.vel.x < 0.0 {
            bounce_axis(&mut ball.vel, 0, params);
        }
    } else if ball.pos.x > params.x_max && !in_goal_mouth(ball.pos.y, params.goal_half_width) {
        ball.pos.x = params.x_max;
        if ball.vel.x > 0.0 {
            bounce_axis(&mut ball.vel, 0, params);
        }
    }
}

fn bounce_axis_if_needed(
    vel: &mut Vec2,
    coordinate: &mut f32,
    min: f32,
    max: f32,
    axis: usize,
    params: &EngineParams,
) {
    if *coordinate < min {
        *coordinate = min;
        if vel[axis] < 0.0 {
            bounce_axis(vel, axis, params);
        }
    } else if *coordinate > max {
        *coordinate = max;
        if vel[axis] > 0.0 {
            bounce_axis(vel, axis, params);
        }
    }
}

fn bounce_axis(vel: &mut Vec2, axis: usize, params: &EngineParams) {
    let into = vel[axis].abs();
    if into <= 5.0 {
        *vel = Vec2::ZERO;
        return;
    }
    vel[axis] *= -params.wall_e;
    let tangent_axis = 1 - axis;
    let tangent = vel[tangent_axis];
    let friction = params.wall_mu * (1.0 + params.wall_e) * into;
    vel[tangent_axis] = tangent.signum() * (tangent.abs() - friction).max(0.0);
}

fn resolve_posts(ball: &mut Ball, params: &EngineParams) {
    if goal_at(ball.pos, params) != EndReason::None {
        return;
    }
    let contact_r = params.post_radius + params.ball_radius;
    for &(x, z) in &[
        (params.posts_x, params.goal_half_width),
        (params.posts_x, -params.goal_half_width),
        (-params.posts_x, params.goal_half_width),
        (-params.posts_x, -params.goal_half_width),
    ] {
        let center = Vec2::new(x, z);
        let delta = ball.pos - center;
        let dist = delta.length();
        if dist >= contact_r || dist < 1e-8 {
            continue;
        }
        let normal = delta / dist;
        ball.pos = center + normal * contact_r;
        let into = ball.vel.dot(normal);
        if into < 0.0 {
            let tangent = ball.vel - normal * into;
            ball.vel = tangent * (1.0 - params.wall_mu).max(0.0)
                + normal * (-into * params.wall_e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ground(vel: Vec2) -> Ball {
        Ball {
            vel,
            ..Ball::default()
        }
    }

    #[test]
    fn kick_curve_is_capped() {
        let p = EngineParams::default();
        let (horiz, lift) = kick_launch_speeds(1.0, &p);
        assert!(horiz <= p.kick_horiz_cap);
        assert!((horiz * horiz + lift * lift).sqrt() <= p.kick_max_speed + 1e-4);
    }

    #[test]
    fn grounded_ball_stops_with_coulomb_slide() {
        let p = EngineParams::default();
        let mut ball = ground(Vec2::new(10.0, 0.0));
        for _ in 0..600 {
            step_free_ball(&mut ball, &p, 1.0 / 60.0);
        }
        assert!(ball.vel.length() < 0.01);
        assert!((ball.pos.x - 8.4).abs() < 0.3);
    }

    #[test]
    fn goal_mouth_is_terminal() {
        let p = EngineParams::default();
        let mut ball = Ball {
            pos: Vec2::new(p.goal_line_x - 1.0, 0.0),
            vel: Vec2::new(20.0, 0.0),
            ..Ball::default()
        };
        let mut reason = EndReason::None;
        for _ in 0..30 {
            reason = step_free_ball(&mut ball, &p, 1.0 / 60.0);
            if reason != EndReason::None {
                break;
            }
        }
        assert_eq!(reason, EndReason::GoalAway);
    }
}
