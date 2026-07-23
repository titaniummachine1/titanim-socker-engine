//! Fixed-step free-ball prediction and reachability helpers.

use crate::ball::{goal_at, kick_launch_speeds, step_free_ball, Ball, EndReason};
use crate::params::EngineParams;
use crate::types::{Vec2, FIXED_DT};

#[derive(Debug, Clone, Copy)]
pub struct BallSample {
    pub t: f32,
    pub pos: Vec2,
    pub vel: Vec2,
    pub end: EndReason,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Intercept {
    pub t: f32,
    pub pos: Vec2,
    pub arriver_dist: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    pub pos: Vec2,
    pub speed: f32,
}

pub fn predict_ball_path(
    ball: &Ball,
    params: &EngineParams,
    dt: f32,
    max_horizon_s: f32,
) -> Vec<BallSample> {
    let mut current = *ball;
    current.held = false;
    let mut path = vec![BallSample {
        t: 0.0,
        pos: current.pos,
        vel: current.vel,
        end: EndReason::None,
    }];
    if dt <= 0.0 || max_horizon_s <= 0.0 {
        return path;
    }

    let mut t = 0.0;
    while t + dt <= max_horizon_s + 1e-6 {
        let end = step_free_ball(&mut current, params, dt);
        t += dt;
        path.push(BallSample {
            t,
            pos: current.pos,
            vel: current.vel,
            end,
        });
        if end != EndReason::None || goal_at(current.pos, params) != EndReason::None {
            break;
        }
        if current.vel.length_squared() < 1e-8 && current.grounded(params) {
            break;
        }
    }
    path
}

pub fn earliest_intercept(
    player_pos: Vec2,
    player_speed: f32,
    path: &[BallSample],
    reach: f32,
) -> Option<Intercept> {
    let speed = player_speed.max(0.0);
    path.iter().skip(1).find_map(|sample| {
        let distance = player_pos.distance(sample.pos);
        (distance <= speed * sample.t + reach + 1e-5).then_some(Intercept {
            t: sample.t,
            pos: sample.pos,
            arriver_dist: distance,
        })
    })
}

pub fn truncate_to_guaranteed_intercept(
    path: &[BallSample],
    candidates: &[Candidate],
    reach: f32,
) -> (Vec<BallSample>, Option<(usize, Intercept)>) {
    let mut best: Option<(usize, Intercept)> = None;
    for (index, candidate) in candidates.iter().enumerate() {
        if let Some(hit) = earliest_intercept(candidate.pos, candidate.speed, path, reach) {
            best = Some(match best {
                Some((best_index, best_hit)) if best_hit.t <= hit.t => (best_index, best_hit),
                _ => (index, hit),
            });
        }
    }
    let Some((index, hit)) = best else {
        return (path.to_vec(), None);
    };
    let truncated = path
        .iter()
        .copied()
        .filter(|sample| sample.t <= hit.t + 1e-6)
        .collect();
    (truncated, Some((index, hit)))
}

pub fn guaranteed_intercept_horizon(
    path: &[BallSample],
    candidates: &[Candidate],
    reach: f32,
) -> Vec<BallSample> {
    truncate_to_guaranteed_intercept(path, candidates, reach).0
}

pub fn gk_cover_point(
    threat: Vec2,
    own_goal_x: f32,
    goal_half_width: f32,
    reach: f32,
) -> Vec2 {
    let left = Vec2::new(own_goal_x, goal_half_width);
    let right = Vec2::new(own_goal_x, -goal_half_width);
    let depth_sign = if own_goal_x > 0.0 { -1.0 } else { 1.0 };
    let cover_x = own_goal_x + depth_sign * (reach * 0.85).clamp(0.8, 2.5);
    let z_at = |post: Vec2| {
        let dx = post.x - threat.x;
        if dx.abs() < 1e-4 {
            post.y
        } else {
            let u = (cover_x - threat.x) / dx;
            threat.y + (post.y - threat.y) * u
        }
    };
    Vec2::new(
        cover_x,
        (0.5 * (z_at(left).min(z_at(right)) + z_at(left).max(z_at(right))))
            .clamp(-goal_half_width, goal_half_width),
    )
}

pub fn path_interceptable_near(
    path: &[BallSample],
    opponents: &[Candidate],
    reach: f32,
    origin: Vec2,
    safe_radius: f32,
) -> bool {
    for sample in path {
        if origin.distance(sample.pos) > safe_radius {
            continue;
        }
        for opponent in opponents {
            if let Some(hit) = earliest_intercept(opponent.pos, opponent.speed, path, reach) {
                if hit.t <= sample.t + 1e-4 && origin.distance(hit.pos) <= safe_radius {
                    return true;
                }
            }
        }
    }
    false
}

pub fn best_safe_clear_dir(
    origin: Vec2,
    attack_sign: f32,
    opponents: &[Candidate],
    charge: f32,
    params: &EngineParams,
    safe_radius: f32,
) -> Vec2 {
    let (horizontal, lift) = kick_launch_speeds(charge.clamp(0.35, 1.0), params);
    let candidates = [
        Vec2::new(attack_sign, 0.85).normalize_or_zero(),
        Vec2::new(attack_sign, -0.85).normalize_or_zero(),
        Vec2::new(attack_sign, 0.45).normalize_or_zero(),
        Vec2::new(attack_sign, -0.45).normalize_or_zero(),
        Vec2::new(attack_sign, 0.0),
    ];
    let mut best = candidates[0];
    let mut best_score = f32::NEG_INFINITY;
    for direction in candidates {
        let ball = Ball {
            pos: origin + direction * 0.15,
            vel: direction * horizontal,
            height: params.ball_rest_height,
            vel_y: lift,
            held: false,
        };
        let path = predict_ball_path(&ball, params, FIXED_DT, 2.0);
        let unsafe_near =
            path_interceptable_near(&path, opponents, params.interact_radius, origin, safe_radius);
        let mut min_separation = f32::MAX;
        for sample in &path {
            for opponent in opponents {
                min_separation = min_separation.min(sample.pos.distance(opponent.pos));
            }
        }
        let mut score = min_separation;
        if !unsafe_near {
            score += 50.0;
        }
        if let Some(last) = path.last() {
            score += (last.pos.x - origin.x) * attack_sign * 0.15;
            score += last.pos.y.abs() * 0.05;
        }
        if score > best_score {
            best_score = score;
            best = direction;
        }
    }
    best
}

pub fn best_forward_pass_dir(
    origin: Vec2,
    mate: Vec2,
    opponents: &[Candidate],
    charge: f32,
    params: &EngineParams,
) -> Option<Vec2> {
    let direct = mate - origin;
    if direct.length_squared() < 1e-4 {
        return None;
    }
    let direct = direct.normalize();
    let direction = (mate + direct * 2.5 - origin).normalize_or_zero();
    let (horizontal, lift) = kick_launch_speeds(charge.clamp(0.25, 0.85), params);
    let ball = Ball {
        pos: origin + direction * 0.15,
        vel: direction * horizontal,
        height: params.ball_rest_height,
        vel_y: lift * 0.35,
        held: false,
    };
    let path = predict_ball_path(&ball, params, FIXED_DT, 2.0);
    let mate_hit = earliest_intercept(mate, 8.0, &path, params.interact_radius)?;
    if opponents.iter().any(|opponent| {
        earliest_intercept(opponent.pos, opponent.speed, &path, params.interact_radius)
            .is_some_and(|hit| hit.t <= mate_hit.t + 0.05)
    }) {
        None
    } else {
        Some(direction)
    }
}

pub fn best_shot_dir_evading(
    origin: Vec2,
    attack_sign: f32,
    opponents: &[Vec2],
    charge: f32,
    params: &EngineParams,
) -> Vec2 {
    let sign = if attack_sign < 0.0 { -1.0 } else { 1.0 };
    let left = Vec2::new(params.goal_line_x * sign, params.goal_half_width);
    let right = Vec2::new(params.goal_line_x * sign, -params.goal_half_width);
    let a0 = (left - origin).normalize_or_zero();
    let a1 = (right - origin).normalize_or_zero();
    let (horizontal, lift) = kick_launch_speeds(charge, params);
    let mut best = Vec2::new(sign, 0.0);
    let mut best_score = f32::NEG_INFINITY;

    for i in 0..9 {
        let u = i as f32 / 8.0;
        let direction = (a0 * (1.0 - u) + a1 * u).normalize_or_zero();
        if direction.length_squared() < 1e-8 {
            continue;
        }
        let mut shot = Ball {
            pos: origin + direction * 0.15,
            vel: direction * horizontal,
            height: params.ball_rest_height,
            vel_y: lift,
            held: false,
        };
        let mut min_separation = f32::MAX;
        let mut score = 0.0;
        for tick in 1..=((2.5 / FIXED_DT).ceil() as usize) {
            let reason = step_free_ball(&mut shot, params, FIXED_DT);
            for opponent in opponents {
                min_separation = min_separation.min(opponent.distance(shot.pos));
            }
            if (reason == EndReason::GoalAway && sign > 0.0)
                || (reason == EndReason::GoalHome && sign < 0.0)
            {
                score += 1000.0 - tick as f32 * 0.05;
                break;
            }
            if opponents
                .iter()
                .any(|opponent| opponent.distance(shot.pos) <= params.interact_radius)
            {
                break;
            }
        }
        score += if min_separation < f32::MAX {
            min_separation * 4.0
        } else {
            1.0
        };
        if score > best_score {
            best_score = score;
            best = direction;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earliest_intercept_respects_speed() {
        let path = vec![
            BallSample {
                t: 0.0,
                pos: Vec2::new(10.0, 0.0),
                vel: Vec2::ZERO,
                end: EndReason::None,
            },
            BallSample {
                t: 1.0,
                pos: Vec2::new(5.0, 0.0),
                vel: Vec2::ZERO,
                end: EndReason::None,
            },
            BallSample {
                t: 2.0,
                pos: Vec2::ZERO,
                vel: Vec2::ZERO,
                end: EndReason::None,
            },
        ];
        let hit = earliest_intercept(Vec2::ZERO, 2.0, &path, 0.1).unwrap();
        assert_eq!(hit.t, 2.0);
    }

    #[test]
    fn gk_cover_tracks_threat_side() {
        let point = gk_cover_point(Vec2::new(0.0, 3.0), 39.5, 6.0, 1.75);
        assert!(point.y > 0.0);
    }

    #[test]
    fn prediction_stops_at_goal() {
        let p = EngineParams::default();
        let ball = Ball {
            pos: Vec2::new(p.goal_line_x - 1.0, 0.0),
            vel: Vec2::new(20.0, 0.0),
            ..Ball::default()
        };
        let path = predict_ball_path(&ball, &p, FIXED_DT, 2.0);
        assert_eq!(path.last().unwrap().end, EndReason::GoalAway);
    }
}
