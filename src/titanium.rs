//! Titanium decision policy over a simulator-provided [`TeamSnapshot`].
//!
//! The engine deliberately has no world/API dependency: callers provide all
//! positions, possession bits, ball state, and charge values in the snapshot.

use crate::ball::Ball;
use crate::params::EngineParams;
use crate::predict::{
    best_forward_pass_dir, best_safe_clear_dir, best_shot_dir_evading, earliest_intercept,
    gk_cover_point, predict_ball_path, truncate_to_guaranteed_intercept, Candidate,
};
use crate::sensors::TeamSnapshot;
use crate::types::{PlayerCommand, PlayerId, TeamCommands, Vec2};

const SPRINT_SPEED: f32 = 8.0;
const HOLD_PROXY: f32 = 0.55;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Attacker,
    LackeyLeft,
    LackeyRight,
    Goalkeeper,
}

#[derive(Debug)]
pub struct TitaniumBrain {
    pub debug: bool,
    side_bias: f32,
    tick: u64,
    pub params: EngineParams,
    flick_dir: Option<Vec2>,
    orbit_side: f32,
}

impl Default for TitaniumBrain {
    fn default() -> Self {
        Self::new(false)
    }
}

impl TitaniumBrain {
    pub fn new(debug: bool) -> Self {
        Self {
            debug,
            side_bias: 1.0,
            tick: 0,
            params: EngineParams::default(),
            flick_dir: None,
            orbit_side: 1.0,
        }
    }

    pub fn think(&mut self, snap: &TeamSnapshot) -> TeamCommands {
        self.tick = self.tick.wrapping_add(1);
        let opponents: Vec<Vec2> = snap.opp_pos.to_vec();
        let carrier = snap.carrier_pos();
        let mut output = TeamCommands::default();

        for id in PlayerId::ALL {
            let index = id.0 as usize - 1;
            let me = snap.team_pos[index];
            let has_ball = snap.team_has[index];
            let charge = snap.shot_charge[index].clamp(0.0, 1.0);
            let command = match Self::role(id) {
                Role::Attacker => self.think_attacker(snap, me, has_ball, charge, &opponents),
                Role::LackeyLeft => self.think_lackey(
                    snap,
                    me,
                    true,
                    has_ball,
                    charge,
                    carrier,
                    &opponents,
                ),
                Role::LackeyRight => self.think_lackey(
                    snap,
                    me,
                    false,
                    has_ball,
                    charge,
                    carrier,
                    &opponents,
                ),
                Role::Goalkeeper => {
                    self.think_gk(snap, me, has_ball, charge, &opponents)
                }
            };
            if self.debug && self.tick % 25 == 0 {
                eprintln!(
                    "[titanium t={} P{}] move=({:.1},{:.1}) has={} ch={:.2} interact={}",
                    self.tick,
                    id.0,
                    command.move_to.x,
                    command.move_to.y,
                    has_ball,
                    charge,
                    command.interact
                );
            }
            output.commands[index] = command;
        }
        output
    }

    fn role(id: PlayerId) -> Role {
        match id.0 {
            1 => Role::Attacker,
            2 => Role::LackeyLeft,
            3 => Role::LackeyRight,
            _ => Role::Goalkeeper,
        }
    }

    fn attack_sign(snap: &TeamSnapshot) -> f32 {
        snap.attack_sign()
    }

    fn own_goal_x(snap: &TeamSnapshot, params: &EngineParams) -> f32 {
        snap.own_goal_x(params.goal_line_x)
    }

    fn nearest_opp(me: Vec2, opponents: &[Vec2]) -> Option<Vec2> {
        let mut best: Option<(f32, Vec2)> = None;
        for &opponent in opponents {
            let distance = me.distance_squared(opponent);
            best = Some(match best {
                // `<=` gives deterministic first-in-array tie behavior and
                // avoids relying on an Ordering::Equal branch.
                Some((best_distance, best_position)) if best_distance <= distance => {
                    (best_distance, best_position)
                }
                _ => (distance, opponent),
            });
        }
        best.map(|(_, position)| position)
    }

    fn pressure_radius(carrier: Vec2, opponents: &[Vec2]) -> f32 {
        opponents
            .iter()
            .map(|position| carrier.distance(*position))
            .fold(f32::INFINITY, f32::min)
    }

    fn cross2(a: Vec2, b: Vec2) -> f32 {
        a.x * b.y - a.y * b.x
    }

    /// Charge while walking; release by putting the shot direction in MoveTo.
    fn flick_shot(
        me: Vec2,
        walk_to: Vec2,
        shot_dir: Vec2,
        charge: f32,
        release_at: f32,
    ) -> PlayerCommand {
        let direction = if shot_dir.length_squared() > 1e-8 {
            shot_dir.normalize()
        } else {
            Vec2::X
        };
        if charge >= release_at {
            PlayerCommand {
                move_to: me + direction * 10.0,
                sprint: true,
                interact: false,
            }
        } else {
            PlayerCommand {
                move_to: walk_to,
                sprint: true,
                interact: true,
            }
        }
    }

    /// Orbit around the nearest tackler. A requested flank change takes the
    /// long route behind the threat before switching sides.
    fn anti_tackle_walk(
        &mut self,
        me: Vec2,
        goal_dir: Vec2,
        threat: Option<Vec2>,
        prefer_side: f32,
    ) -> Vec2 {
        let Some(threat) = threat else {
            return me + goal_dir * 6.0;
        };
        let to_me = me - threat;
        let distance = to_me.length().max(0.2);
        let radial = to_me / distance;
        let goal = if goal_dir.length_squared() > 1e-8 {
            goal_dir.normalize()
        } else {
            radial
        };
        let current_side = {
            let cross = Self::cross2(goal, to_me);
            if cross.abs() < 1e-3 {
                self.orbit_side
            } else {
                cross.signum()
            }
        };
        let preferred = if prefer_side.abs() < 0.5 {
            current_side
        } else {
            prefer_side.signum()
        };
        let behind = radial.dot(goal) < -0.15;
        let side = if (preferred - current_side).abs() > 0.5 && !behind {
            current_side
        } else {
            preferred
        };
        self.orbit_side = side;

        let tangent = Vec2::new(-radial.y, radial.x) * side;
        let face = (goal * 0.55 + tangent * 0.70 - radial * 0.35).normalize_or_zero();
        let hold = me + face * HOLD_PROXY;
        let safe_distance = self.params.interact_radius * 1.35;
        let face = if hold.distance(threat) < safe_distance {
            (tangent * 0.9 - radial * 0.5 + goal * 0.2).normalize_or_zero()
        } else {
            face
        };
        me + face * 5.5
    }

    fn ball_path(&self, snap: &TeamSnapshot, horizon: f32) -> Vec<crate::predict::BallSample> {
        let ball = Ball {
            pos: snap.ball_pos,
            vel: snap.ball_vel,
            height: self.params.ball_rest_height,
            vel_y: 0.0,
            held: false,
        };
        predict_ball_path(&ball, &self.params, crate::types::FIXED_DT, horizon)
    }

    fn think_attacker(
        &mut self,
        snap: &TeamSnapshot,
        me: Vec2,
        has_ball: bool,
        charge: f32,
        opponents: &[Vec2],
    ) -> PlayerCommand {
        let sign = Self::attack_sign(snap);
        let goal_dir = Vec2::new(sign, 0.0);
        let opponent_candidates: Vec<Candidate> = opponents
            .iter()
            .copied()
            .map(|pos| Candidate {
                pos,
                speed: SPRINT_SPEED,
            })
            .collect();

        if !has_ball {
            self.flick_dir = None;
            let path = self.ball_path(snap, 2.5);
            let candidate = [Candidate {
                pos: me,
                speed: SPRINT_SPEED,
            }];
            let target = truncate_to_guaranteed_intercept(&path, &candidate, self.params.interact_radius)
                .1
                .map(|(_, hit)| hit.pos)
                .unwrap_or(snap.ball_pos);
            return PlayerCommand {
                move_to: target,
                sprint: true,
                interact: me.distance(target) <= self.params.interact_radius * 1.15,
            };
        }

        let pressure = Self::pressure_radius(me, opponents);
        let threat = Self::nearest_opp(me, opponents);
        let mut shot = self.flick_dir;
        if shot.is_none() || charge < 0.35 {
            if pressure > 5.0 {
                for teammate_index in [1usize, 2usize] {
                    let mate = snap.team_pos[teammate_index];
                    if (mate.x - me.x) * sign < 3.0 {
                        continue;
                    }
                    if let Some(direction) = best_forward_pass_dir(
                        me,
                        mate,
                        &opponent_candidates,
                        charge.max(0.4),
                        &self.params,
                    ) {
                        shot = Some(direction);
                        break;
                    }
                }
            }
            if shot.is_none() {
                let direction = best_shot_dir_evading(
                    me,
                    sign,
                    opponents,
                    charge.max(0.55),
                    &self.params,
                );
                shot = Some(
                    (direction + Vec2::new(0.0, self.side_bias * 0.12)).normalize_or_zero(),
                );
            }
        }

        let shot_dir = shot.unwrap_or(goal_dir);
        if charge >= 0.40 {
            self.flick_dir = Some(shot_dir);
        }
        let preferred_side = if shot_dir.y.abs() > 0.08 {
            shot_dir.y.signum()
        } else {
            self.side_bias
        };
        let walk = self.anti_tackle_walk(me, goal_dir, threat, preferred_side);
        let release_at = if pressure > 5.0 { 0.55 } else { 0.92 };
        let command = Self::flick_shot(me, walk, shot_dir, charge, release_at);
        if !command.interact {
            self.flick_dir = None;
        }
        command
    }

    fn lackey_spot(carrier: Vec2, sign: f32, left: bool, pressure: f32) -> Vec2 {
        let z_sign = if left { 1.0 } else { -1.0 };
        if pressure > 8.0 {
            Vec2::new(carrier.x + sign * 10.0, carrier.y + z_sign * 7.0)
        } else if pressure > 4.5 {
            Vec2::new(carrier.x + sign * 4.0, carrier.y + z_sign * 9.0)
        } else {
            Vec2::new(carrier.x - sign * 3.0, carrier.y + z_sign * 10.0)
        }
    }

    fn think_lackey(
        &mut self,
        snap: &TeamSnapshot,
        me: Vec2,
        left: bool,
        has_ball: bool,
        charge: f32,
        carrier: Option<Vec2>,
        opponents: &[Vec2],
    ) -> PlayerCommand {
        let sign = Self::attack_sign(snap);
        let goal_dir = Vec2::new(sign, 0.0);
        if has_ball {
            let direction = self.flick_dir.unwrap_or_else(|| {
                best_shot_dir_evading(me, sign, opponents, charge.max(0.55), &self.params)
            });
            if charge >= 0.45 {
                self.flick_dir = Some(direction);
            }
            let command = Self::flick_shot(
                me,
                self.anti_tackle_walk(
                    me,
                    goal_dir,
                    Self::nearest_opp(me, opponents),
                    self.side_bias,
                ),
                direction,
                charge,
                0.90,
            );
            if !command.interact {
                self.flick_dir = None;
            }
            return command;
        }

        if snap.team_has_ball {
            let carrier = carrier.unwrap_or(snap.ball_pos);
            let pressure = Self::pressure_radius(carrier, opponents);
            return PlayerCommand {
                move_to: Self::lackey_spot(carrier, sign, left, pressure),
                sprint: pressure > 5.0,
                interact: false,
            };
        }

        let path = self.ball_path(snap, 2.0);
        if let Some(hit) = earliest_intercept(me, SPRINT_SPEED, &path, self.params.interact_radius) {
            let distance = me.distance(hit.pos);
            PlayerCommand {
                move_to: hit.pos,
                sprint: true,
                interact: distance <= self.params.interact_radius * 1.2,
            }
        } else {
            PlayerCommand {
                move_to: snap.ball_pos,
                sprint: true,
                interact: false,
            }
        }
    }

    fn think_gk(
        &mut self,
        snap: &TeamSnapshot,
        me: Vec2,
        has_ball: bool,
        charge: f32,
        opponents: &[Vec2],
    ) -> PlayerCommand {
        let sign = Self::attack_sign(snap);
        let own_goal_x = Self::own_goal_x(snap, &self.params);
        let goal_dir = Vec2::new(sign, 0.0);
        let opponent_candidates: Vec<Candidate> = opponents
            .iter()
            .copied()
            .map(|pos| Candidate {
                pos,
                speed: SPRINT_SPEED,
            })
            .collect();

        if has_ball {
            let clear = best_safe_clear_dir(
                me,
                sign,
                &opponent_candidates,
                charge.max(0.5),
                &self.params,
                18.0,
            );
            if charge >= 0.40 {
                self.flick_dir = Some(clear);
            }
            let direction = self.flick_dir.unwrap_or(clear);
            let command = Self::flick_shot(
                me,
                self.anti_tackle_walk(
                    me,
                    goal_dir,
                    Self::nearest_opp(me, opponents),
                    direction.y.signum(),
                ),
                direction,
                charge,
                0.70,
            );
            if !command.interact {
                self.flick_dir = None;
            }
            return command;
        }
        self.flick_dir = None;

        let threat = if snap.opp_has_ball {
            snap.opp_pos
                .iter()
                .zip(snap.opp_has.iter())
                .find_map(|(&position, &has)| has.then_some(position))
                .unwrap_or(snap.ball_pos)
        } else {
            snap.ball_pos
        };
        let cover = gk_cover_point(
            threat,
            own_goal_x,
            self.params.goal_half_width,
            self.params.interact_radius,
        );

        if snap.ball_loose || snap.ball_vel.length_squared() > 1.0 {
            let path = self.ball_path(snap, 3.0);
            let mut candidates = vec![Candidate {
                pos: me,
                speed: SPRINT_SPEED,
            }];
            candidates.extend(
                snap.team_pos[..3]
                    .iter()
                    .copied()
                    .map(|pos| Candidate {
                        pos,
                        speed: SPRINT_SPEED,
                    }),
            );
            if let Some((index, hit)) =
                truncate_to_guaranteed_intercept(&path, &candidates, self.params.interact_radius).1
            {
                if index == 0 {
                    return PlayerCommand {
                        move_to: hit.pos,
                        sprint: true,
                        interact: me.distance(hit.pos) <= self.params.interact_radius * 1.2,
                    };
                }
            }
        }

        PlayerCommand {
            move_to: cover,
            sprint: me.distance(cover) > 2.0,
            interact: me.distance(snap.ball_pos) <= self.params.interact_radius * 1.25
                && (snap.ball_loose || snap.opp_has_ball),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flick_snaps_move_to_on_release() {
        let me = Vec2::ZERO;
        let walk = Vec2::new(5.0, 3.0);
        let shot = Vec2::new(0.0, 1.0);
        let charging = TitaniumBrain::flick_shot(me, walk, shot, 0.5, 0.9);
        assert!(charging.interact);
        assert!((charging.move_to - walk).length() < 1e-4);
        let release = TitaniumBrain::flick_shot(me, walk, shot, 0.95, 0.9);
        assert!(!release.interact);
        assert!(release.move_to.y > 5.0);
        assert!(release.move_to.x.abs() < 0.1);
    }

    #[test]
    fn orbit_keeps_side_until_behind_threat() {
        let mut brain = TitaniumBrain::new(false);
        brain.orbit_side = 1.0;
        let me = Vec2::new(0.0, 4.0);
        // Prefer − side but not behind the threat yet → stay on + orbit.
        // With radial = +Y, +orbit tangent is −X, so walk.x < me.x.
        let walk = brain.anti_tackle_walk(me, Vec2::X, Some(Vec2::ZERO), -1.0);
        assert_eq!(brain.orbit_side, 1.0);
        assert!(walk.x < me.x, "walk={walk:?}");
    }

    #[test]
    fn think_emits_four_commands() {
        let snapshot = TeamSnapshot {
            team: crate::types::TeamId::Home,
            is_home: true,
            ball_pos: Vec2::ZERO,
            ball_vel: Vec2::ZERO,
            ball_loose: true,
            team_has_ball: false,
            opp_has_ball: false,
            team_pos: [Vec2::ZERO; 4],
            opp_pos: [Vec2::new(10.0, 10.0); 4],
            team_has: [false; 4],
            opp_has: [false; 4],
            shot_charge: [0.0; 4],
        };
        let commands = TitaniumBrain::default().think(&snapshot);
        assert_eq!(commands.commands.len(), 4);
    }
}
