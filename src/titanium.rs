//! Titanium decision policy over a simulator-provided [`TeamSnapshot`].
//!
//! The engine deliberately has no world/API dependency: callers provide all
//! positions, possession bits, ball state, and charge values in the snapshot.

use crate::ball::Ball;
use crate::params::EngineParams;
use crate::predict::{
    best_forward_pass_dir, best_long_clear_dir, best_shot_dir_evading, earliest_intercept,
    gk_intercept_cover, predict_ball_path, predict_ball_path_until_intercept,
    truncate_to_guaranteed_intercept, Candidate,
};
use crate::sensors::TeamSnapshot;
use crate::types::{PlayerCommand, PlayerId, TeamCommands, Vec2, FIXED_DT};

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
                Role::LackeyLeft => {
                    self.think_lackey(snap, me, true, has_ball, charge, carrier, &opponents)
                }
                Role::LackeyRight => {
                    self.think_lackey(snap, me, false, has_ball, charge, carrier, &opponents)
                }
                Role::Goalkeeper => self.think_gk(snap, me, has_ball, charge, &opponents),
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
            let target =
                truncate_to_guaranteed_intercept(&path, &candidate, self.params.interact_radius)
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

        // 1v1 finisher: ignore parked wide corners; carry wide, then aim far
        // post. Keep the open-play branch below unchanged.
        let near_central = opponents
            .iter()
            .filter(|p| me.distance(**p) < 18.0 && p.y.abs() < 14.0)
            .count();
        let goal_line = sign * self.params.goal_line_x.abs();
        let gk_like = threat
            .map(|t| (t.x - goal_line).abs() < 10.0 && t.y.abs() < 10.0)
            .unwrap_or(false);
        if near_central <= 1 || gk_like {
            let threat_dist = threat.map(|t| me.distance(t)).unwrap_or(999.0);
            let dist_goal = (goal_line - me.x).abs();
            let wide_z = self.side_bias * 7.0;
            let far_post_z = -self.side_bias * self.params.goal_half_width * 0.95;
            let wide_enough = (me.y - wide_z).abs() <= 2.5;
            let can_finish = wide_enough && dist_goal < 11.0;
            let panic = threat_dist < self.params.interact_radius * 1.15;
            let far_aim = (Vec2::new(goal_line, far_post_z) - me).normalize_or_zero();
            let mut shot_dir = if can_finish || panic {
                let evade =
                    best_shot_dir_evading(me, sign, opponents, charge.max(0.85), &self.params);
                (far_aim * 0.55 + evade * 0.45).normalize_or_zero()
            } else {
                far_aim
            };
            if !snap.is_home {
                let mut d = shot_dir.normalize_or_zero();
                if d.x < -0.55 && d.y > -0.55 {
                    let y_sign = if far_post_z >= 0.0 { 1.0 } else { -1.0 };
                    d = Vec2::new(-0.50, y_sign * (0.75f32).sqrt());
                }
                shot_dir = d;
            }
            let walk = if !wide_enough {
                Vec2::new(me.x + sign * 5.0, wide_z)
            } else if !can_finish {
                Vec2::new(me.x + sign * 8.0, wide_z)
            } else if panic {
                self.anti_tackle_walk(me, goal_dir, threat, far_post_z.signum())
            } else {
                me + shot_dir * 5.0
            };
            let release_at: f32 = if panic && charge >= 0.40 {
                0.30
            } else if !can_finish {
                1.05
            } else {
                0.75
            };
            if charge >= 0.20 && (can_finish || panic) {
                self.flick_dir = Some(shot_dir);
            }
            let direction = self.flick_dir.unwrap_or(shot_dir);
            let command = Self::flick_shot(me, walk, direction, charge, release_at);
            if !command.interact {
                self.flick_dir = None;
            }
            return command;
        }

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
                let direction =
                    best_shot_dir_evading(me, sign, opponents, charge.max(0.55), &self.params);
                shot =
                    Some((direction + Vec2::new(0.0, self.side_bias * 0.12)).normalize_or_zero());
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

        // Finishing: drive at goal when the nearest threat is loose / deep (typical
        // 1v1 vs GK). Only hard-orbit when a tackler is close.
        let threat_dist = threat.map(|t| me.distance(t)).unwrap_or(f32::INFINITY);
        // Only "drive and flick early" when truly open (far lone threat / GK).
        let open_look = pressure > 14.0 && threat_dist > 16.0;
        let walk = if open_look {
            me + goal_dir * 7.0 + Vec2::new(0.0, preferred_side * 2.0)
        } else {
            self.anti_tackle_walk(me, goal_dir, threat, preferred_side)
        };
        let release_at = if open_look {
            0.68
        } else if pressure > 5.0 {
            0.55
        } else {
            0.88
        };
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
        if let Some(hit) = earliest_intercept(me, SPRINT_SPEED, &path, self.params.interact_radius)
        {
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
            let clear = best_long_clear_dir(
                me,
                sign,
                &opponent_candidates,
                charge.max(0.7),
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

        // Opponent holds: go to ball, Interact in range — global stam duel.
        if snap.opp_has_ball {
            let reach = self.params.interact_radius;
            let in_reach = me.distance(snap.ball_pos) <= reach;
            return PlayerCommand {
                move_to: snap.ball_pos,
                sprint: true,
                interact: in_reach,
            };
        }

        // Loose / free ball: intercept race.
        if snap.ball_loose || snap.ball_vel.length_squared() > 0.25 {
            let mut cands = opponent_candidates;
            cands.push(Candidate {
                pos: me,
                speed: SPRINT_SPEED,
            });
            for &pos in &snap.team_pos {
                cands.push(Candidate {
                    pos,
                    speed: SPRINT_SPEED,
                });
            }
            let ball = Ball {
                pos: snap.ball_pos,
                vel: snap.ball_vel,
                height: self.params.ball_rest_height,
                vel_y: 0.0,
                held: false,
            };
            let reach = self.params.interact_radius;
            let (path, first) = predict_ball_path_until_intercept(
                &ball,
                &self.params,
                FIXED_DT,
                3.0,
                &cands,
                reach,
            );
            if let Some(hit) = earliest_intercept(me, SPRINT_SPEED, &path, reach) {
                let cut = if own_goal_x > 0.0 {
                    hit.pos.x.max(0.0)
                } else {
                    hit.pos.x.min(0.0)
                };
                return PlayerCommand {
                    move_to: Vec2::new(cut, hit.pos.y),
                    sprint: true,
                    interact: me.distance(hit.pos) <= reach,
                };
            }
            let fallback = first.map(|(_, h)| h.pos).unwrap_or(snap.ball_pos);
            return PlayerCommand {
                move_to: fallback,
                sprint: true,
                interact: me.distance(snap.ball_pos) <= reach,
            };
        }

        // Idle: deepest safe cover.
        let cover = gk_intercept_cover(
            snap.ball_pos,
            own_goal_x,
            self.params.goal_half_width,
            &self.params,
            SPRINT_SPEED,
        );
        PlayerCommand {
            move_to: cover,
            sprint: me.distance(cover) > 1.5,
            interact: false,
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
