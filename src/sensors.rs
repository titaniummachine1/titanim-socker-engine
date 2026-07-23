//! Sensor snapshot the simulator fills each think tick.

use crate::types::{PlayerId, TeamId, Vec2};

#[derive(Debug, Clone)]
pub struct TeamSnapshot {
    pub team: TeamId,
    pub is_home: bool,
    pub ball_pos: Vec2,
    pub ball_vel: Vec2,
    pub ball_loose: bool,
    pub team_has_ball: bool,
    pub opp_has_ball: bool,
    pub team_pos: [Vec2; 4],
    pub opp_pos: [Vec2; 4],
    pub team_stamina: [f32; 4],
    pub opp_stamina: [f32; 4],
    pub team_has: [bool; 4],
    pub opp_has: [bool; 4],
    pub shot_charge: [f32; 4],
}

impl TeamSnapshot {
    pub fn me(&self, id: PlayerId) -> Vec2 {
        self.team_pos[(id.0 as usize - 1).min(3)]
    }

    pub fn attack_sign(&self) -> f32 {
        if self.is_home {
            1.0
        } else {
            -1.0
        }
    }

    pub fn own_goal_x(&self, goal_line_x: f32) -> f32 {
        if self.is_home {
            -goal_line_x.abs()
        } else {
            goal_line_x.abs()
        }
    }

    pub fn carrier_pos(&self) -> Option<Vec2> {
        for i in 0..4 {
            if self.team_has[i] {
                return Some(self.team_pos[i]);
            }
        }
        None
    }

    /// Opponent body center for the current carrier.
    ///
    /// Prefer explicitly marked opponents; if only the aggregate possession
    /// bit is available, use the opponent nearest the held ball. The latter
    /// is still a body position, so the held-ball offset is never used as the
    /// shot origin.
    pub fn opponent_carrier_pos(&self) -> Option<Vec2> {
        let mut best: Option<(f32, Vec2)> = None;
        for i in 0..4 {
            if !self.opp_has[i] {
                continue;
            }
            let distance = self.opp_pos[i].distance_squared(self.ball_pos);
            best = Some(match best {
                Some((best_distance, best_position)) if best_distance <= distance => {
                    (best_distance, best_position)
                }
                _ => (distance, self.opp_pos[i]),
            });
        }
        if best.is_some() {
            return best.map(|(_, position)| position);
        }
        if !self.opp_has_ball {
            return None;
        }
        for &position in &self.opp_pos {
            let distance = position.distance_squared(self.ball_pos);
            best = Some(match best {
                Some((best_distance, best_position)) if best_distance <= distance => {
                    (best_distance, best_position)
                }
                _ => (distance, position),
            });
        }
        best.map(|(_, position)| position)
    }

    /// Stamina of the current opponent carrier.
    ///
    /// Prefer explicitly marked opponents; if only aggregate possession is
    /// available, use the opponent nearest the held ball.
    pub fn opponent_carrier_stamina(&self) -> Option<f32> {
        let mut best: Option<(f32, f32)> = None;
        for i in 0..4 {
            if !self.opp_has[i] {
                continue;
            }
            let distance = self.opp_pos[i].distance_squared(self.ball_pos);
            best = Some(match best {
                Some((best_distance, best_stamina)) if best_distance <= distance => {
                    (best_distance, best_stamina)
                }
                _ => (distance, self.opp_stamina[i]),
            });
        }
        if best.is_some() {
            return best.map(|(_, stamina)| stamina);
        }
        if !self.opp_has_ball {
            return None;
        }
        for i in 0..4 {
            let distance = self.opp_pos[i].distance_squared(self.ball_pos);
            best = Some(match best {
                Some((best_distance, best_stamina)) if best_distance <= distance => {
                    (best_distance, best_stamina)
                }
                _ => (distance, self.opp_stamina[i]),
            });
        }
        best.map(|(_, stamina)| stamina)
    }
}
