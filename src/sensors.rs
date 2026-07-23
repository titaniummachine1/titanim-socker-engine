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
}
