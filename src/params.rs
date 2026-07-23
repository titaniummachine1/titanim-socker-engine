//! Minimal params the engine needs (mirrors soccer sim defaults).

#[derive(Debug, Clone)]
pub struct EngineParams {
    pub goal_half_width: f32,
    pub goal_line_x: f32,
    pub interact_radius: f32,
    pub ball_rest_height: f32,
    pub gravity: f32,
    pub ball_bounce_e: f32,
    pub ball_bounce_settle: f32,
    pub slide_accel: f32,
    pub stop_speed_eps: f32,
    pub kick_speed_base: f32,
    pub kick_speed_per_charge: f32,
    pub kick_horiz_cap: f32,
    pub kick_lift_base: f32,
    pub kick_lift_per_charge: f32,
    pub kick_max_speed: f32,
    pub wall_e: f32,
    pub wall_mu: f32,
    pub x_min: f32,
    pub x_max: f32,
    pub z_min: f32,
    pub z_max: f32,
    pub posts_x: f32,
    pub post_radius: f32,
    pub ball_radius: f32,
}

impl Default for EngineParams {
    fn default() -> Self {
        Self {
            goal_half_width: 6.0,
            goal_line_x: 39.5,
            interact_radius: 1.75,
            ball_rest_height: 0.33,
            gravity: 9.81,
            ball_bounce_e: 0.45,
            ball_bounce_settle: 0.4,
            slide_accel: 5.95,
            stop_speed_eps: 0.05,
            kick_speed_base: 10.0,
            kick_speed_per_charge: 290.0 / 9.0,
            kick_horiz_cap: 35.0,
            kick_lift_base: 0.0,
            kick_lift_per_charge: 12.0,
            kick_max_speed: 40.0,
            wall_e: 0.2,
            wall_mu: 0.35,
            x_min: -39.5,
            x_max: 39.5,
            z_min: -24.7,
            z_max: 24.7,
            posts_x: 40.2,
            post_radius: 0.3,
            ball_radius: 0.4064,
        }
    }
}
