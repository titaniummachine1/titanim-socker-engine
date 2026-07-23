//! Private Titanium soccer decision engine.
//!
//! Consumes [`TeamSnapshot`] each tick; emits [`TeamCommands`]. The match
//! simulator owns real physics — this crate only *predicts* free-ball paths.

pub mod ball;
pub mod params;
pub mod predict;
pub mod sensors;
pub mod titanium;
pub mod types;

pub use ball::{Ball, EndReason};
pub use params::EngineParams;
pub use sensors::TeamSnapshot;
pub use titanium::TitaniumBrain;
pub use types::{PlayerCommand, PlayerId, TeamCommands, TeamId, Vec2, FIXED_DT};
