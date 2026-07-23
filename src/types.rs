//! Shared types. Pitch `Vec2(x,y)` = Unity `(X,Z)`.

pub use glam::Vec2;

/// Confirmed AIComp fixed step (~52.6 Hz). Must match the simulator.
pub const FIXED_DT: f32 = 0.019;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TeamId {
    Home,
    Away,
}

impl TeamId {
    pub fn other(self) -> Self {
        match self {
            TeamId::Home => TeamId::Away,
            TeamId::Away => TeamId::Home,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlayerId(pub u8);

impl PlayerId {
    pub const ALL: [PlayerId; 4] = [PlayerId(1), PlayerId(2), PlayerId(3), PlayerId(4)];
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerCommand {
    pub move_to: Vec2,
    pub sprint: bool,
    pub interact: bool,
}

impl Default for PlayerCommand {
    fn default() -> Self {
        Self {
            move_to: Vec2::ZERO,
            sprint: false,
            interact: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TeamCommands {
    pub commands: [PlayerCommand; 4],
}

impl Default for TeamCommands {
    fn default() -> Self {
        Self {
            commands: [PlayerCommand::default(); 4],
        }
    }
}

impl TeamCommands {
    pub fn for_player(&self, id: PlayerId) -> PlayerCommand {
        let i = (id.0.saturating_sub(1) as usize).min(3);
        self.commands[i]
    }
}
