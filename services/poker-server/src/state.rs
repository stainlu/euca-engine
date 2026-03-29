//! Shared application state.

use crate::room::RoomManager;

/// Top-level application state shared across all handlers.
pub struct AppState {
    pub rooms: RoomManager,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            rooms: RoomManager::new(),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
