//! Game state — match lifecycle and scoring.
//!
//! Resources: `GameState`, `MatchConfig`.
//! Systems: `game_state_system`, `score_system`.

use euca_ecs::{Entity, Events, World};
use std::collections::HashMap;

use crate::health::DeathEvent;

/// Current phase of the match.
#[derive(Clone, Debug, PartialEq)]
pub enum GamePhase {
    /// Waiting for players.
    Lobby,
    /// Counting down to start.
    Countdown { remaining: f32 },
    /// Match in progress.
    Playing,
    /// Match ended.
    PostMatch { winner: Option<Entity> },
}

/// Match configuration (data-driven rules).
#[derive(Clone, Debug)]
pub struct MatchConfig {
    pub mode: String,
    pub score_limit: i32,
    pub time_limit: f32,
    pub respawn_delay: f32,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            mode: "deathmatch".to_string(),
            score_limit: 10,
            time_limit: 300.0,
            respawn_delay: 3.0,
        }
    }
}

/// Global match state resource.
#[derive(Clone, Debug)]
pub struct GameState {
    pub phase: GamePhase,
    pub config: MatchConfig,
    pub scores: HashMap<u32, i32>, // entity index → score
    pub elapsed: f32,
}

impl GameState {
    pub fn new(config: MatchConfig) -> Self {
        Self {
            phase: GamePhase::Lobby,
            config,
            scores: HashMap::new(),
            elapsed: 0.0,
        }
    }

    pub fn start(&mut self) {
        self.phase = GamePhase::Playing;
        self.elapsed = 0.0;
    }

    pub fn add_score(&mut self, entity_index: u32, points: i32) {
        *self.scores.entry(entity_index).or_insert(0) += points;
    }

    /// Check if any player has reached the score limit.
    pub fn check_winner(&self) -> Option<u32> {
        self.scores
            .iter()
            .find(|(_, score)| **score >= self.config.score_limit)
            .map(|(&idx, _)| idx)
    }

    /// Sorted scoreboard (highest first).
    pub fn scoreboard(&self) -> Vec<(u32, i32)> {
        let mut board: Vec<(u32, i32)> = self.scores.iter().map(|(&k, &v)| (k, v)).collect();
        board.sort_by(|a, b| b.1.cmp(&a.1));
        board
    }
}

/// Score event — emitted when a player earns points.
#[derive(Clone, Debug)]
pub struct ScoreEvent {
    pub entity_index: u32,
    pub points: i32,
}

/// Update game state: process scores, check win conditions, advance phase.
pub fn game_state_system(world: &mut World, dt: f32) {
    // Process death events → award kill scores
    let kill_scores: Vec<ScoreEvent> = world
        .resource::<Events>()
        .map(|events| {
            events
                .read::<DeathEvent>()
                .filter_map(|death| {
                    death.killer.map(|killer| ScoreEvent {
                        entity_index: killer.index(),
                        points: 1,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    // Process score events
    let score_events: Vec<ScoreEvent> = world
        .resource::<Events>()
        .map(|e| e.read::<ScoreEvent>().cloned().collect())
        .unwrap_or_default();

    if let Some(state) = world.resource_mut::<GameState>() {
        // Apply kill scores
        for event in &kill_scores {
            state.add_score(event.entity_index, event.points);
        }
        for event in &score_events {
            state.add_score(event.entity_index, event.points);
        }

        // Phase transitions
        match &state.phase {
            GamePhase::Lobby => {
                // Auto-start when game_state_system is called (can be triggered by CLI)
            }
            GamePhase::Countdown { remaining } => {
                let new_remaining = remaining - dt;
                if new_remaining <= 0.0 {
                    state.phase = GamePhase::Playing;
                    state.elapsed = 0.0;
                } else {
                    state.phase = GamePhase::Countdown {
                        remaining: new_remaining,
                    };
                }
            }
            GamePhase::Playing => {
                state.elapsed += dt;

                // Check score limit
                if let Some(winner_idx) = state.check_winner() {
                    // Find the entity — use index as-is for now
                    state.phase = GamePhase::PostMatch {
                        winner: Some(euca_ecs::Entity::from_raw(winner_idx, 0)),
                    };
                }

                // Check time limit
                if state.config.time_limit > 0.0 && state.elapsed >= state.config.time_limit {
                    let winner_idx = state
                        .scoreboard()
                        .first()
                        .map(|(idx, _)| euca_ecs::Entity::from_raw(*idx, 0));
                    state.phase = GamePhase::PostMatch { winner: winner_idx };
                }
            }
            GamePhase::PostMatch { .. } => {
                // Match is over — do nothing
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_state_tracks_scores() {
        let mut state = GameState::new(MatchConfig::default());
        state.add_score(1, 3);
        state.add_score(2, 5);
        state.add_score(1, 2);

        assert_eq!(state.scores[&1], 5);
        assert_eq!(state.scores[&2], 5);
    }

    #[test]
    fn winner_detected_at_score_limit() {
        let mut state = GameState::new(MatchConfig {
            score_limit: 5,
            ..Default::default()
        });
        state.add_score(1, 5);

        assert_eq!(state.check_winner(), Some(1));
    }

    #[test]
    fn no_winner_below_limit() {
        let mut state = GameState::new(MatchConfig {
            score_limit: 10,
            ..Default::default()
        });
        state.add_score(1, 3);
        state.add_score(2, 4);

        assert_eq!(state.check_winner(), None);
    }

    #[test]
    fn scoreboard_sorted() {
        let mut state = GameState::new(MatchConfig::default());
        state.add_score(1, 3);
        state.add_score(2, 7);
        state.add_score(3, 5);

        let board = state.scoreboard();
        assert_eq!(board[0], (2, 7));
        assert_eq!(board[1], (3, 5));
        assert_eq!(board[2], (1, 3));
    }

    #[test]
    fn countdown_transitions_to_playing() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let mut state = GameState::new(MatchConfig::default());
        state.phase = GamePhase::Countdown { remaining: 1.0 };
        world.insert_resource(state);

        game_state_system(&mut world, 1.5);

        let state = world.resource::<GameState>().unwrap();
        assert_eq!(state.phase, GamePhase::Playing);
    }
}
