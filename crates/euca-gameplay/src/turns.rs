//! Turn & phase management — genre-agnostic turn-based game primitives.
//!
//! Resources: [`TurnState`], [`TurnConfig`].
//! Events: [`TurnEvent`].
//! Systems: [`turn_system`].
//!
//! Phases are plain strings, not enums — games define their own phases
//! ("movement", "combat", "diplomacy", "end"). The engine just cycles through them.

use euca_ecs::{Events, World};
use std::collections::HashMap;

/// Current turn state — who is active, what phase, how many action points remain.
#[derive(Clone, Debug)]
pub struct TurnState {
    /// Current turn number (starts at 1).
    pub turn_number: u32,
    /// Which player is currently active (`None` before the first advance).
    pub active_player: Option<u8>,
    /// Current phase name (empty before the first advance).
    pub phase: String,
    /// Action points remaining per player.
    pub action_points: HashMap<u8, i32>,
}

impl TurnState {
    /// Create a new turn state with no active player or phase.
    pub fn new() -> Self {
        Self {
            turn_number: 0,
            active_player: None,
            phase: String::new(),
            action_points: HashMap::new(),
        }
    }
}

impl Default for TurnState {
    fn default() -> Self {
        Self::new()
    }
}

/// Static configuration for turn-based games.
#[derive(Clone, Debug)]
pub struct TurnConfig {
    /// Ordered list of player IDs that take turns.
    pub player_order: Vec<u8>,
    /// Ordered list of phase names each player goes through per turn.
    pub phases_per_turn: Vec<String>,
    /// Action points granted to each player at the start of their turn.
    pub action_points_per_turn: i32,
}

impl Default for TurnConfig {
    fn default() -> Self {
        Self {
            player_order: vec![0],
            phases_per_turn: vec!["main".to_string()],
            action_points_per_turn: 1,
        }
    }
}

/// Events emitted during turn/phase transitions.
#[derive(Clone, Debug, PartialEq)]
pub enum TurnEvent {
    /// A player's turn has started.
    OnTurnStart { player: u8, turn: u32 },
    /// A new phase has begun.
    OnPhaseStart { phase: String },
    /// A player's turn has ended.
    OnTurnEnd { player: u8 },
}

/// Advance to the next phase (and potentially the next player/turn).
///
/// Progression: phase0 → phase1 → ... → last_phase → next_player/phase0.
/// After the last player's last phase → increment turn_number, wrap to first player.
///
/// Emits [`TurnEvent`]s via the [`Events`] resource.
pub fn advance_phase(world: &mut World) {
    let config = match world.resource::<TurnConfig>() {
        Some(c) => c.clone(),
        None => return,
    };

    if config.player_order.is_empty() || config.phases_per_turn.is_empty() {
        return;
    }

    let state = match world.resource_mut::<TurnState>() {
        Some(s) => s,
        None => return,
    };

    // Determine current indices
    let current_player_idx = state
        .active_player
        .and_then(|p| config.player_order.iter().position(|&x| x == p));
    let current_phase_idx = if state.phase.is_empty() {
        None
    } else {
        config
            .phases_per_turn
            .iter()
            .position(|p| *p == state.phase)
    };

    // Calculate next phase/player/turn
    let (next_player_idx, next_phase_idx, new_turn) = match (current_player_idx, current_phase_idx)
    {
        // Not yet started — begin turn 1, first player, first phase
        (None, _) => (0, 0, true),
        // Mid-turn: advance to next phase for same player
        (Some(pi), Some(phi)) if phi + 1 < config.phases_per_turn.len() => (pi, phi + 1, false),
        // Last phase: advance to next player
        (Some(pi), _) if pi + 1 < config.player_order.len() => (pi + 1, 0, true),
        // Last player, last phase: wrap to first player, new turn
        (Some(_), _) => (0, 0, true),
    };

    let prev_player = state.active_player;
    let next_player = config.player_order[next_player_idx];
    let next_phase = config.phases_per_turn[next_phase_idx].clone();

    // Update state — `new_turn` means this player is starting a fresh turn
    // (could be a different player, or the same player wrapping around).
    if new_turn {
        if state.active_player.is_none() {
            // Very first turn
            state.turn_number = 1;
        } else if next_player_idx == 0 {
            // Wrapped around to first player — new round
            state.turn_number += 1;
        }
    }

    state.active_player = Some(next_player);
    state.phase = next_phase.clone();

    // Reset AP when a player starts a new turn
    if new_turn {
        *state.action_points.entry(next_player).or_insert(0) = config.action_points_per_turn;
    }

    // Snapshot values and build events before borrowing Events (disjoint resource borrows).
    let turn_number = state.turn_number;
    let mut pending_events: Vec<TurnEvent> = Vec::new();

    if let Some(prev) = prev_player
        && new_turn
    {
        pending_events.push(TurnEvent::OnTurnEnd { player: prev });
    }
    if new_turn {
        pending_events.push(TurnEvent::OnTurnStart {
            player: next_player,
            turn: turn_number,
        });
    }
    pending_events.push(TurnEvent::OnPhaseStart { phase: next_phase });

    // Emit events
    if let Some(events) = world.resource_mut::<Events>() {
        for event in pending_events {
            events.send(event);
        }
    }
}

/// Deduct action points from a player. Returns `false` if insufficient AP.
pub fn spend_action_points(state: &mut TurnState, player: u8, cost: i32) -> bool {
    let ap = state.action_points.entry(player).or_insert(0);
    if *ap >= cost {
        *ap -= cost;
        true
    } else {
        false
    }
}

/// System that drives turn/phase advancement via events.
///
/// Call [`advance_phase`] to progress the game — this system emits the
/// corresponding [`TurnEvent`]s. Games call `advance_phase` when a player
/// confirms "end phase" or "end turn".
pub fn turn_system(_world: &mut World) {
    // Advancement is driven explicitly by calling advance_phase().
    // This system exists as a hook point for future per-tick logic (e.g., timed turns).
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: set up a world with turn resources and events.
    fn setup_world(players: Vec<u8>, phases: Vec<&str>, ap: i32) -> World {
        let mut world = World::new();
        world.insert_resource(Events::default());
        world.insert_resource(TurnState::new());
        world.insert_resource(TurnConfig {
            player_order: players,
            phases_per_turn: phases.into_iter().map(String::from).collect(),
            action_points_per_turn: ap,
        });
        world
    }

    /// Collect all TurnEvents from the world.
    fn read_turn_events(world: &World) -> Vec<TurnEvent> {
        world
            .resource::<Events>()
            .map(|e| e.read::<TurnEvent>().cloned().collect())
            .unwrap_or_default()
    }

    #[test]
    fn first_advance_starts_turn_one() {
        let mut world = setup_world(vec![0, 1], vec!["move", "combat"], 3);

        advance_phase(&mut world);

        let state = world.resource::<TurnState>().unwrap();
        assert_eq!(state.turn_number, 1);
        assert_eq!(state.active_player, Some(0));
        assert_eq!(state.phase, "move");
    }

    #[test]
    fn phase_cycling_within_player() {
        let mut world = setup_world(vec![0], vec!["move", "combat", "end"], 5);

        advance_phase(&mut world); // move
        advance_phase(&mut world); // combat

        let state = world.resource::<TurnState>().unwrap();
        assert_eq!(state.phase, "combat");
        assert_eq!(state.active_player, Some(0));
        assert_eq!(state.turn_number, 1);
    }

    #[test]
    fn advance_past_last_phase_goes_to_next_player() {
        let mut world = setup_world(vec![0, 1], vec!["main"], 2);

        advance_phase(&mut world); // player 0, main
        advance_phase(&mut world); // player 1, main

        let state = world.resource::<TurnState>().unwrap();
        assert_eq!(state.active_player, Some(1));
        assert_eq!(state.phase, "main");
        assert_eq!(state.turn_number, 1);
    }

    #[test]
    fn wrap_around_last_player_to_first_increments_turn() {
        let mut world = setup_world(vec![0, 1], vec!["main"], 2);

        advance_phase(&mut world); // turn 1, player 0
        advance_phase(&mut world); // turn 1, player 1
        advance_phase(&mut world); // turn 2, player 0

        let state = world.resource::<TurnState>().unwrap();
        assert_eq!(state.turn_number, 2);
        assert_eq!(state.active_player, Some(0));
    }

    #[test]
    fn three_player_rotation() {
        let mut world = setup_world(vec![1, 2, 3], vec!["act"], 1);

        advance_phase(&mut world); // turn 1, player 1
        assert_eq!(
            world.resource::<TurnState>().unwrap().active_player,
            Some(1)
        );

        advance_phase(&mut world); // turn 1, player 2
        assert_eq!(
            world.resource::<TurnState>().unwrap().active_player,
            Some(2)
        );

        advance_phase(&mut world); // turn 1, player 3
        assert_eq!(
            world.resource::<TurnState>().unwrap().active_player,
            Some(3)
        );

        advance_phase(&mut world); // turn 2, player 1
        let state = world.resource::<TurnState>().unwrap();
        assert_eq!(state.active_player, Some(1));
        assert_eq!(state.turn_number, 2);
    }

    #[test]
    fn multi_phase_multi_player_full_cycle() {
        let mut world = setup_world(vec![0, 1], vec!["move", "combat"], 3);

        // Turn 1, Player 0
        advance_phase(&mut world); // move
        advance_phase(&mut world); // combat
        // Turn 1, Player 1
        advance_phase(&mut world); // move
        advance_phase(&mut world); // combat
        // Turn 2, Player 0
        advance_phase(&mut world); // move

        let state = world.resource::<TurnState>().unwrap();
        assert_eq!(state.turn_number, 2);
        assert_eq!(state.active_player, Some(0));
        assert_eq!(state.phase, "move");
    }

    #[test]
    fn action_points_reset_on_new_player() {
        let mut world = setup_world(vec![0, 1], vec!["main"], 5);

        advance_phase(&mut world); // player 0
        let state = world.resource::<TurnState>().unwrap();
        assert_eq!(state.action_points[&0], 5);

        advance_phase(&mut world); // player 1
        let state = world.resource::<TurnState>().unwrap();
        assert_eq!(state.action_points[&1], 5);
    }

    #[test]
    fn spend_action_points_success() {
        let mut state = TurnState::new();
        state.action_points.insert(0, 5);

        assert!(spend_action_points(&mut state, 0, 3));
        assert_eq!(state.action_points[&0], 2);
    }

    #[test]
    fn spend_action_points_exact() {
        let mut state = TurnState::new();
        state.action_points.insert(0, 3);

        assert!(spend_action_points(&mut state, 0, 3));
        assert_eq!(state.action_points[&0], 0);
    }

    #[test]
    fn spend_action_points_insufficient() {
        let mut state = TurnState::new();
        state.action_points.insert(0, 2);

        assert!(!spend_action_points(&mut state, 0, 5));
        // AP should remain unchanged
        assert_eq!(state.action_points[&0], 2);
    }

    #[test]
    fn spend_action_points_no_entry() {
        let mut state = TurnState::new();

        // Player 7 has no AP entry — should fail for any positive cost
        assert!(!spend_action_points(&mut state, 7, 1));
    }

    #[test]
    fn events_emitted_on_first_advance() {
        let mut world = setup_world(vec![0], vec!["main"], 1);

        advance_phase(&mut world);

        let events = read_turn_events(&world);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], TurnEvent::OnTurnStart { player: 0, turn: 1 });
        assert_eq!(
            events[1],
            TurnEvent::OnPhaseStart {
                phase: "main".to_string()
            }
        );
    }

    #[test]
    fn turn_end_event_on_player_change() {
        let mut world = setup_world(vec![0, 1], vec!["main"], 1);

        advance_phase(&mut world); // player 0 starts
        // Clear events from first advance
        world.resource_mut::<Events>().unwrap().update();

        advance_phase(&mut world); // player 1 starts → player 0 ends

        let events = read_turn_events(&world);
        assert!(events.contains(&TurnEvent::OnTurnEnd { player: 0 }));
        assert!(events.contains(&TurnEvent::OnTurnStart { player: 1, turn: 1 }));
    }

    #[test]
    fn phase_start_event_on_phase_change() {
        let mut world = setup_world(vec![0], vec!["move", "combat"], 1);

        advance_phase(&mut world); // move
        world.resource_mut::<Events>().unwrap().update();

        advance_phase(&mut world); // combat

        let events = read_turn_events(&world);
        assert!(events.contains(&TurnEvent::OnPhaseStart {
            phase: "combat".to_string()
        }));
    }

    #[test]
    fn wrap_around_emits_turn_end_and_start() {
        let mut world = setup_world(vec![0, 1], vec!["act"], 1);

        advance_phase(&mut world); // turn 1, player 0
        advance_phase(&mut world); // turn 1, player 1
        world.resource_mut::<Events>().unwrap().update();

        advance_phase(&mut world); // turn 2, player 0

        let events = read_turn_events(&world);
        assert!(events.contains(&TurnEvent::OnTurnEnd { player: 1 }));
        assert!(events.contains(&TurnEvent::OnTurnStart { player: 0, turn: 2 }));
    }

    #[test]
    fn action_points_persist_across_phases() {
        let mut world = setup_world(vec![0], vec!["move", "combat"], 5);

        advance_phase(&mut world); // move — AP set to 5
        {
            let state = world.resource_mut::<TurnState>().unwrap();
            spend_action_points(state, 0, 2); // 5 → 3
        }

        advance_phase(&mut world); // combat — same player, AP should NOT reset

        let state = world.resource::<TurnState>().unwrap();
        assert_eq!(state.action_points[&0], 3);
    }

    #[test]
    fn action_points_reset_on_new_turn_same_player() {
        // Single-player game: AP should reset when turn wraps
        let mut world = setup_world(vec![0], vec!["act"], 5);

        advance_phase(&mut world); // turn 1
        {
            let state = world.resource_mut::<TurnState>().unwrap();
            spend_action_points(state, 0, 4); // 5 → 1
        }

        advance_phase(&mut world); // turn 2 — same player but new turn, AP resets

        let state = world.resource::<TurnState>().unwrap();
        assert_eq!(state.action_points[&0], 5);
    }

    #[test]
    fn no_crash_without_resources() {
        let mut world = World::new();
        // Should not panic even without TurnState/TurnConfig/Events
        advance_phase(&mut world);
    }

    #[test]
    fn empty_player_order_is_safe() {
        let mut world = World::new();
        world.insert_resource(Events::default());
        world.insert_resource(TurnState::new());
        world.insert_resource(TurnConfig {
            player_order: vec![],
            phases_per_turn: vec!["main".to_string()],
            action_points_per_turn: 1,
        });

        advance_phase(&mut world); // Should not panic
        let state = world.resource::<TurnState>().unwrap();
        assert_eq!(state.active_player, None);
    }

    #[test]
    fn empty_phases_is_safe() {
        let mut world = World::new();
        world.insert_resource(Events::default());
        world.insert_resource(TurnState::new());
        world.insert_resource(TurnConfig {
            player_order: vec![0],
            phases_per_turn: vec![],
            action_points_per_turn: 1,
        });

        advance_phase(&mut world); // Should not panic
        let state = world.resource::<TurnState>().unwrap();
        assert_eq!(state.active_player, None);
    }
}
