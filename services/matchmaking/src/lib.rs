//! Core matchmaking types and engine for Euca engine games.
//!
//! This module provides skill-based matchmaking with configurable team sizes,
//! MMR tolerance that widens over time, and an accept/decline flow before
//! matches are confirmed.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A player in the matchmaking queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuedPlayer {
    pub player_id: String,
    pub display_name: String,
    /// Matchmaking rating (default 1000).
    pub mmr: u32,
    /// Unix timestamp when the player joined the queue.
    pub queued_at: u64,
    /// Game mode key, e.g. `"ranked"`, `"casual"`, `"1v1"`, `"5v5"`.
    pub game_mode: String,
}

/// A formed match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Match {
    pub match_id: String,
    pub game_mode: String,
    pub teams: Vec<Team>,
    pub created_at: u64,
    pub server_address: Option<String>,
}

/// One team within a match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Team {
    pub team_id: u32,
    pub players: Vec<QueuedPlayer>,
}

// ---------------------------------------------------------------------------
// WebSocket protocol messages
// ---------------------------------------------------------------------------

/// Messages sent from client to server.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    JoinQueue {
        player_id: String,
        display_name: String,
        mmr: u32,
        game_mode: String,
    },
    LeaveQueue {
        player_id: String,
    },
    AcceptMatch {
        player_id: String,
        match_id: String,
    },
    DeclineMatch {
        player_id: String,
        match_id: String,
    },
    Ping,
}

/// Messages sent from server to client.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    QueueStatus {
        position: u32,
        estimated_wait_secs: u32,
    },
    MatchFound {
        match_id: String,
        teams: Vec<Team>,
    },
    MatchConfirmed {
        match_id: String,
        server_address: String,
    },
    MatchCancelled {
        match_id: String,
        reason: String,
    },
    Error {
        message: String,
    },
    Pong,
}

// ---------------------------------------------------------------------------
// Matchmaker configuration
// ---------------------------------------------------------------------------

/// Per-game-mode configuration for the matchmaker.
#[derive(Debug, Clone)]
pub struct MatchConfig {
    /// Players per team.
    pub team_size: u32,
    /// Number of teams per match.
    pub team_count: u32,
    /// Maximum MMR difference allowed at queue-join time.
    pub max_mmr_gap: u32,
    /// How much the acceptable MMR gap widens per second of waiting.
    pub mmr_gap_growth_per_sec: u32,
    /// Seconds players have to accept a found match.
    pub accept_timeout_secs: u32,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            team_size: 5,
            team_count: 2,
            max_mmr_gap: 100,
            mmr_gap_growth_per_sec: 5,
            accept_timeout_secs: 30,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal state for pending matches
// ---------------------------------------------------------------------------

struct PendingMatch {
    match_data: Match,
    accepted: HashSet<String>,
    created_at: std::time::Instant,
    timeout_secs: u32,
}

// ---------------------------------------------------------------------------
// Match acceptance result
// ---------------------------------------------------------------------------

/// Result of a player accepting or declining a pending match.
#[derive(Debug, PartialEq, Eq)]
pub enum MatchAcceptResult {
    /// Acceptance recorded; still waiting for other players.
    Waiting { remaining: u32 },
    /// All players accepted — the match is confirmed.
    AllAccepted,
    /// The match has expired (timed out).
    Expired,
    /// No pending match with that id exists.
    NotFound,
}

/// Result of a player declining a pending match.
#[derive(Debug, PartialEq, Eq)]
pub enum MatchDeclineResult {
    /// The match has been cancelled because a player declined.
    Cancelled {
        /// Player IDs that were in this pending match (for notification).
        player_ids: Vec<String>,
    },
    /// No pending match with that id exists.
    NotFound,
}

// ---------------------------------------------------------------------------
// Matchmaker engine
// ---------------------------------------------------------------------------

/// The core matchmaking engine.
///
/// Thread-safety is handled externally (the server wraps this in
/// `Arc<Mutex<..>>`).
pub struct Matchmaker {
    /// Players waiting in queue, grouped by game mode.
    queues: HashMap<String, Vec<QueuedPlayer>>,
    /// Pending matches waiting for all players to accept.
    pending_matches: HashMap<String, PendingMatch>,
    /// Configuration per game mode.
    configs: HashMap<String, MatchConfig>,
}

impl Matchmaker {
    /// Create a new empty matchmaker with no game modes configured.
    pub fn new() -> Self {
        Self {
            queues: HashMap::new(),
            pending_matches: HashMap::new(),
            configs: HashMap::new(),
        }
    }

    /// Register (or overwrite) configuration for a game mode.
    pub fn configure_mode(&mut self, mode: impl Into<String>, config: MatchConfig) {
        self.configs.insert(mode.into(), config);
    }

    /// Add a player to the queue. Returns the player's 0-based position.
    pub fn add_player(&mut self, player: QueuedPlayer) -> u32 {
        let queue = self.queues.entry(player.game_mode.clone()).or_default();

        // Prevent duplicate entries.
        if queue.iter().any(|p| p.player_id == player.player_id) {
            return queue
                .iter()
                .position(|p| p.player_id == player.player_id)
                .unwrap_or(0) as u32;
        }

        queue.push(player);
        (queue.len() - 1) as u32
    }

    /// Remove a player from the queue. Returns `true` if the player was found.
    pub fn remove_player(&mut self, player_id: &str) -> bool {
        let mut found = false;
        for queue in self.queues.values_mut() {
            let before = queue.len();
            queue.retain(|p| p.player_id != player_id);
            if queue.len() < before {
                found = true;
            }
        }
        found
    }

    /// Return the number of players currently queued for a game mode.
    pub fn queue_size(&self, mode: &str) -> usize {
        self.queues.get(mode).map_or(0, Vec::len)
    }

    /// Return the total number of queued players across all modes.
    pub fn total_queued(&self) -> usize {
        self.queues.values().map(Vec::len).sum()
    }

    /// Return the number of pending (unconfirmed) matches.
    pub fn pending_match_count(&self) -> usize {
        self.pending_matches.len()
    }

    /// Attempt to form a match for the given game mode.
    ///
    /// Players are sorted by MMR and grouped into a contiguous block whose
    /// MMR spread is within the dynamically-widened tolerance. If enough
    /// players are available, a [`Match`] is created, the players are removed
    /// from the queue, and the match is registered as pending (awaiting
    /// acceptance).
    pub fn try_match(&mut self, mode: &str) -> Option<Match> {
        let config = self.configs.get(mode)?;
        let total_players_needed = (config.team_size * config.team_count) as usize;

        let queue = self.queues.get_mut(mode)?;
        if queue.len() < total_players_needed {
            return None;
        }

        // Sort by MMR so nearby players are adjacent.
        queue.sort_by_key(|p| p.mmr);

        let now_unix = unix_now();

        // Sliding window: find the first contiguous group that fits within
        // the (possibly widened) MMR tolerance.
        let mut best_window: Option<usize> = None;

        for start in 0..=queue.len() - total_players_needed {
            let window = &queue[start..start + total_players_needed];
            let min_mmr = window.first().unwrap().mmr;
            let max_mmr = window.last().unwrap().mmr;

            // Effective tolerance: base + growth * longest_wait_secs.
            let longest_wait = window
                .iter()
                .map(|p| now_unix.saturating_sub(p.queued_at))
                .max()
                .unwrap_or(0);

            let effective_gap =
                config.max_mmr_gap + config.mmr_gap_growth_per_sec * (longest_wait as u32);

            if max_mmr - min_mmr <= effective_gap {
                best_window = Some(start);
                break;
            }
        }

        let start = best_window?;

        // Extract the matched players from the queue.
        let matched: Vec<QueuedPlayer> = queue.drain(start..start + total_players_needed).collect();

        // Distribute into teams round-robin (already sorted by MMR, so this
        // produces reasonably balanced teams).
        let mut teams: Vec<Vec<QueuedPlayer>> =
            (0..config.team_count).map(|_| Vec::new()).collect();
        for (i, player) in matched.into_iter().enumerate() {
            teams[i % config.team_count as usize].push(player);
        }

        let match_id = uuid::Uuid::new_v4().to_string();
        let formed = Match {
            match_id: match_id.clone(),
            game_mode: mode.to_string(),
            teams: teams
                .into_iter()
                .enumerate()
                .map(|(idx, players)| Team {
                    team_id: idx as u32,
                    players,
                })
                .collect(),
            created_at: now_unix,
            server_address: None,
        };

        // Register as pending.
        self.pending_matches.insert(
            match_id,
            PendingMatch {
                match_data: formed.clone(),
                accepted: HashSet::new(),
                created_at: std::time::Instant::now(),
                timeout_secs: config.accept_timeout_secs,
            },
        );

        Some(formed)
    }

    /// Record that a player accepts a pending match.
    pub fn accept_match(&mut self, player_id: &str, match_id: &str) -> MatchAcceptResult {
        let Some(pending) = self.pending_matches.get_mut(match_id) else {
            return MatchAcceptResult::NotFound;
        };

        if pending.created_at.elapsed().as_secs() >= u64::from(pending.timeout_secs) {
            self.pending_matches.remove(match_id);
            return MatchAcceptResult::Expired;
        }

        pending.accepted.insert(player_id.to_string());

        let total_players: usize = pending
            .match_data
            .teams
            .iter()
            .map(|t| t.players.len())
            .sum();

        if pending.accepted.len() >= total_players {
            self.pending_matches.remove(match_id);
            MatchAcceptResult::AllAccepted
        } else {
            let remaining = total_players - pending.accepted.len();
            MatchAcceptResult::Waiting {
                remaining: remaining as u32,
            }
        }
    }

    /// Record that a player declines a pending match.
    pub fn decline_match(&mut self, player_id: &str, match_id: &str) -> MatchDeclineResult {
        let Some(pending) = self.pending_matches.remove(match_id) else {
            return MatchDeclineResult::NotFound;
        };

        let player_ids: Vec<String> = pending
            .match_data
            .teams
            .iter()
            .flat_map(|t| t.players.iter())
            .filter(|p| p.player_id != player_id)
            .map(|p| p.player_id.clone())
            .collect();

        MatchDeclineResult::Cancelled { player_ids }
    }

    /// Periodic tick: attempt to form matches in every mode and expire timed-
    /// out pending matches.
    ///
    /// Returns `(formed_matches, expired_match_ids)`.
    pub fn tick(&mut self) -> (Vec<Match>, Vec<String>) {
        let modes: Vec<String> = self.configs.keys().cloned().collect();
        let mut formed = Vec::new();

        for mode in &modes {
            while let Some(m) = self.try_match(mode) {
                formed.push(m);
            }
        }

        // Expire pending matches.
        let expired: Vec<String> = self
            .pending_matches
            .iter()
            .filter(|(_, pm)| pm.created_at.elapsed().as_secs() >= u64::from(pm.timeout_secs))
            .map(|(id, _)| id.clone())
            .collect();

        for id in &expired {
            self.pending_matches.remove(id);
        }

        (formed, expired)
    }
}

impl Default for Matchmaker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Current Unix timestamp in seconds.
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: make a player with sensible defaults.
    fn player(id: &str, mmr: u32, mode: &str) -> QueuedPlayer {
        QueuedPlayer {
            player_id: id.to_string(),
            display_name: id.to_string(),
            mmr,
            queued_at: unix_now(),
            game_mode: mode.to_string(),
        }
    }

    /// Helper: a simple 1v1 config.
    fn config_1v1() -> MatchConfig {
        MatchConfig {
            team_size: 1,
            team_count: 2,
            max_mmr_gap: 100,
            mmr_gap_growth_per_sec: 5,
            accept_timeout_secs: 30,
        }
    }

    #[test]
    fn test_matchmaker_add_remove() {
        let mut mm = Matchmaker::new();
        mm.configure_mode("1v1", config_1v1());

        let pos = mm.add_player(player("alice", 1000, "1v1"));
        assert_eq!(pos, 0);
        assert_eq!(mm.queue_size("1v1"), 1);

        let removed = mm.remove_player("alice");
        assert!(removed);
        assert_eq!(mm.queue_size("1v1"), 0);

        // Removing again should return false.
        assert!(!mm.remove_player("alice"));
    }

    #[test]
    fn test_matchmaker_basic_match() {
        let mut mm = Matchmaker::new();
        mm.configure_mode("1v1", config_1v1());

        mm.add_player(player("alice", 1000, "1v1"));
        mm.add_player(player("bob", 1050, "1v1"));

        let m = mm.try_match("1v1");
        assert!(m.is_some());

        let m = m.unwrap();
        assert_eq!(m.teams.len(), 2);
        assert_eq!(m.teams[0].players.len(), 1);
        assert_eq!(m.teams[1].players.len(), 1);
        assert_eq!(m.game_mode, "1v1");

        // Queue should be empty now.
        assert_eq!(mm.queue_size("1v1"), 0);
    }

    #[test]
    fn test_matchmaker_mmr_gap() {
        let mut mm = Matchmaker::new();
        mm.configure_mode("1v1", config_1v1());

        // Players are 500 MMR apart — far beyond the 100 base gap.
        // queued_at is *now*, so no gap growth has occurred.
        mm.add_player(player("alice", 500, "1v1"));
        mm.add_player(player("bob", 1000, "1v1"));

        let m = mm.try_match("1v1");
        assert!(m.is_none(), "Players too far apart should not match");
    }

    #[test]
    fn test_matchmaker_accept_flow() {
        let mut mm = Matchmaker::new();
        mm.configure_mode("1v1", config_1v1());

        mm.add_player(player("alice", 1000, "1v1"));
        mm.add_player(player("bob", 1050, "1v1"));

        let m = mm.try_match("1v1").expect("should form match");
        let match_id = m.match_id.clone();

        // First accept: waiting for the other player.
        let result = mm.accept_match("alice", &match_id);
        assert_eq!(result, MatchAcceptResult::Waiting { remaining: 1 });

        // Second accept: all accepted.
        let result = mm.accept_match("bob", &match_id);
        assert_eq!(result, MatchAcceptResult::AllAccepted);
    }

    #[test]
    fn test_matchmaker_decline() {
        let mut mm = Matchmaker::new();
        mm.configure_mode("1v1", config_1v1());

        mm.add_player(player("alice", 1000, "1v1"));
        mm.add_player(player("bob", 1050, "1v1"));

        let m = mm.try_match("1v1").expect("should form match");
        let match_id = m.match_id.clone();

        // Alice declines.
        let result = mm.decline_match("alice", &match_id);
        match result {
            MatchDeclineResult::Cancelled { player_ids } => {
                assert_eq!(player_ids, vec!["bob".to_string()]);
            }
            MatchDeclineResult::NotFound => panic!("expected Cancelled"),
        }

        // The pending match should be gone.
        assert_eq!(mm.pending_match_count(), 0);
    }

    #[test]
    fn test_matchmaker_timeout() {
        let mut mm = Matchmaker::new();
        // Use a 0-second timeout so it expires immediately.
        let mut cfg = config_1v1();
        cfg.accept_timeout_secs = 0;
        mm.configure_mode("1v1", cfg);

        mm.add_player(player("alice", 1000, "1v1"));
        mm.add_player(player("bob", 1050, "1v1"));

        let m = mm.try_match("1v1").expect("should form match");
        let match_id = m.match_id.clone();

        // Even immediate accept should see Expired because timeout is 0.
        // We need a tiny delay for `Instant::elapsed()` to exceed 0.
        std::thread::sleep(std::time::Duration::from_millis(10));

        let result = mm.accept_match("alice", &match_id);
        assert_eq!(result, MatchAcceptResult::Expired);
    }

    #[test]
    fn test_message_serialization() {
        // ClientMessage round-trip.
        let msg = ClientMessage::JoinQueue {
            player_id: "p1".into(),
            display_name: "Player One".into(),
            mmr: 1200,
            game_mode: "ranked".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ClientMessage = serde_json::from_str(&json).unwrap();
        match parsed {
            ClientMessage::JoinQueue {
                player_id,
                mmr,
                game_mode,
                ..
            } => {
                assert_eq!(player_id, "p1");
                assert_eq!(mmr, 1200);
                assert_eq!(game_mode, "ranked");
            }
            _ => panic!("wrong variant"),
        }

        // ServerMessage round-trip.
        let msg = ServerMessage::MatchFound {
            match_id: "m1".into(),
            teams: vec![Team {
                team_id: 0,
                players: vec![QueuedPlayer {
                    player_id: "p1".into(),
                    display_name: "Player One".into(),
                    mmr: 1200,
                    queued_at: 0,
                    game_mode: "ranked".into(),
                }],
            }],
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ServerMessage = serde_json::from_str(&json).unwrap();
        match parsed {
            ServerMessage::MatchFound { match_id, teams } => {
                assert_eq!(match_id, "m1");
                assert_eq!(teams.len(), 1);
                assert_eq!(teams[0].players.len(), 1);
            }
            _ => panic!("wrong variant"),
        }

        // Ping/Pong round-trip.
        let ping_json = serde_json::to_string(&ClientMessage::Ping).unwrap();
        let _: ClientMessage = serde_json::from_str(&ping_json).unwrap();

        let pong_json = serde_json::to_string(&ServerMessage::Pong).unwrap();
        let _: ServerMessage = serde_json::from_str(&pong_json).unwrap();
    }

    #[test]
    fn test_tick_forms_matches_and_expires() {
        let mut mm = Matchmaker::new();
        // Use a generous timeout so matches survive the first tick.
        let mut cfg = config_1v1();
        cfg.accept_timeout_secs = 30;
        mm.configure_mode("1v1", cfg);

        mm.add_player(player("alice", 1000, "1v1"));
        mm.add_player(player("bob", 1050, "1v1"));
        mm.add_player(player("carol", 1020, "1v1"));
        mm.add_player(player("dave", 1030, "1v1"));

        // First tick should form 2 matches.
        let (formed, expired) = mm.tick();
        assert_eq!(formed.len(), 2);
        assert_eq!(expired.len(), 0);
        assert_eq!(mm.queue_size("1v1"), 0);
        assert_eq!(mm.pending_match_count(), 2);
    }

    #[test]
    fn test_tick_expires_timed_out_matches() {
        let mut mm = Matchmaker::new();
        // 0-second timeout: matches expire as soon as they are checked.
        let mut cfg = config_1v1();
        cfg.accept_timeout_secs = 0;
        mm.configure_mode("1v1", cfg);

        mm.add_player(player("alice", 1000, "1v1"));
        mm.add_player(player("bob", 1050, "1v1"));

        // Form the match.
        let m = mm.try_match("1v1").expect("should form match");
        assert_eq!(mm.pending_match_count(), 1);

        // Next tick should expire it (timeout_secs == 0, elapsed >= 0).
        let (_, expired) = mm.tick();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0], m.match_id);
        assert_eq!(mm.pending_match_count(), 0);
    }

    #[test]
    fn test_duplicate_add() {
        let mut mm = Matchmaker::new();
        mm.configure_mode("1v1", config_1v1());

        mm.add_player(player("alice", 1000, "1v1"));
        mm.add_player(player("alice", 1000, "1v1"));

        assert_eq!(mm.queue_size("1v1"), 1);
    }
}
