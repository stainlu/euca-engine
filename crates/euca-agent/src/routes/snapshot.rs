//! World snapshot & diff endpoints — structured handoffs between agent sessions.
//!
//! Snapshots capture a labeled summary of the world state including entity counts
//! by role, team summaries, game phase, and assertion results. Diffs compare two
//! snapshots to show what changed between them.

use axum::Json;
use axum::extract::{Query as AxumQuery, State};
use euca_ecs::{Entity, Query, World};
use serde::{Deserialize, Serialize};

use crate::state::SharedWorld;

/// A labeled world snapshot with computed summaries.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub label: String,
    pub tick: u64,
    pub entity_count: u32,
    pub summary: GameSummary,
    pub assertion_results: Vec<AssertionSnapshot>,
}

/// High-level game state summary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameSummary {
    pub teams: Vec<TeamSummary>,
    pub entity_counts_by_role: std::collections::HashMap<String, u32>,
    pub game_phase: String,
    pub match_time: f32,
    pub total_dead: u32,
}

/// Per-team summary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeamSummary {
    pub team: u8,
    pub entity_count: u32,
    pub heroes_alive: u32,
    pub towers_alive: u32,
    pub score: i32,
}

/// Snapshot of a single assertion result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssertionSnapshot {
    pub name: String,
    pub passed: bool,
    pub message: String,
}

/// In-memory storage for snapshots (stored as World resource).
#[derive(Clone, Default)]
pub struct SnapshotHistory {
    pub snapshots: Vec<WorldSnapshot>,
    pub max_history: usize,
}

impl SnapshotHistory {
    pub fn new(max_history: usize) -> Self {
        Self {
            snapshots: Vec::new(),
            max_history,
        }
    }
}

/// Compute a game summary from the current world state.
fn compute_summary(w: &World) -> GameSummary {
    let mut teams: std::collections::HashMap<u8, TeamSummary> = std::collections::HashMap::new();
    let mut role_counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut total_dead = 0u32;

    let query = Query::<Entity>::new(w);
    for entity in query.iter() {
        // Count by role
        if let Some(role) = w.get::<euca_gameplay::EntityRole>(entity) {
            let role_name = format!("{role:?}");
            *role_counts.entry(role_name).or_default() += 1;
        }

        // Track dead
        if w.get::<euca_gameplay::Dead>(entity).is_some() {
            total_dead += 1;
        }

        // Build team summaries
        if let Some(team) = w.get::<euca_gameplay::Team>(entity) {
            let summary = teams.entry(team.0).or_insert(TeamSummary {
                team: team.0,
                entity_count: 0,
                heroes_alive: 0,
                towers_alive: 0,
                score: 0,
            });
            summary.entity_count += 1;

            let is_dead = w.get::<euca_gameplay::Dead>(entity).is_some();
            if !is_dead {
                if let Some(role) = w.get::<euca_gameplay::EntityRole>(entity) {
                    match role {
                        euca_gameplay::EntityRole::Hero => summary.heroes_alive += 1,
                        euca_gameplay::EntityRole::Tower => summary.towers_alive += 1,
                        _ => {}
                    }
                }
            }
        }
    }

    // Get scores from GameState
    if let Some(gs) = w.resource::<euca_gameplay::GameState>() {
        for (entity_idx, score) in gs.scoreboard() {
            // Find which team this entity is on
            let entity_opt = super::find_entity(w, entity_idx);
            if let Some(entity) = entity_opt {
                if let Some(team) = w.get::<euca_gameplay::Team>(entity) {
                    if let Some(summary) = teams.get_mut(&team.0) {
                        summary.score = score;
                    }
                }
            }
        }
    }

    let game_phase = w
        .resource::<euca_gameplay::GameState>()
        .map(|gs| match &gs.phase {
            euca_gameplay::GamePhase::Lobby => "lobby".to_string(),
            euca_gameplay::GamePhase::Countdown { .. } => "countdown".to_string(),
            euca_gameplay::GamePhase::Playing => "playing".to_string(),
            euca_gameplay::GamePhase::PostMatch { .. } => "post_match".to_string(),
        })
        .unwrap_or_else(|| "none".to_string());

    let match_time = w
        .resource::<euca_gameplay::GameState>()
        .map(|gs| gs.elapsed)
        .unwrap_or(0.0);

    let mut team_list: Vec<TeamSummary> = teams.into_values().collect();
    team_list.sort_by_key(|t| t.team);

    GameSummary {
        teams: team_list,
        entity_counts_by_role: role_counts,
        game_phase,
        match_time,
        total_dead,
    }
}

/// Compute assertion snapshots from current world.
fn snapshot_assertions(w: &World) -> Vec<AssertionSnapshot> {
    let query = Query::<(Entity, &euca_gameplay::assertions::Assertion)>::new(w);
    query
        .iter()
        .filter_map(|(_, a)| {
            a.last_result.as_ref().map(|r| AssertionSnapshot {
                name: a.name.clone(),
                passed: r.passed,
                message: r.message.clone(),
            })
        })
        .collect()
}

/// Capture a snapshot from a world reference (used by probe module).
pub(super) fn capture_snapshot_from_world(w: &World, label: String) -> WorldSnapshot {
    capture_snapshot(w, label)
}

fn capture_snapshot(w: &World, label: String) -> WorldSnapshot {
    WorldSnapshot {
        label,
        tick: w
            .resource::<euca_gameplay::GameState>()
            .map(|gs| gs.elapsed as u64)
            .unwrap_or(0),
        entity_count: w.entity_count(),
        summary: compute_summary(w),
        assertion_results: snapshot_assertions(w),
    }
}

// ── HTTP Handlers ──

/// POST /snapshot — capture a labeled snapshot
pub async fn snapshot_create(
    State(world): State<SharedWorld>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let label = req
        .get("label")
        .and_then(|v| v.as_str())
        .unwrap_or("unnamed")
        .to_string();

    let snapshot = world.with(|w, _| {
        // Ensure SnapshotHistory resource exists
        if w.resource::<SnapshotHistory>().is_none() {
            w.insert_resource(SnapshotHistory::new(50));
        }

        let snap = capture_snapshot(w, label.clone());

        // Store in history
        if let Some(history) = w.resource_mut::<SnapshotHistory>() {
            history.snapshots.push(snap.clone());
            // Trim old snapshots
            if history.snapshots.len() > history.max_history {
                let excess = history.snapshots.len() - history.max_history;
                history.snapshots.drain(..excess);
            }
        }

        snap
    });

    Json(serde_json::json!({
        "ok": true,
        "snapshot": snapshot,
    }))
}

/// GET /snapshot/list — list all snapshots (labels only)
pub async fn snapshot_list(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let labels = world.with_world(|w| {
        w.resource::<SnapshotHistory>()
            .map(|h| {
                h.snapshots
                    .iter()
                    .map(|s| {
                        serde_json::json!({
                            "label": s.label,
                            "tick": s.tick,
                            "entity_count": s.entity_count,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    });

    Json(serde_json::json!({
        "ok": true,
        "snapshots": labels,
        "count": labels.len(),
    }))
}

/// GET /snapshot/latest — get the most recent snapshot with full summary
pub async fn snapshot_latest(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let result = world.with_world(|w| {
        w.resource::<SnapshotHistory>()
            .and_then(|h| h.snapshots.last().cloned())
    });

    match result {
        Some(snap) => Json(serde_json::json!({
            "ok": true,
            "snapshot": snap,
        })),
        None => Json(serde_json::json!({
            "ok": false,
            "error": "No snapshots captured yet. Use POST /snapshot first.",
        })),
    }
}

#[derive(Deserialize)]
pub struct DiffQuery {
    pub from: String,
    pub to: String,
}

/// GET /snapshot/diff?from=X&to=Y — compare two snapshots
pub async fn snapshot_diff(
    State(world): State<SharedWorld>,
    AxumQuery(params): AxumQuery<DiffQuery>,
) -> Json<serde_json::Value> {
    let result = world.with_world(|w| {
        let history = match w.resource::<SnapshotHistory>() {
            Some(h) => h,
            None => {
                return serde_json::json!({
                    "ok": false,
                    "error": "No snapshot history",
                });
            }
        };

        let from_snap = history.snapshots.iter().find(|s| s.label == params.from);
        let to_snap = history.snapshots.iter().find(|s| s.label == params.to);

        match (from_snap, to_snap) {
            (Some(from), Some(to)) => {
                // Compute diff
                let entity_delta = to.entity_count as i32 - from.entity_count as i32;

                // Role count changes
                let mut role_changes: std::collections::HashMap<String, (u32, u32)> =
                    std::collections::HashMap::new();
                for (role, count) in &from.summary.entity_counts_by_role {
                    role_changes.entry(role.clone()).or_insert((0, 0)).0 = *count;
                }
                for (role, count) in &to.summary.entity_counts_by_role {
                    role_changes.entry(role.clone()).or_insert((0, 0)).1 = *count;
                }

                let role_diffs: Vec<_> = role_changes
                    .iter()
                    .filter(|(_, (a, b))| a != b)
                    .map(|(role, (from_count, to_count))| {
                        serde_json::json!({
                            "role": role,
                            "from": from_count,
                            "to": to_count,
                            "delta": *to_count as i32 - *from_count as i32,
                        })
                    })
                    .collect();

                // Assertion changes
                let assertion_changes: Vec<_> = to
                    .assertion_results
                    .iter()
                    .filter_map(|to_a| {
                        let from_a = from
                            .assertion_results
                            .iter()
                            .find(|fa| fa.name == to_a.name);
                        match from_a {
                            Some(fa) if fa.passed != to_a.passed => Some(serde_json::json!({
                                "name": to_a.name,
                                "from_passed": fa.passed,
                                "to_passed": to_a.passed,
                                "message": to_a.message,
                            })),
                            None => Some(serde_json::json!({
                                "name": to_a.name,
                                "from_passed": null,
                                "to_passed": to_a.passed,
                                "message": to_a.message,
                            })),
                            _ => None,
                        }
                    })
                    .collect();

                serde_json::json!({
                    "ok": true,
                    "from": from.label,
                    "to": to.label,
                    "entity_delta": entity_delta,
                    "from_entities": from.entity_count,
                    "to_entities": to.entity_count,
                    "phase_changed": from.summary.game_phase != to.summary.game_phase,
                    "from_phase": from.summary.game_phase,
                    "to_phase": to.summary.game_phase,
                    "role_changes": role_diffs,
                    "assertion_changes": assertion_changes,
                })
            }
            (None, _) => serde_json::json!({
                "ok": false,
                "error": format!("Snapshot '{}' not found", params.from),
            }),
            (_, None) => serde_json::json!({
                "ok": false,
                "error": format!("Snapshot '{}' not found", params.to),
            }),
        }
    });

    Json(result)
}

/// GET /game/summary — get high-level game state summary (no snapshot required)
pub async fn game_summary(State(world): State<SharedWorld>) -> Json<serde_json::Value> {
    let summary = world.with_world(compute_summary);
    let assertions = world.with_world(snapshot_assertions);

    Json(serde_json::json!({
        "ok": true,
        "summary": summary,
        "assertions": assertions,
        "entity_count": summary.teams.iter().map(|t| t.entity_count).sum::<u32>(),
    }))
}
