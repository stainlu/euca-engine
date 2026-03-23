use clap::Subcommand;

use crate::{build_create_body, build_update_body, handle_response, parse_json_flag, post_empty};

#[derive(Subcommand)]
pub(crate) enum EntityCommands {
    /// List all entities
    List {
        /// Filter by component type (e.g. "MeshRenderer")
        #[arg(short, long)]
        filter: Option<String>,
    },
    /// Get a single entity by ID
    Get {
        /// Entity ID
        id: u32,
    },
    /// Create a new entity
    Create {
        /// Mesh: "cube", "sphere", "plane", "cylinder", "cone"
        #[arg(short, long)]
        mesh: Option<String>,
        /// Color: name ("red", "gold") or RGB ("0.5,0.2,0.8")
        #[arg(short, long)]
        color: Option<String>,
        /// Position as "x,y,z"
        #[arg(short, long)]
        position: Option<String>,
        /// Scale as "x,y,z"
        #[arg(long)]
        scale: Option<String>,
        /// Physics body type: Dynamic, Static, Kinematic
        #[arg(long)]
        physics: Option<String>,
        /// Collider: "aabb:hx,hy,hz" or "sphere:radius" or "capsule:radius,half_height"
        #[arg(long)]
        collider: Option<String>,
        /// Initial health (adds Health component)
        #[arg(long)]
        health: Option<f32>,
        /// Team ID (adds Team component)
        #[arg(long)]
        team: Option<u8>,
        /// Enable auto-combat (detect enemies, chase, attack)
        #[arg(long)]
        combat: bool,
        /// Combat damage per hit
        #[arg(long)]
        combat_damage: Option<f32>,
        /// Combat attack range
        #[arg(long)]
        combat_range: Option<f32>,
        /// Combat chase speed (0 for stationary)
        #[arg(long)]
        combat_speed: Option<f32>,
        /// Combat attack cooldown (seconds)
        #[arg(long)]
        combat_cooldown: Option<f32>,
        /// Combat style: "melee" (default) or "stationary" (towers)
        #[arg(long)]
        combat_style: Option<String>,
        /// AI patrol waypoints as "x,y,z:x,y,z:x,y,z"
        #[arg(long)]
        ai_patrol: Option<String>,
        /// Starting gold
        #[arg(long)]
        gold: Option<i32>,
        /// Gold bounty on death
        #[arg(long)]
        gold_bounty: Option<i32>,
        /// XP bounty on death
        #[arg(long)]
        xp_bounty: Option<u32>,
        /// Entity role: hero, minion, tower, structure
        #[arg(long)]
        role: Option<String>,
        /// Spawn point for team (marks as respawn location)
        #[arg(long)]
        spawn_point: Option<u8>,
        /// Mark as player-controlled hero
        #[arg(long)]
        player: bool,
        /// Full JSON body (overrides other flags)
        #[arg(long)]
        json: Option<String>,
        /// Preview without creating
        #[arg(long)]
        dry_run: bool,
    },
    /// Update an entity's components
    Update {
        /// Entity ID
        id: u32,
        /// Color: name ("red", "gold") or RGB ("0.5,0.2,0.8")
        #[arg(short, long)]
        color: Option<String>,
        /// Position as "x,y,z"
        #[arg(short, long)]
        position: Option<String>,
        /// Scale as "x,y,z"
        #[arg(long)]
        scale: Option<String>,
        /// Linear velocity as "x,y,z"
        #[arg(long)]
        velocity: Option<String>,
        /// Physics body type: Dynamic, Static, Kinematic
        #[arg(long)]
        physics: Option<String>,
        /// Collider: "aabb:hx,hy,hz" or "sphere:radius"
        #[arg(long)]
        collider: Option<String>,
        /// Full JSON patch (overrides other flags)
        #[arg(long)]
        json: Option<String>,
        /// Preview without updating
        #[arg(long)]
        dry_run: bool,
    },
    /// Delete an entity
    Delete {
        /// Entity ID (omit with --all to delete everything)
        id: Option<u32>,
        /// Delete all entities
        #[arg(long)]
        all: bool,
    },
    /// Apply damage to an entity
    Damage {
        /// Entity ID
        id: u32,
        /// Damage amount
        #[arg(long)]
        amount: f32,
    },
    /// Heal an entity
    Heal {
        /// Entity ID
        id: u32,
        /// Heal amount
        #[arg(long)]
        amount: f32,
    },
}

pub(crate) fn run_entity(
    command: EntityCommands,
    client: &reqwest::blocking::Client,
    server: &str,
) -> Result<(), String> {
    match command {
        EntityCommands::List { filter: _filter } => {
            let resp = client
                .post(format!("{server}/observe"))
                .header("Content-Type", "application/json")
                .body("{}")
                .send();
            handle_response(resp)
        }
        EntityCommands::Get { id } => {
            let resp = client.get(format!("{server}/entities/{id}")).send();
            handle_response(resp)
        }
        EntityCommands::Create {
            mesh,
            color,
            position,
            scale,
            physics,
            collider,
            health,
            team,
            combat,
            combat_damage,
            combat_range,
            combat_speed,
            combat_cooldown,
            combat_style,
            ai_patrol,
            gold,
            gold_bounty,
            xp_bounty,
            role,
            spawn_point,
            player,
            json,
            dry_run,
        } => {
            let body = if let Some(ref raw) = json {
                parse_json_flag(raw)
            } else {
                build_create_body(
                    &mesh,
                    &color,
                    &position,
                    &scale,
                    &physics,
                    &collider,
                    health,
                    team,
                    combat,
                    combat_damage,
                    combat_range,
                    combat_speed,
                    combat_cooldown,
                    &combat_style,
                    &ai_patrol,
                    gold,
                    gold_bounty,
                    xp_bounty,
                    &role,
                    spawn_point,
                    player,
                )
            };
            if dry_run {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&body).expect("JSON serialization failed")
                );
                println!("(dry-run: not created)");
                Ok(())
            } else {
                let resp = client.post(format!("{server}/spawn")).json(&body).send();
                handle_response(resp)
            }
        }
        EntityCommands::Update {
            id,
            color,
            position,
            scale,
            velocity,
            physics,
            collider,
            json,
            dry_run,
        } => {
            let body = if let Some(ref raw) = json {
                parse_json_flag(raw)
            } else {
                build_update_body(&color, &position, &scale, &velocity, &physics, &collider)
            };
            if dry_run {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&body).expect("JSON serialization failed")
                );
                println!("(dry-run: entity {id} not updated)");
                Ok(())
            } else {
                let resp = client
                    .post(format!("{server}/entities/{id}/components"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
        }
        EntityCommands::Delete { id, all } => {
            if all {
                post_empty(client, server, "/reset")
            } else if let Some(id) = id {
                let body = serde_json::json!({"entity_id": id, "entity_generation": 0});
                let resp = client.post(format!("{server}/despawn")).json(&body).send();
                handle_response(resp)
            } else {
                eprintln!("Specify an entity ID or use --all");
                std::process::exit(1);
            }
        }
        EntityCommands::Damage { id, amount } => {
            let resp = client
                .post(format!("{server}/entity/damage"))
                .json(&serde_json::json!({"entity_id": id, "amount": amount}))
                .send();
            handle_response(resp)
        }
        EntityCommands::Heal { id, amount } => {
            let resp = client
                .post(format!("{server}/entity/heal"))
                .json(&serde_json::json!({"entity_id": id, "amount": amount}))
                .send();
            handle_response(resp)
        }
    }
}
