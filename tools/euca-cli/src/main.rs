use clap::{Parser, Subcommand};
use serde_json::Value;

#[derive(Parser)]
#[command(
    name = "euca",
    about = "Euca Engine CLI — control the engine from the terminal",
    after_help = "Examples:\n  euca entity create --mesh cube --position 0,2,0 --health 100 --team 1 --color red\n  euca rule create --when death --filter team:2 --do-action \"score source +1\"\n  euca sim play\n  euca screenshot"
)]
struct Cli {
    /// Server URL
    #[arg(short, long, default_value = "http://localhost:3917")]
    server: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Entity operations: create, get, list, update, delete
    Entity {
        #[command(subcommand)]
        command: EntityCommands,
    },

    /// Simulation control: play, pause, step, reset
    Sim {
        #[command(subcommand)]
        command: SimCommands,
    },

    /// Scene management: save, load
    Scene {
        #[command(subcommand)]
        command: SceneCommands,
    },

    /// Camera control: get, set
    Camera {
        #[command(subcommand)]
        command: CameraCommands,
    },

    /// Game match: create, state, scoreboard
    Game {
        #[command(subcommand)]
        command: GameCommands,
    },

    /// Trigger zones: create area-based events
    Trigger {
        #[command(subcommand)]
        command: TriggerCommands,
    },

    /// Projectiles: spawn moving damaging entities
    Projectile {
        #[command(subcommand)]
        command: ProjectileCommands,
    },

    /// AI behavior: set entity AI goals
    Ai {
        #[command(subcommand)]
        command: AiCommands,
    },

    /// Game rules: when X happens, do Y
    Rule {
        #[command(subcommand)]
        command: RuleCommands,
    },

    /// Entity templates: define once, spawn many
    Template {
        #[command(subcommand)]
        command: TemplateCommands,
    },

    /// Audio: play, stop, list sounds
    Audio {
        #[command(subcommand)]
        command: AudioCommands,
    },

    /// Authentication via nit identity
    Auth {
        #[command(subcommand)]
        command: AuthCommands,
    },

    /// Engine status
    Status,

    /// Capture a screenshot of the 3D viewport
    Screenshot {
        /// Output file path (default: temp file)
        #[arg(short, long)]
        output: Option<String>,
    },

    /// HUD elements: text, bars, rectangles on screen
    Ui {
        #[command(subcommand)]
        command: UiCommands,
    },

    /// Inspect component schemas
    Schema {
        /// Component name (e.g. "Collider", "PhysicsBody"). Omit to list all.
        component: Option<String>,
    },

    // ── Hidden aliases for backward compatibility ──
    #[command(hide = true)]
    Observe {
        #[arg(short, long)]
        entity: Option<u32>,
    },
    #[command(hide = true)]
    Spawn {
        #[arg(short, long)]
        position: Option<String>,
        #[arg(long)]
        scale: Option<String>,
        #[arg(long)]
        physics: Option<String>,
        #[arg(long)]
        collider: Option<String>,
    },
    #[command(hide = true)]
    Modify {
        id: u32,
        #[arg(short, long)]
        transform: Option<String>,
        #[arg(long)]
        velocity: Option<String>,
        #[arg(long)]
        physics: Option<String>,
        #[arg(long)]
        collider: Option<String>,
        #[arg(long)]
        json: Option<String>,
    },
    #[command(hide = true)]
    Despawn {
        #[arg(default_value = "0")]
        id: u32,
        #[arg(long)]
        all: bool,
    },
    #[command(hide = true)]
    Play,
    #[command(hide = true)]
    Pause,
    #[command(hide = true)]
    Step {
        #[arg(short, long, default_value = "1")]
        ticks: u64,
    },
    #[command(hide = true)]
    Reset,
}

#[derive(Subcommand)]
enum EntityCommands {
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

#[derive(Subcommand)]
enum SimCommands {
    /// Start simulation
    Play,
    /// Pause simulation
    Pause,
    /// Advance simulation by N ticks
    Step {
        /// Number of ticks
        #[arg(short, long, default_value = "1")]
        ticks: u64,
    },
    /// Reset to initial scene
    Reset,
}

#[derive(Subcommand)]
enum SceneCommands {
    /// Save current scene to file
    Save {
        /// Output file path
        path: String,
    },
    /// Load scene from file
    Load {
        /// Input file path
        path: String,
    },
}

#[derive(Subcommand)]
enum CameraCommands {
    /// Get current camera state
    Get,
    /// Set camera position and look-at target
    Set {
        /// Camera position as "x,y,z"
        #[arg(long)]
        eye: Option<String>,
        /// Look-at target as "x,y,z"
        #[arg(long)]
        target: Option<String>,
        /// Field of view in degrees (perspective mode)
        #[arg(long)]
        fov: Option<f32>,
    },
    /// Switch to a preset view: top, front, back, right, left, perspective
    View {
        /// View name
        name: String,
    },
    /// Focus camera on a specific entity
    Focus {
        /// Entity ID to focus on
        id: u32,
    },
}

#[derive(Subcommand)]
enum AudioCommands {
    /// Play a sound file
    Play {
        /// Path to audio file (WAV, MP3, OGG, FLAC)
        path: String,
        /// Position as "x,y,z" (makes it spatial)
        #[arg(long)]
        position: Option<String>,
        /// Volume (0.0-1.0)
        #[arg(long, default_value = "1.0")]
        volume: f32,
        /// Loop the sound
        #[arg(long, name = "loop")]
        looping: bool,
        /// Max audible distance (spatial only)
        #[arg(long, default_value = "50")]
        max_distance: f32,
    },
    /// Stop an audio source
    Stop {
        /// Entity ID of the audio source
        entity_id: u32,
    },
    /// List active audio sources
    List,
}

#[derive(Subcommand)]
enum AuthCommands {
    /// Login with nit Ed25519 identity
    Login,
    /// Check current authentication status
    Status,
}

#[derive(Subcommand)]
enum GameCommands {
    /// Create a new match
    Create {
        /// Game mode (e.g. "deathmatch")
        #[arg(long, default_value = "deathmatch")]
        mode: String,
        /// Score required to win
        #[arg(long, default_value = "10")]
        score_limit: i32,
    },
    /// Get current match state and scores
    State,
}

#[derive(Subcommand)]
enum TriggerCommands {
    /// Create a trigger zone
    Create {
        /// Position as "x,y,z"
        #[arg(long)]
        position: String,
        /// Zone half-extents as "x,y,z"
        #[arg(long, default_value = "1,1,1")]
        zone: String,
        /// Action: "damage:N" or "heal:N"
        #[arg(long, default_value = "damage:10")]
        action: String,
    },
}

#[derive(Subcommand)]
enum ProjectileCommands {
    /// Spawn a projectile
    Spawn {
        /// Origin position as "x,y,z"
        #[arg(long)]
        from: String,
        /// Direction as "x,y,z"
        #[arg(long)]
        direction: String,
        /// Speed (units/sec)
        #[arg(long, default_value = "20")]
        speed: f32,
        /// Damage on hit
        #[arg(long, default_value = "25")]
        damage: f32,
    },
}

#[derive(Subcommand)]
enum AiCommands {
    /// Set AI behavior on an entity
    Set {
        /// Entity ID
        id: u32,
        /// Behavior: idle, patrol, chase, flee
        #[arg(long)]
        behavior: String,
        /// Target entity ID (for chase/flee)
        #[arg(long)]
        target: Option<u32>,
        /// Movement speed
        #[arg(long, default_value = "3")]
        speed: f32,
    },
}

#[derive(Subcommand)]
enum RuleCommands {
    /// Create a game rule: when condition fires, execute actions
    Create {
        /// Condition: "death", "timer:N", "health-below:N"
        #[arg(long)]
        when: String,
        /// Filter: "any", "entity:N", "team:N"
        #[arg(long, default_value = "any")]
        filter: String,
        /// Actions (can repeat): "spawn cube 0,5,0", "score source +1", "damage this 10"
        #[arg(long)]
        do_action: Vec<String>,
    },
    /// List all rules
    List,
}

#[derive(Subcommand)]
enum TemplateCommands {
    /// Define a named entity template
    Create {
        /// Template name
        name: String,
        /// Mesh
        #[arg(long)]
        mesh: Option<String>,
        /// Color
        #[arg(long)]
        color: Option<String>,
        /// Health
        #[arg(long)]
        health: Option<f32>,
        /// Team
        #[arg(long)]
        team: Option<u8>,
        /// Physics body type
        #[arg(long)]
        physics: Option<String>,
        /// Collider
        #[arg(long)]
        collider: Option<String>,
        /// Enable auto-combat
        #[arg(long)]
        combat: bool,
    },
    /// Spawn an entity from a template
    Spawn {
        /// Template name
        name: String,
        /// Position
        #[arg(long)]
        position: Option<String>,
    },
    /// List all templates
    List,
}

#[derive(Subcommand)]
enum UiCommands {
    /// Add text to HUD
    Text {
        /// Text content
        text: String,
        /// X position (0.0-1.0, left to right)
        #[arg(long, default_value = "0.5")]
        x: f32,
        /// Y position (0.0-1.0, top to bottom)
        #[arg(long, default_value = "0.05")]
        y: f32,
        /// Font size in pixels
        #[arg(long, default_value = "20")]
        size: f32,
        /// Color name (red, green, blue, white, yellow, etc.)
        #[arg(long, default_value = "white")]
        color: String,
    },
    /// Add a bar (health bar, progress bar) to HUD
    Bar {
        /// X position (0.0-1.0)
        #[arg(long, default_value = "0.02")]
        x: f32,
        /// Y position (0.0-1.0)
        #[arg(long, default_value = "0.95")]
        y: f32,
        /// Width (0.0-1.0)
        #[arg(long, default_value = "0.2")]
        width: f32,
        /// Height (0.0-1.0)
        #[arg(long, default_value = "0.03")]
        height: f32,
        /// Fill amount (0.0-1.0)
        #[arg(long)]
        fill: f32,
        /// Bar color
        #[arg(long, default_value = "red")]
        color: String,
    },
    /// Remove all HUD elements
    Clear,
    /// List current HUD elements
    List,
}

// ── Helpers ──

fn parse_vec3(s: &str) -> Option<[f32; 3]> {
    let parts: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
    if parts.len() == 3 {
        Some([parts[0], parts[1], parts[2]])
    } else {
        eprintln!("Warning: expected 'x,y,z' format, got '{s}'");
        None
    }
}

fn parse_collider(s: &str) -> Option<Value> {
    if let Some(rest) = s.strip_prefix("sphere:") {
        let radius: f32 = rest.trim().parse().ok()?;
        Some(serde_json::json!({"shape": "Sphere", "radius": radius}))
    } else if let Some(rest) = s.strip_prefix("aabb:") {
        let v = parse_vec3(rest)?;
        Some(serde_json::json!({"shape": "Aabb", "hx": v[0], "hy": v[1], "hz": v[2]}))
    } else if let Some(rest) = s.strip_prefix("capsule:") {
        let parts: Vec<f32> = rest
            .split(',')
            .filter_map(|p| p.trim().parse().ok())
            .collect();
        if parts.len() == 2 {
            Some(
                serde_json::json!({"shape": "Capsule", "radius": parts[0], "half_height": parts[1]}),
            )
        } else {
            eprintln!("Warning: capsule format is 'capsule:radius,half_height'");
            None
        }
    } else {
        eprintln!(
            "Warning: collider format is 'aabb:hx,hy,hz' or 'sphere:radius' or 'capsule:radius,half_height'"
        );
        None
    }
}

/// Build a spawn/create JSON body from friendly flags.
#[allow(clippy::too_many_arguments)]
fn build_create_body(
    mesh: &Option<String>,
    color: &Option<String>,
    position: &Option<String>,
    scale: &Option<String>,
    physics: &Option<String>,
    collider: &Option<String>,
    health: Option<f32>,
    team: Option<u8>,
    combat: bool,
) -> Value {
    let mut body = serde_json::json!({});
    if let Some(m) = mesh {
        body["mesh"] = serde_json::json!(m);
    }
    if let Some(c) = color {
        body["color"] = serde_json::json!(c);
    }
    if let Some(s) = position
        && let Some(v) = parse_vec3(s)
    {
        body["position"] = serde_json::json!(v);
    }
    if let Some(s) = scale
        && let Some(v) = parse_vec3(s)
    {
        body["scale"] = serde_json::json!(v);
    }
    if let Some(pb) = physics {
        body["physics_body"] = serde_json::json!(pb);
    }
    if let Some(c) = collider
        && let Some(v) = parse_collider(c)
    {
        body["collider"] = v;
    }
    if let Some(h) = health {
        body["health"] = serde_json::json!(h);
    }
    if let Some(t) = team {
        body["team"] = serde_json::json!(t);
    }
    if combat {
        body["combat"] = serde_json::json!(true);
    }
    body
}

/// Build an update/patch JSON body from friendly flags.
fn build_update_body(
    color: &Option<String>,
    position: &Option<String>,
    scale: &Option<String>,
    velocity: &Option<String>,
    physics: &Option<String>,
    collider: &Option<String>,
) -> Value {
    let mut body = serde_json::json!({});
    if let Some(c) = color {
        body["color"] = serde_json::json!(c);
    }
    if position.is_some() || scale.is_some() {
        let mut transform = serde_json::json!({});
        if let Some(p) = position
            && let Some(pos) = parse_vec3(p)
        {
            transform["position"] = serde_json::json!(pos);
        }
        if let Some(s) = scale
            && let Some(scl) = parse_vec3(s)
        {
            transform["scale"] = serde_json::json!(scl);
        }
        body["transform"] = transform;
    }
    if let Some(v) = velocity
        && let Some(vel) = parse_vec3(v)
    {
        body["velocity"] = serde_json::json!({"linear": vel, "angular": [0, 0, 0]});
    }
    if let Some(pb) = physics {
        body["physics_body"] = serde_json::json!(pb);
    }
    if let Some(c) = collider
        && let Some(v) = parse_collider(c)
    {
        body["collider"] = v;
    }
    body
}

/// Parse --json flag or exit.
fn parse_json_flag(raw: &str) -> Value {
    match serde_json::from_str::<Value>(raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Invalid JSON: {e}");
            std::process::exit(1);
        }
    }
}

// ── Execution ──

fn main() {
    let cli = Cli::parse();
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .build()
        .expect("Failed to build HTTP client");
    let server = &cli.server;

    let result = match cli.command {
        // ── Entity CRUD ──
        Commands::Entity { command } => match command {
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
                json,
                dry_run,
            } => {
                let body = if let Some(ref raw) = json {
                    parse_json_flag(raw)
                } else {
                    build_create_body(
                        &mesh, &color, &position, &scale, &physics, &collider, health, team, combat,
                    )
                };
                if dry_run {
                    println!("{}", serde_json::to_string_pretty(&body).unwrap());
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
                    println!("{}", serde_json::to_string_pretty(&body).unwrap());
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
                    post_empty(&client, server, "/reset")
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
        },

        // ── Simulation ──
        Commands::Sim { command } => match command {
            SimCommands::Play => post_empty(&client, server, "/play"),
            SimCommands::Pause => post_empty(&client, server, "/pause"),
            SimCommands::Step { ticks } => {
                let resp = client
                    .post(format!("{server}/step"))
                    .json(&serde_json::json!({"ticks": ticks}))
                    .send();
                handle_response(resp)
            }
            SimCommands::Reset => post_empty(&client, server, "/reset"),
        },

        // ── Scene ──
        Commands::Scene { command } => match command {
            SceneCommands::Save { path } => {
                let resp = client
                    .post(format!("{server}/scene/save"))
                    .json(&serde_json::json!({"path": path}))
                    .send();
                handle_response(resp)
            }
            SceneCommands::Load { path } => {
                let resp = client
                    .post(format!("{server}/scene/load"))
                    .json(&serde_json::json!({"path": path}))
                    .send();
                handle_response(resp)
            }
        },

        // ── Camera ──
        Commands::Camera { command } => match command {
            CameraCommands::Get => {
                let resp = client.get(format!("{server}/camera")).send();
                handle_response(resp)
            }
            CameraCommands::Set { eye, target, fov } => {
                let mut body = serde_json::json!({});
                if let Some(e) = eye
                    && let Some(v) = parse_vec3(&e)
                {
                    body["eye"] = serde_json::json!(v);
                }
                if let Some(t) = target
                    && let Some(v) = parse_vec3(&t)
                {
                    body["target"] = serde_json::json!(v);
                }
                if let Some(f) = fov {
                    body["fov"] = serde_json::json!(f);
                }
                let resp = client.post(format!("{server}/camera")).json(&body).send();
                handle_response(resp)
            }
            CameraCommands::View { name } => {
                let resp = client
                    .post(format!("{server}/camera/view"))
                    .json(&serde_json::json!({"view": name}))
                    .send();
                handle_response(resp)
            }
            CameraCommands::Focus { id } => {
                let resp = client
                    .post(format!("{server}/camera/focus"))
                    .json(&serde_json::json!({"entity_id": id}))
                    .send();
                handle_response(resp)
            }
        },

        // ── Auth ──
        // ── Gameplay ──
        Commands::Game { command } => match command {
            GameCommands::Create { mode, score_limit } => {
                let resp = client
                    .post(format!("{server}/game/create"))
                    .json(&serde_json::json!({"mode": mode, "score_limit": score_limit}))
                    .send();
                handle_response(resp)
            }
            GameCommands::State => {
                let resp = client.get(format!("{server}/game/state")).send();
                handle_response(resp)
            }
        },
        Commands::Trigger { command } => match command {
            TriggerCommands::Create {
                position,
                zone,
                action,
            } => {
                let pos = parse_vec3(&position).unwrap_or([0.0, 0.0, 0.0]);
                let z = parse_vec3(&zone).unwrap_or([1.0, 1.0, 1.0]);
                let resp = client
                    .post(format!("{server}/trigger/create"))
                    .json(&serde_json::json!({
                        "position": pos,
                        "zone": z,
                        "action": action,
                    }))
                    .send();
                handle_response(resp)
            }
        },
        Commands::Projectile { command } => match command {
            ProjectileCommands::Spawn {
                from,
                direction,
                speed,
                damage,
            } => {
                let f = parse_vec3(&from).unwrap_or([0.0, 0.0, 0.0]);
                let d = parse_vec3(&direction).unwrap_or([1.0, 0.0, 0.0]);
                let resp = client
                    .post(format!("{server}/projectile/spawn"))
                    .json(&serde_json::json!({
                        "from": f,
                        "direction": d,
                        "speed": speed,
                        "damage": damage,
                    }))
                    .send();
                handle_response(resp)
            }
        },
        Commands::Ai { command } => match command {
            AiCommands::Set {
                id,
                behavior,
                target,
                speed,
            } => {
                let mut body = serde_json::json!({
                    "entity_id": id,
                    "behavior": behavior,
                    "speed": speed,
                });
                if let Some(t) = target {
                    body["target"] = serde_json::json!(t);
                }
                let resp = client.post(format!("{server}/ai/set")).json(&body).send();
                handle_response(resp)
            }
        },

        // ── Templates ──
        Commands::Template { command } => match command {
            TemplateCommands::Create {
                name,
                mesh,
                color,
                health,
                team,
                physics,
                collider,
                combat,
            } => {
                let mut body = serde_json::json!({"name": name});
                if combat {
                    body["combat"] = serde_json::json!(true);
                }
                if let Some(m) = mesh {
                    body["mesh"] = serde_json::json!(m);
                }
                if let Some(c) = color {
                    body["color"] = serde_json::json!(c);
                }
                if let Some(h) = health {
                    body["health"] = serde_json::json!(h);
                }
                if let Some(t) = team {
                    body["team"] = serde_json::json!(t);
                }
                if let Some(p) = physics {
                    body["physics_body"] = serde_json::json!(p);
                }
                if let Some(c) = collider
                    && let Some(v) = parse_collider(&c)
                {
                    body["collider"] = v;
                }
                let resp = client
                    .post(format!("{server}/template/create"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
            TemplateCommands::Spawn { name, position } => {
                let mut body = serde_json::json!({"name": name});
                if let Some(p) = position
                    && let Some(v) = parse_vec3(&p)
                {
                    body["position"] = serde_json::json!(v);
                }
                let resp = client
                    .post(format!("{server}/template/spawn"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
            TemplateCommands::List => {
                let resp = client.get(format!("{server}/template/list")).send();
                handle_response(resp)
            }
        },

        // ── Rules ──
        Commands::Rule { command } => match command {
            RuleCommands::Create {
                when,
                filter,
                do_action,
            } => {
                let resp = client
                    .post(format!("{server}/rule/create"))
                    .json(&serde_json::json!({
                        "when": when,
                        "filter": filter,
                        "actions": do_action,
                    }))
                    .send();
                handle_response(resp)
            }
            RuleCommands::List => {
                let resp = client.get(format!("{server}/rule/list")).send();
                handle_response(resp)
            }
        },

        Commands::Auth { command } => run_auth(command, &client, server),

        // ── Audio ──
        Commands::Audio { command } => match command {
            AudioCommands::Play {
                path,
                position,
                volume,
                looping,
                max_distance,
            } => {
                let mut body = serde_json::json!({
                    "path": path,
                    "volume": volume,
                    "loop": looping,
                    "max_distance": max_distance,
                });
                if let Some(pos_str) = position {
                    let parts: Vec<f32> = pos_str
                        .split(',')
                        .filter_map(|p| p.trim().parse().ok())
                        .collect();
                    if parts.len() == 3 {
                        body["position"] = serde_json::json!([parts[0], parts[1], parts[2]]);
                    }
                }
                let resp = client
                    .post(format!("{server}/audio/play"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
            AudioCommands::Stop { entity_id } => {
                let resp = client
                    .post(format!("{server}/audio/stop"))
                    .json(&serde_json::json!({"entity_id": entity_id}))
                    .send();
                handle_response(resp)
            }
            AudioCommands::List => {
                let resp = client.get(format!("{server}/audio/list")).send();
                handle_response(resp)
            }
        },

        // ── HUD ──
        Commands::Ui { command } => match command {
            UiCommands::Text {
                text,
                x,
                y,
                size,
                color,
            } => {
                let resp = client
                    .post(format!("{server}/ui/text"))
                    .json(&serde_json::json!({"type": "text", "text": text, "x": x, "y": y, "size": size, "color": color}))
                    .send();
                handle_response(resp)
            }
            UiCommands::Bar {
                x,
                y,
                width,
                height,
                fill,
                color,
            } => {
                let resp = client
                    .post(format!("{server}/ui/bar"))
                    .json(&serde_json::json!({"type": "bar", "x": x, "y": y, "width": width, "height": height, "fill": fill, "color": color}))
                    .send();
                handle_response(resp)
            }
            UiCommands::Clear => post_empty(&client, server, "/ui/clear"),
            UiCommands::List => {
                let resp = client.get(format!("{server}/ui/list")).send();
                handle_response(resp)
            }
        },

        // ── Standalone ──
        Commands::Status => {
            let resp = client.get(format!("{server}/")).send();
            handle_response(resp)
        }
        Commands::Screenshot { output } => run_screenshot(&client, server, output),
        Commands::Schema {
            component: _component,
        } => {
            let resp = client.get(format!("{server}/schema")).send();
            handle_response(resp)
        }

        // ── Hidden backward-compat aliases ──
        Commands::Observe { entity } => {
            if let Some(id) = entity {
                let resp = client.get(format!("{server}/entities/{id}")).send();
                handle_response(resp)
            } else {
                let resp = client
                    .post(format!("{server}/observe"))
                    .header("Content-Type", "application/json")
                    .body("{}")
                    .send();
                handle_response(resp)
            }
        }
        Commands::Spawn {
            position,
            scale,
            physics,
            collider,
        } => {
            let body = build_create_body(
                &None, &None, &position, &scale, &physics, &collider, None, None, false,
            );
            let resp = client.post(format!("{server}/spawn")).json(&body).send();
            handle_response(resp)
        }
        Commands::Modify {
            id,
            transform,
            velocity,
            physics,
            collider,
            json,
        } => {
            let body = if let Some(ref raw) = json {
                parse_json_flag(raw)
            } else {
                // Map old --transform to --position
                build_update_body(&None, &transform, &None, &velocity, &physics, &collider)
            };
            let resp = client
                .post(format!("{server}/entities/{id}/components"))
                .json(&body)
                .send();
            handle_response(resp)
        }
        Commands::Despawn { id, all } => {
            if all {
                post_empty(&client, server, "/reset")
            } else {
                let body = serde_json::json!({"entity_id": id, "entity_generation": 0});
                let resp = client.post(format!("{server}/despawn")).json(&body).send();
                handle_response(resp)
            }
        }
        Commands::Play => post_empty(&client, server, "/play"),
        Commands::Pause => post_empty(&client, server, "/pause"),
        Commands::Step { ticks } => {
            let resp = client
                .post(format!("{server}/step"))
                .json(&serde_json::json!({"ticks": ticks}))
                .send();
            handle_response(resp)
        }
        Commands::Reset => post_empty(&client, server, "/reset"),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        eprintln!("Is the engine running? Start with: cargo run -p euca-editor --example editor");
        std::process::exit(1);
    }
}

// ── Shared request helpers ──

fn post_empty(client: &reqwest::blocking::Client, server: &str, path: &str) -> Result<(), String> {
    let resp = client
        .post(format!("{server}{path}"))
        .header("Content-Type", "application/json")
        .body("{}")
        .send();
    handle_response(resp)
}

fn handle_response(
    resp: Result<reqwest::blocking::Response, reqwest::Error>,
) -> Result<(), String> {
    let resp = resp.map_err(|e| e.to_string())?;
    let status = resp.status();
    let text = resp.text().map_err(|e| e.to_string())?;

    match serde_json::from_str::<Value>(&text) {
        Ok(json) => {
            let pretty = serde_json::to_string_pretty(&json).unwrap_or_default();
            println!("{pretty}");
        }
        Err(_) => {
            println!("{text}");
        }
    }

    if !status.is_success() {
        return Err(format!("HTTP {status}"));
    }
    Ok(())
}

fn run_screenshot(
    client: &reqwest::blocking::Client,
    server: &str,
    output: Option<String>,
) -> Result<(), String> {
    let resp = client
        .post(format!("{server}/screenshot"))
        .header("Content-Type", "application/json")
        .body("{}")
        .send();
    match resp {
        Ok(r) if r.status().is_success() => {
            let text = r.text().unwrap_or_default();
            if let Ok(json) = serde_json::from_str::<Value>(&text) {
                let server_path = json["path"].as_str().unwrap_or("");
                if let Some(ref out) = output {
                    if let Err(e) = std::fs::copy(server_path, out) {
                        eprintln!("Failed to copy screenshot: {e}");
                        std::process::exit(1);
                    }
                    println!("{out}");
                } else {
                    println!("{server_path}");
                }
            }
            Ok(())
        }
        Ok(r) => {
            let status = r.status();
            let text = r.text().unwrap_or_default();
            eprintln!("{text}");
            Err(format!("HTTP {status}"))
        }
        Err(e) => Err(e.to_string()),
    }
}

fn run_auth(
    command: AuthCommands,
    client: &reqwest::blocking::Client,
    server: &str,
) -> Result<(), String> {
    match command {
        AuthCommands::Login => {
            let nit_output = std::process::Command::new("nit")
                .args(["sign", "--login", "eucaengine.local"])
                .output();

            let nit_output = match nit_output {
                Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
                Ok(o) => {
                    let err = String::from_utf8_lossy(&o.stderr);
                    eprintln!("nit sign --login failed: {err}");
                    std::process::exit(1);
                }
                Err(_) => {
                    eprintln!("nit not found. Install nit: npm install -g @newtype-ai/nit");
                    std::process::exit(1);
                }
            };

            let nit_data: Value = match serde_json::from_str(nit_output.trim()) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Failed to parse nit output: {e}");
                    std::process::exit(1);
                }
            };

            let home = std::env::var("HOME").unwrap_or_default();
            let pub_key_path = format!("{home}/.nit/identity/agent.pub");
            let public_key = match std::fs::read_to_string(&pub_key_path) {
                Ok(k) => k,
                Err(_) => {
                    eprintln!("Cannot read {pub_key_path}. Run 'nit init' first.");
                    std::process::exit(1);
                }
            };

            let payload = serde_json::json!({
                "agent_id": nit_data["agent_id"],
                "domain": nit_data["domain"],
                "timestamp": nit_data["timestamp"],
                "signature": nit_data["signature"],
                "public_key": public_key.trim(),
            });

            let resp = client
                .post(format!("{server}/auth/login"))
                .json(&payload)
                .send();
            handle_response(resp)
        }
        AuthCommands::Status => {
            let resp = client.get(format!("{server}/auth/status")).send();
            handle_response(resp)
        }
    }
}
