use clap::{Parser, Subcommand};
use serde_json::Value;

#[derive(Parser)]
#[command(
    name = "euca",
    about = "Euca Engine CLI — control the engine from the terminal",
    after_help = "Examples:\n  euca entity list\n  euca entity create --position 1,2,3\n  euca entity update 5 --position 3,0,0\n  euca sim play\n  euca screenshot"
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
        /// Mesh: "cube" or "sphere"
        #[arg(short, long)]
        mesh: Option<String>,
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
enum AuthCommands {
    /// Login with nit Ed25519 identity
    Login,
    /// Check current authentication status
    Status,
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
            Some(serde_json::json!({"shape": "Capsule", "radius": parts[0], "half_height": parts[1]}))
        } else {
            eprintln!("Warning: capsule format is 'capsule:radius,half_height'");
            None
        }
    } else {
        eprintln!("Warning: collider format is 'aabb:hx,hy,hz' or 'sphere:radius' or 'capsule:radius,half_height'");
        None
    }
}

/// Build a spawn/create JSON body from friendly flags.
fn build_create_body(
    mesh: &Option<String>,
    position: &Option<String>,
    scale: &Option<String>,
    physics: &Option<String>,
    collider: &Option<String>,
) -> Value {
    let mut body = serde_json::json!({});
    if let Some(m) = mesh {
        body["mesh"] = serde_json::json!(m);
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
    body
}

/// Build an update/patch JSON body from friendly flags.
fn build_update_body(
    position: &Option<String>,
    scale: &Option<String>,
    velocity: &Option<String>,
    physics: &Option<String>,
    collider: &Option<String>,
) -> Value {
    let mut body = serde_json::json!({});
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
                position,
                scale,
                physics,
                collider,
                json,
                dry_run,
            } => {
                let body = if let Some(ref raw) = json {
                    parse_json_flag(raw)
                } else {
                    build_create_body(&mesh, &position, &scale, &physics, &collider)
                };
                if dry_run {
                    println!("{}", serde_json::to_string_pretty(&body).unwrap());
                    println!("(dry-run: not created)");
                    Ok(())
                } else {
                    let resp = client
                        .post(format!("{server}/spawn"))
                        .json(&body)
                        .send();
                    handle_response(resp)
                }
            }
            EntityCommands::Update {
                id,
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
                    build_update_body(&position, &scale, &velocity, &physics, &collider)
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
                    let body =
                        serde_json::json!({"entity_id": id, "entity_generation": 0});
                    let resp = client
                        .post(format!("{server}/despawn"))
                        .json(&body)
                        .send();
                    handle_response(resp)
                } else {
                    eprintln!("Specify an entity ID or use --all");
                    std::process::exit(1);
                }
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
            SceneCommands::Save { path: _path } => {
                eprintln!("Scene save not yet implemented via CLI");
                Ok(())
            }
            SceneCommands::Load { path: _path } => {
                eprintln!("Scene load not yet implemented via CLI");
                Ok(())
            }
        },

        // ── Auth ──
        Commands::Auth { command } => run_auth(command, &client, server),

        // ── Standalone ──
        Commands::Status => {
            let resp = client.get(format!("{server}/")).send();
            handle_response(resp)
        }
        Commands::Screenshot { output } => run_screenshot(&client, server, output),
        Commands::Schema { component: _component } => {
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
            let body = build_create_body(&None, &position, &scale, &physics, &collider);
            let resp = client
                .post(format!("{server}/spawn"))
                .json(&body)
                .send();
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
                build_update_body(&transform, &None, &velocity, &physics, &collider)
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
                let body =
                    serde_json::json!({"entity_id": id, "entity_generation": 0});
                let resp = client
                    .post(format!("{server}/despawn"))
                    .json(&body)
                    .send();
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

fn post_empty(
    client: &reqwest::blocking::Client,
    server: &str,
    path: &str,
) -> Result<(), String> {
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
                Ok(o) if o.status.success() => {
                    String::from_utf8_lossy(&o.stdout).to_string()
                }
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
