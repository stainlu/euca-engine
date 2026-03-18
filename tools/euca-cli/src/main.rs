use clap::{Parser, Subcommand};
use serde_json::Value;

#[derive(Parser)]
#[command(
    name = "euca",
    about = "Euca Engine CLI — control the engine from the terminal"
)]
struct Cli {
    /// Server URL (default: http://localhost:3917)
    #[arg(short, long, default_value = "http://localhost:3917")]
    server: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show engine status
    Status,

    /// Observe world state (list all entities)
    Observe {
        /// Show a single entity by ID
        #[arg(short, long)]
        entity: Option<u32>,
    },

    /// Advance the simulation by N ticks
    Step {
        /// Number of ticks to advance
        #[arg(short, long, default_value = "1")]
        ticks: u64,
    },

    /// Spawn a new entity
    Spawn {
        /// Position as "x,y,z"
        #[arg(short, long)]
        position: Option<String>,

        /// Scale as "x,y,z"
        #[arg(long)]
        scale: Option<String>,

        /// Physics body type: Dynamic, Static, Kinematic
        #[arg(long)]
        physics: Option<String>,

        /// Collider: "aabb:hx,hy,hz" or "sphere:radius"
        #[arg(long)]
        collider: Option<String>,
    },

    /// Modify an entity's components
    Modify {
        /// Entity ID
        id: u32,

        /// Transform position as "x,y,z"
        #[arg(short, long)]
        transform: Option<String>,

        /// Linear velocity as "x,y,z"
        #[arg(long)]
        velocity: Option<String>,

        /// Physics body type: Dynamic, Static, Kinematic
        #[arg(long)]
        physics: Option<String>,

        /// Collider: "aabb:hx,hy,hz" or "sphere:radius"
        #[arg(long)]
        collider: Option<String>,

        /// Raw JSON patch (overrides other flags)
        #[arg(long)]
        json: Option<String>,
    },

    /// Despawn an entity by ID
    Despawn {
        /// Entity ID
        id: u32,

        /// Despawn all entities
        #[arg(long)]
        all: bool,
    },

    /// Reset the world (despawn all entities)
    Reset,

    /// Start simulation
    Play,

    /// Pause simulation
    Pause,

    /// Capture a screenshot of the 3D viewport
    Screenshot {
        /// Output file path (default: temp file)
        #[arg(short, long)]
        output: Option<String>,
    },

    /// Show available components and actions
    Schema,

    /// Authenticate with the engine via nit identity
    Auth {
        #[command(subcommand)]
        command: AuthCommands,
    },
}

#[derive(Subcommand)]
enum AuthCommands {
    /// Login with nit Ed25519 identity
    Login,
    /// Check current authentication status
    Status,
}

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

fn main() {
    let cli = Cli::parse();
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .build()
        .expect("Failed to build HTTP client");

    let result = match cli.command {
        Commands::Status => {
            let resp = client.get(format!("{}/", cli.server)).send();
            handle_response(resp)
        }
        Commands::Observe { entity } => {
            if let Some(id) = entity {
                let resp = client
                    .get(format!("{}/entities/{}", cli.server, id))
                    .send();
                handle_response(resp)
            } else {
                let resp = client
                    .post(format!("{}/observe", cli.server))
                    .header("Content-Type", "application/json")
                    .body("{}")
                    .send();
                handle_response(resp)
            }
        }
        Commands::Step { ticks } => {
            let resp = client
                .post(format!("{}/step", cli.server))
                .json(&serde_json::json!({"ticks": ticks}))
                .send();
            handle_response(resp)
        }
        Commands::Spawn {
            position,
            scale,
            physics,
            collider,
        } => {
            let mut body = serde_json::json!({});
            if let Some(ref s) = position
                && let Some(v) = parse_vec3(s)
            {
                body["position"] = serde_json::json!(v);
            }
            if let Some(ref s) = scale
                && let Some(v) = parse_vec3(s)
            {
                body["scale"] = serde_json::json!(v);
            }
            if let Some(ref pb) = physics {
                body["physics_body"] = serde_json::json!(pb);
            }
            if let Some(ref c) = collider
                && let Some(v) = parse_collider(c)
            {
                body["collider"] = v;
            }
            let resp = client
                .post(format!("{}/spawn", cli.server))
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
            let body = if let Some(raw) = json {
                match serde_json::from_str::<Value>(&raw) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("Invalid JSON: {e}");
                        std::process::exit(1);
                    }
                }
            } else {
                let mut body = serde_json::json!({});
                if let Some(ref t) = transform
                    && let Some(pos) = parse_vec3(t)
                {
                    body["transform"] = serde_json::json!({"position": pos});
                }
                if let Some(ref v) = velocity
                    && let Some(vel) = parse_vec3(v)
                {
                    body["velocity"] = serde_json::json!({"linear": vel, "angular": [0,0,0]});
                }
                if let Some(ref pb) = physics {
                    body["physics_body"] = serde_json::json!(pb);
                }
                if let Some(ref c) = collider
                    && let Some(v) = parse_collider(c)
                {
                    body["collider"] = v;
                }
                body
            };
            let resp = client
                .post(format!("{}/entities/{}/components", cli.server, id))
                .json(&body)
                .send();
            handle_response(resp)
        }
        Commands::Despawn { id, all } => {
            if all {
                let resp = client
                    .post(format!("{}/reset", cli.server))
                    .header("Content-Type", "application/json")
                    .body("{}")
                    .send();
                handle_response(resp)
            } else {
                // Find entity generation by trying 0-15
                let body = serde_json::json!({"entity_id": id, "entity_generation": 0});
                let resp = client
                    .post(format!("{}/despawn", cli.server))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
        }
        Commands::Reset => {
            let resp = client
                .post(format!("{}/reset", cli.server))
                .header("Content-Type", "application/json")
                .body("{}")
                .send();
            handle_response(resp)
        }
        Commands::Play => {
            let resp = client
                .post(format!("{}/play", cli.server))
                .header("Content-Type", "application/json")
                .body("{}")
                .send();
            handle_response(resp)
        }
        Commands::Pause => {
            let resp = client
                .post(format!("{}/pause", cli.server))
                .header("Content-Type", "application/json")
                .body("{}")
                .send();
            handle_response(resp)
        }
        Commands::Screenshot { output } => {
            let resp = client
                .post(format!("{}/screenshot", cli.server))
                .header("Content-Type", "application/json")
                .body("{}")
                .send();
            match resp {
                Ok(r) if r.status().is_success() => {
                    let text = r.text().unwrap_or_default();
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                        let server_path = json["path"].as_str().unwrap_or("");
                        // If user specified output path, copy the file
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
        Commands::Schema => {
            let resp = client.get(format!("{}/schema", cli.server)).send();
            handle_response(resp)
        }
        Commands::Auth { command } => match command {
            AuthCommands::Login => {
                // Run nit sign --login and read public key
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

                // Parse nit output (JSON)
                let nit_data: Value = match serde_json::from_str(nit_output.trim()) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("Failed to parse nit output: {e}");
                        std::process::exit(1);
                    }
                };

                // Read public key from .nit/identity/agent.pub
                let home = std::env::var("HOME").unwrap_or_default();
                let pub_key_path = format!("{home}/.nit/identity/agent.pub");
                let public_key = match std::fs::read_to_string(&pub_key_path) {
                    Ok(k) => k,
                    Err(_) => {
                        eprintln!("Cannot read {pub_key_path}. Run 'nit init' first.");
                        std::process::exit(1);
                    }
                };

                // Build login payload
                let payload = serde_json::json!({
                    "agent_id": nit_data["agent_id"],
                    "domain": nit_data["domain"],
                    "timestamp": nit_data["timestamp"],
                    "signature": nit_data["signature"],
                    "public_key": public_key.trim(),
                });

                let resp = client
                    .post(format!("{}/auth/login", cli.server))
                    .json(&payload)
                    .send();

                match handle_response(resp) {
                    Ok(()) => {
                        // TODO: Store session token in ~/.euca/auth.json
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            AuthCommands::Status => {
                let resp = client
                    .get(format!("{}/auth/status", cli.server))
                    .send();
                handle_response(resp)
            }
        },
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        eprintln!("Is the engine running? Start with: cargo run -p euca-editor --example editor");
        std::process::exit(1);
    }
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
