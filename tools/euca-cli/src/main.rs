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

    /// Show available components and actions
    Schema,
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
        Commands::Schema => {
            let resp = client.get(format!("{}/schema", cli.server)).send();
            handle_response(resp)
        }
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
