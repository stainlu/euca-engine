use clap::{Parser, Subcommand};
use serde_json::Value;

#[derive(Parser)]
#[command(name = "euca", about = "Euca Engine CLI — control the simulation from the terminal")]
struct Cli {
    /// Server URL (default: http://localhost:8080)
    #[arg(short, long, default_value = "http://localhost:8080")]
    server: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show engine status
    Status,

    /// Observe world state (list all entities)
    Observe,

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
    },

    /// Despawn an entity
    Despawn {
        /// Entity ID
        #[arg(long)]
        id: u32,
        /// Entity generation
        #[arg(long)]
        generation: u32,
    },

    /// Reset the world (despawn all entities)
    Reset,

    /// Show available components and actions
    Schema,
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
        Commands::Observe => {
            let resp = client
                .post(format!("{}/observe", cli.server))
                .header("Content-Type", "application/json")
                .body("{}")
                .send();
            handle_response(resp)
        }
        Commands::Step { ticks } => {
            let resp = client
                .post(format!("{}/step", cli.server))
                .json(&serde_json::json!({"ticks": ticks}))
                .send();
            handle_response(resp)
        }
        Commands::Spawn { position } => {
            let pos = position.map(|s| {
                let parts: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
                if parts.len() == 3 {
                    [parts[0], parts[1], parts[2]]
                } else {
                    eprintln!("Warning: position should be 'x,y,z', using 0,0,0");
                    [0.0, 0.0, 0.0]
                }
            });
            let body = serde_json::json!({"position": pos});
            let resp = client
                .post(format!("{}/spawn", cli.server))
                .json(&body)
                .send();
            handle_response(resp)
        }
        Commands::Despawn { id, generation } => {
            let body = serde_json::json!({"entity_id": id, "entity_generation": generation});
            let resp = client
                .post(format!("{}/despawn", cli.server))
                .json(&body)
                .send();
            handle_response(resp)
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
        eprintln!("Is the Euca Engine server running? Try: cargo run -p euca-agent --example headless_server");
        std::process::exit(1);
    }
}

fn handle_response(resp: Result<reqwest::blocking::Response, reqwest::Error>) -> Result<(), String> {
    let resp = resp.map_err(|e| e.to_string())?;
    let status = resp.status();
    let text = resp.text().map_err(|e| e.to_string())?;

    // Try to parse as JSON for pretty-printing
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
