//! Simple AI agent that plays the game via HTTP.
//!
//! Start the game server first: `cargo run --example agent_game`
//! Then run this: `cargo run --example agent_client`
//!
//! The agent joins the game as a player, observes the state,
//! and moves toward the origin. Same game rules as human players.

use std::thread;
use std::time::Duration;

const SERVER: &str = "http://127.0.0.1:8080";

fn main() {
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .build()
        .unwrap();

    // Join the game
    println!("Joining game...");
    let join_resp: serde_json::Value = client
        .post(format!("{SERVER}/join"))
        .json(&serde_json::json!({"name": "AI Agent"}))
        .send()
        .expect("Failed to connect to game server")
        .json()
        .unwrap();

    let player_id = join_resp["player_id"].as_u64().unwrap();
    println!("Joined as player {player_id} at tick {}", join_resp["tick"]);

    // Game loop: observe → decide → act
    for tick in 0..300 {
        // Observe
        let view: serde_json::Value = client
            .get(format!("{SERVER}/player/{player_id}/view"))
            .send()
            .ok()
            .and_then(|r| r.json().ok())
            .unwrap_or(serde_json::json!({"entities": []}));

        let entities = view["entities"].as_array();

        // Find our position
        let my_pos = entities
            .and_then(|ents| {
                ents.iter()
                    .find(|e| e["network_id"].as_u64() == Some(player_id))
            })
            .and_then(|e| {
                let p = &e["position"];
                Some([
                    p[0].as_f64()? as f32,
                    p[1].as_f64()? as f32,
                    p[2].as_f64()? as f32,
                ])
            });

        // Decide: move toward origin + shoot periodically
        let mut keys = Vec::new();
        if let Some(pos) = my_pos {
            if pos[0] > 0.5 {
                keys.push("a");
            } else if pos[0] < -0.5 {
                keys.push("d");
            }
            if pos[2] > 0.5 {
                keys.push("s");
            } else if pos[2] < -0.5 {
                keys.push("w");
            }

            // Shoot every 30 ticks (0.5 seconds)
            if tick % 30 == 0 {
                keys.push("shoot");
            }

            if tick % 60 == 0 {
                println!(
                    "[tick {tick}] pos=({:.1}, {:.1}, {:.1}) keys={:?}",
                    pos[0], pos[1], pos[2], keys
                );
            }
        } else if tick % 60 == 0 {
            println!("[tick {tick}] waiting for state...");
        }

        // Act
        let _ = client
            .post(format!("{SERVER}/action"))
            .json(&serde_json::json!({
                "player_id": player_id,
                "keys": keys,
            }))
            .send();

        thread::sleep(Duration::from_millis(16)); // ~60Hz
    }

    // Leave
    println!("Agent leaving game.");
    let _ = client
        .post(format!("{SERVER}/leave"))
        .json(&serde_json::json!({"player_id": player_id}))
        .send();
}
