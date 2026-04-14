//! `euca explain <topic>` — progressive-disclosure documentation.
//!
//! Instead of preloading a 638-line `SKILL.md` into every agent
//! session, the engine ships a small topic registry. Agents learn the
//! bootstrap minimum from `SKILL.md`, then pull focused worked
//! examples on demand via `euca explain`.
//!
//! Topics are compiled into the binary via `include_str!` so the CLI
//! has no runtime dependency on a docs directory.

/// One topic in the explain registry.
struct Topic {
    name: &'static str,
    summary: &'static str,
    content: &'static str,
}

/// All available topics. Order matters — it's the order shown in
/// `euca explain` with no argument.
const TOPICS: &[Topic] = &[
    Topic {
        name: "quickstart",
        summary: "Start the engine, confirm it's alive, spawn your first entity, run the sim.",
        content: include_str!("../explain/quickstart.md"),
    },
    Topic {
        name: "entity",
        summary: "Spawning and editing entities — components, templates, CRUD.",
        content: include_str!("../explain/entity.md"),
    },
    Topic {
        name: "combat",
        summary: "Auto-targeting fighters, towers, death + respawn.",
        content: include_str!("../explain/combat.md"),
    },
    Topic {
        name: "rule",
        summary: "Data-driven game logic — when X happens, do Y.",
        content: include_str!("../explain/rule.md"),
    },
    Topic {
        name: "assert",
        summary: "Testable expectations as ECS entities. Evaluate pass/fail.",
        content: include_str!("../explain/assert.md"),
    },
    Topic {
        name: "fork",
        summary: "Counterfactual simulation — clone the world, step it, compare, discard.",
        content: include_str!("../explain/fork.md"),
    },
    Topic {
        name: "scenario",
        summary: "Declarative game setup as a single JSON document. Composes with fork.",
        content: include_str!("../explain/scenario.md"),
    },
];

/// Run `euca explain` with an optional topic name.
///
/// With no topic, prints the list of available topics plus a short
/// summary for each. With a topic name, prints the full content of
/// that topic.
pub(crate) fn run_explain(topic: Option<&str>) -> Result<(), String> {
    match topic {
        None => {
            println!("euca explain — topic index\n");
            println!("Usage: euca explain <topic>\n");
            println!("Available topics:");
            for t in TOPICS {
                println!("  {:<12} {}", t.name, t.summary);
            }
            println!();
            println!("Tip: `euca discover` lists the full CLI command surface.");
            println!("     `euca discover <group>` drills into one group.");
            Ok(())
        }
        Some(name) => {
            for t in TOPICS {
                if t.name == name {
                    print!("{}", t.content);
                    if !t.content.ends_with('\n') {
                        println!();
                    }
                    return Ok(());
                }
            }
            // Unknown topic — suggest the closest match.
            let available: Vec<&str> = TOPICS.iter().map(|t| t.name).collect();
            Err(format!(
                "unknown topic '{name}'\nAvailable topics: {}",
                available.join(", ")
            ))
        }
    }
}
