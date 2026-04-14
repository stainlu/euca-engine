use crate::Cli;

/// Commands that work offline (no engine running).
const OFFLINE_COMMANDS: &[&str] = &["package", "asset", "discover", "explain"];

/// Classification of a CLI group by the agent-facing scope it belongs
/// to. Used by `euca discover --scope <scope>` to filter the output
/// so agents building a puzzle game never see Roshan.
///
/// This is a first pass — the long-term fix is the Genre de-leakage
/// refactor (plan Priority 2), which moves genre-bound modules into
/// separate crates. For now, a simple name-based lookup table is
/// enough to let agents request a scoped view.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    /// Generic engine primitives — no genre vocabulary.
    Core,
    /// Genre-neutral gameplay building blocks (combat, rules, etc.).
    Gameplay,
    /// Rendering / audio / animation / particle / material / etc.
    Media,
    /// MOBA-specific vocabulary (hero, shop, inventory, ...).
    Moba,
    /// Infrastructure and tools (offline commands, diagnostics, auth).
    Tools,
}

impl Scope {
    fn as_str(self) -> &'static str {
        match self {
            Scope::Core => "core",
            Scope::Gameplay => "gameplay",
            Scope::Media => "media",
            Scope::Moba => "moba",
            Scope::Tools => "tools",
        }
    }

    fn parse(s: &str) -> Option<Scope> {
        match s.to_ascii_lowercase().as_str() {
            "core" => Some(Scope::Core),
            "gameplay" => Some(Scope::Gameplay),
            "media" => Some(Scope::Media),
            "moba" => Some(Scope::Moba),
            "tools" => Some(Scope::Tools),
            _ => None,
        }
    }
}

/// Look up the scope for a given top-level group name.
fn scope_of(group: &str) -> Scope {
    match group {
        // Core engine primitives — no genre vocabulary.
        "entity" | "sim" | "scene" | "camera" | "screenshot" | "observe" | "schema"
        | "status" | "trigger" | "projectile" | "fork" | "scenario" | "prefab" => Scope::Core,

        // Genre-neutral gameplay building blocks.
        "game" | "ai" | "rule" | "template" | "ability" | "effect" | "assert" | "manifest"
        | "nav" | "vfx" | "ui" | "input" => Scope::Gameplay,

        // Rendering / audio / animation / visuals.
        "animation" | "audio" | "material" | "postprocess" | "fog" | "terrain" | "foliage"
        | "particle" => Scope::Media,

        // MOBA-specific vocabulary.
        "hero" | "item" | "shop" => Scope::Moba,

        // Tools / infrastructure.
        "package" | "asset" | "discover" | "explain" | "script" | "net" | "auth"
        | "profile" | "diagnose" | "events" | "engine" | "hud" => Scope::Tools,

        // Unknown groups default to Gameplay (visible in gameplay/all views).
        _ => Scope::Gameplay,
    }
}

#[derive(serde::Serialize)]
struct CommandManifest {
    version: String,
    groups: Vec<GroupEntry>,
}

#[derive(serde::Serialize)]
struct GroupEntry {
    name: String,
    description: String,
    requires_engine: bool,
    scope: String,
    commands: Vec<CommandEntry>,
}

#[derive(serde::Serialize)]
struct CommandEntry {
    name: String,
    description: String,
    args: Vec<ArgEntry>,
}

#[derive(serde::Serialize)]
struct ArgEntry {
    name: String,
    #[serde(rename = "type")]
    arg_type: String,
    required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    default: Option<String>,
    description: String,
}

pub(crate) fn run_discover(
    json: bool,
    group_filter: Option<&str>,
    scope_filter: Option<&str>,
) {
    use clap::CommandFactory;
    let cmd = Cli::command();
    let version = cmd.get_version().unwrap_or("0.4.0");

    // Parse the scope filter up-front. An unknown scope string is a
    // no-op (prints everything) with a warning to stderr.
    let scope_parsed: Option<Scope> = match scope_filter {
        Some(s) if !s.is_empty() && s != "all" => match Scope::parse(s) {
            Some(sc) => Some(sc),
            None => {
                eprintln!(
                    "warning: unknown scope '{s}' — valid scopes: core, gameplay, media, moba, tools, all"
                );
                None
            }
        },
        _ => None,
    };

    let mut groups: Vec<GroupEntry> = Vec::new();

    for sub in cmd.get_subcommands() {
        let name = sub.get_name().to_string();

        // Skip hidden commands
        if sub.is_hide_set() {
            continue;
        }

        // Apply group filter
        if let Some(filter) = group_filter
            && !name.contains(filter)
        {
            continue;
        }

        let group_scope = scope_of(&name);

        // Apply scope filter (if any).
        if let Some(target) = scope_parsed
            && group_scope != target
        {
            continue;
        }

        let description = sub.get_about().map(|s| s.to_string()).unwrap_or_default();
        let requires_engine = !OFFLINE_COMMANDS.contains(&name.as_str());

        let mut commands = Vec::new();
        let sub_subs: Vec<_> = sub.get_subcommands().collect();
        if sub_subs.is_empty() {
            // Leaf command (e.g., profile, status)
            commands.push(CommandEntry {
                name: name.clone(),
                description: description.clone(),
                args: collect_args(sub),
            });
        } else {
            for child in sub_subs {
                if child.is_hide_set() {
                    continue;
                }
                commands.push(CommandEntry {
                    name: child.get_name().to_string(),
                    description: child.get_about().map(|s| s.to_string()).unwrap_or_default(),
                    args: collect_args(child),
                });
            }
        }

        groups.push(GroupEntry {
            name,
            description,
            requires_engine,
            scope: group_scope.as_str().to_string(),
            commands,
        });
    }

    if json {
        let manifest = CommandManifest {
            version: version.to_string(),
            groups,
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&manifest).expect("JSON serialization failed")
        );
    } else {
        println!("Euca Engine CLI v{version}\n");
        if group_filter.is_some() && groups.len() == 1 {
            // Detailed view for a single group
            let g = &groups[0];
            let online = if g.requires_engine {
                "requires engine"
            } else {
                "offline"
            };
            println!("  {} — {} ({})\n", g.name, g.description, online);
            for cmd in &g.commands {
                println!("    {} — {}", cmd.name, cmd.description);
                for arg in &cmd.args {
                    let req = if arg.required { " (required)" } else { "" };
                    let def = arg
                        .default
                        .as_ref()
                        .map(|d| format!(" [default: {d}]"))
                        .unwrap_or_default();
                    println!("      --{}: {}{}{}", arg.name, arg.arg_type, req, def);
                }
            }
        } else {
            // Overview of all groups
            for g in &groups {
                let online = if g.requires_engine { "" } else { " (offline)" };
                println!("  {:<14} {}{}", g.name, g.description, online);
            }
            println!();
            println!("Use --json for machine-readable output.");
            println!("Use `euca discover <group>` to see group details.");
        }
    }
}

fn collect_args(cmd: &clap::Command) -> Vec<ArgEntry> {
    cmd.get_arguments()
        .filter(|a| a.get_id() != "help" && a.get_id() != "version")
        .map(|a| {
            let name = a.get_id().to_string();
            let arg_type = if a.get_action().takes_values() {
                "string".to_string()
            } else {
                "flag".to_string()
            };
            let required = a.is_required_set();
            let default = a
                .get_default_values()
                .first()
                .map(|v| v.to_string_lossy().to_string());
            let description = a.get_help().map(|s| s.to_string()).unwrap_or_default();
            ArgEntry {
                name,
                arg_type,
                required,
                default,
                description,
            }
        })
        .collect()
}
