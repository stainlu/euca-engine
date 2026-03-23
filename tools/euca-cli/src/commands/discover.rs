use crate::Cli;

/// Commands that work offline (no engine running).
const OFFLINE_COMMANDS: &[&str] = &["package", "asset", "discover"];

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

pub(crate) fn run_discover(json: bool, group_filter: Option<&str>) {
    use clap::CommandFactory;
    let cmd = Cli::command();
    let version = cmd.get_version().unwrap_or("0.4.0");

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
