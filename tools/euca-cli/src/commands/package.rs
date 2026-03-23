use std::path::Path;

use serde_json::Value;

/// Package a game project into a distributable folder.
///
/// Reads `.eucaproject.json`, copies the game binary, level files, and assets
/// into the output directory.
pub(crate) fn package_game(project_dir: &str, output_dir: &str) {
    let project_path = Path::new(project_dir).join(".eucaproject.json");
    let project_data = match std::fs::read_to_string(&project_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Cannot read project file {}: {e}", project_path.display());
            eprintln!("Make sure .eucaproject.json exists in the project directory.");
            std::process::exit(1);
        }
    };

    let project: Value = match serde_json::from_str(&project_data) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Invalid project JSON: {e}");
            std::process::exit(1);
        }
    };

    let name = project
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("game");
    let default_level = project
        .get("default_level")
        .and_then(|v| v.as_str())
        .unwrap_or("level.json");
    let levels_dir = project
        .get("levels_dir")
        .and_then(|v| v.as_str())
        .unwrap_or("levels");
    let assets_dir = project
        .get("assets_dir")
        .and_then(|v| v.as_str())
        .unwrap_or("assets");

    println!("Packaging: {name}");
    println!("  Project: {}", project_path.display());
    println!("  Output:  {output_dir}/");

    // Create output directory
    let out = Path::new(output_dir);
    std::fs::create_dir_all(out).expect("Failed to create output directory");

    // Copy project file
    let dest_project = out.join(".eucaproject.json");
    std::fs::copy(&project_path, &dest_project).expect("Failed to copy project file");
    println!("  Copied .eucaproject.json");

    // Copy default level
    let src_level = Path::new(project_dir).join(default_level);
    if src_level.exists() {
        std::fs::copy(&src_level, out.join(default_level)).expect("Failed to copy default level");
        println!("  Copied {default_level}");
    } else {
        eprintln!("  Warning: default level {default_level} not found");
    }

    // Copy levels directory
    let src_levels = Path::new(project_dir).join(levels_dir);
    if src_levels.is_dir() {
        let dest_levels = out.join(levels_dir);
        copy_dir_recursive(&src_levels, &dest_levels);
        println!("  Copied {levels_dir}/");
    }

    // Copy assets directory
    let src_assets = Path::new(project_dir).join(assets_dir);
    if src_assets.is_dir() {
        let dest_assets = out.join(assets_dir);
        copy_dir_recursive(&src_assets, &dest_assets);
        println!("  Copied {assets_dir}/");
    }

    // Find the game binary
    let binary_name = if cfg!(target_os = "windows") {
        "euca-game.exe"
    } else {
        "euca-game"
    };

    // Look for the binary in common cargo output locations
    let binary_candidates = [
        Path::new(project_dir)
            .join("target/release")
            .join(binary_name),
        Path::new(project_dir)
            .join("target/debug")
            .join(binary_name),
        Path::new("target/release").join(binary_name),
        Path::new("target/debug").join(binary_name),
    ];

    let mut binary_copied = false;
    for candidate in &binary_candidates {
        if candidate.exists() {
            let dest_name = if cfg!(target_os = "windows") {
                format!("{name}.exe")
            } else {
                name.replace(' ', "-").to_lowercase()
            };
            let dest_binary = out.join(&dest_name);
            std::fs::copy(candidate, &dest_binary).expect("Failed to copy game binary");

            // Make executable on Unix
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = std::fs::metadata(&dest_binary)
                    .expect("metadata")
                    .permissions();
                perms.set_mode(0o755);
                std::fs::set_permissions(&dest_binary, perms).expect("set permissions");
            }

            println!(
                "  Copied binary: {dest_name} (from {})",
                candidate.display()
            );
            binary_copied = true;
            break;
        }
    }

    if !binary_copied {
        eprintln!(
            "  Warning: game binary not found. Build first with: cargo build --release -p euca-game"
        );
    }

    println!();
    println!("Package complete: {}/", out.display());
    if binary_copied {
        let run_name = name.replace(' ', "-").to_lowercase();
        println!("Run with: cd {} && ./{}", out.display(), run_name);
    }
}

/// Recursively copy a directory.
fn copy_dir_recursive(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).expect("Failed to create directory");
    if let Ok(entries) = std::fs::read_dir(src) {
        for entry in entries.flatten() {
            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());
            if src_path.is_dir() {
                copy_dir_recursive(&src_path, &dst_path);
            } else {
                std::fs::copy(&src_path, &dst_path).expect("Failed to copy file");
            }
        }
    }
}
