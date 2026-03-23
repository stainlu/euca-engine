use clap::Subcommand;

#[derive(Subcommand)]
pub(crate) enum AssetCommands {
    /// Show metadata about a glTF/glb asset file
    Info {
        /// Path to the asset file (.gltf or .glb)
        file: String,
    },
    /// Run mesh optimization: dedup vertices, compute tangents, reorder for GPU cache
    Optimize {
        /// Input asset file (.gltf or .glb)
        input: String,
        /// Output stats file (JSON)
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Generate LOD (Level of Detail) chain from a mesh
    Lod {
        /// Input asset file (.gltf or .glb)
        input: String,
        /// Output stats file (JSON)
        #[arg(short, long)]
        output: Option<String>,
        /// Number of LOD levels to generate
        #[arg(short, long, default_value = "4")]
        levels: usize,
    },
}

pub(crate) fn run_asset(command: AssetCommands) {
    match command {
        AssetCommands::Info { file } => run_asset_info(&file),
        AssetCommands::Optimize { input, output } => run_asset_optimize(&input, output.as_deref()),
        AssetCommands::Lod {
            input,
            output,
            levels,
        } => run_asset_lod(&input, output.as_deref(), levels),
    }
}

fn run_asset_info(file: &str) {
    let scene = match euca_asset::load_gltf(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load asset: {e}");
            std::process::exit(1);
        }
    };

    let mut total_vertices: usize = 0;
    let mut total_triangles: usize = 0;
    let mut mesh_details = Vec::new();

    for gm in &scene.meshes {
        let verts = gm.mesh.vertices.len();
        let tris = gm.mesh.indices.len() / 3;
        total_vertices += verts;
        total_triangles += tris;
        mesh_details.push(serde_json::json!({
            "name": gm.name.as_deref().unwrap_or("unnamed"),
            "vertices": verts,
            "triangles": tris,
            "has_skin": gm.joint_indices.is_some(),
        }));
    }

    let image_details: Vec<_> = scene
        .images
        .iter()
        .enumerate()
        .map(|(i, img)| {
            serde_json::json!({
                "index": i,
                "width": img.width,
                "height": img.height,
                "size_bytes": img.pixels.len(),
            })
        })
        .collect();

    let info = serde_json::json!({
        "file": file,
        "mesh_count": scene.meshes.len(),
        "total_vertices": total_vertices,
        "total_triangles": total_triangles,
        "has_skeleton": scene.skeleton.is_some(),
        "animation_count": scene.animations.len(),
        "texture_count": scene.images.len(),
        "meshes": mesh_details,
        "textures": image_details,
    });

    println!("{}", serde_json::to_string_pretty(&info).unwrap());
}

fn run_asset_optimize(input: &str, output: Option<&str>) {
    let scene = match euca_asset::load_gltf(input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load asset: {e}");
            std::process::exit(1);
        }
    };

    let mut results = Vec::new();

    for (i, gm) in scene.meshes.iter().enumerate() {
        let name = gm.name.as_deref().unwrap_or("unnamed").to_string();
        let before_verts = gm.mesh.vertices.len();
        let before_tris = gm.mesh.indices.len() / 3;

        let optimized = euca_asset::optimize_mesh(&gm.mesh);
        let after_verts = optimized.vertices.len();
        let after_tris = optimized.indices.len() / 3;

        let dedup_ratio = if before_verts > 0 {
            1.0 - (after_verts as f64 / before_verts as f64)
        } else {
            0.0
        };

        println!(
            "  Mesh {i} \"{name}\": {before_verts} → {after_verts} vertices ({:.1}% reduction), {before_tris} → {after_tris} triangles",
            dedup_ratio * 100.0,
        );

        results.push(serde_json::json!({
            "name": name,
            "vertices_before": before_verts,
            "vertices_after": after_verts,
            "triangles_before": before_tris,
            "triangles_after": after_tris,
            "dedup_ratio": format!("{:.3}", dedup_ratio),
        }));
    }

    let stats = serde_json::json!({
        "file": input,
        "operation": "optimize",
        "meshes": results,
    });

    if let Some(out_path) = output {
        let json = serde_json::to_string_pretty(&stats).unwrap();
        std::fs::write(out_path, json).expect("Failed to write stats file");
        println!("  Stats written to {out_path}");
    }
}

fn run_asset_lod(input: &str, output: Option<&str>, levels: usize) {
    let scene = match euca_asset::load_gltf(input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load asset: {e}");
            std::process::exit(1);
        }
    };

    // Generate LOD ratios: 1.0, 0.5, 0.25, 0.125, ... for `levels` levels
    let ratios: Vec<f32> = (0..levels).map(|i| 1.0 / (1 << i) as f32).collect();

    let mut results = Vec::new();

    for (i, gm) in scene.meshes.iter().enumerate() {
        let name = gm.name.as_deref().unwrap_or("unnamed").to_string();

        let lod_chain = euca_asset::generate_lod_chain(&gm.mesh, &ratios);

        println!("  Mesh {i} \"{name}\":");
        let mut lod_details = Vec::new();
        for (level, lod_mesh) in lod_chain.iter().enumerate() {
            let verts = lod_mesh.vertices.len();
            let tris = lod_mesh.indices.len() / 3;
            let ratio = ratios[level];
            println!(
                "    LOD {level}: {verts} vertices, {tris} triangles (target {:.0}%)",
                ratio * 100.0
            );
            lod_details.push(serde_json::json!({
                "level": level,
                "target_ratio": ratio,
                "vertices": verts,
                "triangles": tris,
            }));
        }

        results.push(serde_json::json!({
            "name": name,
            "lod_levels": lod_details,
        }));
    }

    let stats = serde_json::json!({
        "file": input,
        "operation": "lod",
        "levels": levels,
        "ratios": ratios,
        "meshes": results,
    });

    if let Some(out_path) = output {
        let json = serde_json::to_string_pretty(&stats).unwrap();
        std::fs::write(out_path, json).expect("Failed to write stats file");
        println!("  Stats written to {out_path}");
    }
}
