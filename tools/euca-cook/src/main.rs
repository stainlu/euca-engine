//! Offline asset cooker for Euca Engine.
//!
//! Pre-processes GLB/glTF files into engine-native `.emesh` binary format with:
//! - LOD generation (4 levels via meshoptimizer simplification)
//! - Vertex cache optimization (via meshoptimizer)
//! - Pre-extracted RGBA8 textures
//!
//! Usage:
//!   euca-cook <input.glb> [-o output_dir]
//!   euca-cook assets/generated/*.glb -o assets/cooked/

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// Re-use the cooked format from euca-asset
use euca_asset::cooked::{CookedLod, CookedMaterial, CookedMesh, CookedTexture};

// ---------------------------------------------------------------------------
// Cook pipeline
// ---------------------------------------------------------------------------

fn cook_glb(input: &Path, output_dir: &Path) -> Result<(), String> {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    log::info!("Cooking '{}'...", input.display());
    let start = std::time::Instant::now();

    // 1. Parse GLB
    let (document, buffers, raw_images) =
        gltf::import(input).map_err(|e| format!("Failed to load: {e}"))?;

    let parse_time = start.elapsed();
    log::info!("  Parsed in {:.1}s", parse_time.as_secs_f32());

    // 2. Extract textures → RGBA8
    let textures: Vec<CookedTexture> = raw_images
        .iter()
        .map(|img| {
            let pixels = convert_to_rgba8(img);
            CookedTexture {
                width: img.width,
                height: img.height,
                pixels,
            }
        })
        .collect();

    // 3. Extract first mesh primitive
    let mesh = document.meshes().next().ok_or("No meshes in GLB")?;
    let primitive = mesh.primitives().next().ok_or("No primitives")?;
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or("No positions")?
        .collect();
    let normals: Vec<[f32; 3]> = reader
        .read_normals()
        .map(|iter| iter.collect())
        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
    let uvs: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .map(|tc| tc.into_f32().collect())
        .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
    let indices: Vec<u32> = reader
        .read_indices()
        .map(|idx| idx.into_u32().collect())
        .unwrap_or_else(|| (0..positions.len() as u32).collect());

    log::info!(
        "  Source: {} vertices, {} indices ({} triangles)",
        positions.len(),
        indices.len(),
        indices.len() / 3
    );

    // 4. Extract material
    let pbr = primitive.material().pbr_metallic_roughness();
    let material = CookedMaterial {
        albedo: pbr.base_color_factor(),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        albedo_tex_index: pbr
            .base_color_texture()
            .map(|t| t.texture().source().index()),
    };

    // 5. Compute bounds
    let (bounds_min, bounds_max, ground_offset) = compute_bounds(&positions);

    // 6. Generate LODs via meshoptimizer
    let lods = generate_lods(&positions, &normals, &uvs, &indices);

    for (i, lod) in lods.iter().enumerate() {
        log::info!(
            "  LOD{}: {} verts, {} tris",
            i,
            lod.vertex_count,
            lod.index_count / 3
        );
    }

    // 7. Serialize
    let cooked = CookedMesh {
        name: stem.to_string(),
        lods,
        textures,
        material,
        bounds_min,
        bounds_max,
        ground_offset,
    };

    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("Failed to create output dir: {e}"))?;
    let output_path = output_dir.join(format!("{stem}.emesh"));
    let bytes =
        bincode::serialize(&cooked).map_err(|e| format!("Serialization failed: {e}"))?;
    std::fs::write(&output_path, &bytes)
        .map_err(|e| format!("Failed to write: {e}"))?;

    let total_time = start.elapsed();
    log::info!(
        "  Output: {} ({:.1} MB) in {:.1}s",
        output_path.display(),
        bytes.len() as f64 / 1_048_576.0,
        total_time.as_secs_f32()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// LOD generation
// ---------------------------------------------------------------------------

/// Interleaved vertex for meshoptimizer (position + normal + uv = 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PackedVertex {
    px: f32,
    py: f32,
    pz: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    u: f32,
    v: f32,
}

fn generate_lods(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<CookedLod> {
    let vertex_count = positions.len();

    // Build interleaved vertex buffer for meshoptimizer
    let packed: Vec<PackedVertex> = (0..vertex_count)
        .map(|i| PackedVertex {
            px: positions[i][0],
            py: positions[i][1],
            pz: positions[i][2],
            nx: normals[i][0],
            ny: normals[i][1],
            nz: normals[i][2],
            u: uvs[i][0],
            v: uvs[i][1],
        })
        .collect();

    // Create VertexDataAdapter for simplification (needs raw bytes + stride)
    let vertex_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            packed.as_ptr() as *const u8,
            packed.len() * std::mem::size_of::<PackedVertex>(),
        )
    };
    let vertex_stride = std::mem::size_of::<PackedVertex>();

    let adapter =
        meshopt::VertexDataAdapter::new(vertex_bytes, vertex_stride, 0).unwrap();

    // LOD targets: 100%, 50%, 25%, 10%
    let original_tri_count = indices.len() / 3;
    let targets = [
        original_tri_count,
        original_tri_count / 2,
        original_tri_count / 4,
        original_tri_count / 10,
    ];

    let mut lods = Vec::new();

    for (level, &target_tris) in targets.iter().enumerate() {
        let target_indices = target_tris * 3;

        let lod_indices = if level == 0 {
            // LOD0: just optimize vertex cache, no simplification
            meshopt::optimize_vertex_cache(&indices, vertex_count)
        } else {
            // Simplify then optimize cache
            let simplified = meshopt::simplify(
                &indices,
                &adapter,
                target_indices,
                1e-2,
                meshopt::SimplifyOptions::empty(),
                None,
            );
            meshopt::optimize_vertex_cache(&simplified, vertex_count)
        };

        // Remap to remove unused vertices
        let (remap_count, remap) =
            meshopt::generate_vertex_remap(&packed, Some(&lod_indices));

        let remapped_indices =
            meshopt::remap_index_buffer(Some(&lod_indices), remap_count, &remap);
        let remapped_packed =
            meshopt::remap_vertex_buffer(&packed, remap_count, &remap);

        // Unpack back to separate arrays
        let lod_positions: Vec<[f32; 3]> =
            remapped_packed.iter().map(|v| [v.px, v.py, v.pz]).collect();
        let lod_normals: Vec<[f32; 3]> =
            remapped_packed.iter().map(|v| [v.nx, v.ny, v.nz]).collect();
        let lod_uvs: Vec<[f32; 2]> =
            remapped_packed.iter().map(|v| [v.u, v.v]).collect();

        lods.push(CookedLod {
            vertex_count: lod_positions.len() as u32,
            index_count: remapped_indices.len() as u32,
            positions: lod_positions,
            normals: lod_normals,
            uvs: lod_uvs,
            indices: remapped_indices,
        });
    }

    lods
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn compute_bounds(positions: &[[f32; 3]]) -> ([f32; 3], [f32; 3], f32) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for p in positions {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }
    let ground_offset = -min[1];
    (min, max, ground_offset)
}

fn convert_to_rgba8(data: &gltf::image::Data) -> Vec<u8> {
    use gltf::image::Format;
    match data.format {
        Format::R8G8B8A8 => data.pixels.clone(),
        Format::R8G8B8 => {
            let count = data.pixels.len() / 3;
            let mut rgba = Vec::with_capacity(count * 4);
            for chunk in data.pixels.chunks_exact(3) {
                rgba.extend_from_slice(chunk);
                rgba.push(255);
            }
            rgba
        }
        Format::R8 => {
            let mut rgba = Vec::with_capacity(data.pixels.len() * 4);
            for &val in &data.pixels {
                rgba.extend_from_slice(&[val, val, val, 255]);
            }
            rgba
        }
        _ => data.pixels.clone(),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: euca-cook <input.glb> [-o output_dir]");
        std::process::exit(1);
    }

    let mut inputs = Vec::new();
    let mut output_dir = PathBuf::from("assets/cooked");
    let mut i = 1;
    while i < args.len() {
        if args[i] == "-o" && i + 1 < args.len() {
            output_dir = PathBuf::from(&args[i + 1]);
            i += 2;
        } else {
            inputs.push(PathBuf::from(&args[i]));
            i += 1;
        }
    }

    if inputs.is_empty() {
        eprintln!("No input files specified");
        std::process::exit(1);
    }

    log::info!("Cooking {} files → {}", inputs.len(), output_dir.display());

    let mut ok = 0;
    let mut fail = 0;
    let total_start = std::time::Instant::now();

    for input in &inputs {
        match cook_glb(input, &output_dir) {
            Ok(()) => ok += 1,
            Err(e) => {
                log::error!("Failed to cook '{}': {e}", input.display());
                fail += 1;
            }
        }
    }

    let total_time = total_start.elapsed();
    log::info!(
        "Done: {ok} cooked, {fail} failed in {:.1}s",
        total_time.as_secs_f32()
    );
}
