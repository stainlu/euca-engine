use euca_render::{Material, Mesh, Vertex};
use std::path::Path;

/// A loaded glTF mesh with its associated material.
pub struct GltfMesh {
    pub mesh: Mesh,
    pub material: Material,
    pub name: Option<String>,
}

/// A complete glTF scene: all meshes with their materials.
pub struct GltfScene {
    pub meshes: Vec<GltfMesh>,
}

/// Load a glTF/glb file, extracting meshes and PBR materials.
///
/// Returns a `GltfScene` containing all meshes found in the file.
pub fn load_gltf(path: impl AsRef<Path>) -> Result<GltfScene, String> {
    let path = path.as_ref();
    let (document, buffers, _images) =
        gltf::import(path).map_err(|e| format!("Failed to load glTF '{}': {e}", path.display()))?;

    let mut scene_meshes = Vec::new();

    for mesh in document.meshes() {
        let mesh_name = mesh.name().map(String::from);

        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            // Read positions (required)
            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(iter) => iter.collect(),
                None => {
                    log::warn!("Mesh primitive has no positions, skipping");
                    continue;
                }
            };

            // Read normals (optional, generate flat [0,1,0] if missing)
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

            // Read UVs (optional, default to [0,0])
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

            // Read indices (optional, generate sequential if missing)
            let indices: Vec<u32> = reader
                .read_indices()
                .map(|iter| iter.into_u32().collect())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());

            // Build vertices
            let vertices: Vec<Vertex> = positions
                .iter()
                .zip(normals.iter())
                .zip(uvs.iter())
                .map(|((pos, norm), uv)| Vertex {
                    position: *pos,
                    normal: *norm,
                    uv: *uv,
                })
                .collect();

            // Extract PBR material
            let pbr = primitive.material().pbr_metallic_roughness();
            let base_color = pbr.base_color_factor();
            let material = Material::new(base_color, pbr.metallic_factor(), pbr.roughness_factor());

            log::info!(
                "Loaded mesh '{}': {} vertices, {} indices, albedo={:?}",
                mesh_name.as_deref().unwrap_or("unnamed"),
                vertices.len(),
                indices.len(),
                &base_color[..3],
            );

            scene_meshes.push(GltfMesh {
                mesh: Mesh { vertices, indices },
                material,
                name: mesh_name.clone(),
            });
        }
    }

    if scene_meshes.is_empty() {
        return Err("No meshes found in glTF file".into());
    }

    Ok(GltfScene {
        meshes: scene_meshes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_nonexistent_file() {
        let result = load_gltf("nonexistent.glb");
        assert!(result.is_err());
    }
}
