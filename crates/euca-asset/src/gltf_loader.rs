use euca_render::{Material, Mesh, Vertex};
use std::path::Path;

use crate::animation::{AnimationClipData, parse_animations};
use crate::skeleton::{Skeleton, parse_skeleton};

/// A loaded glTF mesh with its associated material.
pub struct GltfMesh {
    pub mesh: Mesh,
    pub material: Material,
    pub name: Option<String>,
    /// Per-vertex joint indices (4 joints per vertex). Present if model has a skin.
    pub joint_indices: Option<Vec<[u16; 4]>>,
    /// Per-vertex joint weights (4 weights per vertex). Present if model has a skin.
    pub joint_weights: Option<Vec<[f32; 4]>>,
}

/// A complete glTF scene: all meshes with their materials.
pub struct GltfScene {
    pub meshes: Vec<GltfMesh>,
    /// Skeleton data (if the model has a skin).
    pub skeleton: Option<Skeleton>,
    /// Animation clips (if the model has animations).
    pub animations: Vec<AnimationClipData>,
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

            // Read joint indices (optional, for skinned meshes)
            let joint_indices: Option<Vec<[u16; 4]>> =
                reader.read_joints(0).map(|iter| iter.into_u16().collect());

            // Read joint weights (optional, for skinned meshes)
            let joint_weights: Option<Vec<[f32; 4]>> =
                reader.read_weights(0).map(|iter| iter.into_f32().collect());

            // Build vertices
            let vertices: Vec<Vertex> = positions
                .iter()
                .zip(normals.iter())
                .zip(uvs.iter())
                .map(|((pos, norm), uv)| Vertex {
                    position: *pos,
                    normal: *norm,
                    tangent: [1.0, 0.0, 0.0], // Default tangent; proper computation needs MikkTSpace
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
                joint_indices,
                joint_weights,
            });
        }
    }

    if scene_meshes.is_empty() {
        return Err("No meshes found in glTF file".into());
    }

    // Parse skeleton (first skin in the document)
    let skeleton = parse_skeleton(&document, &buffers);

    // Parse animations (requires joint node indices from skeleton)
    let animations = skeleton
        .as_ref()
        .map(|skel| parse_animations(&document, &buffers, &skel.joint_node_indices))
        .unwrap_or_default();

    if !animations.is_empty() {
        log::info!("Loaded {} animation clips from glTF", animations.len(),);
    }

    Ok(GltfScene {
        meshes: scene_meshes,
        skeleton,
        animations,
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
