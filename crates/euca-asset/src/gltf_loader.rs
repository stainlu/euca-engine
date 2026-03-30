use euca_render::{Material, Mesh, TextureHandle, Vertex};
use std::path::Path;

use crate::animation::{AnimationClipData, parse_animations};
use crate::lod::simplify_mesh;
use crate::mesh_opt::{deduplicate_vertices, optimize_vertex_cache};
use crate::skeleton::{Skeleton, parse_skeleton};

/// Meshes exceeding this vertex count are automatically decimated on load.
///
/// 50K vertices is sufficient for high-quality rendering of most game assets
/// while avoiding the GPU cost of oversized source models (e.g., 280K-vertex
/// tower GLBs that balloon level load times).
// Disabled: QEM decimation on 280K meshes takes longer than just loading them.
// Re-enable when we have faster simplification or pre-processed assets.
const AUTO_DECIMATE_VERTEX_THRESHOLD: usize = usize::MAX;

/// Axis-aligned bounding box computed from mesh vertex positions.
#[derive(Clone, Copy, Debug)]
pub struct MeshBounds {
    /// Minimum vertex position per axis.
    pub min: [f32; 3],
    /// Maximum vertex position per axis.
    pub max: [f32; 3],
}

impl MeshBounds {
    /// Compute the AABB from a slice of vertices.
    pub fn from_vertices(vertices: &[Vertex]) -> Option<Self> {
        if vertices.is_empty() {
            return None;
        }
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for v in vertices {
            for i in 0..3 {
                min[i] = min[i].min(v.position[i]);
                max[i] = max[i].max(v.position[i]);
            }
        }
        Some(Self { min, max })
    }

    /// The vertical (Y-axis) offset needed to place the mesh bottom on the ground plane.
    ///
    /// Returns `-min_y` so that when added to the entity's Y position, the lowest
    /// vertex sits exactly at the entity's logical Y coordinate.
    pub fn ground_offset(&self) -> f32 {
        -self.min[1]
    }
}

/// A loaded glTF mesh with its associated material.
pub struct GltfMesh {
    pub mesh: Mesh,
    pub material: Material,
    pub name: Option<String>,
    /// Axis-aligned bounding box of the mesh in local space.
    pub bounds: Option<MeshBounds>,
    /// Per-vertex joint indices (4 joints per vertex). Present if model has a skin.
    pub joint_indices: Option<Vec<[u16; 4]>>,
    /// Per-vertex joint weights (4 weights per vertex). Present if model has a skin.
    pub joint_weights: Option<Vec<[f32; 4]>>,
    /// Index into `GltfScene::images` for the albedo (base color) texture.
    pub albedo_tex_index: Option<usize>,
    /// Index into `GltfScene::images` for the normal map texture.
    pub normal_tex_index: Option<usize>,
    /// Index into `GltfScene::images` for the metallic-roughness texture.
    pub metallic_roughness_tex_index: Option<usize>,
    /// Index into `GltfScene::images` for the ambient occlusion texture.
    pub ao_tex_index: Option<usize>,
    /// Index into `GltfScene::images` for the emissive texture.
    pub emissive_tex_index: Option<usize>,
}

/// An image extracted from a glTF file, converted to RGBA8.
pub struct GltfImage {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// A complete glTF scene: all meshes with their materials.
pub struct GltfScene {
    pub meshes: Vec<GltfMesh>,
    /// Skeleton data (if the model has a skin).
    pub skeleton: Option<Skeleton>,
    /// Animation clips (if the model has animations).
    pub animations: Vec<AnimationClipData>,
    /// Texture images extracted from the file, in RGBA8 format.
    pub images: Vec<GltfImage>,
}

/// Wire GPU texture handles into a GltfMesh's material using its texture index fields.
///
/// Call this after uploading `GltfScene::images` to the GPU. Pass the resulting
/// `TextureHandle` slice so each mesh's material gets the correct textures.
pub fn apply_texture_handles(mesh: &mut GltfMesh, handles: &[TextureHandle]) {
    if let Some(idx) = mesh.albedo_tex_index
        && let Some(&h) = handles.get(idx)
    {
        mesh.material.albedo_texture = Some(h);
    }
    if let Some(idx) = mesh.normal_tex_index
        && let Some(&h) = handles.get(idx)
    {
        mesh.material.normal_texture = Some(h);
    }
    if let Some(idx) = mesh.metallic_roughness_tex_index
        && let Some(&h) = handles.get(idx)
    {
        mesh.material.metallic_roughness_texture = Some(h);
    }
    if let Some(idx) = mesh.ao_tex_index
        && let Some(&h) = handles.get(idx)
    {
        mesh.material.ao_texture = Some(h);
    }
    if let Some(idx) = mesh.emissive_tex_index
        && let Some(&h) = handles.get(idx)
    {
        mesh.material.emissive_texture = Some(h);
    }
}

/// Convert glTF image data to RGBA8 format.
fn convert_to_rgba8(data: &gltf::image::Data) -> Vec<u8> {
    use gltf::image::Format;
    match data.format {
        Format::R8G8B8A8 => data.pixels.clone(),
        Format::R8G8B8 => {
            let pixel_count = data.pixels.len() / 3;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for chunk in data.pixels.chunks_exact(3) {
                rgba.extend_from_slice(chunk);
                rgba.push(255);
            }
            rgba
        }
        Format::R8 => {
            let mut rgba = Vec::with_capacity(data.pixels.len() * 4);
            for &v in &data.pixels {
                rgba.extend_from_slice(&[v, v, v, 255]);
            }
            rgba
        }
        Format::R8G8 => {
            let pixel_count = data.pixels.len() / 2;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for chunk in data.pixels.chunks_exact(2) {
                rgba.extend_from_slice(&[chunk[0], chunk[1], 0, 255]);
            }
            rgba
        }
        // 16-bit formats: downsample to 8-bit
        Format::R16 => {
            let pixel_count = data.pixels.len() / 2;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for chunk in data.pixels.chunks_exact(2) {
                let v = u16::from_le_bytes([chunk[0], chunk[1]]);
                let v8 = (v >> 8) as u8;
                rgba.extend_from_slice(&[v8, v8, v8, 255]);
            }
            rgba
        }
        Format::R16G16 => {
            let pixel_count = data.pixels.len() / 4;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for chunk in data.pixels.chunks_exact(4) {
                let r = (u16::from_le_bytes([chunk[0], chunk[1]]) >> 8) as u8;
                let g = (u16::from_le_bytes([chunk[2], chunk[3]]) >> 8) as u8;
                rgba.extend_from_slice(&[r, g, 0, 255]);
            }
            rgba
        }
        Format::R16G16B16 => {
            let pixel_count = data.pixels.len() / 6;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for chunk in data.pixels.chunks_exact(6) {
                let r = (u16::from_le_bytes([chunk[0], chunk[1]]) >> 8) as u8;
                let g = (u16::from_le_bytes([chunk[2], chunk[3]]) >> 8) as u8;
                let b = (u16::from_le_bytes([chunk[4], chunk[5]]) >> 8) as u8;
                rgba.extend_from_slice(&[r, g, b, 255]);
            }
            rgba
        }
        Format::R16G16B16A16 => {
            let pixel_count = data.pixels.len() / 8;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for chunk in data.pixels.chunks_exact(8) {
                let r = (u16::from_le_bytes([chunk[0], chunk[1]]) >> 8) as u8;
                let g = (u16::from_le_bytes([chunk[2], chunk[3]]) >> 8) as u8;
                let b = (u16::from_le_bytes([chunk[4], chunk[5]]) >> 8) as u8;
                let a = (u16::from_le_bytes([chunk[6], chunk[7]]) >> 8) as u8;
                rgba.extend_from_slice(&[r, g, b, a]);
            }
            rgba
        }
        // 32-bit float formats: convert to 8-bit with clamping
        Format::R32G32B32FLOAT => {
            let pixel_count = data.pixels.len() / 12;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for chunk in data.pixels.chunks_exact(12) {
                let r = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let g = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                let b = f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
                rgba.extend_from_slice(&[
                    (r.clamp(0.0, 1.0) * 255.0) as u8,
                    (g.clamp(0.0, 1.0) * 255.0) as u8,
                    (b.clamp(0.0, 1.0) * 255.0) as u8,
                    255,
                ]);
            }
            rgba
        }
        Format::R32G32B32A32FLOAT => {
            let pixel_count = data.pixels.len() / 16;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for chunk in data.pixels.chunks_exact(16) {
                let r = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let g = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                let b = f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
                let a = f32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]);
                rgba.extend_from_slice(&[
                    (r.clamp(0.0, 1.0) * 255.0) as u8,
                    (g.clamp(0.0, 1.0) * 255.0) as u8,
                    (b.clamp(0.0, 1.0) * 255.0) as u8,
                    (a.clamp(0.0, 1.0) * 255.0) as u8,
                ]);
            }
            rgba
        }
    }
}

/// Get the image index for a glTF texture reference.
fn texture_image_index(tex_info: &gltf::texture::Info<'_>) -> usize {
    tex_info.texture().source().index()
}

/// Load a glTF/glb file, extracting meshes, PBR materials, and texture images.
///
/// Returns a `GltfScene` containing all meshes and images found in the file.
/// Texture images are returned as RGBA8 data — the caller uploads them to the GPU.
pub fn load_gltf(path: impl AsRef<Path>) -> Result<GltfScene, String> {
    let path = path.as_ref();
    let (document, buffers, raw_images) =
        gltf::import(path).map_err(|e| format!("Failed to load glTF '{}': {e}", path.display()))?;

    // Convert all images to RGBA8
    let images: Vec<GltfImage> = raw_images
        .iter()
        .map(|img| {
            let pixels = convert_to_rgba8(img);
            GltfImage {
                pixels,
                width: img.width,
                height: img.height,
            }
        })
        .collect();

    if !images.is_empty() {
        log::info!("Extracted {} texture images from glTF", images.len());
    }

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

            // Extract PBR material with texture references
            let gltf_mat = primitive.material();
            let pbr = gltf_mat.pbr_metallic_roughness();
            let base_color = pbr.base_color_factor();
            let mut material =
                Material::new(base_color, pbr.metallic_factor(), pbr.roughness_factor());

            // Extract emissive factor
            let emissive = gltf_mat.emissive_factor();
            if emissive != [0.0, 0.0, 0.0] {
                material = material.with_emissive(emissive);
            }

            // Extract texture indices (caller will map these to TextureHandles)
            let albedo_tex_index = pbr.base_color_texture().map(|t| texture_image_index(&t));
            let metallic_roughness_tex_index = pbr
                .metallic_roughness_texture()
                .map(|t| texture_image_index(&t));
            let normal_tex_index = gltf_mat
                .normal_texture()
                .map(|t| t.texture().source().index());
            let ao_tex_index = gltf_mat
                .occlusion_texture()
                .map(|t| t.texture().source().index());
            let emissive_tex_index = gltf_mat.emissive_texture().map(|t| texture_image_index(&t));

            let tex_count = [
                albedo_tex_index,
                normal_tex_index,
                metallic_roughness_tex_index,
                ao_tex_index,
                emissive_tex_index,
            ]
            .iter()
            .filter(|t| t.is_some())
            .count();

            log::info!(
                "Loaded mesh '{}': {} vertices, {} indices, albedo={:?}, textures={}",
                mesh_name.as_deref().unwrap_or("unnamed"),
                vertices.len(),
                indices.len(),
                &base_color[..3],
                tex_count,
            );

            // Auto-decimate oversized meshes to keep load times and GPU cost
            // reasonable.  Skip skinned meshes — their per-vertex joint data
            // would become misaligned after vertex removal.
            let is_skinned = joint_indices.is_some();
            let mut mesh = Mesh { vertices, indices };

            if !is_skinned && mesh.vertices.len() > AUTO_DECIMATE_VERTEX_THRESHOLD {
                let ratio = AUTO_DECIMATE_VERTEX_THRESHOLD as f32 / mesh.vertices.len() as f32;
                log::info!(
                    "Auto-decimating mesh '{}' from {} to ~{} vertices (ratio {:.2})",
                    mesh_name.as_deref().unwrap_or("unnamed"),
                    mesh.vertices.len(),
                    AUTO_DECIMATE_VERTEX_THRESHOLD,
                    ratio,
                );
                mesh = simplify_mesh(&mesh, ratio);
            }

            // Post-process: merge duplicate vertices and reorder triangles for
            // GPU vertex-cache locality.
            mesh = deduplicate_vertices(&mesh);
            optimize_vertex_cache(&mut mesh);

            let bounds = MeshBounds::from_vertices(&mesh.vertices);

            scene_meshes.push(GltfMesh {
                mesh,
                material,
                name: mesh_name.clone(),
                bounds,
                joint_indices,
                joint_weights,
                albedo_tex_index,
                normal_tex_index,
                metallic_roughness_tex_index,
                ao_tex_index,
                emissive_tex_index,
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
        images,
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

    #[test]
    fn convert_rgb_to_rgba() {
        let data = gltf::image::Data {
            pixels: vec![255, 0, 0, 0, 255, 0, 0, 0, 255],
            format: gltf::image::Format::R8G8B8,
            width: 3,
            height: 1,
        };
        let rgba = convert_to_rgba8(&data);
        assert_eq!(rgba.len(), 12); // 3 pixels * 4 channels
        assert_eq!(&rgba[0..4], &[255, 0, 0, 255]); // Red
        assert_eq!(&rgba[4..8], &[0, 255, 0, 255]); // Green
        assert_eq!(&rgba[8..12], &[0, 0, 255, 255]); // Blue
    }

    #[test]
    fn convert_r8_to_rgba() {
        let data = gltf::image::Data {
            pixels: vec![128],
            format: gltf::image::Format::R8,
            width: 1,
            height: 1,
        };
        let rgba = convert_to_rgba8(&data);
        assert_eq!(rgba, vec![128, 128, 128, 255]);
    }

    #[test]
    fn convert_rgba_passthrough() {
        let pixels = vec![10, 20, 30, 40];
        let data = gltf::image::Data {
            pixels: pixels.clone(),
            format: gltf::image::Format::R8G8B8A8,
            width: 1,
            height: 1,
        };
        let rgba = convert_to_rgba8(&data);
        assert_eq!(rgba, pixels);
    }

    #[test]
    fn mesh_bounds_empty_returns_none() {
        let bounds = MeshBounds::from_vertices(&[]);
        assert!(bounds.is_none());
    }

    #[test]
    fn mesh_bounds_ground_offset() {
        let vertices = vec![
            Vertex {
                position: [-1.0, -0.5, 0.0],
                normal: [0.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0],
                uv: [0.0, 0.0],
            },
            Vertex {
                position: [1.0, 1.5, 0.0],
                normal: [0.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0],
                uv: [1.0, 1.0],
            },
        ];
        let bounds = MeshBounds::from_vertices(&vertices).unwrap();
        assert_eq!(bounds.min, [-1.0, -0.5, 0.0]);
        assert_eq!(bounds.max, [1.0, 1.5, 0.0]);
        // ground_offset should be -min_y = 0.5
        assert!((bounds.ground_offset() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mesh_bounds_already_grounded() {
        // Mesh with bottom at y=0 should have zero ground offset.
        let vertices = vec![
            Vertex {
                position: [0.0, 0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0],
                uv: [0.0, 0.0],
            },
            Vertex {
                position: [0.0, 2.0, 0.0],
                normal: [0.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0],
                uv: [0.0, 0.0],
            },
        ];
        let bounds = MeshBounds::from_vertices(&vertices).unwrap();
        assert!((bounds.ground_offset()).abs() < 1e-6);
    }
}
