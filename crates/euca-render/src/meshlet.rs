//! Meshlet data structures and offline meshletizer algorithm.
//!
//! Meshlets are fixed-size clusters of triangles (up to 64 per cluster) that
//! enable fine-grained GPU culling. Each meshlet can be individually
//! frustum-tested, backface-culled, and occlusion-tested, dramatically
//! reducing overdraw.
//!
//! The [`meshletize`] function partitions an indexed triangle mesh into
//! meshlets using a greedy adjacency-aware growth algorithm that maximizes
//! vertex reuse within each cluster.

use crate::vertex::Vertex;
use std::collections::{BinaryHeap, HashMap};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum triangles per meshlet (matches typical compute workgroup size).
pub const MAX_MESHLET_TRIANGLES: u32 = 64;

/// Maximum unique vertices per meshlet.
pub const MAX_MESHLET_VERTICES: u32 = 64;

// ---------------------------------------------------------------------------
// CPU-side meshlet descriptor
// ---------------------------------------------------------------------------

/// A cluster of triangles within a mesh.
///
/// Each meshlet references a contiguous range in [`MeshletMesh::vertices`]
/// (global vertex indices) and [`MeshletMesh::triangles`] (local 8-bit
/// indices into the meshlet's vertex list).
#[derive(Clone, Debug)]
pub struct Meshlet {
    /// Start index in [`MeshletMesh::vertices`].
    pub vertex_offset: u32,
    /// Number of unique vertices referenced by this meshlet.
    pub vertex_count: u32,
    /// Start index in [`MeshletMesh::triangles`].
    pub triangle_offset: u32,
    /// Number of triangles (each triangle is 3 consecutive `u8` indices).
    pub triangle_count: u32,
    /// AABB center in object space.
    pub aabb_center: [f32; 3],
    /// AABB half-extents in object space.
    pub aabb_half_extents: [f32; 3],
    /// Normal cone axis (unit vector, average of triangle normals).
    pub cone_axis: [f32; 3],
    /// Normal cone cutoff (cosine of half-angle). If all triangle normals
    /// are within this cone, backface culling can reject the entire meshlet.
    /// Set to `-1.0` when the cone is too wide for useful culling.
    pub cone_cutoff: f32,
}

// ---------------------------------------------------------------------------
// GPU-side meshlet (compute culling shader)
// ---------------------------------------------------------------------------

/// GPU-side meshlet for the compute culling shader.
///
/// Padded to exactly 64 bytes for aligned buffer access.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMeshlet {
    /// Offset into the global meshlet vertex buffer.
    pub vertex_offset: u32,
    /// Number of unique vertices in this meshlet.
    pub vertex_count: u32,
    /// Offset into the global meshlet triangle buffer.
    pub triangle_offset: u32,
    /// Number of triangles in this meshlet.
    pub triangle_count: u32,
    /// AABB center (xyz) + padding (w).
    pub aabb_center: [f32; 4],
    /// AABB half-extents (xyz) + padding (w).
    pub aabb_half_extents: [f32; 4],
    /// Normal cone axis (xyz) + cutoff (w).
    pub cone_axis_cutoff: [f32; 4],
}

impl Meshlet {
    /// Convert to the GPU-side representation.
    pub fn to_gpu(&self) -> GpuMeshlet {
        GpuMeshlet {
            vertex_offset: self.vertex_offset,
            vertex_count: self.vertex_count,
            triangle_offset: self.triangle_offset,
            triangle_count: self.triangle_count,
            aabb_center: [
                self.aabb_center[0],
                self.aabb_center[1],
                self.aabb_center[2],
                0.0,
            ],
            aabb_half_extents: [
                self.aabb_half_extents[0],
                self.aabb_half_extents[1],
                self.aabb_half_extents[2],
                0.0,
            ],
            cone_axis_cutoff: [
                self.cone_axis[0],
                self.cone_axis[1],
                self.cone_axis[2],
                self.cone_cutoff,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Meshlet mesh (result of meshletization)
// ---------------------------------------------------------------------------

/// Result of meshletizing a mesh.
///
/// Contains the meshlet descriptors, the global vertex index buffer, and the
/// packed local triangle index buffer.
pub struct MeshletMesh {
    /// The meshlet descriptors.
    pub meshlets: Vec<Meshlet>,
    /// Global vertex indices -- each meshlet references a contiguous range.
    /// These index into the original mesh's vertex array.
    pub vertices: Vec<u32>,
    /// Packed local triangle indices (3 bytes per triangle).
    /// Each byte is an index into the meshlet's vertex list (`0..vertex_count`).
    pub triangles: Vec<u8>,
}

impl MeshletMesh {
    /// Convert all meshlets to GPU-side representation.
    pub fn gpu_meshlets(&self) -> Vec<GpuMeshlet> {
        self.meshlets.iter().map(Meshlet::to_gpu).collect()
    }
}

// ---------------------------------------------------------------------------
// Meshletizer algorithm
// ---------------------------------------------------------------------------

/// Partition a triangle mesh into meshlets of up to `max_triangles` triangles
/// and `max_vertices` unique vertices.
///
/// Uses a greedy adjacency-aware growth algorithm:
/// 1. Build triangle adjacency (which triangles share edges).
/// 2. Start from an unvisited seed triangle.
/// 3. Grow the meshlet by adding adjacent triangles that share vertices
///    with the current meshlet, preferring triangles that reuse the most
///    existing vertices (locality optimization).
/// 4. Stop when the meshlet reaches `max_triangles` or `max_vertices`.
/// 5. Compute tight AABB and normal cone for the meshlet.
/// 6. Repeat until all triangles are assigned.
///
/// # Panics
///
/// Panics if `max_triangles` or `max_vertices` is zero, or if `max_vertices`
/// is less than 3.
pub fn meshletize(
    vertices: &[Vertex],
    indices: &[u32],
    max_triangles: u32,
    max_vertices: u32,
) -> MeshletMesh {
    assert!(max_triangles > 0, "max_triangles must be positive");
    assert!(max_vertices >= 3, "max_vertices must be at least 3");

    let triangle_count = indices.len() / 3;
    if triangle_count == 0 {
        return MeshletMesh {
            meshlets: Vec::new(),
            vertices: Vec::new(),
            triangles: Vec::new(),
        };
    }

    // -- Step 1: Build triangle adjacency via shared edges. --
    let adjacency = build_adjacency(indices, triangle_count);

    // -- Step 2-6: Greedy meshlet construction. --
    let mut assigned = vec![false; triangle_count];
    let mut result_meshlets = Vec::new();
    let mut result_vertices: Vec<u32> = Vec::new();
    let mut result_triangles: Vec<u8> = Vec::new();

    for seed in 0..triangle_count {
        if assigned[seed] {
            continue;
        }

        // Local state for this meshlet.
        let mut meshlet_global_verts: Vec<u32> = Vec::new();
        let mut vert_to_local: HashMap<u32, u8> = HashMap::new();
        let mut meshlet_tri_indices: Vec<[u8; 3]> = Vec::new();

        // Priority queue: (score, triangle_index). Higher score = more vertex reuse.
        let mut frontier = BinaryHeap::<(u32, usize)>::new();

        // Seed the meshlet with the first unvisited triangle.
        let added = try_add_triangle(
            seed,
            indices,
            &mut meshlet_global_verts,
            &mut vert_to_local,
            &mut meshlet_tri_indices,
            max_vertices,
        );
        debug_assert!(added, "Seed triangle should always fit in an empty meshlet");
        assigned[seed] = true;

        // Enqueue neighbors of the seed triangle.
        enqueue_neighbors(
            seed,
            indices,
            &adjacency,
            &assigned,
            &vert_to_local,
            &mut frontier,
        );

        // Grow the meshlet.
        while meshlet_tri_indices.len() < max_triangles as usize {
            // Pop the best candidate (highest vertex reuse score).
            let candidate = loop {
                match frontier.pop() {
                    None => break None,
                    Some((_, tri)) if assigned[tri] => continue,
                    Some((_, tri)) => break Some(tri),
                }
            };

            let tri = match candidate {
                Some(t) => t,
                None => break,
            };

            // Check if this triangle fits (respects vertex limit).
            if !try_add_triangle(
                tri,
                indices,
                &mut meshlet_global_verts,
                &mut vert_to_local,
                &mut meshlet_tri_indices,
                max_vertices,
            ) {
                // Triangle would exceed vertex limit -- skip it.
                // Don't mark as assigned; another meshlet will take it.
                continue;
            }

            assigned[tri] = true;
            enqueue_neighbors(
                tri,
                indices,
                &adjacency,
                &assigned,
                &vert_to_local,
                &mut frontier,
            );
        }

        // -- Finalize meshlet: compute AABB and normal cone. --
        let vertex_offset = result_vertices.len() as u32;
        let triangle_offset = result_triangles.len() as u32;
        let vertex_count = meshlet_global_verts.len() as u32;
        let triangle_count_val = meshlet_tri_indices.len() as u32;

        let (aabb_center, aabb_half_extents) = compute_aabb(vertices, &meshlet_global_verts);
        let (cone_axis, cone_cutoff) =
            compute_normal_cone(vertices, &meshlet_tri_indices, &meshlet_global_verts);

        result_vertices.extend_from_slice(&meshlet_global_verts);
        for tri_local in &meshlet_tri_indices {
            result_triangles.push(tri_local[0]);
            result_triangles.push(tri_local[1]);
            result_triangles.push(tri_local[2]);
        }

        result_meshlets.push(Meshlet {
            vertex_offset,
            vertex_count,
            triangle_offset,
            triangle_count: triangle_count_val,
            aabb_center,
            aabb_half_extents,
            cone_axis,
            cone_cutoff,
        });
    }

    MeshletMesh {
        meshlets: result_meshlets,
        vertices: result_vertices,
        triangles: result_triangles,
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build triangle adjacency: for each triangle, which other triangles share
/// at least one edge.
fn build_adjacency(indices: &[u32], triangle_count: usize) -> Vec<Vec<usize>> {
    // Map each sorted edge to the list of triangles that contain it.
    let mut edge_to_tris: HashMap<(u32, u32), Vec<usize>> = HashMap::new();

    for tri in 0..triangle_count {
        let base = tri * 3;
        let v = [indices[base], indices[base + 1], indices[base + 2]];
        for &(a, b) in &[(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
            let edge = if a < b { (a, b) } else { (b, a) };
            edge_to_tris.entry(edge).or_default().push(tri);
        }
    }

    let mut adjacency = vec![Vec::new(); triangle_count];
    for neighbors in edge_to_tris.values() {
        for (i, &t0) in neighbors.iter().enumerate() {
            for &t1 in &neighbors[i + 1..] {
                adjacency[t0].push(t1);
                adjacency[t1].push(t0);
            }
        }
    }

    // Deduplicate (cheap insurance for non-manifold meshes where a triangle
    // pair might be recorded more than once).
    for adj in &mut adjacency {
        adj.sort_unstable();
        adj.dedup();
    }

    adjacency
}

/// Try to add a triangle to the current meshlet. Returns `true` if
/// the triangle was added, `false` if it would exceed `max_vertices`.
fn try_add_triangle(
    tri: usize,
    indices: &[u32],
    meshlet_global_verts: &mut Vec<u32>,
    vert_to_local: &mut HashMap<u32, u8>,
    meshlet_tri_indices: &mut Vec<[u8; 3]>,
    max_vertices: u32,
) -> bool {
    let base = tri * 3;
    let v = [indices[base], indices[base + 1], indices[base + 2]];

    // Count how many new vertices this triangle would introduce.
    let new_verts = v
        .iter()
        .filter(|&&vi| !vert_to_local.contains_key(&vi))
        .count();
    if meshlet_global_verts.len() + new_verts > max_vertices as usize {
        return false;
    }

    // Add vertices and build local indices.
    let mut local = [0u8; 3];
    for (i, &vi) in v.iter().enumerate() {
        local[i] = *vert_to_local.entry(vi).or_insert_with(|| {
            let idx = meshlet_global_verts.len() as u8;
            meshlet_global_verts.push(vi);
            idx
        });
    }

    meshlet_tri_indices.push(local);
    true
}

/// Score and enqueue the neighbors of a triangle into the frontier.
///
/// The score is the number of the neighbor's vertices already present in the
/// current meshlet (0-3). Higher scores are popped first, maximizing vertex
/// reuse and spatial locality.
fn enqueue_neighbors(
    tri: usize,
    indices: &[u32],
    adjacency: &[Vec<usize>],
    assigned: &[bool],
    vert_to_local: &HashMap<u32, u8>,
    frontier: &mut BinaryHeap<(u32, usize)>,
) {
    for &neighbor in &adjacency[tri] {
        if assigned[neighbor] {
            continue;
        }
        let base = neighbor * 3;
        let v = [indices[base], indices[base + 1], indices[base + 2]];
        let reuse = v.iter().filter(|vi| vert_to_local.contains_key(vi)).count() as u32;
        frontier.push((reuse, neighbor));
    }
}

/// Compute AABB center and half-extents for a set of global vertex indices.
fn compute_aabb(vertices: &[Vertex], global_indices: &[u32]) -> ([f32; 3], [f32; 3]) {
    debug_assert!(!global_indices.is_empty());

    let first = vertices[global_indices[0] as usize].position;
    let mut min = first;
    let mut max = first;

    for &gi in &global_indices[1..] {
        let p = vertices[gi as usize].position;
        for axis in 0..3 {
            if p[axis] < min[axis] {
                min[axis] = p[axis];
            }
            if p[axis] > max[axis] {
                max[axis] = p[axis];
            }
        }
    }

    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let half = [
        (max[0] - min[0]) * 0.5,
        (max[1] - min[1]) * 0.5,
        (max[2] - min[2]) * 0.5,
    ];

    (center, half)
}

/// Compute the normal cone axis and cutoff for a meshlet's triangles.
///
/// Returns `(axis, cutoff)` where `axis` is the average geometric triangle
/// normal (unit vector) and `cutoff` is the cosine of the half-angle of the
/// tightest cone containing all triangle normals. If the cone is too wide
/// (any normal points away from the average), `cutoff` is `-1.0`.
fn compute_normal_cone(
    vertices: &[Vertex],
    meshlet_tri_indices: &[[u8; 3]],
    meshlet_global_verts: &[u32],
) -> ([f32; 3], f32) {
    if meshlet_tri_indices.is_empty() {
        return ([0.0, 0.0, 1.0], -1.0);
    }

    // Compute per-triangle geometric normals and accumulate the average.
    let mut normals = Vec::with_capacity(meshlet_tri_indices.len());
    let mut avg = [0.0f32; 3];

    for tri_local in meshlet_tri_indices {
        let p0 = vertices[meshlet_global_verts[tri_local[0] as usize] as usize].position;
        let p1 = vertices[meshlet_global_verts[tri_local[1] as usize] as usize].position;
        let p2 = vertices[meshlet_global_verts[tri_local[2] as usize] as usize].position;

        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        let len = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        if len < 1e-12 {
            // Degenerate triangle -- skip.
            continue;
        }

        let n = [cross[0] / len, cross[1] / len, cross[2] / len];
        normals.push(n);
        avg[0] += n[0];
        avg[1] += n[1];
        avg[2] += n[2];
    }

    if normals.is_empty() {
        return ([0.0, 0.0, 1.0], -1.0);
    }

    let avg_len = (avg[0] * avg[0] + avg[1] * avg[1] + avg[2] * avg[2]).sqrt();
    if avg_len < 1e-12 {
        // Normals cancel out -- cone is too wide.
        return ([0.0, 0.0, 1.0], -1.0);
    }

    let axis = [avg[0] / avg_len, avg[1] / avg_len, avg[2] / avg_len];

    // Find the minimum dot product between any triangle normal and the axis.
    let mut min_dot = 1.0f32;
    for n in &normals {
        let dot = n[0] * axis[0] + n[1] * axis[1] + n[2] * axis[2];
        if dot < min_dot {
            min_dot = dot;
        }
    }

    // If any normal is more than 90 degrees from the axis, the cone is useless.
    let cutoff = if min_dot < 0.0 { -1.0 } else { min_dot };

    (axis, cutoff)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;

    #[test]
    fn test_gpu_meshlet_size() {
        assert_eq!(
            std::mem::size_of::<GpuMeshlet>(),
            64,
            "GpuMeshlet must be exactly 64 bytes for aligned GPU access"
        );
    }

    #[test]
    fn test_meshletize_cube() {
        let cube = Mesh::cube();
        let result = meshletize(
            &cube.vertices,
            &cube.indices,
            MAX_MESHLET_TRIANGLES,
            MAX_MESHLET_VERTICES,
        );

        // A cube has 12 triangles. They should all be covered.
        let total_tris: u32 = result.meshlets.iter().map(|m| m.triangle_count).sum();
        assert_eq!(total_tris, 12, "All 12 cube triangles must be assigned");

        // Verify no triangle appears in multiple meshlets by checking total
        // local index count matches.
        assert_eq!(
            result.triangles.len(),
            (total_tris * 3) as usize,
            "Triangle index buffer length must be 3 * total triangles"
        );

        // Verify all global vertex indices and local triangle indices are valid.
        for meshlet in &result.meshlets {
            let v_start = meshlet.vertex_offset as usize;
            let v_end = v_start + meshlet.vertex_count as usize;
            for &vi in &result.vertices[v_start..v_end] {
                assert!(
                    (vi as usize) < cube.vertices.len(),
                    "Global vertex index {vi} out of range"
                );
            }

            let t_start = meshlet.triangle_offset as usize;
            let t_end = t_start + (meshlet.triangle_count as usize) * 3;
            for &li in &result.triangles[t_start..t_end] {
                assert!(
                    (li as u32) < meshlet.vertex_count,
                    "Local index {li} >= vertex_count {}",
                    meshlet.vertex_count
                );
            }
        }
    }

    #[test]
    fn test_meshletize_respects_limits() {
        // Create a sphere with enough triangles to require multiple meshlets.
        let sphere = Mesh::sphere(1.0, 16, 32);
        let max_tri = 16u32;
        let max_vert = 20u32;
        let result = meshletize(&sphere.vertices, &sphere.indices, max_tri, max_vert);

        for (i, meshlet) in result.meshlets.iter().enumerate() {
            assert!(
                meshlet.triangle_count <= max_tri,
                "Meshlet {i} has {} triangles, exceeding limit {max_tri}",
                meshlet.triangle_count
            );
            assert!(
                meshlet.vertex_count <= max_vert,
                "Meshlet {i} has {} vertices, exceeding limit {max_vert}",
                meshlet.vertex_count
            );
        }

        // Verify all triangles are covered.
        let total_tris: u32 = result.meshlets.iter().map(|m| m.triangle_count).sum();
        let expected = (sphere.indices.len() / 3) as u32;
        assert_eq!(total_tris, expected, "All triangles must be assigned");
    }

    #[test]
    fn test_meshlet_aabb() {
        let cube = Mesh::cube();
        let result = meshletize(
            &cube.vertices,
            &cube.indices,
            MAX_MESHLET_TRIANGLES,
            MAX_MESHLET_VERTICES,
        );

        for (i, meshlet) in result.meshlets.iter().enumerate() {
            let v_start = meshlet.vertex_offset as usize;
            let v_end = v_start + meshlet.vertex_count as usize;
            let aabb_min = [
                meshlet.aabb_center[0] - meshlet.aabb_half_extents[0],
                meshlet.aabb_center[1] - meshlet.aabb_half_extents[1],
                meshlet.aabb_center[2] - meshlet.aabb_half_extents[2],
            ];
            let aabb_max = [
                meshlet.aabb_center[0] + meshlet.aabb_half_extents[0],
                meshlet.aabb_center[1] + meshlet.aabb_half_extents[1],
                meshlet.aabb_center[2] + meshlet.aabb_half_extents[2],
            ];

            for &vi in &result.vertices[v_start..v_end] {
                let p = cube.vertices[vi as usize].position;
                for axis in 0..3 {
                    assert!(
                        p[axis] >= aabb_min[axis] - 1e-5 && p[axis] <= aabb_max[axis] + 1e-5,
                        "Meshlet {i}: vertex position {p:?} outside AABB on axis {axis}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_meshlet_normal_cone() {
        let cube = Mesh::cube();
        let result = meshletize(
            &cube.vertices,
            &cube.indices,
            MAX_MESHLET_TRIANGLES,
            MAX_MESHLET_VERTICES,
        );

        for meshlet in &result.meshlets {
            let axis = meshlet.cone_axis;
            let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            // Axis should be unit length (or zero if degenerate).
            assert!(
                (len - 1.0).abs() < 1e-4 || len < 1e-6,
                "Cone axis should be unit length, got {len}"
            );
            // Cutoff should be in [-1, 1].
            assert!(
                meshlet.cone_cutoff >= -1.0 && meshlet.cone_cutoff <= 1.0,
                "Cone cutoff out of range: {}",
                meshlet.cone_cutoff
            );
        }
    }

    #[test]
    fn test_empty_mesh() {
        let result = meshletize(&[], &[], MAX_MESHLET_TRIANGLES, MAX_MESHLET_VERTICES);
        assert!(result.meshlets.is_empty());
        assert!(result.vertices.is_empty());
        assert!(result.triangles.is_empty());
    }

    #[test]
    fn test_single_triangle() {
        let vertices = vec![
            Vertex {
                position: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                tangent: [1.0, 0.0, 0.0],
                uv: [0.0, 0.0],
            },
            Vertex {
                position: [1.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                tangent: [1.0, 0.0, 0.0],
                uv: [1.0, 0.0],
            },
            Vertex {
                position: [0.0, 1.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                tangent: [1.0, 0.0, 0.0],
                uv: [0.0, 1.0],
            },
        ];
        let indices = vec![0, 1, 2];

        let result = meshletize(
            &vertices,
            &indices,
            MAX_MESHLET_TRIANGLES,
            MAX_MESHLET_VERTICES,
        );
        assert_eq!(result.meshlets.len(), 1, "One triangle = one meshlet");
        assert_eq!(result.meshlets[0].triangle_count, 1);
        assert_eq!(result.meshlets[0].vertex_count, 3);
        assert_eq!(result.triangles.len(), 3);
        assert_eq!(result.vertices.len(), 3);
    }
}
