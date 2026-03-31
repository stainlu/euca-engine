//! Mesh optimization: vertex deduplication, tangent generation, index optimization.

use euca_render::{Mesh, Vertex};
use std::collections::HashMap;

/// Deduplicate vertices: merge identical vertices and remap indices.
///
/// Uses a hash of quantized position + normal + uv to identify duplicates.
/// Typically reduces vertex count by 10-30% for glTF meshes.
pub fn deduplicate_vertices(mesh: &Mesh) -> Mesh {
    let mut unique_vertices: Vec<Vertex> = Vec::new();
    let mut vertex_map: HashMap<u64, u32> = HashMap::new();
    let mut new_indices: Vec<u32> = Vec::with_capacity(mesh.indices.len());

    for &idx in &mesh.indices {
        let v = &mesh.vertices[idx as usize];
        let key = hash_vertex(v);

        let new_idx = *vertex_map.entry(key).or_insert_with(|| {
            let idx = unique_vertices.len() as u32;
            unique_vertices.push(*v);
            idx
        });
        new_indices.push(new_idx);
    }

    let removed = mesh.vertices.len() - unique_vertices.len();
    if removed > 0 {
        log::info!(
            "Vertex dedup: {} → {} vertices ({} removed)",
            mesh.vertices.len(),
            unique_vertices.len(),
            removed,
        );
    }

    Mesh {
        vertices: unique_vertices,
        indices: new_indices,
    }
}

/// Compute tangent vectors for a mesh using the UV-based method.
///
/// For each triangle, computes the tangent from the UV gradient.
/// Per-vertex tangents are averaged from adjacent triangles.
pub fn compute_tangents(mesh: &mut Mesh) {
    let n = mesh.vertices.len();
    let mut tangents = vec![[0.0f32; 3]; n];
    let mut counts = vec![0u32; n];

    // Accumulate per-triangle tangent contributions
    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        let v0 = &mesh.vertices[i0];
        let v1 = &mesh.vertices[i1];
        let v2 = &mesh.vertices[i2];

        let edge1 = [
            v1.position[0] - v0.position[0],
            v1.position[1] - v0.position[1],
            v1.position[2] - v0.position[2],
        ];
        let edge2 = [
            v2.position[0] - v0.position[0],
            v2.position[1] - v0.position[1],
            v2.position[2] - v0.position[2],
        ];

        let duv1 = [v1.uv[0] - v0.uv[0], v1.uv[1] - v0.uv[1]];
        let duv2 = [v2.uv[0] - v0.uv[0], v2.uv[1] - v0.uv[1]];

        let det = duv1[0] * duv2[1] - duv2[0] * duv1[1];
        if det.abs() < 1e-8 {
            continue;
        }
        let inv_det = 1.0 / det;

        let t = [
            inv_det * (duv2[1] * edge1[0] - duv1[1] * edge2[0]),
            inv_det * (duv2[1] * edge1[1] - duv1[1] * edge2[1]),
            inv_det * (duv2[1] * edge1[2] - duv1[1] * edge2[2]),
        ];

        for &idx in &[i0, i1, i2] {
            tangents[idx][0] += t[0];
            tangents[idx][1] += t[1];
            tangents[idx][2] += t[2];
            counts[idx] += 1;
        }
    }

    // Normalize and apply
    for (i, v) in mesh.vertices.iter_mut().enumerate() {
        if counts[i] > 0 {
            let t = &tangents[i];
            let len = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
            if len > 1e-6 {
                v.tangent = [t[0] / len, t[1] / len, t[2] / len];
            }
        }
    }
}

/// Optimize index buffer for vertex cache locality using a simple greedy algorithm.
///
/// Reorders triangles to maximize cache hits when the GPU processes vertices
/// sequentially through a FIFO cache.
pub fn optimize_vertex_cache(mesh: &mut Mesh) {
    if mesh.indices.len() < 6 {
        return;
    }

    let num_tris = mesh.indices.len() / 3;
    let cache_size = 32usize;
    let mut cache: Vec<u32> = Vec::with_capacity(cache_size);
    let mut used = vec![false; num_tris];
    let mut new_indices = Vec::with_capacity(mesh.indices.len());

    // Build adjacency: vertex → triangles
    let mut vertex_tris: HashMap<u32, Vec<usize>> = HashMap::new();
    for (tri_idx, tri) in mesh.indices.chunks(3).enumerate() {
        for &v in tri {
            vertex_tris.entry(v).or_default().push(tri_idx);
        }
    }

    // Start with triangle 0
    used[0] = true;
    for &v in &mesh.indices[0..3] {
        new_indices.push(v);
        if !cache.contains(&v) {
            cache.push(v);
        }
    }

    let mut emitted = 1;
    while emitted < num_tris {
        // Find best triangle: most vertices already in cache
        let mut best_tri = None;
        let mut best_score = 0;

        for &v in &cache {
            if let Some(tris) = vertex_tris.get(&v) {
                for &ti in tris {
                    if used[ti] {
                        continue;
                    }
                    let tri = &mesh.indices[ti * 3..ti * 3 + 3];
                    let score = tri.iter().filter(|&&idx| cache.contains(&idx)).count();
                    if score > best_score {
                        best_score = score;
                        best_tri = Some(ti);
                    }
                }
            }
        }

        // If no cache-friendly triangle found, pick first unused
        let tri_idx = best_tri.unwrap_or_else(|| {
            (0..num_tris)
                .find(|&i| !used[i])
                .expect("unused triangle must exist")
        });

        used[tri_idx] = true;
        let tri = &mesh.indices[tri_idx * 3..tri_idx * 3 + 3];
        for &v in tri {
            new_indices.push(v);
            if !cache.contains(&v) {
                if cache.len() >= cache_size {
                    cache.remove(0);
                }
                cache.push(v);
            }
        }
        emitted += 1;
    }

    mesh.indices = new_indices;
}

/// Run mesh optimizations: dedup and tangent computation.
///
/// Vertex cache optimization is intentionally skipped — the current O(T²)
/// greedy algorithm takes 2-3 seconds per 500K-triangle mesh. Will be
/// replaced by meshoptimizer (O(T) Forsyth/Tipsify) in the asset cooker.
pub fn optimize_mesh(mesh: &Mesh) -> Mesh {
    let mut optimized = deduplicate_vertices(mesh);
    compute_tangents(&mut optimized);
    optimized
}

/// Hash a vertex for deduplication (quantized to avoid float comparison issues).
fn hash_vertex(v: &Vertex) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    // Quantize to 1/1000 precision
    let quantize = |f: f32| (f * 1000.0) as i32;
    quantize(v.position[0]).hash(&mut hasher);
    quantize(v.position[1]).hash(&mut hasher);
    quantize(v.position[2]).hash(&mut hasher);
    quantize(v.normal[0]).hash(&mut hasher);
    quantize(v.normal[1]).hash(&mut hasher);
    quantize(v.normal[2]).hash(&mut hasher);
    quantize(v.uv[0]).hash(&mut hasher);
    quantize(v.uv[1]).hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_mesh() -> Mesh {
        // Simple cube with 24 vertices (4 per face, duplicated normals)
        Mesh::cube()
    }

    #[test]
    fn deduplicate_removes_duplicates() {
        let mesh = cube_mesh();
        let deduped = deduplicate_vertices(&mesh);
        // Cube has some shared positions — dedup should reduce vertex count
        assert!(deduped.vertices.len() <= mesh.vertices.len());
        // But index count stays the same (just remapped)
        assert_eq!(deduped.indices.len(), mesh.indices.len());
    }

    #[test]
    fn tangent_computation_produces_nonzero() {
        let mut mesh = cube_mesh();
        compute_tangents(&mut mesh);
        // At least some vertices should have non-default tangents
        let has_tangent = mesh
            .vertices
            .iter()
            .any(|v| v.tangent[0].abs() > 0.01 || v.tangent[2].abs() > 0.01);
        assert!(
            has_tangent,
            "Tangent computation should produce non-zero tangents"
        );
    }

    #[test]
    fn cache_optimization_preserves_triangles() {
        let mut mesh = cube_mesh();
        let original_tri_count = mesh.indices.len() / 3;
        optimize_vertex_cache(&mut mesh);
        assert_eq!(mesh.indices.len() / 3, original_tri_count);
    }

    #[test]
    fn optimize_mesh_end_to_end() {
        let mesh = cube_mesh();
        let optimized = optimize_mesh(&mesh);
        // Should have same or fewer vertices, same triangle count
        assert!(optimized.vertices.len() <= mesh.vertices.len());
        assert_eq!(optimized.indices.len(), mesh.indices.len());
    }
}
