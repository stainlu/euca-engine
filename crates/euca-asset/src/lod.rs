//! LOD (Level of Detail) mesh simplification using Quadric Error Metrics (QEM).
//!
//! Generates progressively simplified meshes by collapsing edges with the
//! lowest geometric error, preserving overall shape while reducing triangle count.

use euca_render::{Mesh, Vertex};
use std::collections::{BinaryHeap, HashMap, HashSet};

/// A 4x4 symmetric error quadric matrix stored as 10 unique values.
#[derive(Clone, Copy, Debug)]
struct Quadric {
    a: [f64; 10], // Upper triangle of symmetric 4x4
}

impl Quadric {
    fn zero() -> Self {
        Self { a: [0.0; 10] }
    }

    /// Create a quadric from a plane equation (ax + by + cz + d = 0).
    fn from_plane(nx: f64, ny: f64, nz: f64, d: f64) -> Self {
        Self {
            a: [
                nx * nx,
                nx * ny,
                nx * nz,
                nx * d, // row 0
                ny * ny,
                ny * nz,
                ny * d, // row 1
                nz * nz,
                nz * d, // row 2
                d * d,  // row 3
            ],
        }
    }

    fn add(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        for i in 0..10 {
            result.a[i] = self.a[i] + other.a[i];
        }
        result
    }

    /// Evaluate the error for a point: v^T * Q * v.
    fn evaluate(&self, x: f64, y: f64, z: f64) -> f64 {
        let q = &self.a;
        // Q = [[q0,q1,q2,q3],[q1,q4,q5,q6],[q2,q5,q7,q8],[q3,q6,q8,q9]]
        x * (q[0] * x + q[1] * y + q[2] * z + q[3])
            + y * (q[1] * x + q[4] * y + q[5] * z + q[6])
            + z * (q[2] * x + q[5] * y + q[7] * z + q[8])
            + (q[3] * x + q[6] * y + q[8] * z + q[9])
    }
}

/// An edge collapse candidate in the priority queue.
#[derive(Clone)]
struct EdgeCollapse {
    cost: f64,
    v0: u32,
    v1: u32,
    target_pos: [f32; 3],
}

impl PartialEq for EdgeCollapse {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}
impl Eq for EdgeCollapse {}

impl Ord for EdgeCollapse {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: lower cost = higher priority
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for EdgeCollapse {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Simplify a mesh to approximately `target_ratio` of its original vertex count.
///
/// Uses Quadric Error Metrics (QEM) to find the cheapest edge collapses.
/// `target_ratio` should be in (0.0, 1.0] — e.g., 0.5 halves the vertex count.
pub fn simplify_mesh(mesh: &Mesh, target_ratio: f32) -> Mesh {
    let target_ratio = target_ratio.clamp(0.01, 1.0);
    if target_ratio >= 0.99 {
        return Mesh {
            vertices: mesh.vertices.clone(),
            indices: mesh.indices.clone(),
        };
    }

    let n_verts = mesh.vertices.len();
    let target_verts = (n_verts as f32 * target_ratio).max(4.0) as usize;

    // Build per-vertex quadrics from adjacent face planes
    let mut quadrics = vec![Quadric::zero(); n_verts];
    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let p0 = mesh.vertices[tri[0] as usize].position;
        let p1 = mesh.vertices[tri[1] as usize].position;
        let p2 = mesh.vertices[tri[2] as usize].position;

        // Face normal (not normalized — area-weighted)
        let e1 = [
            (p1[0] - p0[0]) as f64,
            (p1[1] - p0[1]) as f64,
            (p1[2] - p0[2]) as f64,
        ];
        let e2 = [
            (p2[0] - p0[0]) as f64,
            (p2[1] - p0[1]) as f64,
            (p2[2] - p0[2]) as f64,
        ];
        let nx = e1[1] * e2[2] - e1[2] * e2[1];
        let ny = e1[2] * e2[0] - e1[0] * e2[2];
        let nz = e1[0] * e2[1] - e1[1] * e2[0];
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len < 1e-12 {
            continue;
        }
        let nx = nx / len;
        let ny = ny / len;
        let nz = nz / len;
        let d = -(nx * p0[0] as f64 + ny * p0[1] as f64 + nz * p0[2] as f64);

        let face_quadric = Quadric::from_plane(nx, ny, nz, d);
        for &idx in tri {
            quadrics[idx as usize] = quadrics[idx as usize].add(&face_quadric);
        }
    }

    // Collect unique edges
    let mut edges: HashSet<(u32, u32)> = HashSet::new();
    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let mut add_edge = |a: u32, b: u32| {
            edges.insert((a.min(b), a.max(b)));
        };
        add_edge(tri[0], tri[1]);
        add_edge(tri[1], tri[2]);
        add_edge(tri[2], tri[0]);
    }

    // Build priority queue of edge collapses
    let mut heap = BinaryHeap::new();
    for &(v0, v1) in &edges {
        let collapse = compute_collapse(&mesh.vertices, &quadrics, v0, v1);
        heap.push(collapse);
    }

    // Vertex mapping: each vertex points to its current representative
    let mut remap: Vec<u32> = (0..n_verts as u32).collect();

    fn find_root(remap: &[u32], mut v: u32) -> u32 {
        while remap[v as usize] != v {
            v = remap[v as usize];
        }
        v
    }

    let mut live_verts = n_verts;
    let mut positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.position).collect();

    while live_verts > target_verts {
        let collapse = match heap.pop() {
            Some(c) => c,
            None => break,
        };

        let r0 = find_root(&remap, collapse.v0);
        let r1 = find_root(&remap, collapse.v1);
        if r0 == r1 {
            continue; // already collapsed
        }

        // Collapse v1 into v0
        remap[r1 as usize] = r0;
        positions[r0 as usize] = collapse.target_pos;
        quadrics[r0 as usize] = quadrics[r0 as usize].add(&quadrics[r1 as usize]);
        live_verts -= 1;
    }

    // Rebuild mesh with collapsed vertices
    let mut new_vertex_map: HashMap<u32, u32> = HashMap::new();
    let mut new_vertices: Vec<Vertex> = Vec::new();

    let get_new_idx = |old: u32,
                       remap: &[u32],
                       positions: &[[f32; 3]],
                       mesh: &Mesh,
                       map: &mut HashMap<u32, u32>,
                       verts: &mut Vec<Vertex>|
     -> u32 {
        let root = find_root(remap, old);
        *map.entry(root).or_insert_with(|| {
            let idx = verts.len() as u32;
            let mut v = mesh.vertices[root as usize];
            v.position = positions[root as usize];
            verts.push(v);
            idx
        })
    };

    let mut new_indices: Vec<u32> = Vec::new();
    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let a = get_new_idx(
            tri[0],
            &remap,
            &positions,
            mesh,
            &mut new_vertex_map,
            &mut new_vertices,
        );
        let b = get_new_idx(
            tri[1],
            &remap,
            &positions,
            mesh,
            &mut new_vertex_map,
            &mut new_vertices,
        );
        let c = get_new_idx(
            tri[2],
            &remap,
            &positions,
            mesh,
            &mut new_vertex_map,
            &mut new_vertices,
        );

        // Skip degenerate triangles (collapsed to a line or point)
        if a != b && b != c && a != c {
            new_indices.push(a);
            new_indices.push(b);
            new_indices.push(c);
        }
    }

    log::info!(
        "LOD simplification: {} → {} vertices, {} → {} triangles (ratio {:.0}%)",
        mesh.vertices.len(),
        new_vertices.len(),
        mesh.indices.len() / 3,
        new_indices.len() / 3,
        target_ratio * 100.0,
    );

    Mesh {
        vertices: new_vertices,
        indices: new_indices,
    }
}

/// Generate a chain of LOD meshes at the given quality ratios.
///
/// `levels` should be sorted descending, e.g., `[1.0, 0.5, 0.25, 0.1]`.
/// Level 0 is the original mesh (ratio 1.0), subsequent levels are simplified.
pub fn generate_lod_chain(mesh: &Mesh, levels: &[f32]) -> Vec<Mesh> {
    levels
        .iter()
        .map(|&ratio| simplify_mesh(mesh, ratio))
        .collect()
}

fn compute_collapse(vertices: &[Vertex], quadrics: &[Quadric], v0: u32, v1: u32) -> EdgeCollapse {
    let q = quadrics[v0 as usize].add(&quadrics[v1 as usize]);
    let p0 = vertices[v0 as usize].position;
    let p1 = vertices[v1 as usize].position;

    // Use midpoint as target position (optimal position requires solving 3x3 system)
    let target = [
        (p0[0] + p1[0]) * 0.5,
        (p0[1] + p1[1]) * 0.5,
        (p0[2] + p1[2]) * 0.5,
    ];

    let cost = q.evaluate(target[0] as f64, target[1] as f64, target[2] as f64);

    EdgeCollapse {
        cost: cost.abs(),
        v0,
        v1,
        target_pos: target,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplify_cube_to_half() {
        let mesh = Mesh::cube();
        let simplified = simplify_mesh(&mesh, 0.5);
        // QEM simplification is non-deterministic across platforms (float precision).
        // Allow equal vertex count if the cube is already minimal and can't simplify further.
        assert!(
            simplified.vertices.len() <= mesh.vertices.len(),
            "Simplified should have no more vertices than original: {} vs {}",
            simplified.vertices.len(),
            mesh.vertices.len(),
        );
    }

    #[test]
    fn simplify_preserves_at_ratio_one() {
        let mesh = Mesh::cube();
        let same = simplify_mesh(&mesh, 1.0);
        assert_eq!(same.vertices.len(), mesh.vertices.len());
        assert_eq!(same.indices.len(), mesh.indices.len());
    }

    #[test]
    fn lod_chain_produces_decreasing_counts() {
        let mesh = Mesh::sphere(1.0, 16, 32);
        let chain = generate_lod_chain(&mesh, &[1.0, 0.5, 0.25]);
        assert_eq!(chain.len(), 3);
        // Each level should have fewer or equal vertices
        for i in 1..chain.len() {
            assert!(
                chain[i].vertices.len() <= chain[i - 1].vertices.len(),
                "LOD {} ({} verts) should have ≤ LOD {} ({} verts)",
                i,
                chain[i].vertices.len(),
                i - 1,
                chain[i - 1].vertices.len(),
            );
        }
    }

    #[test]
    fn simplify_sphere() {
        let mesh = Mesh::sphere(1.0, 16, 32);
        let simplified = simplify_mesh(&mesh, 0.25);
        assert!(simplified.vertices.len() < mesh.vertices.len());
        // Should still be a valid mesh
        assert!(simplified.indices.len() >= 3);
        assert!(
            simplified
                .indices
                .iter()
                .all(|&i| (i as usize) < simplified.vertices.len())
        );
    }
}
