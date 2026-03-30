//! Procedural mesh generation for MOBA creep types.
//!
//! Provides helper primitives (`make_box`, `make_sphere`, `make_cylinder`)
//! and composite creep mesh builders for melee, ranged, and siege creeps.
//! Each builder returns a [`Mesh`] ready for GPU upload.
//!
//! Designed for visual clarity at typical MOBA camera distances (20-40 units
//! above ground). Creeps are assembled from simple geometric parts so they
//! read well as distinct silhouettes.

use crate::Mesh;
use crate::vertex::Vertex;

/// Offset applied to every vertex position in a part.
type Offset = [f32; 3];

// ── Primitive helpers ───────────────────────────────────────────────────────

/// Generate a box (rectangular cuboid) centered at `offset` with the given
/// half-extents along each axis.
///
/// `w` = X extent, `h` = Y extent, `d` = Z extent.
/// Returns vertex and index data ready for combining via [`combine_meshes`].
pub fn make_box(w: f32, h: f32, d: f32, offset: Offset) -> (Vec<Vertex>, Vec<u32>) {
    let hw = w / 2.0;
    let hh = h / 2.0;
    let hd = d / 2.0;
    let [ox, oy, oz] = offset;

    let face = |positions: [[f32; 3]; 4], normal: [f32; 3]| -> [Vertex; 4] {
        let tangent = [
            positions[1][0] - positions[0][0],
            positions[1][1] - positions[0][1],
            positions[1][2] - positions[0][2],
        ];
        [
            Vertex {
                position: positions[0],
                normal,
                tangent,
                uv: [0.0, 0.0],
            },
            Vertex {
                position: positions[1],
                normal,
                tangent,
                uv: [1.0, 0.0],
            },
            Vertex {
                position: positions[2],
                normal,
                tangent,
                uv: [1.0, 1.0],
            },
            Vertex {
                position: positions[3],
                normal,
                tangent,
                uv: [0.0, 1.0],
            },
        ]
    };

    let mut vertices = Vec::with_capacity(24);

    // Front (+Z)
    vertices.extend_from_slice(&face(
        [
            [ox - hw, oy - hh, oz + hd],
            [ox + hw, oy - hh, oz + hd],
            [ox + hw, oy + hh, oz + hd],
            [ox - hw, oy + hh, oz + hd],
        ],
        [0.0, 0.0, 1.0],
    ));
    // Back (-Z)
    vertices.extend_from_slice(&face(
        [
            [ox + hw, oy - hh, oz - hd],
            [ox - hw, oy - hh, oz - hd],
            [ox - hw, oy + hh, oz - hd],
            [ox + hw, oy + hh, oz - hd],
        ],
        [0.0, 0.0, -1.0],
    ));
    // Top (+Y)
    vertices.extend_from_slice(&face(
        [
            [ox - hw, oy + hh, oz + hd],
            [ox + hw, oy + hh, oz + hd],
            [ox + hw, oy + hh, oz - hd],
            [ox - hw, oy + hh, oz - hd],
        ],
        [0.0, 1.0, 0.0],
    ));
    // Bottom (-Y)
    vertices.extend_from_slice(&face(
        [
            [ox - hw, oy - hh, oz - hd],
            [ox + hw, oy - hh, oz - hd],
            [ox + hw, oy - hh, oz + hd],
            [ox - hw, oy - hh, oz + hd],
        ],
        [0.0, -1.0, 0.0],
    ));
    // Right (+X)
    vertices.extend_from_slice(&face(
        [
            [ox + hw, oy - hh, oz + hd],
            [ox + hw, oy - hh, oz - hd],
            [ox + hw, oy + hh, oz - hd],
            [ox + hw, oy + hh, oz + hd],
        ],
        [1.0, 0.0, 0.0],
    ));
    // Left (-X)
    vertices.extend_from_slice(&face(
        [
            [ox - hw, oy - hh, oz - hd],
            [ox - hw, oy - hh, oz + hd],
            [ox - hw, oy + hh, oz + hd],
            [ox - hw, oy + hh, oz - hd],
        ],
        [-1.0, 0.0, 0.0],
    ));

    let indices = vec![
        0, 1, 2, 0, 2, 3, // front
        4, 5, 6, 4, 6, 7, // back
        8, 9, 10, 8, 10, 11, // top
        12, 13, 14, 12, 14, 15, // bottom
        16, 17, 18, 16, 18, 19, // right
        20, 21, 22, 20, 22, 23, // left
    ];

    (vertices, indices)
}

/// Generate a UV sphere centered at `offset` with the given radius and
/// segment counts.
///
/// `segments` controls both latitude stacks and longitude sectors for
/// uniform tessellation.
pub fn make_sphere(radius: f32, segments: u32, offset: Offset) -> (Vec<Vertex>, Vec<u32>) {
    let [ox, oy, oz] = offset;
    let stacks = segments;
    let sectors = segments * 2;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for i in 0..=stacks {
        let phi = std::f32::consts::PI * i as f32 / stacks as f32;
        let y = radius * phi.cos();
        let r = radius * phi.sin();

        for j in 0..=sectors {
            let theta = 2.0 * std::f32::consts::PI * j as f32 / sectors as f32;
            let x = r * theta.cos();
            let z = r * theta.sin();

            let nx = x / radius;
            let ny = y / radius;
            let nz = z / radius;
            let tx = -theta.sin();
            let tz = theta.cos();

            vertices.push(Vertex {
                position: [ox + x, oy + y, oz + z],
                normal: [nx, ny, nz],
                tangent: [tx, 0.0, tz],
                uv: [j as f32 / sectors as f32, i as f32 / stacks as f32],
            });
        }
    }

    for i in 0..stacks {
        for j in 0..sectors {
            let a = i * (sectors + 1) + j;
            let b = a + sectors + 1;
            indices.push(a);
            indices.push(b);
            indices.push(a + 1);
            indices.push(a + 1);
            indices.push(b);
            indices.push(b + 1);
        }
    }

    (vertices, indices)
}

/// Generate a cylinder along the Y axis, centered at `offset`, with the
/// given radius, height, and circumference segment count.
///
/// Includes top and bottom caps.
pub fn make_cylinder(
    radius: f32,
    height: f32,
    segments: u32,
    offset: Offset,
) -> (Vec<Vertex>, Vec<u32>) {
    let [ox, oy, oz] = offset;
    let half_h = height / 2.0;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Side vertices: two rings (bottom and top)
    for ring in 0..2u32 {
        let y = if ring == 0 { -half_h } else { half_h };
        for j in 0..=segments {
            let theta = 2.0 * std::f32::consts::PI * j as f32 / segments as f32;
            let x = radius * theta.cos();
            let z = radius * theta.sin();
            vertices.push(Vertex {
                position: [ox + x, oy + y, oz + z],
                normal: [theta.cos(), 0.0, theta.sin()],
                tangent: [-theta.sin(), 0.0, theta.cos()],
                uv: [j as f32 / segments as f32, ring as f32],
            });
        }
    }
    let ring_verts = segments + 1;
    for j in 0..segments {
        let a = j;
        let b = j + ring_verts;
        indices.push(a);
        indices.push(b);
        indices.push(a + 1);
        indices.push(a + 1);
        indices.push(b);
        indices.push(b + 1);
    }

    // Top cap
    let top_center = vertices.len() as u32;
    vertices.push(Vertex {
        position: [ox, oy + half_h, oz],
        normal: [0.0, 1.0, 0.0],
        tangent: [1.0, 0.0, 0.0],
        uv: [0.5, 0.5],
    });
    for j in 0..=segments {
        let theta = 2.0 * std::f32::consts::PI * j as f32 / segments as f32;
        vertices.push(Vertex {
            position: [
                ox + radius * theta.cos(),
                oy + half_h,
                oz + radius * theta.sin(),
            ],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            uv: [0.5 + 0.5 * theta.cos(), 0.5 + 0.5 * theta.sin()],
        });
    }
    for j in 0..segments {
        indices.push(top_center);
        indices.push(top_center + 1 + j);
        indices.push(top_center + 2 + j);
    }

    // Bottom cap
    let bot_center = vertices.len() as u32;
    vertices.push(Vertex {
        position: [ox, oy - half_h, oz],
        normal: [0.0, -1.0, 0.0],
        tangent: [1.0, 0.0, 0.0],
        uv: [0.5, 0.5],
    });
    for j in 0..=segments {
        let theta = 2.0 * std::f32::consts::PI * j as f32 / segments as f32;
        vertices.push(Vertex {
            position: [
                ox + radius * theta.cos(),
                oy - half_h,
                oz + radius * theta.sin(),
            ],
            normal: [0.0, -1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            uv: [0.5 + 0.5 * theta.cos(), 0.5 + 0.5 * theta.sin()],
        });
    }
    for j in 0..segments {
        indices.push(bot_center);
        indices.push(bot_center + 2 + j);
        indices.push(bot_center + 1 + j);
    }

    (vertices, indices)
}

/// Combine multiple mesh parts (each with their own vertices and indices)
/// into a single [`Mesh`]. Index offsets are adjusted automatically.
pub fn combine_meshes(parts: &[(Vec<Vertex>, Vec<u32>)]) -> Mesh {
    let total_verts: usize = parts.iter().map(|(v, _)| v.len()).sum();
    let total_idx: usize = parts.iter().map(|(_, i)| i.len()).sum();

    let mut vertices = Vec::with_capacity(total_verts);
    let mut indices = Vec::with_capacity(total_idx);

    for (part_verts, part_indices) in parts {
        let base = vertices.len() as u32;
        vertices.extend_from_slice(part_verts);
        indices.extend(part_indices.iter().map(|&i| i + base));
    }

    Mesh { vertices, indices }
}

// ── Creep mesh builders ─────────────────────────────────────────────────────

/// Melee creep: small armored soldier.
///
/// - Body: box (0.3 x 0.5 x 0.2) — the torso
/// - Head: sphere (radius 0.12) sitting on top of the body
/// - Shield: flat box (0.05 x 0.3 x 0.25) offset to one side
/// - Total height ~0.7 units
pub fn melee_creep_mesh() -> Mesh {
    let body = make_box(0.3, 0.5, 0.2, [0.0, 0.25, 0.0]);
    let head = make_sphere(0.12, 8, [0.0, 0.62, 0.0]);
    let shield = make_box(0.05, 0.3, 0.25, [0.2, 0.3, 0.0]);
    combine_meshes(&[body, head, shield])
}

/// Ranged creep: taller, thinner figure with a bow.
///
/// - Body: tall thin box (0.2 x 0.6 x 0.15)
/// - Head: sphere (radius 0.1) on top
/// - Bow: thin box (0.02 x 0.3 x 0.02) sticking out to the side
/// - Total height ~0.8 units
pub fn ranged_creep_mesh() -> Mesh {
    let body = make_box(0.2, 0.6, 0.15, [0.0, 0.3, 0.0]);
    let head = make_sphere(0.1, 8, [0.0, 0.72, 0.0]);
    let bow = make_box(0.02, 0.3, 0.02, [0.2, 0.4, 0.0]);
    combine_meshes(&[body, head, bow])
}

/// Siege creep: big cart with a barrel cannon and wheels.
///
/// - Body: wide box (0.5 x 0.3 x 0.4) — the cart
/// - Barrel: cylinder on top pointing forward (radius 0.08, height 0.4)
///   — rotated to lie along Z by placing it horizontally via offset
/// - Wheels: 4 small cylinders on the sides
/// - Total height ~0.5 units, width ~0.5
pub fn siege_creep_mesh() -> Mesh {
    // Cart body
    let body = make_box(0.5, 0.3, 0.4, [0.0, 0.15, 0.0]);

    // Barrel on top, pointing forward (+Z). We use a cylinder along Y then
    // approximate a horizontal barrel via a thin box (actual rotation would
    // need matrix transforms; a box reads well at camera distance).
    let barrel = make_box(0.16, 0.16, 0.4, [0.0, 0.38, 0.15]);

    // Four wheels — small cylinders at the corners.
    // Wheels are along Y axis (upright) and placed at the four bottom corners.
    let wheel_r = 0.08;
    let wheel_h = 0.06;
    let wx = 0.28; // X offset from center
    let wz = 0.15; // Z offset from center
    let wy = 0.08; // Y center of wheel

    let wheel_fl = make_cylinder(wheel_r, wheel_h, 8, [-wx, wy, wz]);
    let wheel_fr = make_cylinder(wheel_r, wheel_h, 8, [wx, wy, wz]);
    let wheel_bl = make_cylinder(wheel_r, wheel_h, 8, [-wx, wy, -wz]);
    let wheel_br = make_cylinder(wheel_r, wheel_h, 8, [wx, wy, -wz]);

    combine_meshes(&[body, barrel, wheel_fl, wheel_fr, wheel_bl, wheel_br])
}

/// Return the mesh-name key for a given creep type and team.
///
/// Format: `"creep_{type}_{team}"` where type is `melee`/`ranged`/`siege`/`super`
/// and team is `radiant` or `dire`.
///
/// Super creeps reuse the melee mesh (they are upgraded melee creeps).
pub fn creep_mesh_name(creep_type_tag: &str, team: u8) -> String {
    let team_tag = if team == 1 { "radiant" } else { "dire" };
    format!("creep_{creep_type_tag}_{team_tag}")
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_box_vertex_and_index_counts() {
        let (verts, indices) = make_box(1.0, 1.0, 1.0, [0.0, 0.0, 0.0]);
        assert_eq!(verts.len(), 24, "box should have 24 vertices (6 faces x 4)");
        assert_eq!(
            indices.len(),
            36,
            "box should have 36 indices (6 faces x 2 tris x 3)"
        );
    }

    #[test]
    fn make_box_offset_applied() {
        let (verts, _) = make_box(1.0, 1.0, 1.0, [5.0, 10.0, 15.0]);
        // All vertex positions should be offset by (5, 10, 15) from the
        // centered range of [-0.5, 0.5].
        for v in &verts {
            assert!(v.position[0] >= 4.5 && v.position[0] <= 5.5);
            assert!(v.position[1] >= 9.5 && v.position[1] <= 10.5);
            assert!(v.position[2] >= 14.5 && v.position[2] <= 15.5);
        }
    }

    #[test]
    fn make_box_normals_are_unit_length() {
        let (verts, _) = make_box(2.0, 3.0, 4.0, [0.0, 0.0, 0.0]);
        for v in &verts {
            let len = (v.normal[0].powi(2) + v.normal[1].powi(2) + v.normal[2].powi(2)).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "Normal should be unit length, got {len}"
            );
        }
    }

    #[test]
    fn make_sphere_non_empty() {
        let (verts, indices) = make_sphere(0.5, 8, [0.0, 0.0, 0.0]);
        assert!(!verts.is_empty(), "sphere should produce vertices");
        assert!(!indices.is_empty(), "sphere should produce indices");
    }

    #[test]
    fn make_sphere_offset_applied() {
        let (verts, _) = make_sphere(0.5, 8, [3.0, 4.0, 5.0]);
        // All vertices should be within radius 0.5 of the offset center.
        for v in &verts {
            let dx = v.position[0] - 3.0;
            let dy = v.position[1] - 4.0;
            let dz = v.position[2] - 5.0;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(
                dist <= 0.5 + 1e-4,
                "vertex should be within radius of offset center, got dist={dist}"
            );
        }
    }

    #[test]
    fn make_cylinder_non_empty() {
        let (verts, indices) = make_cylinder(0.1, 0.5, 12, [0.0, 0.0, 0.0]);
        assert!(!verts.is_empty());
        assert!(!indices.is_empty());
    }

    #[test]
    fn combine_meshes_adjusts_indices() {
        let a = make_box(1.0, 1.0, 1.0, [0.0, 0.0, 0.0]);
        let b = make_box(1.0, 1.0, 1.0, [3.0, 0.0, 0.0]);
        let combined = combine_meshes(&[a.clone(), b.clone()]);

        assert_eq!(
            combined.vertices.len(),
            a.0.len() + b.0.len(),
            "combined vertex count should be sum of parts"
        );
        assert_eq!(
            combined.indices.len(),
            a.1.len() + b.1.len(),
            "combined index count should be sum of parts"
        );

        // Second part's indices should be offset by the first part's vertex count.
        let base = a.0.len() as u32;
        let second_start = a.1.len();
        for (i, &idx) in combined.indices[second_start..].iter().enumerate() {
            assert!(
                idx >= base,
                "index {i} of second part should be >= {base}, got {idx}"
            );
        }
    }

    #[test]
    fn melee_creep_mesh_non_empty() {
        let mesh = melee_creep_mesh();
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        // Height should be approximately 0.7 units (body top at 0.5 + head radius 0.12 + gap).
        let max_y = mesh
            .vertices
            .iter()
            .map(|v| v.position[1])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_y > 0.6 && max_y < 0.85,
            "melee creep max height should be ~0.7, got {max_y}"
        );
    }

    #[test]
    fn ranged_creep_mesh_non_empty() {
        let mesh = ranged_creep_mesh();
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        let max_y = mesh
            .vertices
            .iter()
            .map(|v| v.position[1])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_y > 0.7 && max_y < 0.95,
            "ranged creep max height should be ~0.8, got {max_y}"
        );
    }

    #[test]
    fn siege_creep_mesh_non_empty() {
        let mesh = siege_creep_mesh();
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        // Width should be approximately 0.5 units.
        let max_x = mesh
            .vertices
            .iter()
            .map(|v| v.position[0])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_x > 0.2 && max_x < 0.4,
            "siege creep half-width should be ~0.28, got {max_x}"
        );
    }

    #[test]
    fn creep_mesh_name_formatting() {
        assert_eq!(creep_mesh_name("melee", 1), "creep_melee_radiant");
        assert_eq!(creep_mesh_name("melee", 2), "creep_melee_dire");
        assert_eq!(creep_mesh_name("ranged", 1), "creep_ranged_radiant");
        assert_eq!(creep_mesh_name("siege", 2), "creep_siege_dire");
        assert_eq!(creep_mesh_name("super", 1), "creep_super_radiant");
    }
}
