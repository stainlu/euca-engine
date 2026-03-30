use crate::vertex::Vertex;
use euca_reflect::Reflect;

/// A mesh handle referencing a GPU-uploaded mesh by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Reflect)]
pub struct MeshHandle(pub u32);

/// Component that marks an entity for rendering with a specific mesh.
#[derive(Clone, Copy, Debug, Reflect)]
pub struct MeshRenderer {
    /// Handle to the GPU-uploaded mesh this entity should be drawn with.
    pub mesh: MeshHandle,
}

/// Visual vertical offset applied at render time so that a mesh's bottom
/// sits on the ground plane, without altering the entity's logical position.
///
/// When present on an entity, the render extraction layer adds this value
/// to the model matrix's Y translation. This keeps the entity's position
/// at ground level for physics and gameplay calculations while the visual
/// mesh is shifted upward.
#[derive(Clone, Copy, Debug)]
pub struct GroundOffset(pub f32);

/// CPU-side mesh geometry (vertices and triangle indices).
///
/// Upload to the GPU via [`Renderer::upload_mesh`] to obtain a [`MeshHandle`]
/// that can be referenced in [`DrawCommand`](crate::DrawCommand)s.
pub struct Mesh {
    /// Interleaved vertex data (position, normal, tangent, UV).
    pub vertices: Vec<Vertex>,
    /// Triangle indices into `vertices`. Every three consecutive values form
    /// one triangle (counter-clockwise winding).
    pub indices: Vec<u32>,
}

impl Mesh {
    /// Create a unit cube centered at origin with proper normals and tangents per face.
    pub fn cube() -> Self {
        let face = |positions: [[f32; 3]; 4], normal: [f32; 3]| -> [Vertex; 4] {
            // Tangent is along the U direction (edge from vertex 0 → vertex 1)
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

        // Front (z = +0.5), normal = +Z
        vertices.extend_from_slice(&face(
            [
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
            [0.0, 0.0, 1.0],
        ));
        // Back (z = -0.5), normal = -Z
        vertices.extend_from_slice(&face(
            [
                [0.5, -0.5, -0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
            ],
            [0.0, 0.0, -1.0],
        ));
        // Top (y = +0.5), normal = +Y
        vertices.extend_from_slice(&face(
            [
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
            ],
            [0.0, 1.0, 0.0],
        ));
        // Bottom (y = -0.5), normal = -Y
        vertices.extend_from_slice(&face(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [-0.5, -0.5, 0.5],
            ],
            [0.0, -1.0, 0.0],
        ));
        // Right (x = +0.5), normal = +X
        vertices.extend_from_slice(&face(
            [
                [0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ],
            [1.0, 0.0, 0.0],
        ));
        // Left (x = -0.5), normal = -X
        vertices.extend_from_slice(&face(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
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

        Self { vertices, indices }
    }

    /// Create a UV sphere with the given radius, stacks, and sectors.
    pub fn sphere(radius: f32, stacks: u32, sectors: u32) -> Self {
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

                // Tangent: derivative of position w.r.t. theta (longitude)
                let tx = -theta.sin();
                let tz = theta.cos();

                vertices.push(Vertex {
                    position: [x, y, z],
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

        Self { vertices, indices }
    }

    /// Create a flat plane (quad) on the XZ plane centered at origin.
    pub fn plane(size: f32) -> Self {
        let h = size / 2.0;
        // Tangent along X (U direction on XZ plane)
        let tangent = [1.0, 0.0, 0.0];
        let vertices = vec![
            Vertex {
                position: [-h, 0.0, -h],
                normal: [0.0, 1.0, 0.0],
                tangent,
                uv: [0.0, 0.0],
            },
            Vertex {
                position: [h, 0.0, -h],
                normal: [0.0, 1.0, 0.0],
                tangent,
                uv: [1.0, 0.0],
            },
            Vertex {
                position: [h, 0.0, h],
                normal: [0.0, 1.0, 0.0],
                tangent,
                uv: [1.0, 1.0],
            },
            Vertex {
                position: [-h, 0.0, h],
                normal: [0.0, 1.0, 0.0],
                tangent,
                uv: [0.0, 1.0],
            },
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];
        Self { vertices, indices }
    }

    /// Create a flat disc (filled circle) on the XZ plane centered at origin.
    ///
    /// `radius` is the disc radius; `segments` controls smoothness (more = rounder).
    pub fn disc(radius: f32, segments: u32) -> Self {
        let segments = segments.max(3);
        let mut vertices = Vec::with_capacity(segments as usize + 1);
        let mut indices = Vec::with_capacity(segments as usize * 3);

        let normal = [0.0, 1.0, 0.0];
        let tangent = [1.0, 0.0, 0.0];

        // Center vertex
        vertices.push(Vertex {
            position: [0.0, 0.0, 0.0],
            normal,
            tangent,
            uv: [0.5, 0.5],
        });

        // Ring vertices
        for i in 0..segments {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / segments as f32;
            let x = angle.cos() * radius;
            let z = angle.sin() * radius;
            vertices.push(Vertex {
                position: [x, 0.0, z],
                normal,
                tangent,
                uv: [0.5 + 0.5 * angle.cos(), 0.5 + 0.5 * angle.sin()],
            });
        }

        // Triangle fan from center
        for i in 0..segments {
            indices.push(0);
            indices.push(1 + i);
            indices.push(1 + (i + 1) % segments);
        }

        Self { vertices, indices }
    }

    /// Create a cylinder along the Y axis with given radius and height.
    pub fn cylinder(radius: f32, height: f32, sectors: u32) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let half_h = height / 2.0;

        // Side vertices: two rings (bottom and top)
        for ring in 0..2u32 {
            let y = if ring == 0 { -half_h } else { half_h };
            for j in 0..=sectors {
                let theta = 2.0 * std::f32::consts::PI * j as f32 / sectors as f32;
                let x = radius * theta.cos();
                let z = radius * theta.sin();
                let nx = theta.cos();
                let nz = theta.sin();
                let tx = -theta.sin();
                let tz = theta.cos();
                vertices.push(Vertex {
                    position: [x, y, z],
                    normal: [nx, 0.0, nz],
                    tangent: [tx, 0.0, tz],
                    uv: [j as f32 / sectors as f32, ring as f32],
                });
            }
        }
        let ring_verts = sectors + 1;
        for j in 0..sectors {
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
            position: [0.0, half_h, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            uv: [0.5, 0.5],
        });
        for j in 0..=sectors {
            let theta = 2.0 * std::f32::consts::PI * j as f32 / sectors as f32;
            let x = radius * theta.cos();
            let z = radius * theta.sin();
            vertices.push(Vertex {
                position: [x, half_h, z],
                normal: [0.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0],
                uv: [0.5 + 0.5 * theta.cos(), 0.5 + 0.5 * theta.sin()],
            });
        }
        for j in 0..sectors {
            indices.push(top_center);
            indices.push(top_center + 1 + j);
            indices.push(top_center + 2 + j);
        }

        // Bottom cap
        let bot_center = vertices.len() as u32;
        vertices.push(Vertex {
            position: [0.0, -half_h, 0.0],
            normal: [0.0, -1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            uv: [0.5, 0.5],
        });
        for j in 0..=sectors {
            let theta = 2.0 * std::f32::consts::PI * j as f32 / sectors as f32;
            let x = radius * theta.cos();
            let z = radius * theta.sin();
            vertices.push(Vertex {
                position: [x, -half_h, z],
                normal: [0.0, -1.0, 0.0],
                tangent: [1.0, 0.0, 0.0],
                uv: [0.5 + 0.5 * theta.cos(), 0.5 + 0.5 * theta.sin()],
            });
        }
        for j in 0..sectors {
            indices.push(bot_center);
            indices.push(bot_center + 2 + j);
            indices.push(bot_center + 1 + j);
        }

        Self { vertices, indices }
    }

    /// Create a cone along the Y axis with given radius and height.
    /// The base is at `-height/2` and the apex at `+height/2`.
    pub fn cone(radius: f32, height: f32, sectors: u32) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let half_h = height / 2.0;
        let slope = radius / height;

        // Side vertices: base ring + apex duplicated per sector for correct normals
        for j in 0..=sectors {
            let theta = 2.0 * std::f32::consts::PI * j as f32 / sectors as f32;
            let x = radius * theta.cos();
            let z = radius * theta.sin();
            // Normal: outward and upward along the slope
            let ny = slope;
            let nx = theta.cos();
            let nz = theta.sin();
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            vertices.push(Vertex {
                position: [x, -half_h, z],
                normal: [nx / len, ny / len, nz / len],
                tangent: [-theta.sin(), 0.0, theta.cos()],
                uv: [j as f32 / sectors as f32, 1.0],
            });
        }
        // Apex vertices (one per sector for smooth shading)
        for j in 0..=sectors {
            let theta = 2.0 * std::f32::consts::PI * j as f32 / sectors as f32;
            let ny = slope;
            let nx = theta.cos();
            let nz = theta.sin();
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            vertices.push(Vertex {
                position: [0.0, half_h, 0.0],
                normal: [nx / len, ny / len, nz / len],
                tangent: [-theta.sin(), 0.0, theta.cos()],
                uv: [j as f32 / sectors as f32, 0.0],
            });
        }
        let ring_verts = sectors + 1;
        for j in 0..sectors {
            indices.push(j);
            indices.push(j + 1);
            indices.push(ring_verts + j);
        }

        // Bottom cap
        let bot_center = vertices.len() as u32;
        vertices.push(Vertex {
            position: [0.0, -half_h, 0.0],
            normal: [0.0, -1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            uv: [0.5, 0.5],
        });
        for j in 0..=sectors {
            let theta = 2.0 * std::f32::consts::PI * j as f32 / sectors as f32;
            let x = radius * theta.cos();
            let z = radius * theta.sin();
            vertices.push(Vertex {
                position: [x, -half_h, z],
                normal: [0.0, -1.0, 0.0],
                tangent: [1.0, 0.0, 0.0],
                uv: [0.5 + 0.5 * theta.cos(), 0.5 + 0.5 * theta.sin()],
            });
        }
        for j in 0..sectors {
            indices.push(bot_center);
            indices.push(bot_center + 2 + j);
            indices.push(bot_center + 1 + j);
        }

        Self { vertices, indices }
    }
    // ── Offset builders & multi-part combiner ──────────────────────────────

    /// Create a box at the given offset, with dimensions `(w, h, d)`.
    pub fn offset_box(w: f32, h: f32, d: f32, offset: [f32; 3]) -> Self {
        let mut m = Self::cube();
        for v in &mut m.vertices {
            v.position[0] = v.position[0] * w + offset[0];
            v.position[1] = v.position[1] * h + offset[1];
            v.position[2] = v.position[2] * d + offset[2];
        }
        m
    }

    /// Create a sphere at the given offset.
    pub fn offset_sphere(radius: f32, stacks: u32, sectors: u32, offset: [f32; 3]) -> Self {
        let mut m = Self::sphere(radius, stacks, sectors);
        for v in &mut m.vertices {
            v.position[0] += offset[0];
            v.position[1] += offset[1];
            v.position[2] += offset[2];
        }
        m
    }

    /// Create a cone at the given offset with an optional rotation.
    pub fn offset_cone(
        radius: f32,
        height: f32,
        sectors: u32,
        offset: [f32; 3],
        rotation: Option<[f32; 3]>,
    ) -> Self {
        let mut m = Self::cone(radius, height, sectors);
        if let Some([pitch, yaw, roll]) = rotation {
            let (sp, cp) = (pitch.sin(), pitch.cos());
            let (sy, cy) = (yaw.sin(), yaw.cos());
            let (sr, cr) = (roll.sin(), roll.cos());
            // Rotation matrix: Ry * Rx * Rz
            let r00 = cy * cr + sy * sp * sr;
            let r01 = -cy * sr + sy * sp * cr;
            let r02 = sy * cp;
            let r10 = cp * sr;
            let r11 = cp * cr;
            let r12 = -sp;
            let r20 = -sy * cr + cy * sp * sr;
            let r21 = sy * sr + cy * sp * cr;
            let r22 = cy * cp;
            for v in &mut m.vertices {
                let [x, y, z] = v.position;
                v.position = [
                    r00 * x + r01 * y + r02 * z,
                    r10 * x + r11 * y + r12 * z,
                    r20 * x + r21 * y + r22 * z,
                ];
                let [nx, ny, nz] = v.normal;
                v.normal = [
                    r00 * nx + r01 * ny + r02 * nz,
                    r10 * nx + r11 * ny + r12 * nz,
                    r20 * nx + r21 * ny + r22 * nz,
                ];
                let [tx, ty, tz] = v.tangent;
                v.tangent = [
                    r00 * tx + r01 * ty + r02 * tz,
                    r10 * tx + r11 * ty + r12 * tz,
                    r20 * tx + r21 * ty + r22 * tz,
                ];
            }
        }
        for v in &mut m.vertices {
            v.position[0] += offset[0];
            v.position[1] += offset[1];
            v.position[2] += offset[2];
        }
        m
    }

    /// Combine multiple meshes into a single mesh, merging vertex and index buffers.
    pub fn combine(parts: Vec<Self>) -> Self {
        let total_verts: usize = parts.iter().map(|p| p.vertices.len()).sum();
        let total_indices: usize = parts.iter().map(|p| p.indices.len()).sum();
        let mut vertices = Vec::with_capacity(total_verts);
        let mut indices = Vec::with_capacity(total_indices);
        for part in parts {
            let base = vertices.len() as u32;
            vertices.extend(part.vertices);
            indices.extend(part.indices.iter().map(|i| i + base));
        }
        Self { vertices, indices }
    }

    // ── Procedural creature meshes ───────────────────────────────────────

    /// Procedural Roshan boss model: large body sphere, head, two horns, two arms.
    /// Total height ~2.5 units, width ~2 units. Centered at origin, resting on Y=0.
    pub fn roshan() -> Self {
        let segs = 12;
        // Body: large sphere, center at Y=1.0 (radius 1.0, bottom at Y=0)
        let body = Self::offset_sphere(1.0, segs, segs * 2, [0.0, 1.0, 0.0]);
        // Head: smaller sphere in front, center at Y=1.8, Z=-0.8
        let head = Self::offset_sphere(0.5, segs, segs * 2, [0.0, 1.8, -0.8]);
        // Left horn: cone angled outward-left, from head
        let horn_l = Self::offset_cone(
            0.15,
            0.8,
            8,
            [-0.3, 2.2, -0.9],
            Some([0.0, 0.0, 0.5]), // tilt outward left
        );
        // Right horn: cone angled outward-right
        let horn_r = Self::offset_cone(
            0.15,
            0.8,
            8,
            [0.3, 2.2, -0.9],
            Some([0.0, 0.0, -0.5]), // tilt outward right
        );
        // Left arm
        let arm_l = Self::offset_box(0.15, 0.6, 0.2, [-1.0, 0.8, 0.0]);
        // Right arm
        let arm_r = Self::offset_box(0.15, 0.6, 0.2, [1.0, 0.8, 0.0]);

        Self::combine(vec![body, head, horn_l, horn_r, arm_l, arm_r])
    }

    /// Procedural wolf model: horizontal body, head, 4 legs.
    /// Medium quadruped beast. Centered at origin, resting on Y=0.
    pub fn wolf() -> Self {
        let segs = 8;
        // Body: horizontal box, center at Y=0.3
        let body = Self::offset_box(0.4, 0.25, 0.6, [0.0, 0.3, 0.0]);
        // Head: small sphere at front
        let head = Self::offset_sphere(0.12, segs, segs * 2, [0.0, 0.38, -0.4]);
        // Front-left leg
        let fl = Self::offset_box(0.06, 0.18, 0.06, [-0.14, 0.09, -0.2]);
        // Front-right leg
        let fr = Self::offset_box(0.06, 0.18, 0.06, [0.14, 0.09, -0.2]);
        // Back-left leg
        let bl = Self::offset_box(0.06, 0.18, 0.06, [-0.14, 0.09, 0.2]);
        // Back-right leg
        let br = Self::offset_box(0.06, 0.18, 0.06, [0.14, 0.09, 0.2]);

        Self::combine(vec![body, head, fl, fr, bl, br])
    }

    /// Procedural troll model: upright body, head, two arms, club in one hand.
    /// Bipedal monster. Centered at origin, resting on Y=0.
    pub fn troll() -> Self {
        let segs = 8;
        // Body: upright box, center at Y=0.35
        let body = Self::offset_box(0.25, 0.5, 0.2, [0.0, 0.35, 0.0]);
        // Head: sphere on top
        let head = Self::offset_sphere(0.15, segs, segs * 2, [0.0, 0.7, 0.0]);
        // Left arm
        let arm_l = Self::offset_box(0.08, 0.35, 0.08, [-0.22, 0.35, 0.0]);
        // Right arm
        let arm_r = Self::offset_box(0.08, 0.35, 0.08, [0.22, 0.35, 0.0]);
        // Club in right hand: long thin box extending down and forward
        let club = Self::offset_box(0.05, 0.45, 0.05, [0.28, 0.15, -0.1]);
        // Left leg
        let leg_l = Self::offset_box(0.08, 0.2, 0.08, [-0.08, 0.1, 0.0]);
        // Right leg
        let leg_r = Self::offset_box(0.08, 0.2, 0.08, [0.08, 0.1, 0.0]);

        Self::combine(vec![body, head, arm_l, arm_r, club, leg_l, leg_r])
    }

    /// Merge another mesh into this one, offsetting its indices.
    pub fn merge(&mut self, other: &Mesh) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.indices.extend(other.indices.iter().map(|i| i + base));
    }

    /// Procedural tree: cylinder trunk + sphere canopy in one mesh.
    pub fn tree(
        trunk_radius: f32, trunk_height: f32, trunk_sectors: u32,
        canopy_radius: f32, canopy_centre_y: f32, canopy_stacks: u32, canopy_sectors: u32,
    ) -> Self {
        let mut trunk = Self::cylinder(trunk_radius, trunk_height, trunk_sectors);
        let trunk_offset_y = trunk_height / 2.0;
        for v in &mut trunk.vertices { v.position[1] += trunk_offset_y; }
        let mut canopy = Self::sphere(canopy_radius, canopy_stacks, canopy_sectors);
        for v in &mut canopy.vertices { v.position[1] += canopy_centre_y; }
        trunk.merge(&canopy);
        trunk
    }

    /// Convenience wrapper with MOBA-style tree defaults.
    pub fn tree_default() -> Self {
        Self::tree(0.15, 1.5, 8, 0.6, 1.8, 8, 16)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_has_24_vertices_and_36_indices() {
        let cube = Mesh::cube();
        assert_eq!(cube.vertices.len(), 24); // 6 faces * 4 vertices
        assert_eq!(cube.indices.len(), 36); // 6 faces * 2 triangles * 3
    }

    #[test]
    fn cube_normals_are_unit_length() {
        let cube = Mesh::cube();
        for v in &cube.vertices {
            let len = (v.normal[0].powi(2) + v.normal[1].powi(2) + v.normal[2].powi(2)).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "Normal should be unit length, got {len}"
            );
        }
    }

    #[test]
    fn sphere_vertex_count() {
        let sphere = Mesh::sphere(1.0, 8, 16);
        assert!(!sphere.vertices.is_empty());
        assert!(!sphere.indices.is_empty());
    }

    #[test]
    fn plane_has_4_vertices() {
        let plane = Mesh::plane(10.0);
        assert_eq!(plane.vertices.len(), 4);
        assert_eq!(plane.indices.len(), 6);
    }

    #[test]
    fn offset_box_applies_offset() {
        let m = Mesh::offset_box(1.0, 1.0, 1.0, [5.0, 0.0, 0.0]);
        // All vertices should be centered around x=5
        let avg_x: f32 =
            m.vertices.iter().map(|v| v.position[0]).sum::<f32>() / m.vertices.len() as f32;
        assert!(
            (avg_x - 5.0).abs() < 0.01,
            "avg_x should be ~5.0, got {avg_x}"
        );
    }

    #[test]
    fn combine_merges_meshes() {
        let a = Mesh::cube();
        let b = Mesh::cube();
        let a_verts = a.vertices.len();
        let a_indices = a.indices.len();
        let combined = Mesh::combine(vec![a, b]);
        assert_eq!(combined.vertices.len(), a_verts * 2);
        assert_eq!(combined.indices.len(), a_indices * 2);
        // Second mesh's indices should be offset by the first mesh's vertex count.
        assert!(combined.indices[a_indices] >= a_verts as u32);
    }

    #[test]
    fn roshan_mesh_is_nonempty() {
        let m = Mesh::roshan();
        assert!(m.vertices.len() > 50, "Roshan should have many vertices");
        assert!(!m.indices.is_empty());
    }

    #[test]
    fn wolf_mesh_is_nonempty() {
        let m = Mesh::wolf();
        assert!(m.vertices.len() > 30, "Wolf should have multiple parts");
        assert!(!m.indices.is_empty());
    }

    #[test]
    fn troll_mesh_is_nonempty() {
        let m = Mesh::troll();
        assert!(m.vertices.len() > 30, "Troll should have multiple parts");
        assert!(!m.indices.is_empty());
    }

    #[test]
    fn roshan_rests_on_ground() {
        let m = Mesh::roshan();
        let min_y = m
            .vertices
            .iter()
            .map(|v| v.position[1])
            .fold(f32::INFINITY, f32::min);
        assert!(
            min_y >= -0.1,
            "Roshan should rest on or above ground (Y=0), min_y={min_y}"
        );
    }

    #[test]
    fn offset_box_applies_offset() {
        let m = Mesh::offset_box(1.0, 1.0, 1.0, [5.0, 0.0, 0.0]);
        // All vertices should be centered around x=5
        let avg_x: f32 =
            m.vertices.iter().map(|v| v.position[0]).sum::<f32>() / m.vertices.len() as f32;
        assert!(
            (avg_x - 5.0).abs() < 0.01,
            "avg_x should be ~5.0, got {avg_x}"
        );
    }

    #[test]
    fn combine_merges_meshes() {
        let a = Mesh::cube();
        let b = Mesh::cube();
        let a_verts = a.vertices.len();
        let a_indices = a.indices.len();
        let combined = Mesh::combine(vec![a, b]);
        assert_eq!(combined.vertices.len(), a_verts * 2);
        assert_eq!(combined.indices.len(), a_indices * 2);
        // Second mesh's indices should be offset by the first mesh's vertex count.
        assert!(combined.indices[a_indices] >= a_verts as u32);
    }

    #[test]
    fn roshan_mesh_is_nonempty() {
        let m = Mesh::roshan();
        assert!(m.vertices.len() > 50, "Roshan should have many vertices");
        assert!(!m.indices.is_empty());
    }

    #[test]
    fn wolf_mesh_is_nonempty() {
        let m = Mesh::wolf();
        assert!(m.vertices.len() > 30, "Wolf should have multiple parts");
        assert!(!m.indices.is_empty());
    }

    #[test]
    fn troll_mesh_is_nonempty() {
        let m = Mesh::troll();
        assert!(m.vertices.len() > 30, "Troll should have multiple parts");
        assert!(!m.indices.is_empty());
    }

    #[test]
    fn roshan_rests_on_ground() {
        let m = Mesh::roshan();
        let min_y = m
            .vertices
            .iter()
            .map(|v| v.position[1])
            .fold(f32::INFINITY, f32::min);
        assert!(
            min_y >= -0.1,
            "Roshan should rest on or above ground (Y=0), min_y={min_y}"
        );
    }
}
