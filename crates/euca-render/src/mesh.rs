use crate::vertex::Vertex;
use euca_reflect::Reflect;

/// A mesh handle referencing a GPU-uploaded mesh by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub u32);

/// Component that marks an entity for rendering with a specific mesh.
#[derive(Clone, Copy, Debug, Reflect)]
pub struct MeshRenderer {
    pub mesh: MeshHandle,
}

/// CPU-side mesh data (vertices + indices).
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    /// Create a unit cube centered at origin with proper normals per face.
    pub fn cube() -> Self {
        let face = |positions: [[f32; 3]; 4], normal: [f32; 3]| -> [Vertex; 4] {
            [
                Vertex {
                    position: positions[0],
                    normal,
                    uv: [0.0, 0.0],
                },
                Vertex {
                    position: positions[1],
                    normal,
                    uv: [1.0, 0.0],
                },
                Vertex {
                    position: positions[2],
                    normal,
                    uv: [1.0, 1.0],
                },
                Vertex {
                    position: positions[3],
                    normal,
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

                vertices.push(Vertex {
                    position: [x, y, z],
                    normal: [nx, ny, nz],
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
        let vertices = vec![
            Vertex {
                position: [-h, 0.0, -h],
                normal: [0.0, 1.0, 0.0],
                uv: [0.0, 0.0],
            },
            Vertex {
                position: [h, 0.0, -h],
                normal: [0.0, 1.0, 0.0],
                uv: [1.0, 0.0],
            },
            Vertex {
                position: [h, 0.0, h],
                normal: [0.0, 1.0, 0.0],
                uv: [1.0, 1.0],
            },
            Vertex {
                position: [-h, 0.0, h],
                normal: [0.0, 1.0, 0.0],
                uv: [0.0, 1.0],
            },
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];
        Self { vertices, indices }
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
}
