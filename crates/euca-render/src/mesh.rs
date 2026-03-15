use crate::vertex::Vertex;

/// A mesh handle referencing a GPU-uploaded mesh by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub u32);

/// Component that marks an entity for rendering with a specific mesh.
#[derive(Clone, Copy, Debug)]
pub struct MeshRenderer {
    pub mesh: MeshHandle,
}

/// CPU-side mesh data (vertices + indices).
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    /// Create a unit cube centered at origin with the given base color.
    /// Each face is tinted slightly differently for 3D depth cues.
    pub fn cube(color: [f32; 3]) -> Self {
        let tint = |base: [f32; 3], factor: f32| -> [f32; 3] {
            [base[0] * factor, base[1] * factor, base[2] * factor]
        };

        let face = |pos: [[f32; 3]; 4], c: [f32; 3]| -> [Vertex; 4] {
            [
                Vertex { position: pos[0], color: c },
                Vertex { position: pos[1], color: c },
                Vertex { position: pos[2], color: c },
                Vertex { position: pos[3], color: c },
            ]
        };

        let top    = tint(color, 1.0);   // brightest
        let front  = tint(color, 0.85);
        let right  = tint(color, 0.7);
        let back   = tint(color, 0.6);
        let left   = tint(color, 0.55);
        let bottom = tint(color, 0.4);   // darkest

        let mut vertices = Vec::with_capacity(24);

        // Front (z = 0.5)
        vertices.extend_from_slice(&face(
            [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]],
            front,
        ));
        // Back (z = -0.5)
        vertices.extend_from_slice(&face(
            [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]],
            back,
        ));
        // Top (y = 0.5)
        vertices.extend_from_slice(&face(
            [[-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]],
            top,
        ));
        // Bottom (y = -0.5)
        vertices.extend_from_slice(&face(
            [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5]],
            bottom,
        ));
        // Right (x = 0.5)
        vertices.extend_from_slice(&face(
            [[0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]],
            right,
        ));
        // Left (x = -0.5)
        vertices.extend_from_slice(&face(
            [[-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5]],
            left,
        ));

        let indices = vec![
            0, 1, 2, 0, 2, 3,       // front
            4, 6, 5, 4, 7, 6,       // back
            8, 9, 10, 8, 10, 11,    // top
            12, 14, 13, 12, 15, 14, // bottom
            16, 17, 18, 16, 18, 19, // right
            20, 22, 21, 20, 23, 22, // left
        ];

        Self { vertices, indices }
    }
}
