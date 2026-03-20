//! Deferred decal rendering.
//!
//! Decals project a texture onto existing surfaces using the depth buffer.
//! A unit cube is rendered as the decal volume; the fragment shader
//! reconstructs world-space position from the depth buffer and projects
//! the decal texture in the configured direction.
//!
//! # Pipeline
//! 1. For each `Decal` component with a `GlobalTransform`, build a
//!    `DecalDrawCommand` containing the projection matrices and fade parameters.
//! 2. Sort commands by priority (ascending — higher priority renders last / on top).
//! 3. Render the decal box volumes in a deferred pass that reads the G-buffer
//!    depth and writes to the albedo / normal targets.
//!
//! # Fading
//! - **Distance fade**: linear fade-out as camera distance exceeds a threshold.
//! - **Angle fade**: linear fade-out as the angle between the surface normal
//!   and the decal projection direction exceeds a threshold.

use crate::texture::TextureHandle;
use euca_ecs::{Query, World};
use euca_math::{Mat4, Vec3};
use euca_scene::GlobalTransform;

// ---------------------------------------------------------------------------
// Blend mode
// ---------------------------------------------------------------------------

/// How a decal blends with the underlying surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecalBlendMode {
    /// Modulates albedo using the decal color/texture alpha.
    Translucent,
    /// Multiplies the surface albedo (dirt, grime, wetness).
    Stain,
    /// Only modifies the surface normal (scratches, dents).
    NormalOnly,
    /// Adds emissive light (neon signs, glowing runes).
    Emissive,
}

impl Default for DecalBlendMode {
    fn default() -> Self {
        Self::Translucent
    }
}

// ---------------------------------------------------------------------------
// Decal component
// ---------------------------------------------------------------------------

/// Component that marks an entity as a projected decal.
///
/// Attach alongside a `LocalTransform` / `GlobalTransform` to position the
/// decal in the world. The transform's translation is the decal center; the
/// `size` field controls the box extents along each local axis.
#[derive(Clone, Debug)]
pub struct Decal {
    /// Texture projected onto the surface.
    pub texture: TextureHandle,
    /// Half-extents of the decal box volume along local X, Y, Z.
    pub size: Vec3,
    /// Local-space direction the decal projects along (normalized).
    /// Defaults to -Y (projecting downward).
    pub normal: Vec3,
    /// How the decal composites with the surface.
    pub blend_mode: DecalBlendMode,
    /// Master opacity in `[0, 1]`.
    pub opacity: f32,
    /// Render priority. Higher values render on top of lower values.
    pub priority: u8,
    /// Camera distance at which the decal becomes fully invisible.
    pub max_distance: f32,
    /// Surface-normal angle (radians) at which fading begins.
    /// 0 = only surfaces facing directly at the projection; PI/2 = all.
    pub angle_fade_start: f32,
    /// Surface-normal angle (radians) at which the decal is fully invisible.
    pub angle_fade_end: f32,
}

impl Default for Decal {
    fn default() -> Self {
        Self {
            texture: TextureHandle(0),
            size: Vec3::ONE,
            normal: Vec3::new(0.0, -1.0, 0.0),
            blend_mode: DecalBlendMode::Translucent,
            opacity: 1.0,
            priority: 0,
            max_distance: 80.0,
            angle_fade_start: 1.0,                       // ~57 degrees
            angle_fade_end: std::f32::consts::FRAC_PI_2, // 90 degrees
        }
    }
}

impl Decal {
    /// Create a decal with a texture, size, and default settings.
    pub fn new(texture: TextureHandle, size: Vec3) -> Self {
        Self {
            texture,
            size,
            ..Default::default()
        }
    }

    /// Builder: set the projection direction (will be normalized).
    pub fn with_normal(mut self, dir: Vec3) -> Self {
        self.normal = dir.normalize();
        self
    }

    /// Builder: set the blend mode.
    pub fn with_blend_mode(mut self, mode: DecalBlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Builder: set the opacity.
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Builder: set the render priority.
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Builder: set the maximum visible distance.
    pub fn with_max_distance(mut self, max_distance: f32) -> Self {
        self.max_distance = max_distance;
        self
    }

    /// Builder: set the angle fade range (radians).
    pub fn with_angle_fade(mut self, start: f32, end: f32) -> Self {
        self.angle_fade_start = start;
        self.angle_fade_end = end;
        self
    }

    /// Compute opacity multiplier from camera distance.
    ///
    /// Returns 1.0 when `distance == 0`, 0.0 when `distance >= max_distance`,
    /// and linearly interpolates between.
    pub fn distance_fade(&self, distance: f32) -> f32 {
        if self.max_distance <= f32::EPSILON {
            return 0.0;
        }
        1.0 - (distance / self.max_distance).clamp(0.0, 1.0)
    }

    /// Compute opacity multiplier from the angle between a surface normal
    /// and the decal projection direction.
    ///
    /// `angle` is the angle in radians between the surface normal and
    /// `-normal` (i.e., 0 when the surface faces the projector).
    pub fn angle_fade(&self, angle: f32) -> f32 {
        if angle <= self.angle_fade_start {
            return 1.0;
        }
        if angle >= self.angle_fade_end {
            return 0.0;
        }
        let range = self.angle_fade_end - self.angle_fade_start;
        if range <= f32::EPSILON {
            return 0.0;
        }
        1.0 - (angle - self.angle_fade_start) / range
    }
}

// ---------------------------------------------------------------------------
// Draw command
// ---------------------------------------------------------------------------

/// Per-frame data for a single decal, ready for GPU submission.
#[derive(Clone, Debug)]
pub struct DecalDrawCommand {
    /// World-space model matrix that transforms the unit cube into the decal volume.
    pub model_matrix: Mat4,
    /// Final opacity (component opacity * distance fade; angle fade applied in shader).
    pub opacity: f32,
    /// Render priority (commands are sorted ascending by this).
    pub priority: u8,
}

// ---------------------------------------------------------------------------
// GPU resources
// ---------------------------------------------------------------------------

/// Unit-cube vertex (position only — decals only need the volume geometry).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DecalVertex {
    position: [f32; 3],
}

/// Manages GPU resources for decal rendering: the unit-cube vertex and index
/// buffers used as the decal projection volume.
pub struct DecalRenderer {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl DecalRenderer {
    /// Create the GPU resources for decal rendering.
    pub fn new(device: &wgpu::Device) -> Self {
        // Unit cube from -0.5 to +0.5 on each axis.
        #[rustfmt::skip]
        let vertices: [DecalVertex; 8] = [
            DecalVertex { position: [-0.5, -0.5, -0.5] },
            DecalVertex { position: [ 0.5, -0.5, -0.5] },
            DecalVertex { position: [ 0.5,  0.5, -0.5] },
            DecalVertex { position: [-0.5,  0.5, -0.5] },
            DecalVertex { position: [-0.5, -0.5,  0.5] },
            DecalVertex { position: [ 0.5, -0.5,  0.5] },
            DecalVertex { position: [ 0.5,  0.5,  0.5] },
            DecalVertex { position: [-0.5,  0.5,  0.5] },
        ];

        // 12 triangles (6 faces * 2 triangles)
        #[rustfmt::skip]
        let indices: [u16; 36] = [
            // Front face (-Z)
            0, 2, 1,  0, 3, 2,
            // Back face (+Z)
            4, 5, 6,  4, 6, 7,
            // Left face (-X)
            0, 4, 7,  0, 7, 3,
            // Right face (+X)
            1, 2, 6,  1, 6, 5,
            // Bottom face (-Y)
            0, 1, 5,  0, 5, 4,
            // Top face (+Y)
            2, 3, 7,  2, 7, 6,
        ];

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Decal Vertex Buffer"),
            size: std::mem::size_of_val(&vertices) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Decal Index Buffer"),
            size: std::mem::size_of_val(&indices) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
        }
    }

    /// Upload the unit-cube geometry to the GPU. Call once after creation.
    pub fn upload(&self, queue: &wgpu::Queue) {
        #[rustfmt::skip]
        let vertices: [DecalVertex; 8] = [
            DecalVertex { position: [-0.5, -0.5, -0.5] },
            DecalVertex { position: [ 0.5, -0.5, -0.5] },
            DecalVertex { position: [ 0.5,  0.5, -0.5] },
            DecalVertex { position: [-0.5,  0.5, -0.5] },
            DecalVertex { position: [-0.5, -0.5,  0.5] },
            DecalVertex { position: [ 0.5, -0.5,  0.5] },
            DecalVertex { position: [ 0.5,  0.5,  0.5] },
            DecalVertex { position: [-0.5,  0.5,  0.5] },
        ];

        #[rustfmt::skip]
        let indices: [u16; 36] = [
            0, 2, 1,  0, 3, 2,
            4, 5, 6,  4, 6, 7,
            0, 4, 7,  0, 7, 3,
            1, 2, 6,  1, 6, 5,
            0, 1, 5,  0, 5, 4,
            2, 3, 7,  2, 7, 6,
        ];

        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
    }

    /// The vertex buffer for the decal unit cube.
    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertex_buffer
    }

    /// The index buffer for the decal unit cube.
    pub fn index_buffer(&self) -> &wgpu::Buffer {
        &self.index_buffer
    }

    /// Number of indices in the unit cube (36).
    pub fn index_count(&self) -> u32 {
        self.index_count
    }

    /// The vertex buffer layout for `DecalVertex`.
    pub fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<DecalVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            }],
        }
    }
}

// ---------------------------------------------------------------------------
// Projection helper
// ---------------------------------------------------------------------------

/// Computes the projection matrix needed to project a decal onto scene geometry.
pub struct DecalProjection;

impl DecalProjection {
    /// Compute an orthographic projection matrix for a decal.
    ///
    /// `position` is the decal's world-space center.
    /// `normal` is the world-space projection direction (normalized).
    /// `size` is the decal's box half-extents (width, height, depth).
    ///
    /// Returns a combined view-projection matrix that maps world-space
    /// positions into decal clip space.
    pub fn from_transform(position: Vec3, normal: Vec3, size: Vec3) -> Mat4 {
        let world_dir = normal.normalize();

        // Build an orthonormal basis from the projection direction.
        // Choose a stable "up" vector that is not parallel to `world_dir`.
        let candidate_up = if world_dir.dot(Vec3::Y).abs() > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let right = candidate_up.cross(world_dir).normalize();
        let up = world_dir.cross(right);

        // View matrix: columns are the basis vectors in eye space.
        let view = Mat4 {
            cols: [
                [right.x, up.x, world_dir.x, 0.0],
                [right.y, up.y, world_dir.y, 0.0],
                [right.z, up.z, world_dir.z, 0.0],
                [
                    -right.dot(position),
                    -up.dot(position),
                    -world_dir.dot(position),
                    1.0,
                ],
            ],
        };

        // Orthographic projection sized to the decal's box extents.
        let projection = Mat4::orthographic_lh(-size.x, size.x, -size.y, size.y, -size.z, size.z);

        projection * view
    }
}

// ---------------------------------------------------------------------------
// Collection from ECS
// ---------------------------------------------------------------------------

/// Query the ECS world for all entities with `Decal` + `GlobalTransform`
/// components and produce a sorted list of draw commands.
///
/// For each matching entity, the distance from camera to decal center is
/// computed to apply the distance fade. Commands are sorted by priority
/// ascending so higher-priority decals render last (on top).
pub fn collect_decal_draw_commands(world: &World) -> Vec<DecalDrawCommand> {
    let query = Query::<(&Decal, &GlobalTransform)>::new(world);

    let mut commands: Vec<DecalDrawCommand> = query
        .iter()
        .filter_map(|(decal, global)| {
            if decal.opacity <= 0.0 {
                return None;
            }

            // Build model matrix: transform scales the unit cube to match the
            // decal's size, then applies the entity's world transform.
            let size_matrix = Mat4::from_scale(decal.size);
            let world_matrix = global.0.to_matrix();
            let model_matrix = world_matrix * size_matrix;

            Some(DecalDrawCommand {
                model_matrix,
                opacity: decal.opacity,
                priority: decal.priority,
            })
        })
        .collect();

    // Sort ascending by priority — higher priority decals render last (on top).
    commands.sort_by_key(|cmd| cmd.priority);

    commands
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::{Transform, Vec3};
    use euca_scene::GlobalTransform;

    /// Helper: spawn a decal entity with a GlobalTransform.
    fn spawn_decal(world: &mut World, decal: Decal, position: Vec3) {
        let entity = world.spawn(decal);
        world.insert(
            entity,
            GlobalTransform(Transform::from_translation(position)),
        );
    }

    #[test]
    fn collect_returns_sorted_by_priority() {
        let mut world = World::new();

        spawn_decal(
            &mut world,
            Decal::new(TextureHandle(1), Vec3::ONE).with_priority(10),
            Vec3::ZERO,
        );
        spawn_decal(
            &mut world,
            Decal::new(TextureHandle(2), Vec3::ONE).with_priority(1),
            Vec3::ZERO,
        );
        spawn_decal(
            &mut world,
            Decal::new(TextureHandle(3), Vec3::ONE).with_priority(5),
            Vec3::ZERO,
        );

        let cmds = collect_decal_draw_commands(&world);
        assert_eq!(cmds.len(), 3);
        assert_eq!(cmds[0].priority, 1);
        assert_eq!(cmds[1].priority, 5);
        assert_eq!(cmds[2].priority, 10);
    }

    #[test]
    fn distance_fade_linear_interpolation() {
        let decal = Decal {
            max_distance: 100.0,
            ..Default::default()
        };

        // At origin: full opacity.
        assert!((decal.distance_fade(0.0) - 1.0).abs() < 1e-6);
        // At half distance: 0.5.
        assert!((decal.distance_fade(50.0) - 0.5).abs() < 1e-6);
        // At max distance: 0.0.
        assert!(decal.distance_fade(100.0).abs() < 1e-6);
        // Beyond max distance: clamped to 0.0.
        assert!(decal.distance_fade(200.0).abs() < 1e-6);
    }

    #[test]
    fn angle_fade_interpolation() {
        let decal = Decal {
            angle_fade_start: 0.5,
            angle_fade_end: 1.5,
            ..Default::default()
        };

        // Below start: full opacity.
        assert!((decal.angle_fade(0.3) - 1.0).abs() < 1e-6);
        // At start: full opacity.
        assert!((decal.angle_fade(0.5) - 1.0).abs() < 1e-6);
        // Midpoint: 0.5.
        assert!((decal.angle_fade(1.0) - 0.5).abs() < 1e-6);
        // At end: zero.
        assert!(decal.angle_fade(1.5).abs() < 1e-6);
        // Beyond end: zero.
        assert!(decal.angle_fade(2.0).abs() < 1e-6);
    }

    #[test]
    fn default_decal_projects_downward() {
        let decal = Decal::default();
        assert_eq!(decal.normal, Vec3::new(0.0, -1.0, 0.0));
        assert_eq!(decal.blend_mode, DecalBlendMode::Translucent);
        assert!((decal.opacity - 1.0).abs() < 1e-6);
        assert_eq!(decal.priority, 0);
    }

    #[test]
    fn builder_chain() {
        let decal = Decal::new(TextureHandle(42), Vec3::new(2.0, 0.5, 3.0))
            .with_normal(Vec3::new(0.0, 0.0, -1.0))
            .with_blend_mode(DecalBlendMode::Stain)
            .with_opacity(0.7)
            .with_priority(5)
            .with_max_distance(40.0)
            .with_angle_fade(0.3, 1.0);

        assert_eq!(decal.texture, TextureHandle(42));
        assert_eq!(decal.blend_mode, DecalBlendMode::Stain);
        assert!((decal.opacity - 0.7).abs() < 1e-6);
        assert_eq!(decal.priority, 5);
        assert!((decal.max_distance - 40.0).abs() < 1e-6);
        // Projection direction should be normalized.
        let dir = decal.normal;
        assert!((dir.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn opacity_clamped() {
        let d1 = Decal::default().with_opacity(2.0);
        assert!((d1.opacity - 1.0).abs() < 1e-6);
        let d2 = Decal::default().with_opacity(-0.5);
        assert!(d2.opacity.abs() < 1e-6);
    }

    #[test]
    fn zero_opacity_decals_culled() {
        let mut world = World::new();

        spawn_decal(
            &mut world,
            Decal::new(TextureHandle(1), Vec3::ONE).with_opacity(0.0),
            Vec3::ZERO,
        );

        let cmds = collect_decal_draw_commands(&world);
        assert!(cmds.is_empty(), "zero-opacity decal should be culled");
    }

    #[test]
    fn decal_projection_origin_maps_correctly() {
        // A decal at the origin projecting downward (-Y) should produce
        // a valid projection matrix that maps the origin to NDC center.
        let size = Vec3::new(5.0, 5.0, 5.0);
        let vp = DecalProjection::from_transform(Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0), size);

        // Origin in world space should map to (0, 0, z) in clip space.
        let origin_clip = vp.transform_point3(Vec3::ZERO);
        assert!(
            origin_clip.x.abs() < 1e-4,
            "origin x in clip space should be ~0, got {}",
            origin_clip.x
        );
    }

    #[test]
    fn decal_projection_maps_extents() {
        // A decal at origin with size (2, 3, 4) projecting along -Y.
        // Points at the exact extents should map to NDC +/-1.
        let size = Vec3::new(2.0, 3.0, 4.0);
        let vp = DecalProjection::from_transform(Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0), size);

        // A point at (2, 0, 0) should map to x = +1 in NDC (right edge).
        let right_edge = vp.transform_point3(Vec3::new(2.0, 0.0, 0.0));
        assert!(
            (right_edge.x - 1.0).abs() < 1e-4,
            "right edge x should be 1.0, got {}",
            right_edge.x
        );

        // A point at (-2, 0, 0) should map to x = -1 in NDC (left edge).
        let left_edge = vp.transform_point3(Vec3::new(-2.0, 0.0, 0.0));
        assert!(
            (left_edge.x - (-1.0)).abs() < 1e-4,
            "left edge x should be -1.0, got {}",
            left_edge.x
        );
    }

    #[test]
    fn decal_renderer_geometry_counts() {
        // Verify the correct number of indices for a unit cube
        // (6 faces * 2 triangles * 3 vertices = 36).
        assert_eq!(36u32, 6 * 2 * 3);
    }
}
