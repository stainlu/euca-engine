//! GPU particle rendering module.
//!
//! Produces renderer-agnostic draw data from particle emitters. The renderer
//! consumes [`ParticleRenderBatch`] batches (one per emitter) containing
//! per-particle vertex data and blend mode.
//!
//! # Architecture
//!
//! This module follows the same pattern as `euca_ui::UiDrawCommand`: the
//! simulation crate owns no GPU resources. Instead, it outputs a flat data
//! structure that the renderer interprets to build vertex/instance buffers,
//! bind textures, and select the correct pipeline.
//!
//! Billboard quads are assembled from camera-relative axes so every particle
//! always faces the viewer.

use euca_ecs::{Query, World};
use euca_math::Vec3;
use serde::{Deserialize, Serialize};

use crate::ParticleEmitter;

// ── Blend mode ──────────────────────────────────────────────────────────────

/// How particle quads are composited into the framebuffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParticleBlendMode {
    /// Standard alpha blending: `src * src_a + dst * (1 - src_a)`.
    /// Suitable for smoke, dust, fog, and soft effects.
    AlphaBlend,
    /// Additive blending: `src * src_a + dst`.
    /// Suitable for fire, sparks, lightning, magic effects.
    Additive,
}

impl Default for ParticleBlendMode {
    fn default() -> Self {
        Self::AlphaBlend
    }
}

// ── Per-particle vertex data ────────────────────────────────────────────────

/// Per-particle data produced for the renderer.
///
/// Contains everything the GPU needs to draw one particle: world-space
/// position, billboard size, vertex color, and UV bounds within a texture
/// atlas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleVertex {
    /// World-space center of the particle.
    pub position: Vec3,
    /// Uniform scale (half-extent of the billboard quad).
    pub size: f32,
    /// RGBA color.
    pub color: [f32; 4],
    /// Top-left UV coordinate within the bound texture (atlas sub-region).
    pub uv_min: [f32; 2],
    /// Bottom-right UV coordinate within the bound texture (atlas sub-region).
    pub uv_max: [f32; 2],
}

// ── Render data output ──────────────────────────────────────────────────────

/// Renderer-consumable batch of particle draw data for a single emitter.
///
/// Follows the same pattern as `UiDrawCommand`: the simulation crate produces
/// this struct; the render crate consumes it to issue GPU draw calls.
#[derive(Clone, Debug)]
pub struct ParticleRenderBatch {
    /// Per-particle vertex data, sorted back-to-front for correct
    /// alpha-blending (the collector handles sorting).
    pub vertices: Vec<ParticleVertex>,
    /// How to composite these particles.
    pub blend_mode: ParticleBlendMode,
}

impl ParticleRenderBatch {
    /// Number of particles in this batch.
    pub fn particle_count(&self) -> usize {
        self.vertices.len()
    }

    /// Whether this batch has any particles to draw.
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Build billboard vertex + index buffers for the entire batch.
    ///
    /// Returns `(vertices, indices)` ready for GPU upload.
    pub fn build_billboard_geometry(
        &self,
        axes: &BillboardAxes,
    ) -> (Vec<BillboardVertex>, Vec<u32>) {
        let count = self.vertices.len();
        let mut verts = Vec::with_capacity(count * 4);
        let mut indices = Vec::with_capacity(count * 6);

        for (i, pv) in self.vertices.iter().enumerate() {
            let quad = axes.quad_vertices(pv.position, pv.size, pv.uv_min, pv.uv_max);
            verts.extend_from_slice(&quad);

            let base = (i as u32) * 4;
            for &offset in &QUAD_INDICES {
                indices.push(base + offset);
            }
        }

        (verts, indices)
    }
}

// ── Billboard quad vertices ─────────────────────────────────────────────────

/// A single vertex of a billboard quad (position + UV).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BillboardVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

/// Camera orientation used to construct billboard quads.
///
/// Both axes must be unit-length and perpendicular to the camera's forward
/// direction.
#[derive(Clone, Copy, Debug)]
pub struct BillboardAxes {
    /// Camera's local right direction in world space.
    pub right: Vec3,
    /// Camera's local up direction in world space.
    pub up: Vec3,
}

impl BillboardAxes {
    /// Derive billboard axes from camera eye and target positions.
    ///
    /// `world_up` is typically `Vec3::new(0.0, 1.0, 0.0)`.
    pub fn from_camera(eye: Vec3, target: Vec3, world_up: Vec3) -> Self {
        let forward = (target - eye).normalize();
        let right = forward.cross(world_up).normalize();
        let up = right.cross(forward).normalize();
        Self { right, up }
    }

    /// Build the four corners of a camera-facing quad for one particle.
    ///
    /// Vertices are in counter-clockwise order (bottom-left, bottom-right,
    /// top-right, top-left) suitable for a two-triangle strip or indexed draw.
    pub fn quad_vertices(
        &self,
        center: Vec3,
        half_size: f32,
        uv_min: [f32; 2],
        uv_max: [f32; 2],
    ) -> [BillboardVertex; 4] {
        let r = self.right * half_size;
        let u = self.up * half_size;

        let bl = center - r - u;
        let br = center + r - u;
        let tr = center + r + u;
        let tl = center - r + u;

        [
            BillboardVertex {
                position: [bl.x, bl.y, bl.z],
                uv: [uv_min[0], uv_max[1]],
            },
            BillboardVertex {
                position: [br.x, br.y, br.z],
                uv: [uv_max[0], uv_max[1]],
            },
            BillboardVertex {
                position: [tr.x, tr.y, tr.z],
                uv: [uv_max[0], uv_min[1]],
            },
            BillboardVertex {
                position: [tl.x, tl.y, tl.z],
                uv: [uv_min[0], uv_min[1]],
            },
        ]
    }
}

/// Indices for a single quad (two triangles, counter-clockwise winding).
/// Offset each set by `quad_index * 4`.
pub const QUAD_INDICES: [u32; 6] = [0, 1, 2, 2, 3, 0];

/// Build index data for `quad_count` quads.
pub fn build_quad_indices(quad_count: u32) -> Vec<u32> {
    let mut indices = Vec::with_capacity(quad_count as usize * 6);
    for i in 0..quad_count {
        let base = i * 4;
        for &offset in &QUAD_INDICES {
            indices.push(base + offset);
        }
    }
    indices
}

// ── Atlas UV computation ────────────────────────────────────────────────────

/// Compute the UV sub-region for a particle based on its age fraction and
/// the emitter's atlas grid dimensions.
///
/// When `animate` is true and the atlas has multiple cells (`cols * rows > 1`),
/// the particle's age fraction [0..1] selects a frame index that advances
/// linearly through the atlas (left-to-right, top-to-bottom).
///
/// Returns `(uv_min, uv_max)`.
fn atlas_uv_for_particle(
    age_fraction: f32,
    cols: u32,
    rows: u32,
    animate: bool,
) -> ([f32; 2], [f32; 2]) {
    let total_frames = cols * rows;
    if !animate || total_frames <= 1 {
        return ([0.0, 0.0], [1.0, 1.0]);
    }

    let frame = (age_fraction * total_frames as f32).min(total_frames as f32 - 1.0) as u32;
    let col = frame % cols;
    let row = frame / cols;

    let tile_w = 1.0 / cols as f32;
    let tile_h = 1.0 / rows as f32;

    let u_min = col as f32 * tile_w;
    let v_min = row as f32 * tile_h;
    let u_max = (col + 1) as f32 * tile_w;
    let v_max = (row + 1) as f32 * tile_h;

    ([u_min, v_min], [u_max, v_max])
}

// ── Collection system ───────────────────────────────────────────────────────

/// Collect render data from all active particle emitters in the world.
///
/// Each emitter produces one [`ParticleRenderBatch`] batch. Particles within
/// each batch are sorted back-to-front relative to `camera_pos` for correct
/// transparency rendering.
///
/// Billboard orientation is derived from `camera_pos` using a standard
/// world-up vector of `(0, 1, 0)`.
pub fn collect_particle_render_data(world: &World, camera_pos: Vec3) -> Vec<ParticleRenderBatch> {
    let mut batches = Vec::new();

    let query = Query::<&ParticleEmitter>::new(world);

    for emitter in query.iter() {
        if emitter.particles.is_empty() {
            continue;
        }

        // Build per-particle vertex data.
        let mut vertices: Vec<ParticleVertex> = emitter
            .particles
            .iter()
            .map(|p| {
                let age_frac = p.age_fraction();
                let color = emitter.color_at(age_frac);
                let (uv_min, uv_max) = atlas_uv_for_particle(
                    age_frac,
                    emitter.atlas_cols,
                    emitter.atlas_rows,
                    emitter.animate_atlas,
                );

                ParticleVertex {
                    position: p.position,
                    size: p.size,
                    color,
                    uv_min,
                    uv_max,
                }
            })
            .collect();

        // Sort back-to-front by squared distance to camera (avoids sqrt).
        vertices.sort_by(|a, b| {
            let dist_a = sq_dist(a.position, camera_pos);
            let dist_b = sq_dist(b.position, camera_pos);
            // Reverse: farther particles first.
            dist_b
                .partial_cmp(&dist_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        batches.push(ParticleRenderBatch {
            vertices,
            blend_mode: emitter.blend_mode,
        });
    }

    batches
}

/// Squared distance from a particle position to the camera.
fn sq_dist(pos: Vec3, cam: Vec3) -> f32 {
    let dx = pos.x - cam.x;
    let dy = pos.y - cam.y;
    let dz = pos.z - cam.z;
    dx * dx + dy * dy + dz * dz
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EmitterConfig, Particle, ParticleEmitter};
    use euca_math::Vec3;
    use euca_scene::GlobalTransform;

    /// Helper: create a test emitter with pre-populated particles.
    fn emitter_with_particles(particles: Vec<Particle>) -> ParticleEmitter {
        let mut em = ParticleEmitter::new(EmitterConfig::default());
        em.particles = particles;
        em
    }

    fn make_particle(pos: Vec3, size: f32, age: f32, lifetime: f32) -> Particle {
        Particle {
            position: pos,
            velocity: Vec3::ZERO,
            age,
            lifetime,
            size,
        }
    }

    // ── Test 1: Atlas UV grid computation ────────────────────────────────

    #[test]
    fn atlas_uv_first_and_last_frame() {
        // 4x2 atlas = 8 frames
        let (uv_min, uv_max) = atlas_uv_for_particle(0.0, 4, 2, true);
        assert!((uv_min[0] - 0.0).abs() < 1e-6);
        assert!((uv_min[1] - 0.0).abs() < 1e-6);
        assert!((uv_max[0] - 0.25).abs() < 1e-6);
        assert!((uv_max[1] - 0.5).abs() < 1e-6);

        // age_fraction=1.0 -> last frame (frame 7 = col 3, row 1)
        let (uv_min, uv_max) = atlas_uv_for_particle(1.0, 4, 2, true);
        assert!((uv_min[0] - 0.75).abs() < 1e-6);
        assert!((uv_min[1] - 0.5).abs() < 1e-6);
        assert!((uv_max[0] - 1.0).abs() < 1e-6);
        assert!((uv_max[1] - 1.0).abs() < 1e-6);
    }

    // ── Test 2: Non-animated atlas returns full UV ───────────────────────

    #[test]
    fn non_animated_atlas_returns_full_uv() {
        let (uv_min, uv_max) = atlas_uv_for_particle(0.5, 4, 4, false);
        assert_eq!(uv_min, [0.0, 0.0]);
        assert_eq!(uv_max, [1.0, 1.0]);

        // Single-cell atlas always returns full UV even when animated.
        let (uv_min, uv_max) = atlas_uv_for_particle(0.5, 1, 1, true);
        assert_eq!(uv_min, [0.0, 0.0]);
        assert_eq!(uv_max, [1.0, 1.0]);
    }

    // ── Test 3: Billboard quad vertices are centered on particle ────────

    #[test]
    fn billboard_quad_vertices_are_centered_on_particle() {
        let axes = BillboardAxes::from_camera(
            Vec3::new(0.0, 0.0, -5.0),
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
        );

        let center = Vec3::new(1.0, 2.0, 3.0);
        let verts = axes.quad_vertices(center, 0.5, [0.0, 0.0], [1.0, 1.0]);

        // The average of the four corners should be the particle center.
        let avg_x: f32 = verts.iter().map(|v| v.position[0]).sum::<f32>() / 4.0;
        let avg_y: f32 = verts.iter().map(|v| v.position[1]).sum::<f32>() / 4.0;
        let avg_z: f32 = verts.iter().map(|v| v.position[2]).sum::<f32>() / 4.0;

        assert!((avg_x - center.x).abs() < 1e-5);
        assert!((avg_y - center.y).abs() < 1e-5);
        assert!((avg_z - center.z).abs() < 1e-5);
    }

    // ── Test 4: Quad index generation ───────────────────────────────────

    #[test]
    fn quad_indices_correct_for_multiple_quads() {
        let indices = build_quad_indices(3);
        assert_eq!(indices.len(), 18); // 3 quads * 6 indices

        // First quad: 0,1,2,2,3,0
        assert_eq!(&indices[0..6], &[0, 1, 2, 2, 3, 0]);
        // Second quad: 4,5,6,6,7,4
        assert_eq!(&indices[6..12], &[4, 5, 6, 6, 7, 4]);
        // Third quad: 8,9,10,10,11,8
        assert_eq!(&indices[12..18], &[8, 9, 10, 10, 11, 8]);
    }

    // ── Test 5: Render data collection with back-to-front sorting ───────

    #[test]
    fn collect_render_data_produces_sorted_vertices() {
        let mut world = World::new();

        // Emitter with particles at known positions.
        let p_near = make_particle(Vec3::new(0.0, 0.0, 1.0), 0.5, 0.0, 2.0);
        let p_far = make_particle(Vec3::new(0.0, 0.0, 10.0), 0.5, 0.0, 2.0);
        let p_mid = make_particle(Vec3::new(0.0, 0.0, 5.0), 0.5, 0.0, 2.0);

        let em = emitter_with_particles(vec![p_near, p_far, p_mid]);
        let entity = world.spawn(em);
        world.insert(entity, GlobalTransform::default());

        let camera_pos = Vec3::ZERO;
        let batches = collect_particle_render_data(&world, camera_pos);

        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.vertices.len(), 3);

        // Back-to-front: farthest first (z=10), then z=5, then z=1.
        assert!((batch.vertices[0].position.z - 10.0).abs() < 1e-5);
        assert!((batch.vertices[1].position.z - 5.0).abs() < 1e-5);
        assert!((batch.vertices[2].position.z - 1.0).abs() < 1e-5);
    }

    // ── Test 6: Blend mode propagation ──────────────────────────────────

    #[test]
    fn blend_mode_propagates_from_emitter() {
        let mut world = World::new();

        let p = make_particle(Vec3::ZERO, 1.0, 0.0, 1.0);
        let mut em = emitter_with_particles(vec![p]);
        em.blend_mode = ParticleBlendMode::Additive;

        let entity = world.spawn(em);
        world.insert(entity, GlobalTransform::default());

        let batches = collect_particle_render_data(&world, Vec3::ZERO);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].blend_mode, ParticleBlendMode::Additive);
    }

    // ── Test 7: Empty emitters produce no batches ───────────────────────

    #[test]
    fn empty_emitters_produce_no_batches() {
        let mut world = World::new();
        let em = ParticleEmitter::new(EmitterConfig::default()); // no particles
        let entity = world.spawn(em);
        world.insert(entity, GlobalTransform::default());

        let batches = collect_particle_render_data(&world, Vec3::ZERO);
        assert!(batches.is_empty());
    }

    // ── Test 8: Geometry builder produces correct buffer sizes ──────────

    #[test]
    fn render_data_builds_correct_geometry_count() {
        let data = ParticleRenderBatch {
            vertices: vec![
                ParticleVertex {
                    position: Vec3::ZERO,
                    size: 1.0,
                    color: [1.0; 4],
                    uv_min: [0.0, 0.0],
                    uv_max: [1.0, 1.0],
                },
                ParticleVertex {
                    position: Vec3::new(1.0, 0.0, 0.0),
                    size: 0.5,
                    color: [1.0; 4],
                    uv_min: [0.0, 0.0],
                    uv_max: [1.0, 1.0],
                },
            ],
            blend_mode: ParticleBlendMode::Additive,
        };

        let axes = BillboardAxes {
            right: Vec3::new(1.0, 0.0, 0.0),
            up: Vec3::new(0.0, 1.0, 0.0),
        };

        let (verts, indices) = data.build_billboard_geometry(&axes);
        assert_eq!(verts.len(), 8); // 2 particles * 4 vertices
        assert_eq!(indices.len(), 12); // 2 particles * 6 indices
        assert_eq!(data.particle_count(), 2);
        assert!(!data.is_empty());
    }

    // ── Test 9: Atlas animation selects correct frame by age ────────────

    #[test]
    fn atlas_animation_mid_frame_selection() {
        // 4x1 atlas = 4 frames
        // age_fraction=0.5 -> frame 2 -> col 2, row 0
        let (uv_min, _uv_max) = atlas_uv_for_particle(0.5, 4, 1, true);
        assert!((uv_min[0] - 0.5).abs() < 1e-6);
        assert!((uv_min[1] - 0.0).abs() < 1e-6);
    }

    // ── Test 10: Color interpolation appears in vertex output ───────────

    #[test]
    fn particle_vertex_color_reflects_age() {
        let mut world = World::new();

        let mut em = ParticleEmitter::new(EmitterConfig {
            color_start: [1.0, 1.0, 1.0, 1.0],
            color_end: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        });
        // Insert a particle at half its lifetime.
        em.particles.push(Particle {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            age: 1.0,
            lifetime: 2.0,
            size: 1.0,
        });

        let entity = world.spawn(em);
        world.insert(entity, GlobalTransform::default());

        let batches = collect_particle_render_data(&world, Vec3::ZERO);
        assert_eq!(batches.len(), 1);
        let color = batches[0].vertices[0].color;
        // At age_fraction=0.5, each channel should be ~0.5.
        for ch in &color {
            assert!((*ch - 0.5).abs() < 0.01);
        }
    }
}
