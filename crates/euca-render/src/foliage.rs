//! Instanced foliage rendering system.
//!
//! Manages vegetation layers (grass, bushes, trees) as instanced geometry.
//! Each `FoliageLayer` defines a mesh/material pair plus placement parameters.
//! `scatter_foliage` distributes instances via Poisson disk sampling for
//! natural-looking distribution, and `FoliageRenderer` culls instances per
//! frame to produce `FoliageDrawData` for the GPU pipeline.
//!
//! # Pipeline
//! 1. Define a `FoliageLayer` with mesh, material, density, scale range, etc.
//! 2. Call `scatter_foliage` to populate instances within an area.
//! 3. Each frame, call `FoliageRenderer::collect_visible_instances` to cull
//!    by distance and frustum, producing `FoliageDrawData` with model matrices.

use crate::camera::Frustum;
use crate::material::MaterialHandle;
use crate::mesh::MeshHandle;
use euca_math::{Mat4, Quat, Vec3};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A single foliage instance placed in the world.
#[derive(Clone, Debug)]
pub struct FoliageInstance {
    /// World-space position.
    pub position: Vec3,
    /// Y-axis rotation in radians.
    pub rotation: f32,
    /// Uniform scale factor.
    pub scale: f32,
}

/// A layer of instanced foliage sharing the same mesh and material.
#[derive(Clone, Debug)]
pub struct FoliageLayer {
    /// Mesh used for all instances in this layer.
    pub mesh: MeshHandle,
    /// Material used for all instances in this layer.
    pub material: MaterialHandle,
    /// Target instances per square unit (XZ plane).
    pub density: f32,
    /// Minimum random scale factor.
    pub min_scale: f32,
    /// Maximum random scale factor.
    pub max_scale: f32,
    /// Instances beyond this distance from the camera are culled.
    pub max_distance: f32,
    /// Placed instances.
    pub instances: Vec<FoliageInstance>,
}

/// Renderer-consumable draw data for one foliage layer, ready for GPU submission.
///
/// Follows the same pattern as `DecalDrawCommand` and `ParticleRenderBatch`:
/// the foliage system produces this struct; the render backend consumes it to
/// issue instanced draw calls.
#[derive(Clone, Debug)]
pub struct FoliageDrawData {
    /// Mesh to draw.
    pub mesh: MeshHandle,
    /// Material to bind.
    pub material: MaterialHandle,
    /// Per-instance model matrices for all visible instances.
    pub instance_matrices: Vec<Mat4>,
}

impl FoliageDrawData {
    /// Number of visible instances to draw.
    pub fn instance_count(&self) -> usize {
        self.instance_matrices.len()
    }

    /// Whether there are any visible instances.
    pub fn is_empty(&self) -> bool {
        self.instance_matrices.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Poisson disk sampling (2D on XZ plane)
// ---------------------------------------------------------------------------

/// Simple seeded pseudo-random number generator (xoshiro128+).
/// Avoids external RNG dependencies.
struct Rng {
    state: [u32; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        // Split 64-bit seed into four 32-bit values via mixing.
        let s0 = (seed & 0xFFFF_FFFF) as u32 | 1;
        let s1 = ((seed >> 32) & 0xFFFF_FFFF) as u32 | 1;
        let s2 = s0.wrapping_mul(0x9E3779B9);
        let s3 = s1.wrapping_mul(0x9E3779B9);
        Self {
            state: [s0, s1, s2, s3],
        }
    }

    /// Generate a random u32 (xoshiro128+).
    fn next_u32(&mut self) -> u32 {
        let result = self.state[0].wrapping_add(self.state[3]);
        let t = self.state[1] << 9;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(11);

        result
    }

    /// Generate a random f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Generate a random f32 in [lo, hi).
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

/// Populate a foliage layer with instances using Poisson disk sampling.
///
/// Samples are distributed on the XZ plane between `area_min` and `area_max`.
/// The Y component of each instance is linearly interpolated between the min
/// and max Y values (useful for flat terrain; for heightmap terrain, a
/// post-placement height adjustment pass is expected).
///
/// The `seed` parameter ensures deterministic results.
pub fn scatter_foliage(layer: &mut FoliageLayer, area_min: Vec3, area_max: Vec3, seed: u64) {
    layer.instances.clear();

    if layer.density <= 0.0 {
        return;
    }

    let width = area_max.x - area_min.x;
    let depth = area_max.z - area_min.z;

    if width <= 0.0 || depth <= 0.0 {
        return;
    }

    // Minimum distance between samples derived from density.
    // density = instances / unit^2, so average area per instance = 1/density.
    // For Poisson disk, the disk radius is related to the cell spacing.
    let min_dist = (1.0 / layer.density).sqrt();

    // Grid-accelerated Poisson disk sampling (Bridson's algorithm).
    let cell_size = min_dist / std::f32::consts::SQRT_2;
    let grid_w = ((width / cell_size).ceil() as usize).max(1);
    let grid_h = ((depth / cell_size).ceil() as usize).max(1);

    // -1 means empty cell
    let mut grid = vec![-1i32; grid_w * grid_h];
    let mut points: Vec<[f32; 2]> = Vec::new();
    let mut active: Vec<usize> = Vec::new();

    let mut rng = Rng::new(seed);

    let grid_index = |x: f32, z: f32| -> (usize, usize) {
        let gx = ((x - area_min.x) / cell_size) as usize;
        let gz = ((z - area_min.z) / cell_size) as usize;
        (gx.min(grid_w - 1), gz.min(grid_h - 1))
    };

    // Seed point
    let start_x = rng.range(area_min.x, area_max.x);
    let start_z = rng.range(area_min.z, area_max.z);
    let (gx, gz) = grid_index(start_x, start_z);
    grid[gz * grid_w + gx] = 0;
    points.push([start_x, start_z]);
    active.push(0);

    let max_attempts = 30u32;
    let min_dist_sq = min_dist * min_dist;

    while !active.is_empty() {
        // Pick a random active point.
        let active_idx = (rng.next_u32() as usize) % active.len();
        let point_idx = active[active_idx];
        let [px, pz] = points[point_idx];

        let mut found = false;
        for _ in 0..max_attempts {
            // Generate a candidate in the annulus [min_dist, 2 * min_dist].
            let angle = rng.next_f32() * std::f32::consts::TAU;
            let radius = rng.range(min_dist, 2.0 * min_dist);
            let cx = px + radius * angle.cos();
            let cz = pz + radius * angle.sin();

            // Bounds check.
            if cx < area_min.x || cx >= area_max.x || cz < area_min.z || cz >= area_max.z {
                continue;
            }

            let (cgx, cgz) = grid_index(cx, cz);

            // Check neighbors in a 5x5 grid window.
            let mut too_close = false;
            let search_min_x = cgx.saturating_sub(2);
            let search_min_z = cgz.saturating_sub(2);
            let search_max_x = (cgx + 3).min(grid_w);
            let search_max_z = (cgz + 3).min(grid_h);

            'outer: for nz in search_min_z..search_max_z {
                for nx in search_min_x..search_max_x {
                    let ni = grid[nz * grid_w + nx];
                    if ni >= 0 {
                        let [nx_pos, nz_pos] = points[ni as usize];
                        let dx = cx - nx_pos;
                        let dz = cz - nz_pos;
                        if dx * dx + dz * dz < min_dist_sq {
                            too_close = true;
                            break 'outer;
                        }
                    }
                }
            }

            if !too_close {
                let new_idx = points.len() as i32;
                grid[cgz * grid_w + cgx] = new_idx;
                points.push([cx, cz]);
                active.push(new_idx as usize);
                found = true;
            }
        }

        if !found {
            active.swap_remove(active_idx);
        }
    }

    // Convert 2D points to FoliageInstances.
    layer.instances.reserve(points.len());
    for [px, pz] in &points {
        let y = area_min.y + (area_max.y - area_min.y) * rng.next_f32();
        let rotation = rng.range(0.0, std::f32::consts::TAU);
        let scale = rng.range(layer.min_scale, layer.max_scale);

        layer.instances.push(FoliageInstance {
            position: Vec3::new(*px, y, *pz),
            rotation,
            scale,
        });
    }
}

// ---------------------------------------------------------------------------
// Foliage renderer
// ---------------------------------------------------------------------------

/// Manages per-frame visibility determination for foliage layers.
///
/// Stateless: all culling state is computed fresh each frame from the camera
/// parameters. This keeps the renderer simple and avoids stale-state bugs.
pub struct FoliageRenderer;

impl FoliageRenderer {
    /// Cull a foliage layer's instances by distance and frustum, returning
    /// model matrices for all visible instances.
    ///
    /// The returned matrices encode position, Y-axis rotation, and uniform
    /// scale -- ready for instanced draw submission.
    pub fn collect_visible_instances(
        layer: &FoliageLayer,
        camera_pos: Vec3,
        frustum: &Frustum,
    ) -> Vec<Mat4> {
        let max_dist_sq = layer.max_distance * layer.max_distance;

        layer
            .instances
            .iter()
            .filter_map(|inst| {
                // Distance cull (squared distance avoids sqrt).
                let dx = inst.position.x - camera_pos.x;
                let dy = inst.position.y - camera_pos.y;
                let dz = inst.position.z - camera_pos.z;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                if dist_sq > max_dist_sq {
                    return None;
                }

                // Frustum cull: treat each instance as a point-sized AABB
                // scaled by the instance's scale (conservative approximation).
                let half = Vec3::new(inst.scale, inst.scale, inst.scale);
                if !frustum.intersects_aabb(inst.position, half) {
                    return None;
                }

                // Build model matrix: scale * rotation(Y) * translation.
                let rotation = Quat::from_axis_angle(Vec3::Y, inst.rotation);
                let scale = Vec3::new(inst.scale, inst.scale, inst.scale);
                Some(Mat4::from_scale_rotation_translation(
                    scale,
                    rotation,
                    inst.position,
                ))
            })
            .collect()
    }

    /// Produce a `FoliageDrawData` for a layer, ready for GPU consumption.
    pub fn build_draw_data(
        layer: &FoliageLayer,
        camera_pos: Vec3,
        frustum: &Frustum,
    ) -> FoliageDrawData {
        let instance_matrices = Self::collect_visible_instances(layer, camera_pos, frustum);
        FoliageDrawData {
            mesh: layer.mesh,
            material: layer.material,
            instance_matrices,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Vec3;

    /// Helper: create a layer with default settings for testing.
    fn test_layer(density: f32) -> FoliageLayer {
        FoliageLayer {
            mesh: MeshHandle(0),
            material: MaterialHandle(0),
            density,
            min_scale: 0.5,
            max_scale: 1.5,
            max_distance: 100.0,
            instances: Vec::new(),
        }
    }

    /// Helper: create a wide frustum that contains everything in a large area.
    fn wide_frustum() -> Frustum {
        let cam = crate::camera::Camera {
            eye: Vec3::new(0.0, 10.0, 0.0),
            target: Vec3::new(0.0, 0.0, 0.01),
            up: Vec3::Y,
            fov_y: std::f32::consts::FRAC_PI_2,
            near: 0.1,
            far: 1000.0,
            orthographic: false,
            ortho_size: 10.0,
        };
        let vp = cam.view_projection_matrix(1.0);
        Frustum::from_view_projection(&vp)
    }

    // ── Test 1: Scatter density produces reasonable instance count ────────

    #[test]
    fn scatter_density_produces_expected_count() {
        let mut layer = test_layer(1.0); // 1 instance per square unit
        let area_min = Vec3::new(0.0, 0.0, 0.0);
        let area_max = Vec3::new(10.0, 0.0, 10.0);

        scatter_foliage(&mut layer, area_min, area_max, 42);

        // Area = 100 sq units, density = 1.0, so expect ~100 instances.
        // Poisson disk sampling yields fewer than uniform grid due to spacing
        // constraints, but should be within a reasonable range.
        let count = layer.instances.len();
        assert!(
            count >= 50 && count <= 150,
            "Expected ~100 instances for density=1.0 on 10x10 area, got {count}"
        );
    }

    // ── Test 2: Distance culling removes far instances ───────────────────

    #[test]
    fn distance_culling_removes_far_instances() {
        let mut layer = test_layer(1.0);
        layer.max_distance = 10.0;

        // Place instances: one near, one far.
        layer.instances = vec![
            FoliageInstance {
                position: Vec3::new(5.0, 0.0, 0.0),
                rotation: 0.0,
                scale: 1.0,
            },
            FoliageInstance {
                position: Vec3::new(50.0, 0.0, 0.0),
                rotation: 0.0,
                scale: 1.0,
            },
        ];

        let camera_pos = Vec3::ZERO;
        let frustum = wide_frustum();
        let visible = FoliageRenderer::collect_visible_instances(&layer, camera_pos, &frustum);

        // Only the near instance (at distance 5) should survive; the far one
        // (at distance 50) exceeds max_distance=10.
        assert_eq!(
            visible.len(),
            1,
            "Expected 1 visible instance after distance cull, got {}",
            visible.len()
        );
    }

    // ── Test 3: Scale range is respected ─────────────────────────────────

    #[test]
    fn scatter_respects_scale_range() {
        let mut layer = test_layer(4.0);
        layer.min_scale = 0.8;
        layer.max_scale = 1.2;

        let area_min = Vec3::new(0.0, 0.0, 0.0);
        let area_max = Vec3::new(10.0, 0.0, 10.0);

        scatter_foliage(&mut layer, area_min, area_max, 123);

        assert!(
            !layer.instances.is_empty(),
            "Should have scattered some instances"
        );
        for inst in &layer.instances {
            assert!(
                inst.scale >= 0.8 && inst.scale <= 1.2,
                "Instance scale {} is outside range [0.8, 1.2]",
                inst.scale
            );
        }
    }

    // ── Test 4: Frustum culling removes off-screen instances ─────────────

    #[test]
    fn frustum_culling_removes_outside_instances() {
        let mut layer = test_layer(1.0);
        layer.max_distance = 1000.0;

        // Camera looks along +Z from origin. Place one instance in front
        // of the camera and one far behind.
        let cam = crate::camera::Camera {
            eye: Vec3::ZERO,
            target: Vec3::new(0.0, 0.0, 10.0),
            up: Vec3::Y,
            fov_y: std::f32::consts::FRAC_PI_4,
            near: 0.1,
            far: 500.0,
            orthographic: false,
            ortho_size: 10.0,
        };
        let vp = cam.view_projection_matrix(1.0);
        let frustum = Frustum::from_view_projection(&vp);

        layer.instances = vec![
            // In front of camera, on-screen.
            FoliageInstance {
                position: Vec3::new(0.0, 0.0, 20.0),
                rotation: 0.0,
                scale: 1.0,
            },
            // Behind the camera, should be culled.
            FoliageInstance {
                position: Vec3::new(0.0, 0.0, -50.0),
                rotation: 0.0,
                scale: 1.0,
            },
        ];

        let visible = FoliageRenderer::collect_visible_instances(&layer, cam.eye, &frustum);

        assert_eq!(
            visible.len(),
            1,
            "Expected 1 visible instance (behind-camera instance should be frustum-culled), got {}",
            visible.len()
        );
    }

    // ── Test 5: Seed determinism ─────────────────────────────────────────

    #[test]
    fn scatter_is_deterministic_with_same_seed() {
        let area_min = Vec3::new(0.0, 0.0, 0.0);
        let area_max = Vec3::new(20.0, 0.0, 20.0);

        let mut layer_a = test_layer(2.0);
        scatter_foliage(&mut layer_a, area_min, area_max, 999);

        let mut layer_b = test_layer(2.0);
        scatter_foliage(&mut layer_b, area_min, area_max, 999);

        assert_eq!(
            layer_a.instances.len(),
            layer_b.instances.len(),
            "Same seed should produce same instance count"
        );

        for (a, b) in layer_a.instances.iter().zip(layer_b.instances.iter()) {
            assert!(
                (a.position.x - b.position.x).abs() < 1e-6
                    && (a.position.y - b.position.y).abs() < 1e-6
                    && (a.position.z - b.position.z).abs() < 1e-6,
                "Positions should be identical for the same seed"
            );
            assert!(
                (a.rotation - b.rotation).abs() < 1e-6,
                "Rotations should be identical for the same seed"
            );
            assert!(
                (a.scale - b.scale).abs() < 1e-6,
                "Scales should be identical for the same seed"
            );
        }
    }

    // ── Test 6: Draw data output struct ──────────────────────────────────

    #[test]
    fn build_draw_data_populates_mesh_and_material() {
        let mut layer = test_layer(1.0);
        layer.mesh = MeshHandle(7);
        layer.material = MaterialHandle(3);
        layer.instances.push(FoliageInstance {
            position: Vec3::new(0.0, 0.0, 5.0),
            rotation: 0.0,
            scale: 1.0,
        });

        let frustum = wide_frustum();
        let draw_data = FoliageRenderer::build_draw_data(&layer, Vec3::ZERO, &frustum);

        assert_eq!(draw_data.mesh, MeshHandle(7));
        assert_eq!(draw_data.material, MaterialHandle(3));
        assert!(!draw_data.is_empty());
        assert_eq!(draw_data.instance_count(), 1);
    }

    // ── Test 7: Empty layer produces empty draw data ─────────────────────

    #[test]
    fn empty_layer_produces_empty_draw_data() {
        let layer = test_layer(1.0);
        let frustum = wide_frustum();
        let draw_data = FoliageRenderer::build_draw_data(&layer, Vec3::ZERO, &frustum);

        assert!(draw_data.is_empty());
        assert_eq!(draw_data.instance_count(), 0);
    }

    // ── Test 8: Zero density produces no instances ───────────────────────

    #[test]
    fn zero_density_produces_no_instances() {
        let mut layer = test_layer(0.0);
        let area_min = Vec3::new(0.0, 0.0, 0.0);
        let area_max = Vec3::new(10.0, 0.0, 10.0);

        scatter_foliage(&mut layer, area_min, area_max, 42);

        assert!(
            layer.instances.is_empty(),
            "Zero density should produce no instances"
        );
    }

    // ── Test 9: Model matrix encodes position correctly ──────────────────

    #[test]
    fn model_matrix_encodes_position() {
        let mut layer = test_layer(1.0);
        layer.max_distance = 1000.0;
        // Place on the XZ plane directly below the wide_frustum camera,
        // which looks down from (0,10,0) toward the origin.
        let pos = Vec3::new(1.0, 0.0, 1.0);
        layer.instances.push(FoliageInstance {
            position: pos,
            rotation: 0.0,
            scale: 1.0,
        });

        let frustum = wide_frustum();
        let matrices =
            FoliageRenderer::collect_visible_instances(&layer, Vec3::new(0.0, 10.0, 0.0), &frustum);

        assert_eq!(matrices.len(), 1);
        // Translation is stored in column 3 of the model matrix.
        let mat = &matrices[0];
        assert!(
            (mat.cols[3][0] - pos.x).abs() < 1e-5,
            "Translation X mismatch"
        );
        assert!(
            (mat.cols[3][1] - pos.y).abs() < 1e-5,
            "Translation Y mismatch"
        );
        assert!(
            (mat.cols[3][2] - pos.z).abs() < 1e-5,
            "Translation Z mismatch"
        );
    }
}
