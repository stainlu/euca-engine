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
use crate::compute::{ComputePipeline, ComputePipelineDesc, GpuBuffer};
use crate::gpu_driven::GpuFrustumData;
use crate::material::MaterialHandle;
use crate::mesh::MeshHandle;
use euca_math::{Mat4, Quat, Vec3};

// ---------------------------------------------------------------------------
// GPU foliage culling — shader & constants
// ---------------------------------------------------------------------------

/// WGSL compute shader source for GPU foliage instance culling.
pub const FOLIAGE_CULL_SHADER: &str = include_str!("../shaders/foliage_cull.wgsl");

/// Workgroup size used by the foliage cull shader (must match `@workgroup_size` in WGSL).
const FOLIAGE_CULL_WORKGROUP_SIZE: u32 = 64;

/// Size of one `ModelMatrix` in bytes (4 x vec4<f32> = 64 bytes).
const MODEL_MATRIX_SIZE: u64 = 64;

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
// GPU foliage culling — CPU-side structs mirroring WGSL layout
// ---------------------------------------------------------------------------

/// GPU-side foliage instance for upload to storage buffer.
///
/// Mirrors the `FoliageInstance` struct in `foliage_cull.wgsl` exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuFoliageInstance {
    /// Position (xyz) + rotation angle (w).
    pub position_rotation: [f32; 4],
    /// Scale (xyz) + max_distance (w).
    pub scale_distance: [f32; 4],
}

impl GpuFoliageInstance {
    /// Create a GPU foliage instance with the given max draw distance.
    pub fn from_instance(inst: &FoliageInstance, max_distance: f32) -> Self {
        Self {
            position_rotation: [
                inst.position.x,
                inst.position.y,
                inst.position.z,
                inst.rotation,
            ],
            scale_distance: [inst.scale, inst.scale, inst.scale, max_distance],
        }
    }
}

impl From<&FoliageInstance> for GpuFoliageInstance {
    fn from(inst: &FoliageInstance) -> Self {
        Self {
            position_rotation: [
                inst.position.x,
                inst.position.y,
                inst.position.z,
                inst.rotation,
            ],
            // max_distance defaults to 0; callers should use `from_instance` with the layer's
            // max_distance, or patch `scale_distance[3]` after conversion.
            scale_distance: [inst.scale, inst.scale, inst.scale, 0.0],
        }
    }
}

/// Uniform data for the foliage cull compute shader.
///
/// Mirrors the `CullUniforms` struct in `foliage_cull.wgsl` exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FoliageCullUniforms {
    /// Six frustum planes: `(nx, ny, nz, d)` each.
    pub frustum_planes: [[f32; 4]; 6],
    /// Camera eye position (xyz), w unused.
    pub camera_position: [f32; 4],
    /// Number of instances to process.
    pub instance_count: u32,
    /// Padding to align to 16 bytes.
    pub _pad: [u32; 3],
}

impl FoliageCullUniforms {
    /// Build from `GpuFrustumData` and an instance count.
    pub fn from_frustum_data(frustum: &GpuFrustumData, instance_count: u32) -> Self {
        Self {
            frustum_planes: frustum.planes,
            camera_position: frustum.camera_position,
            instance_count,
            _pad: [0; 3],
        }
    }
}

/// Bind-group layout entries for the foliage cull compute shader (group 0).
///
/// - binding 0: storage read (foliage instances)
/// - binding 1: uniform (cull uniforms)
/// - binding 2: storage read_write (visible model matrices output)
/// - binding 3: storage read_write (draw count atomic)
pub const FOLIAGE_CULL_BINDINGS: &[euca_rhi::BindGroupLayoutEntry] = &[
    euca_rhi::BindGroupLayoutEntry {
        binding: 0,
        visibility: euca_rhi::ShaderStages::COMPUTE,
        ty: euca_rhi::BindingType::Buffer {
            ty: euca_rhi::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
    euca_rhi::BindGroupLayoutEntry {
        binding: 1,
        visibility: euca_rhi::ShaderStages::COMPUTE,
        ty: euca_rhi::BindingType::Buffer {
            ty: euca_rhi::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
    euca_rhi::BindGroupLayoutEntry {
        binding: 2,
        visibility: euca_rhi::ShaderStages::COMPUTE,
        ty: euca_rhi::BindingType::Buffer {
            ty: euca_rhi::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
    euca_rhi::BindGroupLayoutEntry {
        binding: 3,
        visibility: euca_rhi::ShaderStages::COMPUTE,
        ty: euca_rhi::BindingType::Buffer {
            ty: euca_rhi::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
];

// ---------------------------------------------------------------------------
// GpuFoliageCuller
// ---------------------------------------------------------------------------

/// GPU-accelerated foliage culling resources.
///
/// Holds all GPU buffers and the compute pipeline needed to perform per-instance
/// frustum + distance culling entirely on the GPU. Visible instances are compacted
/// into an output model matrix buffer suitable for instanced draw calls.
///
/// # Usage
///
/// ```text
/// let culler = GpuFoliageCuller::new(&device, max_instances);
/// culler.upload_instances(&device, &gpu_instances);
/// // Each frame:
/// culler.cull(&device, &mut encoder, &frustum_data, instance_count);
/// // Then bind culler.visible_matrix_buffer() in the render pass.
/// ```
pub struct GpuFoliageCuller<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    instance_buffer: GpuBuffer<D>,
    output_matrix_buffer: GpuBuffer<D>,
    draw_count_buffer: GpuBuffer<D>,
    uniforms_buffer: GpuBuffer<D>,
    pipeline: ComputePipeline<D>,
    instance_count: u32,
    capacity: u32,
}

impl<D: euca_rhi::RenderDevice> GpuFoliageCuller<D> {
    /// Create a foliage culler sized for up to `max_instances` foliage instances.
    pub fn new(device: &D, max_instances: u32) -> Self {
        let pipeline = ComputePipeline::new(
            device,
            &ComputePipelineDesc {
                label: "foliage_cull_pipeline",
                shader_source: FOLIAGE_CULL_SHADER,
                entry_point: "main",
            },
            FOLIAGE_CULL_BINDINGS,
        );

        let instance_buf_size =
            (max_instances as u64) * std::mem::size_of::<GpuFoliageInstance>() as u64;
        let instance_buffer =
            GpuBuffer::new_storage(device, instance_buf_size, "foliage_instances");

        let output_buf_size = (max_instances as u64) * MODEL_MATRIX_SIZE;
        let output_matrix_buffer =
            GpuBuffer::new_storage(device, output_buf_size, "foliage_visible_matrices");

        let draw_count_buffer = GpuBuffer::new_storage(device, 4, "foliage_draw_count");

        let uniforms_buffer = GpuBuffer::new_uniform_with_data(
            device,
            &FoliageCullUniforms {
                frustum_planes: [[0.0; 4]; 6],
                camera_position: [0.0; 4],
                instance_count: 0,
                _pad: [0; 3],
            },
            "foliage_cull_uniforms",
        );

        Self {
            instance_buffer,
            output_matrix_buffer,
            draw_count_buffer,
            uniforms_buffer,
            pipeline,
            instance_count: 0,
            capacity: max_instances,
        }
    }

    /// Upload all foliage instances to the GPU storage buffer.
    ///
    /// Call this when the foliage layer changes (e.g. after scattering).
    /// The instance data persists across frames until the next upload.
    pub fn upload_instances(&mut self, device: &D, instances: &[GpuFoliageInstance]) {
        assert!(
            instances.len() <= self.capacity as usize,
            "Too many foliage instances ({}) for culler capacity ({})",
            instances.len(),
            self.capacity
        );
        self.instance_buffer.write(device, instances);
        self.instance_count = instances.len() as u32;
    }

    /// Dispatch the foliage cull compute shader.
    ///
    /// Clears the draw count, uploads the frustum uniforms, and dispatches the
    /// compute shader. After submission, `visible_matrix_buffer` contains the
    /// compacted model matrices and `draw_count_buffer` holds the count.
    pub fn cull(&self, device: &D, encoder: &mut D::CommandEncoder, frustum: &GpuFrustumData) {
        if self.instance_count == 0 {
            return;
        }

        // Upload uniforms for this frame.
        let uniforms = FoliageCullUniforms::from_frustum_data(frustum, self.instance_count);
        self.uniforms_buffer
            .write(device, std::slice::from_ref(&uniforms));

        // Clear the atomic draw count to zero.
        device.clear_buffer(encoder, self.draw_count_buffer.raw(), 0, None);

        // Create bind group.
        let bind_group = device.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("foliage_cull_bind_group"),
            layout: self.pipeline.bind_group_layout(),
            entries: &[
                euca_rhi::BindGroupEntry {
                    binding: 0,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: self.instance_buffer.raw(),
                        offset: 0,
                        size: None,
                    }),
                },
                euca_rhi::BindGroupEntry {
                    binding: 1,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: self.uniforms_buffer.raw(),
                        offset: 0,
                        size: None,
                    }),
                },
                euca_rhi::BindGroupEntry {
                    binding: 2,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: self.output_matrix_buffer.raw(),
                        offset: 0,
                        size: None,
                    }),
                },
                euca_rhi::BindGroupEntry {
                    binding: 3,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: self.draw_count_buffer.raw(),
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        // Dispatch.
        let workgroup_count = self.instance_count.div_ceil(FOLIAGE_CULL_WORKGROUP_SIZE);
        crate::compute::dispatch_compute_generic(
            device,
            encoder,
            &self.pipeline,
            &[&bind_group],
            [workgroup_count, 1, 1],
            None,
        );
    }

    /// The output buffer containing compacted model matrices for visible instances.
    ///
    /// Bind this as a vertex/storage buffer in the render pass for instanced drawing.
    pub fn visible_matrix_buffer(&self) -> &D::Buffer {
        self.output_matrix_buffer.raw()
    }

    /// The buffer containing the number of visible instances (single `u32`).
    ///
    /// Can be used for `multi_draw_indirect_count` or read back to the CPU.
    pub fn draw_count_buffer(&self) -> &D::Buffer {
        self.draw_count_buffer.raw()
    }

    /// The number of instances currently uploaded.
    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }

    /// Maximum number of instances this culler can handle.
    pub fn capacity(&self) -> u32 {
        self.capacity
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
// World resource wrapper
// ---------------------------------------------------------------------------

/// Collection of foliage layers stored as a world resource.
///
/// The editor and agent routes insert/read this resource to manage foliage.
#[derive(Clone, Debug, Default)]
pub struct FoliageLayers {
    /// All active foliage layers.
    pub layers: Vec<FoliageLayer>,
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
            jitter: [0.0, 0.0],
            prev_view_proj: None,
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
            jitter: [0.0, 0.0],
            prev_view_proj: None,
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

// ---------------------------------------------------------------------------
// Integration tests for FoliageLayers and render pipeline wiring
// ---------------------------------------------------------------------------

#[cfg(test)]
mod integration_tests {
    use super::*;
    use euca_math::Vec3;

    // ── Test: FoliageLayers default is empty ──

    #[test]
    fn foliage_layers_default_is_empty() {
        let layers = FoliageLayers::default();
        assert!(
            layers.layers.is_empty(),
            "Default FoliageLayers should have no layers"
        );
    }

    // ── Test: Scatter and collect produces DrawCommand-compatible output ──

    #[test]
    fn scatter_then_collect_produces_draw_commands() {
        let mesh = MeshHandle(5);
        let material = MaterialHandle(2);

        let mut layer = FoliageLayer {
            mesh,
            material,
            density: 2.0,
            min_scale: 0.9,
            max_scale: 1.1,
            max_distance: 200.0,
            instances: Vec::new(),
        };

        // Scatter on a 10x10 area
        let area_min = Vec3::new(-5.0, 0.0, -5.0);
        let area_max = Vec3::new(5.0, 0.0, 5.0);
        scatter_foliage(&mut layer, area_min, area_max, 77);

        assert!(
            !layer.instances.is_empty(),
            "Scatter should produce instances"
        );

        // Store in FoliageLayers
        let layers = FoliageLayers {
            layers: vec![layer.clone()],
        };
        assert_eq!(layers.layers.len(), 1);

        // Collect visible instances (camera above, looking down)
        let cam = crate::camera::Camera {
            eye: Vec3::new(0.0, 50.0, 0.0),
            target: Vec3::new(0.0, 0.0, 0.01),
            up: Vec3::Y,
            fov_y: std::f32::consts::FRAC_PI_2,
            near: 0.1,
            far: 1000.0,
            orthographic: false,
            ortho_size: 10.0,
            jitter: [0.0, 0.0],
            prev_view_proj: None,
        };
        let vp = cam.view_projection_matrix(1.0);
        let frustum = crate::camera::Frustum::from_view_projection(&vp);

        let matrices =
            FoliageRenderer::collect_visible_instances(&layers.layers[0], cam.eye, &frustum);

        // All instances should be visible (within 200m distance, camera sees the whole area)
        assert_eq!(
            matrices.len(),
            layer.instances.len(),
            "All scattered instances should be visible from directly above"
        );

        // Each matrix should be a valid 4x4 with the correct mesh/material preserved
        assert_eq!(layers.layers[0].mesh, MeshHandle(5));
        assert_eq!(layers.layers[0].material, MaterialHandle(2));
    }
}

// ---------------------------------------------------------------------------
// Tests for GPU foliage culling types
// ---------------------------------------------------------------------------

#[cfg(test)]
mod gpu_cull_tests {
    use super::*;

    // ── Struct layout tests ──────────────────────────────────────────────

    #[test]
    fn gpu_foliage_instance_layout() {
        // 2 x vec4<f32> = 32 bytes
        assert_eq!(std::mem::size_of::<GpuFoliageInstance>(), 32);
    }

    #[test]
    fn foliage_cull_uniforms_layout() {
        // 6 x vec4<f32> (planes) + vec4<f32> (camera) + u32 + 3 x u32 (pad)
        // = 96 + 16 + 16 = 128 bytes
        assert_eq!(std::mem::size_of::<FoliageCullUniforms>(), 128);
    }

    // ── Conversion tests ─────────────────────────────────────────────────

    #[test]
    fn gpu_foliage_instance_from_foliage_instance() {
        let inst = FoliageInstance {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: 1.57,
            scale: 0.8,
        };

        let gpu: GpuFoliageInstance = (&inst).into();
        assert_eq!(gpu.position_rotation, [1.0, 2.0, 3.0, 1.57]);
        assert_eq!(gpu.scale_distance, [0.8, 0.8, 0.8, 0.0]);
    }

    #[test]
    fn gpu_foliage_instance_from_instance_with_distance() {
        let inst = FoliageInstance {
            position: Vec3::new(10.0, 0.0, -5.0),
            rotation: 3.14,
            scale: 1.5,
        };

        let gpu = GpuFoliageInstance::from_instance(&inst, 200.0);
        assert_eq!(gpu.position_rotation, [10.0, 0.0, -5.0, 3.14]);
        assert_eq!(gpu.scale_distance, [1.5, 1.5, 1.5, 200.0]);
    }

    // ── Uniforms construction ────────────────────────────────────────────

    #[test]
    fn foliage_cull_uniforms_from_frustum_data() {
        let frustum_data = GpuFrustumData {
            planes: [
                [1.0, 0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0, -1.0],
                [0.0, -1.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, -0.1],
                [0.0, 0.0, -1.0, -100.0],
            ],
            camera_position: [10.0, 20.0, 30.0, 0.0],
        };

        let uniforms = FoliageCullUniforms::from_frustum_data(&frustum_data, 500);
        assert_eq!(uniforms.frustum_planes, frustum_data.planes);
        assert_eq!(uniforms.camera_position, [10.0, 20.0, 30.0, 0.0]);
        assert_eq!(uniforms.instance_count, 500);
        assert_eq!(uniforms._pad, [0; 3]);
    }

    // ── Bytemuck roundtrip ───────────────────────────────────────────────

    #[test]
    fn gpu_foliage_instance_bytemuck_roundtrip() {
        let inst = GpuFoliageInstance {
            position_rotation: [1.0, 2.0, 3.0, 0.5],
            scale_distance: [0.8, 0.8, 0.8, 100.0],
        };
        let bytes = bytemuck::bytes_of(&inst);
        assert_eq!(bytes.len(), 32);
        let restored: &GpuFoliageInstance = bytemuck::from_bytes(bytes);
        assert_eq!(restored.position_rotation, inst.position_rotation);
        assert_eq!(restored.scale_distance, inst.scale_distance);
    }

    #[test]
    fn foliage_cull_uniforms_bytemuck_roundtrip() {
        let uniforms = FoliageCullUniforms {
            frustum_planes: [[1.0; 4]; 6],
            camera_position: [5.0, 10.0, 15.0, 0.0],
            instance_count: 1000,
            _pad: [0; 3],
        };
        let bytes = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), 128);
        let restored: &FoliageCullUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(restored.instance_count, 1000);
        assert_eq!(restored.camera_position, [5.0, 10.0, 15.0, 0.0]);
    }

    // ── Shader source sanity ─────────────────────────────────────────────

    #[test]
    fn foliage_cull_shader_is_valid_wgsl() {
        assert!(!FOLIAGE_CULL_SHADER.is_empty());
        assert!(FOLIAGE_CULL_SHADER.contains("@compute"));
        assert!(FOLIAGE_CULL_SHADER.contains("@workgroup_size(64)"));
        assert!(FOLIAGE_CULL_SHADER.contains("fn main"));
        assert!(FOLIAGE_CULL_SHADER.contains("FoliageInstance"));
        assert!(FOLIAGE_CULL_SHADER.contains("CullUniforms"));
        assert!(FOLIAGE_CULL_SHADER.contains("ModelMatrix"));
        assert!(FOLIAGE_CULL_SHADER.contains("frustum_test_sphere"));
        assert!(FOLIAGE_CULL_SHADER.contains("rotation_y"));
        assert!(FOLIAGE_CULL_SHADER.contains("atomicAdd"));
    }

    // ── Bind group layout ────────────────────────────────────────────────

    #[test]
    fn foliage_cull_bindings_count() {
        assert_eq!(FOLIAGE_CULL_BINDINGS.len(), 4);
    }

    // ── Workgroup count rounding ─────────────────────────────────────────

    #[test]
    fn foliage_workgroup_count_rounding() {
        assert_eq!(0_u32.div_ceil(FOLIAGE_CULL_WORKGROUP_SIZE), 0);
        assert_eq!(1_u32.div_ceil(FOLIAGE_CULL_WORKGROUP_SIZE), 1);
        assert_eq!(64_u32.div_ceil(FOLIAGE_CULL_WORKGROUP_SIZE), 1);
        assert_eq!(65_u32.div_ceil(FOLIAGE_CULL_WORKGROUP_SIZE), 2);
        assert_eq!(10000_u32.div_ceil(FOLIAGE_CULL_WORKGROUP_SIZE), 157);
    }
}
