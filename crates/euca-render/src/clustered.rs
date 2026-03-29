//! Clustered light culling for Forward+ rendering.
//!
//! Divides the view frustum into a 3D grid of clusters and assigns lights to
//! each cluster via a GPU compute shader. The PBR fragment shader reads the
//! per-cluster light lists to shade only relevant lights, enabling 256+ lights.

use crate::compute::{ComputePipeline, ComputePipelineDesc, GpuBuffer};
use euca_rhi::pass::ComputePassOps;

/// Number of horizontal tiles in the cluster grid.
pub const TILES_X: u32 = 16;
/// Number of vertical tiles in the cluster grid.
pub const TILES_Y: u32 = 9;
/// Number of depth slices (exponential distribution).
pub const DEPTH_SLICES: u32 = 24;
/// Total number of clusters: 16 * 9 * 24 = 3456.
pub const CLUSTER_COUNT: u32 = TILES_X * TILES_Y * DEPTH_SLICES;
/// Maximum lights that can influence a single cluster.
pub const MAX_LIGHTS_PER_CLUSTER: u32 = 32;
/// Maximum total lights supported.
pub const MAX_LIGHTS: u32 = 256;

/// GPU-friendly light data (matches `LightData` in `light_assign.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLightData {
    pub position_range: [f32; 4],
    pub color_intensity: [f32; 4],
    pub direction_type: [f32; 4],
    pub cone_angles: [f32; 4],
}

impl Default for GpuLightData {
    fn default() -> Self {
        Self {
            position_range: [0.0; 4],
            color_intensity: [0.0; 4],
            direction_type: [0.0; 4],
            cone_angles: [0.0; 4],
        }
    }
}

/// Light type discriminant stored in `direction_type.w`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u32)]
pub enum LightType {
    Point = 0,
    Spot = 1,
}

impl GpuLightData {
    /// Create a point light.
    pub fn point(position: [f32; 3], range: f32, color: [f32; 3], intensity: f32) -> Self {
        Self {
            position_range: [position[0], position[1], position[2], range],
            color_intensity: [color[0], color[1], color[2], intensity],
            direction_type: [0.0, 0.0, 0.0, LightType::Point as u32 as f32],
            cone_angles: [0.0; 4],
        }
    }

    /// Create a spot light. `inner_cone` and `outer_cone` are in radians.
    pub fn spot(
        position: [f32; 3],
        range: f32,
        color: [f32; 3],
        intensity: f32,
        direction: [f32; 3],
        inner_cone: f32,
        outer_cone: f32,
    ) -> Self {
        Self {
            position_range: [position[0], position[1], position[2], range],
            color_intensity: [color[0], color[1], color[2], intensity],
            direction_type: [
                direction[0],
                direction[1],
                direction[2],
                LightType::Spot as u32 as f32,
            ],
            cone_angles: [inner_cone.cos(), outer_cone.cos(), 0.0, 0.0],
        }
    }
}

/// GPU-side cluster configuration uniform (matches `ClusterConfig` in WGSL).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ClusterConfig {
    pub view: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub screen_size: [f32; 2],
    pub near_z: f32,
    pub far_z: f32,
    pub num_lights: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            view: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            inv_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            screen_size: [1920.0, 1080.0],
            near_z: 0.1,
            far_z: 1000.0,
            num_lights: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

/// Embedded WGSL compute shader for light-cluster assignment.
pub const LIGHT_ASSIGN_SHADER: &str = include_str!("../shaders/light_assign.wgsl");

/// Parameters for [`ClusteredLightGrid::update`].
pub struct UpdateParams<'a> {
    pub lights: &'a [GpuLightData],
    pub view: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub screen_size: [f32; 2],
    pub near_z: f32,
    pub far_z: f32,
}

/// Bind-group layout entries for the light assignment compute shader (group 0).
///
/// - binding 0: uniform (ClusterConfig)
/// - binding 1: storage read (lights)
/// - binding 2: storage read_write (light_indices)
/// - binding 3: storage read_write (cluster_light_counts)
pub const LIGHT_ASSIGN_BINDINGS: &[euca_rhi::BindGroupLayoutEntry] = &[
    euca_rhi::BindGroupLayoutEntry {
        binding: 0,
        visibility: euca_rhi::ShaderStages::COMPUTE,
        ty: euca_rhi::BindingType::Buffer {
            ty: euca_rhi::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
    euca_rhi::BindGroupLayoutEntry {
        binding: 1,
        visibility: euca_rhi::ShaderStages::COMPUTE,
        ty: euca_rhi::BindingType::Buffer {
            ty: euca_rhi::BufferBindingType::Storage { read_only: true },
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

/// Manages the GPU resources for clustered light culling.
pub struct ClusteredLightGrid<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    pipeline: ComputePipeline<D>,
    config_buffer: GpuBuffer<D>,
    lights_buffer: GpuBuffer<D>,
    light_indices_buffer: GpuBuffer<D>,
    cluster_counts_buffer: GpuBuffer<D>,
}

impl<D: euca_rhi::RenderDevice> ClusteredLightGrid<D> {
    /// Create the clustered light grid with all GPU resources.
    pub fn new(device: &D) -> Self {
        let pipeline = ComputePipeline::new(
            device,
            &ComputePipelineDesc {
                label: "light_assign_pipeline",
                shader_source: LIGHT_ASSIGN_SHADER,
                entry_point: "main",
            },
            LIGHT_ASSIGN_BINDINGS,
        );
        let config_buffer =
            GpuBuffer::new_uniform_with_data(device, &ClusterConfig::default(), "cluster_config");
        let lights_size = (MAX_LIGHTS as u64) * std::mem::size_of::<GpuLightData>() as u64;
        let lights_buffer = GpuBuffer::new_storage(device, lights_size, "cluster_lights");
        let indices_size = (CLUSTER_COUNT as u64)
            * (MAX_LIGHTS_PER_CLUSTER as u64)
            * std::mem::size_of::<u32>() as u64;
        let light_indices_buffer =
            GpuBuffer::new_storage(device, indices_size, "cluster_light_indices");
        let counts_size = (CLUSTER_COUNT as u64) * std::mem::size_of::<u32>() as u64;
        let cluster_counts_buffer =
            GpuBuffer::new_storage(device, counts_size, "cluster_light_counts");
        Self {
            pipeline,
            config_buffer,
            lights_buffer,
            light_indices_buffer,
            cluster_counts_buffer,
        }
    }

    /// Upload lights and cluster config, then dispatch the light assignment compute shader.
    pub fn update(&self, device: &D, encoder: &mut D::CommandEncoder, params: &UpdateParams) {
        let num_lights = params.lights.len().min(MAX_LIGHTS as usize) as u32;
        let config = ClusterConfig {
            view: params.view,
            inv_proj: params.inv_proj,
            screen_size: params.screen_size,
            near_z: params.near_z,
            far_z: params.far_z,
            num_lights,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        device.write_buffer(self.config_buffer.raw(), 0, bytemuck::bytes_of(&config));
        if num_lights > 0 {
            let byte_count = num_lights as usize * std::mem::size_of::<GpuLightData>();
            device.write_buffer(
                self.lights_buffer.raw(),
                0,
                &bytemuck::cast_slice(params.lights)[..byte_count],
            );
        }
        device.clear_buffer(encoder, self.cluster_counts_buffer.raw(), 0, None);

        let bind_group = self.create_bind_group(device, "light_assign_bind_group");

        let wg_x = TILES_X.div_ceil(4);
        let wg_y = TILES_Y.div_ceil(4);
        let wg_z = DEPTH_SLICES.div_ceil(4);
        let mut pass = device.begin_compute_pass(encoder, Some("light_assign_pass"));
        pass.set_pipeline(self.pipeline.raw());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, wg_z);
    }

    /// Create a bind group that the PBR shader can read to access cluster data.
    pub fn bind_group(&self, device: &D) -> D::BindGroup {
        self.create_bind_group(device, "clustered_light_read_bind_group")
    }

    /// The compute pipeline's bind group layout.
    pub fn bind_group_layout(&self) -> &D::BindGroupLayout {
        self.pipeline.bind_group_layout()
    }

    /// Helper: create a bind group referencing all four cluster buffers.
    fn create_bind_group(&self, device: &D, label: &str) -> D::BindGroup {
        device.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some(label),
            layout: self.pipeline.bind_group_layout(),
            entries: &[
                euca_rhi::BindGroupEntry {
                    binding: 0,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: self.config_buffer.raw(),
                        offset: 0,
                        size: None,
                    }),
                },
                euca_rhi::BindGroupEntry {
                    binding: 1,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: self.lights_buffer.raw(),
                        offset: 0,
                        size: None,
                    }),
                },
                euca_rhi::BindGroupEntry {
                    binding: 2,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: self.light_indices_buffer.raw(),
                        offset: 0,
                        size: None,
                    }),
                },
                euca_rhi::BindGroupEntry {
                    binding: 3,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: self.cluster_counts_buffer.raw(),
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        })
    }
}

/// Compute the flat cluster index from screen UV and linear depth.
pub fn cluster_index(
    screen_uv: [f32; 2],
    linear_depth: f32,
    near_z: f32,
    far_z: f32,
) -> Option<u32> {
    if linear_depth < near_z || linear_depth > far_z || near_z <= 0.0 || far_z <= near_z {
        return None;
    }
    let tile_x = ((screen_uv[0] * TILES_X as f32) as u32).min(TILES_X - 1);
    let tile_y = ((screen_uv[1] * TILES_Y as f32) as u32).min(TILES_Y - 1);
    let log_ratio = (far_z / near_z).ln();
    let slice = ((linear_depth / near_z).ln() / log_ratio * DEPTH_SLICES as f32).floor() as u32;
    let slice = slice.min(DEPTH_SLICES - 1);
    Some(tile_x + tile_y * TILES_X + slice * TILES_X * TILES_Y)
}

/// Compute the view-space AABB for a cluster (CPU-side, for testing/debug).
pub fn cluster_aabb(
    tile_x: u32,
    tile_y: u32,
    depth_slice: u32,
    near_z: f32,
    far_z: f32,
    inv_proj: &[[f32; 4]; 4],
) -> ([f32; 3], [f32; 3]) {
    let tile_size_x = 1.0 / TILES_X as f32;
    let tile_size_y = 1.0 / TILES_Y as f32;
    let uv_min = [tile_x as f32 * tile_size_x, tile_y as f32 * tile_size_y];
    let uv_max = [
        (tile_x + 1) as f32 * tile_size_x,
        (tile_y + 1) as f32 * tile_size_y,
    ];
    let log_ratio = (far_z / near_z).ln();
    let near_depth = near_z * (log_ratio * depth_slice as f32 / DEPTH_SLICES as f32).exp();
    let far_depth = near_z * (log_ratio * (depth_slice + 1) as f32 / DEPTH_SLICES as f32).exp();
    let depth_range = far_z - near_z;
    let near_ndc = (near_depth - near_z) / depth_range;
    let far_ndc = (far_depth - near_z) / depth_range;
    let corners_uv = [
        (uv_min, near_ndc),
        ([uv_max[0], uv_min[1]], near_ndc),
        ([uv_min[0], uv_max[1]], near_ndc),
        (uv_max, near_ndc),
        (uv_min, far_ndc),
        ([uv_max[0], uv_min[1]], far_ndc),
        ([uv_min[0], uv_max[1]], far_ndc),
        (uv_max, far_ndc),
    ];
    let mut aabb_min = [f32::MAX; 3];
    let mut aabb_max = [f32::MIN; 3];
    for (uv, depth) in corners_uv {
        let view = screen_to_view_cpu(uv, depth, inv_proj);
        for i in 0..3 {
            aabb_min[i] = aabb_min[i].min(view[i]);
            aabb_max[i] = aabb_max[i].max(view[i]);
        }
    }
    (aabb_min, aabb_max)
}

fn screen_to_view_cpu(uv: [f32; 2], depth: f32, inv_proj: &[[f32; 4]; 4]) -> [f32; 3] {
    let ndc = [uv[0] * 2.0 - 1.0, (1.0 - uv[1]) * 2.0 - 1.0, depth, 1.0];
    let mut result = [0.0f32; 4];
    for (r, out) in result.iter_mut().enumerate() {
        for c in 0..4 {
            *out += inv_proj[c][r] * ndc[c];
        }
    }
    let w = result[3];
    [result[0] / w, result[1] / w, result[2] / w]
}

/// CPU-side sphere-AABB intersection test (mirrors WGSL `sphere_aabb_intersect`).
pub fn sphere_aabb_intersect(
    center: [f32; 3],
    radius: f32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> bool {
    let mut dist_sq = 0.0f32;
    for i in 0..3 {
        let clamped = center[i].clamp(aabb_min[i], aabb_max[i]);
        let diff = center[i] - clamped;
        dist_sq += diff * diff;
    }
    dist_sq <= radius * radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_index_center_screen() {
        let idx = cluster_index([0.5, 0.5], 1.0, 0.1, 1000.0).unwrap();
        let tile_x = idx % TILES_X;
        let tile_y = (idx / TILES_X) % TILES_Y;
        let slice = idx / (TILES_X * TILES_Y);
        assert_eq!(tile_x, 8);
        assert_eq!(tile_y, 4);
        assert!(slice < DEPTH_SLICES);
    }

    #[test]
    fn cluster_index_out_of_range() {
        assert!(cluster_index([0.5, 0.5], 0.01, 0.1, 1000.0).is_none());
        assert!(cluster_index([0.5, 0.5], 1500.0, 0.1, 1000.0).is_none());
        assert!(cluster_index([0.5, 0.5], 1.0, 0.0, 1000.0).is_none());
        assert!(cluster_index([0.5, 0.5], 1.0, 100.0, 10.0).is_none());
    }

    #[test]
    fn cluster_index_corners() {
        let depth = 50.0;
        let idx_tl = cluster_index([0.0, 0.0], depth, 0.1, 1000.0).unwrap();
        assert_eq!(idx_tl % TILES_X, 0);
        assert_eq!((idx_tl / TILES_X) % TILES_Y, 0);
        let idx_br = cluster_index([0.999, 0.999], depth, 0.1, 1000.0).unwrap();
        assert_eq!(idx_br % TILES_X, TILES_X - 1);
        assert_eq!((idx_br / TILES_X) % TILES_Y, TILES_Y - 1);
    }

    #[test]
    fn cluster_aabb_non_degenerate() {
        let inv_proj = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let (aabb_min, aabb_max) = cluster_aabb(0, 0, 0, 0.1, 100.0, &inv_proj);
        for i in 0..3 {
            assert!(aabb_max[i] > aabb_min[i]);
        }
    }

    #[test]
    fn sphere_aabb_intersection_basic() {
        let mn = [-1.0, -1.0, -1.0];
        let mx = [1.0, 1.0, 1.0];
        assert!(sphere_aabb_intersect([0.0, 0.0, 0.0], 0.5, mn, mx));
        assert!(sphere_aabb_intersect([2.0, 0.0, 0.0], 1.0, mn, mx));
        assert!(!sphere_aabb_intersect([3.0, 0.0, 0.0], 1.0, mn, mx));
        assert!(!sphere_aabb_intersect([10.0, 10.0, 10.0], 1.0, mn, mx));
    }

    #[test]
    fn gpu_light_data_point() {
        let l = GpuLightData::point([1.0, 2.0, 3.0], 10.0, [1.0, 0.5, 0.0], 2.0);
        assert_eq!(l.position_range, [1.0, 2.0, 3.0, 10.0]);
        assert_eq!(l.color_intensity, [1.0, 0.5, 0.0, 2.0]);
        assert_eq!(l.direction_type[3], 0.0);
    }

    #[test]
    fn gpu_light_data_spot() {
        let (inner, outer) = (0.3_f32, 0.5_f32);
        let l = GpuLightData::spot(
            [1.0, 2.0, 3.0],
            15.0,
            [1.0, 1.0, 1.0],
            1.0,
            [0.0, -1.0, 0.0],
            inner,
            outer,
        );
        assert_eq!(l.position_range, [1.0, 2.0, 3.0, 15.0]);
        assert_eq!(l.direction_type[3], 1.0);
        assert!((l.cone_angles[0] - inner.cos()).abs() < 1e-6);
        assert!((l.cone_angles[1] - outer.cos()).abs() < 1e-6);
    }

    #[test]
    fn cluster_config_default_values() {
        let c = ClusterConfig::default();
        assert_eq!(c.screen_size, [1920.0, 1080.0]);
        assert!((c.near_z - 0.1).abs() < 1e-6);
        assert!((c.far_z - 1000.0).abs() < 1e-6);
        assert_eq!(c.num_lights, 0);
    }

    #[test]
    fn cluster_config_gpu_size() {
        assert_eq!(std::mem::size_of::<ClusterConfig>(), 160);
    }

    #[test]
    fn gpu_light_data_gpu_size() {
        assert_eq!(std::mem::size_of::<GpuLightData>(), 64);
    }

    #[test]
    fn cluster_count_is_correct() {
        assert_eq!(CLUSTER_COUNT, 3456);
    }

    #[test]
    fn light_assign_shader_is_valid_wgsl_source() {
        assert!(!LIGHT_ASSIGN_SHADER.is_empty());
        assert!(LIGHT_ASSIGN_SHADER.contains("@compute"));
        assert!(LIGHT_ASSIGN_SHADER.contains("@workgroup_size(4, 4, 4)"));
        assert!(LIGHT_ASSIGN_SHADER.contains("fn main"));
        assert!(LIGHT_ASSIGN_SHADER.contains("sphere_aabb_intersect"));
    }

    #[test]
    fn exponential_depth_slicing_covers_full_range() {
        let (near, far) = (0.1_f32, 1000.0_f32);
        let slice_near = cluster_index([0.5, 0.5], near, near, far).unwrap() / (TILES_X * TILES_Y);
        assert_eq!(slice_near, 0);
        let slice_far = cluster_index([0.5, 0.5], far, near, far).unwrap() / (TILES_X * TILES_Y);
        assert_eq!(slice_far, DEPTH_SLICES - 1);
    }
}
