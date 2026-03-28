//! GPU-driven rendering pipeline using indirect draw calls.
//!
//! Instead of the CPU issuing one `draw_indexed` per entity, this module:
//!
//! 1. Uploads all entity draw commands to a GPU storage buffer.
//! 2. Dispatches a compute shader (`gpu_cull.wgsl`) that performs frustum
//!    culling and LOD selection entirely on the GPU.
//! 3. The compute shader writes `DrawIndexedIndirect` arguments into an
//!    output storage buffer.
//! 4. The render pass calls `draw_indexed_indirect` once per entity slot,
//!    where culled entities have `index_count = 0` (zero-cost no-op).
//!
//! This replaces per-entity CPU frustum culling and draw-call submission
//! with a single compute dispatch, dramatically reducing CPU overhead for
//! large scenes.

use crate::camera::Frustum;
use crate::compute::{ComputePipeline, ComputePipelineDesc, GpuBuffer};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// WGSL compute shader source for GPU culling and indirect argument generation.
pub const GPU_CULL_SHADER: &str = include_str!("../shaders/gpu_cull.wgsl");

/// Size of one `DrawIndexedIndirect` struct in bytes (5 x u32 = 20 bytes).
pub const DRAW_INDEXED_INDIRECT_SIZE: u64 = 20;

/// Workgroup size used by the compute shader (must match `@workgroup_size` in WGSL).
const WORKGROUP_SIZE: u32 = 64;

// ---------------------------------------------------------------------------
// CPU-side structs that mirror GPU layout
// ---------------------------------------------------------------------------

/// A single draw command uploaded to the GPU, one per entity.
///
/// This struct mirrors the `DrawCommandGpu` in `gpu_cull.wgsl` exactly.
/// The compute shader reads this and produces `DrawIndexedIndirect` arguments.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawCommandGpu {
    /// Model matrix column 0.
    pub model_col0: [f32; 4],
    /// Model matrix column 1.
    pub model_col1: [f32; 4],
    /// Model matrix column 2.
    pub model_col2: [f32; 4],
    /// Model matrix column 3.
    pub model_col3: [f32; 4],
    /// AABB center in world space (xyz), w unused.
    pub aabb_center: [f32; 4],
    /// AABB half-extents in world space (xyz), w unused.
    pub aabb_half_extents: [f32; 4],
    /// Mesh ID (index into the mesh table).
    pub mesh_id: u32,
    /// Material ID (index into the material table).
    pub material_id: u32,
    /// Number of indices to draw (for the base LOD).
    pub index_count: u32,
    /// Offset into the index buffer (for the base LOD).
    pub first_index: u32,
    /// Added to each index value before indexing into the vertex buffer.
    pub vertex_offset: i32,
    /// Number of valid LOD levels (1 = no LOD, up to 4).
    pub lod_count: u32,
    /// Index counts for each LOD level (up to 4).
    pub lod_index_counts: [u32; 4],
    /// First index for each LOD level.
    pub lod_first_indices: [u32; 4],
    /// Vertex offsets for each LOD level.
    pub lod_vertex_offsets: [i32; 4],
    /// Squared distance thresholds for LOD transitions (ascending order).
    pub lod_distance_sq: [f32; 4],
}

/// Mirrors `wgpu::DrawIndexedIndirect` layout -- the output of the compute shader.
///
/// Each visible entity gets one of these with valid draw parameters.
/// Culled entities get all-zero fields (producing zero triangles).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirectArgs {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

/// Frustum data uploaded as a uniform buffer for the compute shader.
///
/// Contains 6 frustum planes plus the camera position (for LOD distance).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuFrustumData {
    /// Six frustum planes: `(nx, ny, nz, d)` each.
    pub planes: [[f32; 4]; 6],
    /// Camera eye position (xyz), w unused.
    pub camera_position: [f32; 4],
}

impl GpuFrustumData {
    /// Build from a `Frustum` and a camera eye position.
    pub fn from_frustum(frustum: &Frustum, eye: [f32; 3]) -> Self {
        Self {
            planes: frustum.planes,
            camera_position: [eye[0], eye[1], eye[2], 0.0],
        }
    }
}

/// Parameters uniform for the compute shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCullParams {
    pub entity_count: u32,
    pub _pad: [u32; 3],
}

// ---------------------------------------------------------------------------
// IndirectDrawBuffer
// ---------------------------------------------------------------------------

/// Storage buffer holding `DrawIndexedIndirect` structs -- the output of the
/// GPU cull pass, consumed by `draw_indexed_indirect` in the render pass.
pub struct IndirectDrawBuffer<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    buffer: D::Buffer,
    capacity: u32,
}

impl IndirectDrawBuffer {
    /// Create an indirect draw buffer that can hold up to `max_draws` entries.
    pub fn new(device: &wgpu::Device, max_draws: u32) -> Self {
        let size = (max_draws as u64) * DRAW_INDEXED_INDIRECT_SIZE;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indirect_draw_buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            capacity: max_draws,
        }
    }

    /// The underlying wgpu buffer.
    pub fn raw(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Maximum number of indirect draw entries.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Size in bytes of the entire buffer.
    pub fn size_bytes(&self) -> u64 {
        (self.capacity as u64) * DRAW_INDEXED_INDIRECT_SIZE
    }
}

// ---------------------------------------------------------------------------
// GpuDrivenPipeline
// ---------------------------------------------------------------------------

/// Manages the complete GPU-driven rendering pipeline: command upload,
/// compute-based culling, and indirect draw dispatch.
pub struct GpuDrivenPipeline<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    cull_pipeline: ComputePipeline,
    command_buffer: GpuBuffer,
    frustum_buffer: GpuBuffer,
    params_buffer: GpuBuffer,
    draw_count_buffer: GpuBuffer,
    indirect_buffer: IndirectDrawBuffer<D>,
    max_entities: u32,
}

impl GpuDrivenPipeline {
    /// Create a new GPU-driven pipeline sized for up to `max_entities`.
    pub fn new(device: &wgpu::Device, max_entities: u32) -> Self {
        let cull_pipeline = ComputePipeline::new(
            device,
            &ComputePipelineDesc {
                label: "gpu_cull_pipeline",
                shader_source: GPU_CULL_SHADER,
                entry_point: "main",
            },
        );

        let command_buf_size = (max_entities as u64) * std::mem::size_of::<DrawCommandGpu>() as u64;
        let command_buffer = GpuBuffer::new_storage(device, command_buf_size, "gpu_draw_commands");

        let frustum_buffer = GpuBuffer::new_uniform_with_data(
            device,
            &GpuFrustumData {
                planes: [[0.0; 4]; 6],
                camera_position: [0.0; 4],
            },
            "gpu_cull_frustum",
        );

        let params_buffer = GpuBuffer::new_uniform_with_data(
            device,
            &GpuCullParams {
                entity_count: 0,
                _pad: [0; 3],
            },
            "gpu_cull_params",
        );

        let draw_count_buffer = GpuBuffer::new_storage(device, 4, "gpu_draw_count");
        let indirect_buffer = IndirectDrawBuffer::new(device, max_entities);

        Self {
            cull_pipeline,
            command_buffer,
            frustum_buffer,
            params_buffer,
            draw_count_buffer,
            indirect_buffer,
            max_entities,
        }
    }

    /// Upload draw commands for this frame.
    pub fn upload_commands(&self, queue: &wgpu::Queue, commands: &[DrawCommandGpu]) {
        assert!(
            commands.len() <= self.max_entities as usize,
            "Too many draw commands ({}) for pipeline capacity ({})",
            commands.len(),
            self.max_entities
        );
        self.command_buffer.write(queue, commands);
    }

    /// Upload frustum data for this frame.
    pub fn upload_frustum(&self, queue: &wgpu::Queue, frustum_data: &GpuFrustumData) {
        self.frustum_buffer
            .write(queue, std::slice::from_ref(frustum_data));
    }

    /// Upload the entity count parameter. Call before `cull_and_prepare`.
    pub fn upload_params(&self, queue: &wgpu::Queue, entity_count: u32) {
        let params = GpuCullParams {
            entity_count,
            _pad: [0; 3],
        };
        self.params_buffer
            .write(queue, std::slice::from_ref(&params));
    }

    /// Dispatch the GPU cull compute shader.
    ///
    /// The caller must have called `upload_commands`, `upload_frustum`, and
    /// `upload_params` via the queue before beginning the command encoder.
    pub fn cull_and_prepare(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        entity_count: u32,
    ) {
        assert!(
            entity_count <= self.max_entities,
            "entity_count ({entity_count}) exceeds pipeline capacity ({})",
            self.max_entities
        );

        encoder.clear_buffer(self.draw_count_buffer.raw(), 0, None);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_cull_bind_group"),
            layout: self.cull_pipeline.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.command_buffer.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.frustum_buffer.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.indirect_buffer.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.draw_count_buffer.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params_buffer.raw().as_entire_binding(),
                },
            ],
        });

        let workgroup_count = entity_count.div_ceil(WORKGROUP_SIZE);
        crate::compute::dispatch_compute(
            encoder,
            &self.cull_pipeline,
            &[&bind_group],
            [workgroup_count, 1, 1],
            None,
        );
    }

    /// Issue indirect draw calls from the indirect buffer.
    ///
    /// Call inside a render pass after `cull_and_prepare` has been submitted.
    pub fn draw_indirect<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, entity_count: u32) {
        for i in 0..entity_count {
            let offset = (i as u64) * DRAW_INDEXED_INDIRECT_SIZE;
            render_pass.draw_indexed_indirect(self.indirect_buffer.raw(), offset);
        }
    }

    /// Access the indirect draw buffer.
    pub fn indirect_buffer(&self) -> &IndirectDrawBuffer {
        &self.indirect_buffer
    }

    /// Access the draw count buffer.
    pub fn draw_count_buffer(&self) -> &GpuBuffer {
        &self.draw_count_buffer
    }

    /// Maximum entities this pipeline can handle.
    pub fn max_entities(&self) -> u32 {
        self.max_entities
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    #[test]
    fn draw_command_gpu_layout() {
        assert_eq!(std::mem::size_of::<DrawCommandGpu>(), 184);

        let cmd = DrawCommandGpu {
            model_col0: [1.0, 0.0, 0.0, 0.0],
            model_col1: [0.0, 1.0, 0.0, 0.0],
            model_col2: [0.0, 0.0, 1.0, 0.0],
            model_col3: [0.0, 0.0, 0.0, 1.0],
            aabb_center: [5.0, 5.0, 5.0, 0.0],
            aabb_half_extents: [1.0, 1.0, 1.0, 0.0],
            mesh_id: 42,
            material_id: 7,
            index_count: 36,
            first_index: 0,
            vertex_offset: 0,
            lod_count: 1,
            lod_index_counts: [36, 0, 0, 0],
            lod_first_indices: [0, 0, 0, 0],
            lod_vertex_offsets: [0, 0, 0, 0],
            lod_distance_sq: [100.0, 400.0, 1600.0, 0.0],
        };

        let bytes = bytemuck::bytes_of(&cmd);
        assert_eq!(bytes.len(), 184);

        let mesh_id_bytes = &bytes[96..100];
        assert_eq!(u32::from_le_bytes(mesh_id_bytes.try_into().unwrap()), 42);
    }

    #[test]
    fn indirect_args_layout() {
        assert_eq!(
            std::mem::size_of::<DrawIndexedIndirectArgs>(),
            DRAW_INDEXED_INDIRECT_SIZE as usize,
        );

        let args = DrawIndexedIndirectArgs {
            index_count: 36,
            instance_count: 1,
            first_index: 0,
            base_vertex: -5,
            first_instance: 0,
        };

        let bytes = bytemuck::bytes_of(&args);
        assert_eq!(bytes.len(), 20);
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 36);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 1);
        assert_eq!(i32::from_le_bytes(bytes[12..16].try_into().unwrap()), -5);
    }

    #[test]
    fn frustum_data_layout_and_construction() {
        assert_eq!(std::mem::size_of::<GpuFrustumData>(), 112);

        let frustum = Frustum {
            planes: [
                [1.0, 0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0, -1.0],
                [0.0, -1.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, -0.1],
                [0.0, 0.0, -1.0, -100.0],
            ],
        };

        let data = GpuFrustumData::from_frustum(&frustum, [10.0, 20.0, 30.0]);
        assert_eq!(data.planes, frustum.planes);
        assert_eq!(data.camera_position, [10.0, 20.0, 30.0, 0.0]);
    }

    #[test]
    fn cull_params_layout() {
        assert_eq!(std::mem::size_of::<GpuCullParams>(), 16);

        let params = GpuCullParams {
            entity_count: 1000,
            _pad: [0; 3],
        };
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 1000);
    }

    #[test]
    fn gpu_cull_shader_is_valid_wgsl() {
        assert!(!GPU_CULL_SHADER.is_empty());
        assert!(GPU_CULL_SHADER.contains("@compute"));
        assert!(GPU_CULL_SHADER.contains("@workgroup_size(64)"));
        assert!(GPU_CULL_SHADER.contains("fn main"));
        assert!(GPU_CULL_SHADER.contains("DrawCommandGpu"));
        assert!(GPU_CULL_SHADER.contains("DrawIndexedIndirect"));
        assert!(GPU_CULL_SHADER.contains("frustum_test"));
        assert!(GPU_CULL_SHADER.contains("select_lod"));
        assert!(GPU_CULL_SHADER.contains("atomicAdd"));
    }

    #[test]
    fn workgroup_count_rounding() {
        assert_eq!(0_u32.div_ceil(WORKGROUP_SIZE), 0);
        assert_eq!(1_u32.div_ceil(WORKGROUP_SIZE), 1);
        assert_eq!(64_u32.div_ceil(WORKGROUP_SIZE), 1);
        assert_eq!(65_u32.div_ceil(WORKGROUP_SIZE), 2);
        assert_eq!(128_u32.div_ceil(WORKGROUP_SIZE), 2);
        assert_eq!(10000_u32.div_ceil(WORKGROUP_SIZE), 157);
    }

    #[test]
    fn indirect_buffer_size_calculation() {
        assert_eq!(
            DRAW_INDEXED_INDIRECT_SIZE as usize,
            std::mem::size_of::<DrawIndexedIndirectArgs>(),
        );

        let capacity = 1024u32;
        let expected_size = capacity as u64 * DRAW_INDEXED_INDIRECT_SIZE;
        assert_eq!(expected_size, 1024 * 20);
    }

    #[test]
    fn draw_command_gpu_zeroed() {
        let cmd = DrawCommandGpu::zeroed();
        assert_eq!(cmd.mesh_id, 0);
        assert_eq!(cmd.material_id, 0);
        assert_eq!(cmd.index_count, 0);
        assert_eq!(cmd.lod_count, 0);
        assert_eq!(cmd.model_col0, [0.0; 4]);
        assert_eq!(cmd.aabb_center, [0.0; 4]);
        assert_eq!(cmd.lod_distance_sq, [0.0; 4]);
    }
}
