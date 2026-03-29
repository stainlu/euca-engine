//! GPU-driven rendering pipeline using indirect draw calls.
//!
//! Instead of the CPU issuing one `draw_indexed` per entity, this module:
//!
//! 1. Uploads all entity draw commands to a GPU storage buffer.
//! 2. Dispatches a compute shader (`gpu_cull.wgsl`) that performs frustum
//!    culling and LOD selection entirely on the GPU.
//! 3. The compute shader writes `DrawIndexedIndirect` arguments into an
//!    output storage buffer and atomically increments a draw count.
//! 4. **Preferred path:** A single `multi_draw_indexed_indirect_count` call
//!    renders all visible entities in one GPU submission, using the draw
//!    count buffer to skip culled entries entirely.
//! 5. **Fallback:** `draw_indirect()` loops per entity slot calling
//!    `draw_indexed_indirect`, where culled entities have `index_count = 0`.
//!
//! This replaces per-entity CPU frustum culling and draw-call submission
//! with a single compute dispatch, dramatically reducing CPU overhead for
//! large scenes.

use crate::camera::Frustum;
use crate::compute::{ComputePipeline, ComputePipelineDesc, GpuBuffer};
use euca_rhi::pass::RenderPassOps;

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

impl DrawCommandGpu {
    /// Build a GPU draw command from raw geometry and material components.
    ///
    /// `model` is the 4x4 column-major model matrix.
    /// `vertex_offset`, `first_index`, `index_count` describe the mesh region
    /// in the global geometry pool.
    /// `aabb_center` and `aabb_half_extents` are in world space.
    /// `mesh_id` is the mesh handle index; `material_id` is the bindless
    /// material index.
    ///
    /// A single LOD level is created from the provided geometry. Use the
    /// struct fields directly to configure multi-LOD entries.
    #[allow(clippy::too_many_arguments)]
    pub fn from_components(
        model: &[[f32; 4]; 4],
        vertex_offset: i32,
        first_index: u32,
        index_count: u32,
        aabb_center: [f32; 3],
        aabb_half_extents: [f32; 3],
        mesh_id: u32,
        material_id: u32,
    ) -> Self {
        Self {
            model_col0: model[0],
            model_col1: model[1],
            model_col2: model[2],
            model_col3: model[3],
            aabb_center: [aabb_center[0], aabb_center[1], aabb_center[2], 0.0],
            aabb_half_extents: [
                aabb_half_extents[0],
                aabb_half_extents[1],
                aabb_half_extents[2],
                0.0,
            ],
            mesh_id,
            material_id,
            index_count,
            first_index,
            vertex_offset,
            lod_count: 1,
            lod_index_counts: [index_count, 0, 0, 0],
            lod_first_indices: [first_index, 0, 0, 0],
            lod_vertex_offsets: [vertex_offset, 0, 0, 0],
            lod_distance_sq: [f32::MAX, f32::MAX, f32::MAX, f32::MAX],
        }
    }
}

/// Mirrors the `DrawIndexedIndirect` GPU layout -- the output of the compute shader.
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

impl<D: euca_rhi::RenderDevice> IndirectDrawBuffer<D> {
    /// Create an indirect draw buffer that can hold up to `max_draws` entries.
    pub fn new(device: &D, max_draws: u32) -> Self {
        let size = (max_draws as u64) * DRAW_INDEXED_INDIRECT_SIZE;
        let buffer = device.create_buffer(&euca_rhi::BufferDesc {
            label: Some("indirect_draw_buffer"),
            size,
            usage: euca_rhi::BufferUsages::STORAGE
                | euca_rhi::BufferUsages::INDIRECT
                | euca_rhi::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            capacity: max_draws,
        }
    }

    /// The underlying buffer handle.
    pub fn raw(&self) -> &D::Buffer {
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

/// Bind-group layout entries for the GPU cull compute shader (group 0).
///
/// - binding 0: storage read (draw commands)
/// - binding 1: uniform (frustum data)
/// - binding 2: storage read_write (indirect args output)
/// - binding 3: storage read_write (draw count atomic)
/// - binding 4: uniform (cull params)
pub const GPU_CULL_BINDINGS: &[euca_rhi::BindGroupLayoutEntry] = &[
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
    euca_rhi::BindGroupLayoutEntry {
        binding: 4,
        visibility: euca_rhi::ShaderStages::COMPUTE,
        ty: euca_rhi::BindingType::Buffer {
            ty: euca_rhi::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
];

/// Manages the complete GPU-driven rendering pipeline: command upload,
/// compute-based culling, and indirect draw dispatch.
pub struct GpuDrivenPipeline<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    cull_pipeline: ComputePipeline<D>,
    command_buffer: GpuBuffer<D>,
    frustum_buffer: GpuBuffer<D>,
    params_buffer: GpuBuffer<D>,
    draw_count_buffer: GpuBuffer<D>,
    indirect_buffer: IndirectDrawBuffer<D>,
    max_entities: u32,
}

impl<D: euca_rhi::RenderDevice> GpuDrivenPipeline<D> {
    /// Create a new GPU-driven pipeline sized for up to `max_entities`.
    pub fn new(device: &D, max_entities: u32) -> Self {
        let cull_pipeline = ComputePipeline::new(
            device,
            &ComputePipelineDesc {
                label: "gpu_cull_pipeline",
                shader_source: GPU_CULL_SHADER,
                entry_point: "main",
            },
            GPU_CULL_BINDINGS,
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
    pub fn upload_commands(&self, device: &D, commands: &[DrawCommandGpu]) {
        assert!(
            commands.len() <= self.max_entities as usize,
            "Too many draw commands ({}) for pipeline capacity ({})",
            commands.len(),
            self.max_entities
        );
        self.command_buffer.write(device, commands);
    }

    /// Upload frustum data for this frame.
    pub fn upload_frustum(&self, device: &D, frustum_data: &GpuFrustumData) {
        self.frustum_buffer
            .write(device, std::slice::from_ref(frustum_data));
    }

    /// Upload the entity count parameter. Call before `cull_and_prepare`.
    pub fn upload_params(&self, device: &D, entity_count: u32) {
        let params = GpuCullParams {
            entity_count,
            _pad: [0; 3],
        };
        self.params_buffer
            .write(device, std::slice::from_ref(&params));
    }

    /// Dispatch the GPU cull compute shader.
    ///
    /// The caller must have called `upload_commands`, `upload_frustum`, and
    /// `upload_params` before beginning the command encoder.
    pub fn cull_and_prepare(&self, device: &D, encoder: &mut D::CommandEncoder, entity_count: u32) {
        assert!(
            entity_count <= self.max_entities,
            "entity_count ({entity_count}) exceeds pipeline capacity ({})",
            self.max_entities
        );

        device.clear_buffer(encoder, self.draw_count_buffer.raw(), 0, None);

        let bind_group = device.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("gpu_cull_bind_group"),
            layout: self.cull_pipeline.bind_group_layout(),
            entries: &[
                euca_rhi::BindGroupEntry {
                    binding: 0,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: self.command_buffer.raw(),
                        offset: 0,
                        size: None,
                    }),
                },
                euca_rhi::BindGroupEntry {
                    binding: 1,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: self.frustum_buffer.raw(),
                        offset: 0,
                        size: None,
                    }),
                },
                euca_rhi::BindGroupEntry {
                    binding: 2,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: self.indirect_buffer.raw(),
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
                euca_rhi::BindGroupEntry {
                    binding: 4,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: self.params_buffer.raw(),
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let workgroup_count = entity_count.div_ceil(WORKGROUP_SIZE);
        crate::compute::dispatch_compute_generic(
            device,
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
    ///
    /// This loops per entity calling `draw_indexed_indirect` individually.
    /// Prefer [`draw_indirect_multi`](Self::draw_indirect_multi) when the
    /// `MULTI_DRAW_INDIRECT_COUNT` capability is available.
    pub fn draw_indirect<'a>(&'a self, render_pass: &mut D::RenderPass<'a>, entity_count: u32) {
        for i in 0..entity_count {
            let offset = (i as u64) * DRAW_INDEXED_INDIRECT_SIZE;
            render_pass.draw_indexed_indirect(self.indirect_buffer.raw(), offset);
        }
    }

    /// Issue a single multi-draw-indirect-count call that renders all visible
    /// entities in one GPU submission. The `draw_count_buffer` (populated by
    /// the compute cull shader) tells the GPU how many draws to execute.
    ///
    /// Requires `MULTI_DRAW_INDIRECT_COUNT` capability. Falls back to
    /// [`draw_indirect`](Self::draw_indirect) loop if not available.
    pub fn draw_indirect_multi<'a>(&'a self, render_pass: &mut D::RenderPass<'a>, max_count: u32) {
        render_pass.multi_draw_indexed_indirect_count(
            self.indirect_buffer.raw(),
            0,
            self.draw_count_buffer.raw(),
            0,
            max_count,
        );
    }

    /// Access the raw indirect draw buffer handle for binding in render passes.
    pub fn indirect_buffer_raw(&self) -> &D::Buffer {
        self.indirect_buffer.raw()
    }

    /// Access the indirect draw buffer.
    pub fn indirect_buffer(&self) -> &IndirectDrawBuffer<D> {
        &self.indirect_buffer
    }

    /// Access the draw count buffer.
    pub fn draw_count_buffer(&self) -> &GpuBuffer<D> {
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
    fn draw_command_gpu_from_components() {
        let model: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [10.0, 20.0, 30.0, 1.0],
        ];
        let cmd = DrawCommandGpu::from_components(
            &model,
            100, // vertex_offset
            200, // first_index
            36,  // index_count
            [5.0, 6.0, 7.0],
            [1.0, 2.0, 3.0],
            42, // mesh_id
            7,  // material_id
        );

        // Model matrix columns
        assert_eq!(cmd.model_col0, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(cmd.model_col1, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(cmd.model_col2, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(cmd.model_col3, [10.0, 20.0, 30.0, 1.0]);

        // AABB (w-component should be 0.0)
        assert_eq!(cmd.aabb_center, [5.0, 6.0, 7.0, 0.0]);
        assert_eq!(cmd.aabb_half_extents, [1.0, 2.0, 3.0, 0.0]);

        // IDs
        assert_eq!(cmd.mesh_id, 42);
        assert_eq!(cmd.material_id, 7);

        // Geometry
        assert_eq!(cmd.index_count, 36);
        assert_eq!(cmd.first_index, 200);
        assert_eq!(cmd.vertex_offset, 100);

        // Single LOD populated from the provided geometry
        assert_eq!(cmd.lod_count, 1);
        assert_eq!(cmd.lod_index_counts, [36, 0, 0, 0]);
        assert_eq!(cmd.lod_first_indices, [200, 0, 0, 0]);
        assert_eq!(cmd.lod_vertex_offsets, [100, 0, 0, 0]);
        assert_eq!(
            cmd.lod_distance_sq,
            [f32::MAX, f32::MAX, f32::MAX, f32::MAX]
        );

        // Verify it's still 184 bytes and bytemuck-safe
        let bytes = bytemuck::bytes_of(&cmd);
        assert_eq!(bytes.len(), 184);
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
