//! GPU-accelerated terrain mesh generation via compute shader.
//!
//! Instead of generating terrain vertices on the CPU, this module uploads
//! the heightmap to a GPU storage buffer and dispatches a compute shader
//! that fills vertex and index buffers directly on the GPU.
//!
//! The output vertex layout exactly matches [`euca_render::vertex::Vertex`]:
//! position (`[f32; 3]`), normal (`[f32; 3]`), tangent (`[f32; 3]`), UV
//! (`[f32; 2]`) — 44 bytes per vertex, tightly packed in `#[repr(C)]` order.

use euca_rhi::{
    BindGroupDesc, BindGroupEntry, BindGroupLayoutDesc, BindGroupLayoutEntry, BindingResource,
    BindingType, BufferBinding, BufferBindingType, BufferDesc, BufferUsages, ComputePipelineDesc,
    RenderDevice, ShaderDesc, ShaderSource, ShaderStages, pass::ComputePassOps,
};

/// WGSL compute shader for terrain mesh generation.
const TERRAIN_GEN_SHADER: &str = include_str!("../shaders/terrain_gen.wgsl");

/// Floats per vertex in the output buffer (position=3 + normal=3 + tangent=3 + uv=2).
const FLOATS_PER_VERTEX: u32 = 11;

/// Bytes per vertex (44 bytes, matching the engine's `Vertex` struct).
const BYTES_PER_VERTEX: u64 = (FLOATS_PER_VERTEX as u64) * std::mem::size_of::<f32>() as u64;

/// Indices per quad (two triangles).
const INDICES_PER_QUAD: u32 = 6;

/// Parameters for GPU terrain generation, uploaded as a uniform buffer.
///
/// This struct is 48 bytes (12 x `f32`), matching the WGSL `TerrainParams`
/// uniform layout. The three padding fields align the struct to a 16-byte
/// boundary as required by the uniform address space.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainGenParams {
    /// Number of vertex columns in this chunk's grid.
    pub grid_cols: u32,
    /// Number of vertex rows in this chunk's grid.
    pub grid_rows: u32,
    /// World-space distance between adjacent grid vertices.
    pub cell_size: f32,
    /// LOD step (1 = full detail, 2 = half, 4 = quarter, etc.).
    pub step: u32,
    /// World-space X origin of this chunk.
    pub origin_x: f32,
    /// World-space Z origin of this chunk.
    pub origin_z: f32,
    /// Heightmap width (number of columns in the source data).
    pub heightmap_width: u32,
    /// Heightmap height (number of rows in the source data).
    pub heightmap_height: u32,
    /// Height scale multiplier applied to raw heightmap values.
    pub height_scale: f32,
    /// Padding to reach 48-byte (16-byte-aligned) size.
    pub _pad: [f32; 3],
}

/// Manages GPU-side terrain generation resources: heightmap storage,
/// parameter uniform, compute pipeline, and bind group layout.
///
/// The generator is generic over [`RenderDevice`] so it works with any
/// backend (wgpu, native Metal, etc.).
pub struct GpuTerrainGenerator<D: RenderDevice> {
    heightmap_buffer: D::Buffer,
    params_buffer: D::Buffer,
    bind_group_layout: D::BindGroupLayout,
    vertex_pipeline: D::ComputePipeline,
    index_pipeline: D::ComputePipeline,
}

/// Shared bind group layout entries for the terrain generation pipeline.
///
/// Binding 0: heightmap (storage, read-only)
/// Binding 1: params (uniform)
/// Binding 2: vertex output (storage, read-write)
/// Binding 3: index output (storage, read-write)
fn layout_entries() -> [BindGroupLayoutEntry; 4] {
    [
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ]
}

impl<D: RenderDevice> GpuTerrainGenerator<D> {
    /// Create a new GPU terrain generator.
    ///
    /// `heightmap_data` is the flat row-major array of `f32` height values
    /// (normalised to `[0, 1]`). `width` and `height` are the heightmap
    /// grid dimensions.
    pub fn new(device: &D, heightmap_data: &[f32], width: u32, height: u32) -> Self {
        assert_eq!(
            heightmap_data.len(),
            (width as usize) * (height as usize),
            "heightmap_data length must equal width * height"
        );

        // Heightmap storage buffer (read-only by the shader).
        let heightmap_buffer = device.create_buffer(&BufferDesc {
            label: Some("Terrain Heightmap"),
            size: (heightmap_data.len() as u64) * std::mem::size_of::<f32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device.write_buffer(&heightmap_buffer, 0, bytemuck::cast_slice(heightmap_data));

        // Uniform buffer for generation parameters.
        let params_buffer = device.create_buffer(&BufferDesc {
            label: Some("Terrain Gen Params"),
            size: std::mem::size_of::<TerrainGenParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layout (shared between vertex and index pipelines).
        let entries = layout_entries();
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("Terrain Gen Layout"),
            entries: &entries,
        });

        // Compile the shader module.
        let shader = device.create_shader(&ShaderDesc {
            label: Some("Terrain Gen Shader"),
            source: ShaderSource::Wgsl(TERRAIN_GEN_SHADER.into()),
        });

        // Two pipelines: one for vertex generation, one for index generation.
        let vertex_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            label: Some("Terrain Gen Vertices"),
            layout: &[&bind_group_layout],
            module: &shader,
            entry_point: "generate_vertices",
        });

        let index_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            label: Some("Terrain Gen Indices"),
            layout: &[&bind_group_layout],
            module: &shader,
            entry_point: "generate_indices",
        });

        Self {
            heightmap_buffer,
            params_buffer,
            bind_group_layout,
            vertex_pipeline,
            index_pipeline,
        }
    }

    /// Upload new heightmap data (for dynamic terrain modification).
    ///
    /// The new data must have the same length as the original heightmap.
    pub fn update_heightmap(&self, device: &D, data: &[f32]) {
        device.write_buffer(&self.heightmap_buffer, 0, bytemuck::cast_slice(data));
    }

    /// Generate terrain mesh into the provided vertex and index buffers.
    ///
    /// The caller must provide pre-allocated output buffers with sufficient
    /// capacity:
    /// - vertex buffer: `grid_cols * grid_rows * 44` bytes
    /// - index buffer: `(grid_cols - 1) * (grid_rows - 1) * 6 * 4` bytes
    ///
    /// Both output buffers must have `BufferUsages::STORAGE` set.
    pub fn generate(
        &self,
        device: &D,
        encoder: &mut D::CommandEncoder,
        params: &TerrainGenParams,
        output_vertex_buffer: &D::Buffer,
        output_index_buffer: &D::Buffer,
    ) {
        // Upload generation parameters.
        device.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));

        // Create bind group referencing all four buffers.
        let bind_group = device.create_bind_group(&BindGroupDesc {
            label: Some("Terrain Gen Bind"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &self.heightmap_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &self.params_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: output_vertex_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: output_index_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let total_vertices = params.grid_cols * params.grid_rows;
        let total_quads = params.grid_cols.saturating_sub(1) * params.grid_rows.saturating_sub(1);

        // Dispatch vertex generation.
        {
            let workgroups = total_vertices.div_ceil(64);
            let mut pass = device.begin_compute_pass(encoder, Some("Terrain Gen Vertices"));
            pass.set_pipeline(&self.vertex_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Dispatch index generation.
        {
            let workgroups = total_quads.div_ceil(64);
            let mut pass = device.begin_compute_pass(encoder, Some("Terrain Gen Indices"));
            pass.set_pipeline(&self.index_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    /// Compute the required vertex buffer size in bytes for the given grid.
    pub fn vertex_buffer_size(grid_cols: u32, grid_rows: u32) -> u64 {
        (grid_cols as u64) * (grid_rows as u64) * BYTES_PER_VERTEX
    }

    /// Compute the required index buffer size in bytes for the given grid.
    pub fn index_buffer_size(grid_cols: u32, grid_rows: u32) -> u64 {
        let quads = (grid_cols.saturating_sub(1) as u64) * (grid_rows.saturating_sub(1) as u64);
        quads * (INDICES_PER_QUAD as u64) * std::mem::size_of::<u32>() as u64
    }

    /// Generate terrain mesh for a chunk, returning GPU buffers ready for rendering.
    ///
    /// This is a higher-level wrapper around [`generate`](Self::generate) that
    /// allocates output buffers with the correct usage flags
    /// (`STORAGE | VERTEX` / `STORAGE | INDEX`) so they can serve as both
    /// compute shader output and render pipeline input, then dispatches the
    /// compute kernels.
    ///
    /// The returned [`GpuTerrainChunkOutput`] contains the buffer handles and
    /// index count needed to register the mesh with the renderer via
    /// `Renderer::register_gpu_mesh`.
    pub fn generate_chunk(
        &self,
        device: &D,
        encoder: &mut D::CommandEncoder,
        params: &TerrainGenParams,
    ) -> GpuTerrainChunkOutput<D> {
        let vb_size = Self::vertex_buffer_size(params.grid_cols, params.grid_rows);
        let ib_size = Self::index_buffer_size(params.grid_cols, params.grid_rows);

        let vertex_buffer = device.create_buffer(&BufferDesc {
            label: Some("Terrain Chunk Vertices"),
            size: vb_size,
            usage: BufferUsages::STORAGE | BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&BufferDesc {
            label: Some("Terrain Chunk Indices"),
            size: ib_size,
            usage: BufferUsages::STORAGE | BufferUsages::INDEX,
            mapped_at_creation: false,
        });

        self.generate(device, encoder, params, &vertex_buffer, &index_buffer);

        let index_count = (params.grid_cols.saturating_sub(1))
            * (params.grid_rows.saturating_sub(1))
            * INDICES_PER_QUAD;

        GpuTerrainChunkOutput {
            vertex_buffer,
            vertex_buffer_size: vb_size,
            index_buffer,
            index_buffer_size: ib_size,
            index_count,
        }
    }
}

/// Output from [`GpuTerrainGenerator::generate_chunk`]: GPU buffers containing
/// the generated terrain mesh, ready to be registered with the renderer.
pub struct GpuTerrainChunkOutput<D: RenderDevice> {
    /// Vertex buffer (layout matches `euca_render::Vertex`: 44 bytes per vertex).
    pub vertex_buffer: D::Buffer,
    /// Size of the vertex buffer in bytes.
    pub vertex_buffer_size: u64,
    /// Index buffer (`u32` triangle indices).
    pub index_buffer: D::Buffer,
    /// Size of the index buffer in bytes.
    pub index_buffer_size: u64,
    /// Number of indices in the index buffer.
    pub index_count: u32,
}

#[cfg(test)]
mod tests {
    use bytemuck::Zeroable;

    use super::*;

    #[test]
    fn terrain_gen_params_is_48_bytes() {
        assert_eq!(std::mem::size_of::<TerrainGenParams>(), 48);
    }

    #[test]
    fn terrain_gen_params_is_pod_and_zeroable() {
        // Verify bytemuck traits by constructing a zeroed instance.
        let zeroed = TerrainGenParams::zeroed();
        assert_eq!(zeroed.grid_cols, 0);
        assert_eq!(zeroed.grid_rows, 0);
        assert_eq!(zeroed.cell_size, 0.0);
        assert_eq!(zeroed.step, 0);

        // Verify round-trip through bytes.
        let params = TerrainGenParams {
            grid_cols: 32,
            grid_rows: 32,
            cell_size: 1.0,
            step: 1,
            origin_x: 0.0,
            origin_z: 0.0,
            heightmap_width: 128,
            heightmap_height: 128,
            height_scale: 50.0,
            _pad: [0.0; 3],
        };
        let bytes = bytemuck::bytes_of(&params);
        let restored: &TerrainGenParams = bytemuck::from_bytes(bytes);
        assert_eq!(restored.grid_cols, 32);
        assert_eq!(restored.height_scale, 50.0);
    }

    #[test]
    fn vertex_buffer_size_calculation() {
        // 4x4 grid = 16 vertices * 44 bytes = 704
        assert_eq!(
            GpuTerrainGenerator::<DummyDevice>::vertex_buffer_size(4, 4),
            704
        );
    }

    #[test]
    fn index_buffer_size_calculation() {
        // 4x4 grid = 3x3 quads = 9 quads * 6 indices * 4 bytes = 216
        assert_eq!(
            GpuTerrainGenerator::<DummyDevice>::index_buffer_size(4, 4),
            216
        );
    }

    #[test]
    fn index_buffer_size_degenerate_grid() {
        // 1x1 grid = 0 quads = 0 bytes
        assert_eq!(
            GpuTerrainGenerator::<DummyDevice>::index_buffer_size(1, 1),
            0
        );
        // 2x2 grid = 1 quad = 6 * 4 = 24 bytes
        assert_eq!(
            GpuTerrainGenerator::<DummyDevice>::index_buffer_size(2, 2),
            24
        );
    }

    // Minimal dummy device to allow calling associated functions.
    // The actual GPU pipeline tests require a real device.
    struct DummyDevice;

    impl RenderDevice for DummyDevice {
        type Buffer = ();
        type Texture = ();
        type TextureView = ();
        type Sampler = ();
        type BindGroupLayout = ();
        type BindGroup = ();
        type ShaderModule = ();
        type RenderPipeline = ();
        type ComputePipeline = ();
        type CommandEncoder = ();
        type RenderPass<'a> = DummyRenderPass;
        type ComputePass<'a> = DummyComputePass;
        type SurfaceTexture = ();

        fn capabilities(&self) -> &euca_rhi::Capabilities {
            unimplemented!()
        }
        fn create_buffer(&self, _: &BufferDesc) -> Self::Buffer {}
        fn create_texture(&self, _: &euca_rhi::TextureDesc) -> Self::Texture {}
        fn create_texture_view(
            &self,
            _: &Self::Texture,
            _: &euca_rhi::TextureViewDesc,
        ) -> Self::TextureView {
        }
        fn create_sampler(&self, _: &euca_rhi::SamplerDesc) -> Self::Sampler {}
        fn create_shader(&self, _: &ShaderDesc) -> Self::ShaderModule {}
        fn create_bind_group_layout(&self, _: &BindGroupLayoutDesc) -> Self::BindGroupLayout {}
        fn create_bind_group(&self, _: &BindGroupDesc<Self>) -> Self::BindGroup {}
        fn create_render_pipeline(
            &self,
            _: &euca_rhi::RenderPipelineDesc<Self>,
        ) -> Self::RenderPipeline {
        }
        fn create_compute_pipeline(&self, _: &ComputePipelineDesc<Self>) -> Self::ComputePipeline {}
        fn write_buffer(&self, _: &Self::Buffer, _: u64, _: &[u8]) {}
        fn write_texture(
            &self,
            _: &euca_rhi::TexelCopyTextureInfo<Self>,
            _: &[u8],
            _: &euca_rhi::TextureDataLayout,
            _: euca_rhi::Extent3d,
        ) {
        }
        fn create_command_encoder(&self, _: Option<&str>) -> Self::CommandEncoder {}
        fn begin_render_pass<'a>(
            &self,
            _: &'a mut Self::CommandEncoder,
            _: &euca_rhi::RenderPassDesc<'_, Self>,
        ) -> Self::RenderPass<'a> {
            DummyRenderPass
        }
        fn begin_compute_pass<'a>(
            &self,
            _: &'a mut Self::CommandEncoder,
            _: Option<&str>,
        ) -> Self::ComputePass<'a> {
            DummyComputePass
        }
        fn clear_buffer(
            &self,
            _: &mut Self::CommandEncoder,
            _: &Self::Buffer,
            _: u64,
            _: Option<u64>,
        ) {
        }
        fn copy_texture_to_texture(
            &self,
            _: &mut Self::CommandEncoder,
            _: &euca_rhi::TexelCopyTextureInfo<Self>,
            _: &euca_rhi::TexelCopyTextureInfo<Self>,
            _: euca_rhi::Extent3d,
        ) {
        }
        fn submit(&self, _: Self::CommandEncoder) {}
        fn get_current_texture(&self) -> Result<Self::SurfaceTexture, euca_rhi::SurfaceError> {
            Err(euca_rhi::SurfaceError::Lost)
        }
        fn surface_texture_view(&self, _: &Self::SurfaceTexture) -> Self::TextureView {}
        fn present(&self, _: Self::SurfaceTexture) {}
        fn resize_surface(&mut self, _: u32, _: u32) {}
        fn surface_format(&self) -> euca_rhi::TextureFormat {
            euca_rhi::TextureFormat::Bgra8Unorm
        }
        fn surface_size(&self) -> (u32, u32) {
            (1, 1)
        }
    }

    struct DummyRenderPass;
    impl euca_rhi::pass::RenderPassOps<DummyDevice> for DummyRenderPass {
        fn set_pipeline(&mut self, _: &()) {}
        fn set_bind_group(&mut self, _: u32, _: &(), _: &[u32]) {}
        fn set_vertex_buffer(&mut self, _: u32, _: &(), _: u64, _: u64) {}
        fn set_index_buffer(&mut self, _: &(), _: euca_rhi::IndexFormat, _: u64, _: u64) {}
        fn draw(&mut self, _: std::ops::Range<u32>, _: std::ops::Range<u32>) {}
        fn draw_indexed(&mut self, _: std::ops::Range<u32>, _: i32, _: std::ops::Range<u32>) {}
        fn draw_indexed_indirect(&mut self, _: &(), _: u64) {}
        fn multi_draw_indexed_indirect(&mut self, _: &(), _: u64, _: u32) {}
        fn multi_draw_indexed_indirect_count(&mut self, _: &(), _: u64, _: &(), _: u64, _: u32) {}
        fn set_viewport(&mut self, _: f32, _: f32, _: f32, _: f32, _: f32, _: f32) {}
        fn set_scissor_rect(&mut self, _: u32, _: u32, _: u32, _: u32) {}
    }

    struct DummyComputePass;
    impl euca_rhi::pass::ComputePassOps<DummyDevice> for DummyComputePass {
        fn set_pipeline(&mut self, _: &()) {}
        fn set_bind_group(&mut self, _: u32, _: &(), _: &[u32]) {}
        fn dispatch_workgroups(&mut self, _: u32, _: u32, _: u32) {}
        fn dispatch_workgroups_indirect(&mut self, _: &(), _: u64) {}
    }
}
