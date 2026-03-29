use std::borrow::Cow;
use std::collections::HashMap;

use euca_rhi::RenderDevice;

use crate::metal_hints::{ComputeOptimizer, MetalRenderHints};

// ---------------------------------------------------------------------------
// ComputePipeline
// ---------------------------------------------------------------------------

/// Descriptor for creating a compute pipeline from WGSL source.
pub struct ComputePipelineDesc {
    /// Debug label for the pipeline.
    pub label: &'static str,
    /// WGSL shader source code.
    pub shader_source: &'static str,
    /// Entry-point function name inside the shader.
    pub entry_point: &'static str,
}

/// A compiled compute pipeline together with its bind-group layout.
pub struct ComputePipeline<D: RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    pipeline: D::ComputePipeline,
    bind_group_layout: D::BindGroupLayout,
}

impl<D: RenderDevice> ComputePipeline<D> {
    /// Create a compute pipeline from `desc` with explicit bind-group layout entries.
    ///
    /// The bind-group layout is built from `bindings` (group 0 only).
    /// Callers create bind groups against [`bind_group_layout()`](Self::bind_group_layout).
    pub fn new(
        device: &D,
        desc: &ComputePipelineDesc,
        bindings: &[euca_rhi::BindGroupLayoutEntry],
    ) -> Self {
        let shader = device.create_shader(&euca_rhi::ShaderDesc {
            label: Some(desc.label),
            source: euca_rhi::ShaderSource::Wgsl(Cow::Borrowed(desc.shader_source)),
        });

        let bind_group_layout = device.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some(desc.label),
            entries: bindings,
        });

        let pipeline = device.create_compute_pipeline(&euca_rhi::ComputePipelineDesc {
            label: Some(desc.label),
            layout: &[&bind_group_layout],
            module: &shader,
            entry_point: desc.entry_point,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// The bind-group layout for group 0.
    pub fn bind_group_layout(&self) -> &D::BindGroupLayout {
        &self.bind_group_layout
    }

    /// The underlying compute pipeline handle.
    pub fn raw(&self) -> &D::ComputePipeline {
        &self.pipeline
    }
}

// ---------------------------------------------------------------------------
// ComputePipeline — wgpu backward-compatibility
// ---------------------------------------------------------------------------
// Subsystems not yet generic can call these methods with raw wgpu types.
// They will be removed once all subsystems use the generic RenderDevice path.

impl ComputePipeline {
    /// Create a compute pipeline using wgpu auto-layout.
    ///
    /// Backward-compatible constructor that uses wgpu's shader reflection to
    /// derive the bind-group layout. Prefer [`ComputePipeline::new`] with
    /// explicit bindings for backend-portable code.
    pub fn from_wgpu(device: &wgpu::Device, desc: &ComputePipelineDesc) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(desc.label),
            source: wgpu::ShaderSource::Wgsl(desc.shader_source.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(desc.label),
            layout: None, // auto layout
            module: &shader_module,
            entry_point: Some(desc.entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        Self {
            pipeline,
            bind_group_layout,
        }
    }
}

// ---------------------------------------------------------------------------
// GpuBuffer
// ---------------------------------------------------------------------------

/// A GPU buffer wrapper that tracks its size.
pub struct GpuBuffer<D: RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    buffer: D::Buffer,
    size: u64,
}

impl<D: RenderDevice> GpuBuffer<D> {
    /// Create a buffer with the given `usage` flags, pre-filled with `initial_data`.
    fn init_with_bytes(
        device: &D,
        initial_data: &[u8],
        usage: euca_rhi::BufferUsages,
        label: &str,
    ) -> Self {
        let size = initial_data.len() as u64;
        let buffer = device.create_buffer(&euca_rhi::BufferDesc {
            label: Some(label),
            size,
            usage: usage | euca_rhi::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device.write_buffer(&buffer, 0, initial_data);
        Self { buffer, size }
    }

    /// Create an uninitialised storage buffer.
    ///
    /// The buffer is usable as storage (read/write from compute shaders) and as
    /// a copy-source/destination so the CPU can read back results.
    pub fn new_storage(device: &D, size: u64, label: &str) -> Self {
        let buffer = device.create_buffer(&euca_rhi::BufferDesc {
            label: Some(label),
            size,
            usage: euca_rhi::BufferUsages::STORAGE
                | euca_rhi::BufferUsages::COPY_DST
                | euca_rhi::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self { buffer, size }
    }

    /// Create a storage buffer pre-filled with `data`.
    pub fn new_storage_with_data<T: bytemuck::Pod>(device: &D, data: &[T], label: &str) -> Self {
        Self::init_with_bytes(
            device,
            bytemuck::cast_slice(data),
            euca_rhi::BufferUsages::STORAGE | euca_rhi::BufferUsages::COPY_SRC,
            label,
        )
    }

    /// Create a uniform buffer pre-filled with `data`.
    pub fn new_uniform_with_data<T: bytemuck::Pod>(device: &D, data: &T, label: &str) -> Self {
        Self::init_with_bytes(
            device,
            bytemuck::bytes_of(data),
            euca_rhi::BufferUsages::UNIFORM,
            label,
        )
    }

    /// Overwrite the buffer contents from the CPU side.
    pub fn write<T: bytemuck::Pod>(&self, device: &D, data: &[T]) {
        device.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
    }

    /// The underlying buffer handle.
    pub fn raw(&self) -> &D::Buffer {
        &self.buffer
    }

    /// Size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }
}

// ---------------------------------------------------------------------------
// GpuBuffer — wgpu backward-compatibility
// ---------------------------------------------------------------------------

impl GpuBuffer {
    /// Create a buffer pre-filled with `initial_data` via raw wgpu types.
    ///
    /// Uses `mapped_at_creation` for efficient zero-copy upload.
    fn init_with_bytes_wgpu(
        device: &wgpu::Device,
        initial_data: &[u8],
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> Self {
        let size = initial_data.len() as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(initial_data);
        buffer.unmap();
        Self { buffer, size }
    }

    /// Create an uninitialised storage buffer via raw wgpu.
    pub fn new_storage_wgpu(device: &wgpu::Device, size: u64, label: &str) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self { buffer, size }
    }

    /// Create a storage buffer pre-filled with `data` via raw wgpu.
    pub fn new_storage_with_data_wgpu<T: bytemuck::Pod>(
        device: &wgpu::Device,
        data: &[T],
        label: &str,
    ) -> Self {
        Self::init_with_bytes_wgpu(
            device,
            bytemuck::cast_slice(data),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            label,
        )
    }

    /// Create a uniform buffer pre-filled with `data` via raw wgpu.
    pub fn new_uniform_with_data_wgpu<T: bytemuck::Pod>(
        device: &wgpu::Device,
        data: &T,
        label: &str,
    ) -> Self {
        Self::init_with_bytes_wgpu(
            device,
            bytemuck::bytes_of(data),
            wgpu::BufferUsages::UNIFORM,
            label,
        )
    }

    /// Overwrite the buffer contents via raw wgpu `Queue`.
    pub fn write_wgpu<T: bytemuck::Pod>(&self, queue: &wgpu::Queue, data: &[T]) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
    }
}

// ---------------------------------------------------------------------------
// Dispatch helpers — shared workgroup resolution
// ---------------------------------------------------------------------------

/// Resolve workgroup counts, optionally applying Metal-specific optimizations.
fn resolve_workgroups(workgroups: [u32; 3], hints: Option<&MetalRenderHints>) -> [u32; 3] {
    match hints {
        Some(h) => {
            let dispatch = ComputeOptimizer::new(h).optimal_dispatch(workgroups[0]);
            [dispatch[0], workgroups[1], workgroups[2]]
        }
        None => workgroups,
    }
}

// ---------------------------------------------------------------------------
// Dispatch helper — generic
// ---------------------------------------------------------------------------

/// Record a compute dispatch into `encoder`.
///
/// `bind_groups` are bound in order starting at group 0.
/// `workgroups` is `[x, y, z]` — the number of workgroups to dispatch.
pub fn dispatch_compute_generic<D: RenderDevice>(
    device: &D,
    encoder: &mut D::CommandEncoder,
    pipeline: &ComputePipeline<D>,
    bind_groups: &[&D::BindGroup],
    workgroups: [u32; 3],
    hints: Option<&MetalRenderHints>,
) {
    let resolved = resolve_workgroups(workgroups, hints);
    let mut pass = device.begin_compute_pass(encoder, None);
    euca_rhi::ComputePassOps::set_pipeline(&mut pass, pipeline.raw());
    for (i, bg) in bind_groups.iter().enumerate() {
        euca_rhi::ComputePassOps::set_bind_group(&mut pass, i as u32, bg, &[]);
    }
    euca_rhi::ComputePassOps::dispatch_workgroups(&mut pass, resolved[0], resolved[1], resolved[2]);
}

/// Optimized generic compute dispatch that auto-computes workgroup counts.
pub fn dispatch_compute_optimized_generic<D: RenderDevice>(
    device: &D,
    encoder: &mut D::CommandEncoder,
    pipeline: &ComputePipeline<D>,
    bind_groups: &[&D::BindGroup],
    total_items: u32,
    hints: &MetalRenderHints,
) {
    let workgroups = ComputeOptimizer::new(hints).optimal_dispatch(total_items);
    let mut pass = device.begin_compute_pass(encoder, None);
    euca_rhi::ComputePassOps::set_pipeline(&mut pass, pipeline.raw());
    for (i, bg) in bind_groups.iter().enumerate() {
        euca_rhi::ComputePassOps::set_bind_group(&mut pass, i as u32, bg, &[]);
    }
    euca_rhi::ComputePassOps::dispatch_workgroups(
        &mut pass,
        workgroups[0],
        workgroups[1],
        workgroups[2],
    );
}

// ---------------------------------------------------------------------------
// Dispatch helper — wgpu backward-compatibility
// ---------------------------------------------------------------------------

/// Record a compute dispatch into `encoder` (wgpu backward-compatible).
///
/// `bind_groups` are bound in order starting at group 0.
/// `workgroups` is `[x, y, z]` — the number of workgroups to dispatch.
pub fn dispatch_compute(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &ComputePipeline,
    bind_groups: &[&wgpu::BindGroup],
    workgroups: [u32; 3],
    hints: Option<&MetalRenderHints>,
) {
    let resolved = resolve_workgroups(workgroups, hints);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline.raw());
    for (i, bg) in bind_groups.iter().enumerate() {
        pass.set_bind_group(i as u32, Some(*bg), &[]);
    }
    pass.dispatch_workgroups(resolved[0], resolved[1], resolved[2]);
}

/// Optimized compute dispatch that auto-computes workgroup counts (wgpu backward-compatible).
pub fn dispatch_compute_optimized(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &ComputePipeline,
    bind_groups: &[&wgpu::BindGroup],
    total_items: u32,
    hints: &MetalRenderHints,
) {
    let workgroups = ComputeOptimizer::new(hints).optimal_dispatch(total_items);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline.raw());
    for (i, bg) in bind_groups.iter().enumerate() {
        pass.set_bind_group(i as u32, Some(*bg), &[]);
    }
    pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
}

// ---------------------------------------------------------------------------
// ComputeManager
// ---------------------------------------------------------------------------

/// Manages a named collection of compute pipelines and GPU buffers.
///
/// Game systems register pipelines/buffers by name and look them up later for
/// dispatch. This keeps ownership centralized and avoids duplicating resources.
pub struct ComputeManager<D: RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    pipelines: HashMap<String, ComputePipeline<D>>,
    buffers: HashMap<String, GpuBuffer<D>>,
}

impl<D: RenderDevice> ComputeManager<D> {
    /// Create an empty manager.
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            buffers: HashMap::new(),
        }
    }

    /// Register a compute pipeline under `name`.
    pub fn add_pipeline(&mut self, name: impl Into<String>, pipeline: ComputePipeline<D>) {
        self.pipelines.insert(name.into(), pipeline);
    }

    /// Register a GPU buffer under `name`.
    pub fn add_buffer(&mut self, name: impl Into<String>, buffer: GpuBuffer<D>) {
        self.buffers.insert(name.into(), buffer);
    }

    /// Look up a pipeline by name.
    pub fn pipeline(&self, name: &str) -> Option<&ComputePipeline<D>> {
        self.pipelines.get(name)
    }

    /// Look up a buffer by name.
    pub fn buffer(&self, name: &str) -> Option<&GpuBuffer<D>> {
        self.buffers.get(name)
    }

    /// Remove and return a pipeline.
    pub fn remove_pipeline(&mut self, name: &str) -> Option<ComputePipeline<D>> {
        self.pipelines.remove(name)
    }

    /// Remove and return a buffer.
    pub fn remove_buffer(&mut self, name: &str) -> Option<GpuBuffer<D>> {
        self.buffers.remove(name)
    }
}

impl<D: RenderDevice> Default for ComputeManager<D> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GPU Frustum Culling — proof-of-concept
// ---------------------------------------------------------------------------

/// WGSL compute shader that performs per-entity frustum culling on the GPU.
///
/// Bindings (group 0):
///   @binding(0) `frustum_planes` — uniform `FrustumPlanes` (6 x vec4<f32>)
///   @binding(1) `aabbs`          — storage (read) array of `Aabb` structs
///   @binding(2) `visibility`     — storage (read_write) array of `atomic<u32>` bitset
///   @binding(3) `params`         — uniform `CullParams { entity_count: u32 }`
pub const FRUSTUM_CULL_SHADER: &str = include_str!("../shaders/frustum_cull.wgsl");

/// Bind-group layout entries for the frustum-culling shader (group 0).
pub const FRUSTUM_CULL_BINDINGS: &[euca_rhi::BindGroupLayoutEntry] = &[
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
            ty: euca_rhi::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    },
];

/// GPU-side AABB for the frustum culling shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAabb {
    pub center: [f32; 4],
    pub half_extents: [f32; 4],
}

/// GPU-side culling parameters.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CullParams {
    pub entity_count: u32,
    pub _pad: [u32; 3],
}

/// GPU-side frustum planes (6 planes, each a `vec4`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuFrustumPlanes {
    pub planes: [[f32; 4]; 6],
}

/// Set up the frustum-culling pipeline and its buffers inside a
/// `ComputeManager`. Returns the pipeline name so the caller can dispatch.
///
/// * `max_entities` — upper bound on the number of entities (determines buffer sizes).
pub fn setup_frustum_culling(
    device: &wgpu::Device,
    manager: &mut ComputeManager,
    max_entities: u32,
) -> &'static str {
    const NAME: &str = "frustum_cull";

    let pipeline = ComputePipeline::from_wgpu(
        device,
        &ComputePipelineDesc {
            label: "frustum_cull_pipeline",
            shader_source: FRUSTUM_CULL_SHADER,
            entry_point: "main",
        },
    );
    manager.add_pipeline(NAME, pipeline);

    // Frustum planes uniform (6 * vec4 = 96 bytes).
    let frustum_buf = GpuBuffer::new_uniform_with_data_wgpu(
        device,
        &GpuFrustumPlanes {
            planes: [[0.0; 4]; 6],
        },
        "frustum_planes",
    );
    manager.add_buffer("frustum_planes", frustum_buf);

    // AABB storage buffer.
    let aabb_size = (max_entities as u64) * std::mem::size_of::<GpuAabb>() as u64;
    let aabb_buf = GpuBuffer::new_storage_wgpu(device, aabb_size, "cull_aabbs");
    manager.add_buffer("cull_aabbs", aabb_buf);

    // Visibility bitset: one bit per entity, packed into u32 words.
    let vis_words = max_entities.div_ceil(32);
    let vis_size = (vis_words as u64) * 4;
    let vis_buf = GpuBuffer::new_storage_wgpu(device, vis_size, "cull_visibility");
    manager.add_buffer("cull_visibility", vis_buf);

    // CullParams uniform.
    let params_buf = GpuBuffer::new_uniform_with_data_wgpu(
        device,
        &CullParams {
            entity_count: 0,
            _pad: [0; 3],
        },
        "cull_params",
    );
    manager.add_buffer("cull_params", params_buf);

    NAME
}

/// Create the bind group for one frustum-cull dispatch.
///
/// Returns `None` if frustum culling has not been set up (pipeline or buffers missing).
pub fn create_frustum_cull_bind_group(
    device: &wgpu::Device,
    manager: &ComputeManager,
) -> Option<wgpu::BindGroup> {
    let pipeline = manager.pipeline("frustum_cull")?;
    let frustum_buf = manager.buffer("frustum_planes")?;
    let aabb_buf = manager.buffer("cull_aabbs")?;
    let vis_buf = manager.buffer("cull_visibility")?;
    let params_buf = manager.buffer("cull_params")?;

    Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("frustum_cull_bind_group"),
        layout: pipeline.bind_group_layout(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: frustum_buf.raw().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: aabb_buf.raw().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: vis_buf.raw().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.raw().as_entire_binding(),
            },
        ],
    }))
}

/// Dispatch a frustum-culling compute pass.
///
/// Before calling this, the caller should:
/// 1. Write frustum planes via `manager.buffer("frustum_planes").write(...)`.
/// 2. Write AABBs via `manager.buffer("cull_aabbs").write(...)`.
/// 3. Write entity count via `manager.buffer("cull_params").write(...)`.
/// 4. Clear the visibility buffer to 0.
pub fn dispatch_frustum_culling(
    encoder: &mut wgpu::CommandEncoder,
    manager: &ComputeManager,
    bind_group: &wgpu::BindGroup,
    entity_count: u32,
    hints: Option<&MetalRenderHints>,
) {
    let pipeline = manager
        .pipeline("frustum_cull")
        .expect("frustum_cull pipeline not set up");

    match hints {
        Some(h) => {
            dispatch_compute_optimized(encoder, pipeline, &[bind_group], entity_count, h);
        }
        None => {
            let workgroup_count = entity_count.div_ceil(64);
            dispatch_compute(
                encoder,
                pipeline,
                &[bind_group],
                [workgroup_count, 1, 1],
                None,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_manager_insert_and_lookup() {
        let mut mgr = ComputeManager::<euca_rhi::wgpu_backend::WgpuDevice>::new();
        assert!(mgr.pipeline("foo").is_none());
        assert!(mgr.buffer("bar").is_none());

        // We can't create real pipelines without a device, but we can test
        // the HashMap plumbing with remove on empty.
        assert!(mgr.remove_pipeline("x").is_none());
        assert!(mgr.remove_buffer("x").is_none());
    }

    #[test]
    fn compute_manager_default() {
        let mgr = ComputeManager::<euca_rhi::wgpu_backend::WgpuDevice>::default();
        assert!(mgr.pipeline("anything").is_none());
        assert!(mgr.buffer("anything").is_none());
    }

    #[test]
    fn gpu_aabb_layout() {
        // Verify the GPU struct sizes match shader expectations.
        assert_eq!(std::mem::size_of::<GpuAabb>(), 32); // 2 * vec4<f32>
        assert_eq!(std::mem::size_of::<CullParams>(), 16); // 1 * vec4<u32>
        assert_eq!(std::mem::size_of::<GpuFrustumPlanes>(), 96); // 6 * vec4<f32>
    }

    #[test]
    fn cull_params_padding() {
        let params = CullParams {
            entity_count: 42,
            _pad: [0; 3],
        };
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 16);
        // First 4 bytes should be entity_count = 42 in little-endian.
        assert_eq!(
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            42
        );
    }

    #[test]
    fn frustum_cull_shader_is_valid_wgsl_source() {
        // Sanity-check: the shader source is valid UTF-8 and non-empty.
        assert!(!FRUSTUM_CULL_SHADER.is_empty());
        assert!(FRUSTUM_CULL_SHADER.contains("@compute"));
        assert!(FRUSTUM_CULL_SHADER.contains("@workgroup_size(64)"));
        assert!(FRUSTUM_CULL_SHADER.contains("fn main"));
    }

    #[test]
    fn workgroup_count_rounding() {
        // Verify the ceil(entity_count / 64) formula.
        assert_eq!(0_u32.div_ceil(64), 0);
        assert_eq!(1_u32.div_ceil(64), 1);
        assert_eq!(64_u32.div_ceil(64), 1);
        assert_eq!(65_u32.div_ceil(64), 2);
        assert_eq!(128_u32.div_ceil(64), 2);
        assert_eq!(129_u32.div_ceil(64), 3);
    }

    #[test]
    fn dispatch_compute_resolves_hints_apple() {
        let hints = MetalRenderHints {
            is_apple_gpu: true,
            prefer_single_pass: true,
            optimal_threadgroup_size: 32,
            supports_memoryless: true,
        };
        let optimizer = ComputeOptimizer::new(&hints);
        assert_eq!(optimizer.optimal_dispatch(100), [4, 1, 1]);
    }

    #[test]
    fn dispatch_compute_resolves_hints_discrete() {
        let hints = MetalRenderHints {
            is_apple_gpu: false,
            prefer_single_pass: false,
            optimal_threadgroup_size: 64,
            supports_memoryless: false,
        };
        let optimizer = ComputeOptimizer::new(&hints);
        assert_eq!(optimizer.optimal_dispatch(100), [2, 1, 1]);
    }

    #[test]
    fn dispatch_compute_no_hints_passthrough() {
        let workgroups = [4_u32, 1, 1];
        assert_eq!(workgroups, [4, 1, 1]);
    }

    #[test]
    fn frustum_cull_dispatch_with_apple_hints() {
        let hints = MetalRenderHints {
            is_apple_gpu: true,
            prefer_single_pass: true,
            optimal_threadgroup_size: 32,
            supports_memoryless: true,
        };
        let optimizer = ComputeOptimizer::new(&hints);
        assert_eq!(optimizer.optimal_dispatch(1000), [32, 1, 1]);
        assert_eq!(optimizer.optimal_dispatch(64), [2, 1, 1]);
    }

    #[test]
    fn frustum_cull_dispatch_without_hints_uses_64() {
        assert_eq!(1000_u32.div_ceil(64), 16);
        assert_eq!(64_u32.div_ceil(64), 1);
        assert_eq!(65_u32.div_ceil(64), 2);
    }

    #[test]
    fn frustum_cull_bindings_count() {
        assert_eq!(FRUSTUM_CULL_BINDINGS.len(), 4);
    }
}
