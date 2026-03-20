use std::collections::HashMap;

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

/// A compiled compute pipeline together with its auto-derived bind-group layout.
pub struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ComputePipeline {
    /// Create a compute pipeline from `desc`.
    ///
    /// The bind-group layout is derived automatically from the shader source
    /// (group 0 only). Callers create bind groups against `bind_group_layout()`.
    pub fn new(device: &wgpu::Device, desc: &ComputePipelineDesc) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(desc.label),
            source: wgpu::ShaderSource::Wgsl(desc.shader_source.into()),
        });

        // Use auto layout so the bind-group layout is derived from the shader.
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

    /// The bind-group layout for group 0, auto-derived from the shader.
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// The underlying `wgpu::ComputePipeline`.
    pub fn raw(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }
}

// ---------------------------------------------------------------------------
// GpuBuffer
// ---------------------------------------------------------------------------

/// A GPU buffer wrapper that tracks its size.
pub struct GpuBuffer {
    buffer: wgpu::Buffer,
    size: u64,
}

impl GpuBuffer {
    /// Create a buffer with the given `usage` flags, pre-filled with `initial_data`.
    fn init_with_bytes(
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

    /// Create an uninitialised storage buffer.
    ///
    /// The buffer is usable as storage (read/write from compute shaders) and as
    /// a copy-source/destination so the CPU can read back results.
    pub fn new_storage(device: &wgpu::Device, size: u64, label: &str) -> Self {
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

    /// Create a storage buffer pre-filled with `data`.
    pub fn new_storage_with_data<T: bytemuck::Pod>(
        device: &wgpu::Device,
        data: &[T],
        label: &str,
    ) -> Self {
        Self::init_with_bytes(
            device,
            bytemuck::cast_slice(data),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            label,
        )
    }

    /// Create a uniform buffer pre-filled with `data`.
    pub fn new_uniform_with_data<T: bytemuck::Pod>(
        device: &wgpu::Device,
        data: &T,
        label: &str,
    ) -> Self {
        Self::init_with_bytes(
            device,
            bytemuck::bytes_of(data),
            wgpu::BufferUsages::UNIFORM,
            label,
        )
    }

    /// Overwrite the buffer contents from the CPU side.
    pub fn write<T: bytemuck::Pod>(&self, queue: &wgpu::Queue, data: &[T]) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
    }

    /// The underlying `wgpu::Buffer`.
    pub fn raw(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }
}

// ---------------------------------------------------------------------------
// Dispatch helper
// ---------------------------------------------------------------------------

/// Record a compute dispatch into `encoder`.
///
/// `bind_groups` are bound in order starting at group 0.
/// `workgroups` is `[x, y, z]` — the number of workgroups to dispatch.
pub fn dispatch_compute(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &ComputePipeline,
    bind_groups: &[&wgpu::BindGroup],
    workgroups: [u32; 3],
) {
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
pub struct ComputeManager {
    pipelines: HashMap<String, ComputePipeline>,
    buffers: HashMap<String, GpuBuffer>,
}

impl ComputeManager {
    /// Create an empty manager.
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            buffers: HashMap::new(),
        }
    }

    /// Register a compute pipeline under `name`.
    pub fn add_pipeline(&mut self, name: impl Into<String>, pipeline: ComputePipeline) {
        self.pipelines.insert(name.into(), pipeline);
    }

    /// Register a GPU buffer under `name`.
    pub fn add_buffer(&mut self, name: impl Into<String>, buffer: GpuBuffer) {
        self.buffers.insert(name.into(), buffer);
    }

    /// Look up a pipeline by name.
    pub fn pipeline(&self, name: &str) -> Option<&ComputePipeline> {
        self.pipelines.get(name)
    }

    /// Look up a buffer by name.
    pub fn buffer(&self, name: &str) -> Option<&GpuBuffer> {
        self.buffers.get(name)
    }

    /// Remove and return a pipeline.
    pub fn remove_pipeline(&mut self, name: &str) -> Option<ComputePipeline> {
        self.pipelines.remove(name)
    }

    /// Remove and return a buffer.
    pub fn remove_buffer(&mut self, name: &str) -> Option<GpuBuffer> {
        self.buffers.remove(name)
    }
}

impl Default for ComputeManager {
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
pub const FRUSTUM_CULL_SHADER: &str = r#"
struct FrustumPlanes {
    planes: array<vec4<f32>, 6>,
}

struct Aabb {
    center: vec4<f32>,       // xyz = center, w unused
    half_extents: vec4<f32>, // xyz = half-extents, w unused
}

struct CullParams {
    entity_count: u32,
}

@group(0) @binding(0) var<uniform> frustum: FrustumPlanes;
@group(0) @binding(1) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(2) var<storage, read_write> visibility: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: CullParams;

fn is_visible(entity: u32) -> bool {
    let c = aabbs[entity].center.xyz;
    let h = aabbs[entity].half_extents.xyz;

    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = frustum.planes[i];
        let n = plane.xyz;
        let d = plane.w;

        // Effective radius projected onto the plane normal
        let r = dot(h, abs(n));
        // Signed distance from center to plane
        let dist = dot(n, c) + d;

        if dist < -r {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let entity = gid.x;
    if entity >= params.entity_count {
        return;
    }

    let word_index = entity / 32u;
    let bit_index = entity % 32u;

    if is_visible(entity) {
        // Atomically set the bit.
        atomicOr(&visibility[word_index], 1u << bit_index);
    }
    // Bits default to 0 (invisible) — caller must clear the buffer before dispatch.
}
"#;

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

    let pipeline = ComputePipeline::new(
        device,
        &ComputePipelineDesc {
            label: "frustum_cull_pipeline",
            shader_source: FRUSTUM_CULL_SHADER,
            entry_point: "main",
        },
    );
    manager.add_pipeline(NAME, pipeline);

    // Frustum planes uniform (6 * vec4 = 96 bytes).
    let frustum_buf = GpuBuffer::new_uniform_with_data(
        device,
        &GpuFrustumPlanes {
            planes: [[0.0; 4]; 6],
        },
        "frustum_planes",
    );
    manager.add_buffer("frustum_planes", frustum_buf);

    // AABB storage buffer.
    let aabb_size = (max_entities as u64) * std::mem::size_of::<GpuAabb>() as u64;
    let aabb_buf = GpuBuffer::new_storage(device, aabb_size, "cull_aabbs");
    manager.add_buffer("cull_aabbs", aabb_buf);

    // Visibility bitset: one bit per entity, packed into u32 words.
    let vis_words = (max_entities + 31) / 32;
    let vis_size = (vis_words as u64) * 4;
    let vis_buf = GpuBuffer::new_storage(device, vis_size, "cull_visibility");
    manager.add_buffer("cull_visibility", vis_buf);

    // CullParams uniform.
    let params_buf = GpuBuffer::new_uniform_with_data(
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
/// The caller must have previously called `setup_frustum_culling`.
pub fn create_frustum_cull_bind_group(
    device: &wgpu::Device,
    manager: &ComputeManager,
) -> wgpu::BindGroup {
    let pipeline = manager
        .pipeline("frustum_cull")
        .expect("frustum_cull pipeline not set up");
    let frustum_buf = manager.buffer("frustum_planes").expect("buffer missing");
    let aabb_buf = manager.buffer("cull_aabbs").expect("buffer missing");
    let vis_buf = manager.buffer("cull_visibility").expect("buffer missing");
    let params_buf = manager.buffer("cull_params").expect("buffer missing");

    device.create_bind_group(&wgpu::BindGroupDescriptor {
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
    })
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
) {
    let pipeline = manager
        .pipeline("frustum_cull")
        .expect("frustum_cull pipeline not set up");

    let workgroup_count = entity_count.div_ceil(64);
    dispatch_compute(encoder, pipeline, &[bind_group], [workgroup_count, 1, 1]);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_manager_insert_and_lookup() {
        let mut mgr = ComputeManager::new();
        assert!(mgr.pipeline("foo").is_none());
        assert!(mgr.buffer("bar").is_none());

        // We can't create real pipelines without a device, but we can test
        // the HashMap plumbing with remove on empty.
        assert!(mgr.remove_pipeline("x").is_none());
        assert!(mgr.remove_buffer("x").is_none());
    }

    #[test]
    fn compute_manager_default() {
        let mgr = ComputeManager::default();
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
}
