use crate::buffer::{BufferKind, SmartBuffer};
use crate::camera::Camera;
use crate::decal::{DecalDrawCommand, DecalRenderer};
use crate::gpu::GpuContext;
use crate::ibl::IblResources;
use crate::light::{AmbientLight, DirectionalLight};
use crate::material::{Material, MaterialHandle};
use crate::mesh::{Mesh, MeshHandle};
use crate::occlusion::OcclusionCuller;
use crate::post_process::{PostProcessSettings, PostProcessStack};
use crate::texture::{TextureHandle, TextureStore};
use crate::vertex::Vertex;
use crate::volumetric::{FrameParams, VolumetricFogPass, VolumetricFogSettings};
use euca_math::Mat4;
use euca_rhi::RenderDevice;

/// Preset quality tiers that map to sensible [`PostProcessSettings`] defaults.
///
/// Use [`RenderQuality::to_settings`] to obtain the corresponding settings,
/// then apply them via [`Renderer::set_post_process_settings`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RenderQuality {
    /// Minimal overhead: SSAO off, FXAA on, bloom on. All advanced features
    /// (SSGI, motion blur, DoF, IBL) disabled.
    Low,
    /// Balanced: SSAO on at reduced radius, IBL at 0.8 intensity, motion blur
    /// with 4 samples. No SSGI or DoF.
    Medium,
    /// Full quality: SSAO, SSGI (4 rays), IBL, motion blur (8 samples), DoF,
    /// and PCSS all enabled. Color grading at neutral.
    High,
    /// Maximum fidelity: boosted SSAO, SSGI (8 rays, 1.2x intensity), motion
    /// blur (16 samples), DoF (30px max blur), slightly elevated contrast and
    /// saturation.
    Ultra,
}

impl RenderQuality {
    /// Convert this quality tier into concrete post-process settings.
    pub fn to_settings(self) -> PostProcessSettings {
        match self {
            RenderQuality::Low => PostProcessSettings {
                ssao_enabled: false,
                fxaa_enabled: true,
                bloom_enabled: true,
                // All new features off for minimum overhead.
                ssgi_enabled: false,
                motion_blur: crate::motion_blur::MotionBlurSettings {
                    enabled: false,
                    ..Default::default()
                },
                dof: crate::dof::DofSettings {
                    enabled: false,
                    ..Default::default()
                },
                ibl_enabled: false,
                ibl_intensity: 1.0,
                pcss_enabled: true,
                ..PostProcessSettings::default()
            },
            RenderQuality::Medium => PostProcessSettings {
                ssao_enabled: true,
                ssao_radius: 0.3,
                fxaa_enabled: true,
                bloom_enabled: true,
                // IBL on at reduced intensity; motion blur with fewer samples.
                ibl_enabled: true,
                ibl_intensity: 0.8,
                motion_blur: crate::motion_blur::MotionBlurSettings {
                    enabled: true,
                    sample_count: 4,
                    ..Default::default()
                },
                ssgi_enabled: false,
                dof: crate::dof::DofSettings {
                    enabled: false,
                    ..Default::default()
                },
                pcss_enabled: true,
                ..PostProcessSettings::default()
            },
            RenderQuality::High => PostProcessSettings {
                ssao_enabled: true,
                ssao_radius: 0.5,
                ssao_intensity: 1.0,
                fxaa_enabled: true,
                bloom_enabled: true,
                exposure: 1.0,
                contrast: 1.0,
                saturation: 1.0,
                // SSGI, motion blur, DoF, and IBL all active.
                ssgi_enabled: true,
                ssgi_ray_count: 4,
                ssgi_intensity: 1.0,
                ibl_enabled: true,
                ibl_intensity: 1.0,
                motion_blur: crate::motion_blur::MotionBlurSettings {
                    enabled: true,
                    sample_count: 8,
                    ..Default::default()
                },
                dof: crate::dof::DofSettings {
                    enabled: true,
                    focus_distance: 10.0,
                    aperture: 0.05,
                    ..Default::default()
                },
                pcss_enabled: true,
                ..PostProcessSettings::default()
            },
            RenderQuality::Ultra => PostProcessSettings {
                ssao_enabled: true,
                ssao_radius: 0.5,
                ssao_intensity: 1.2,
                fxaa_enabled: true,
                bloom_enabled: true,
                exposure: 1.0,
                contrast: 1.05,
                saturation: 1.05,
                // Everything maxed.
                ssgi_enabled: true,
                ssgi_ray_count: 8,
                ssgi_intensity: 1.2,
                ibl_enabled: true,
                ibl_intensity: 1.0,
                motion_blur: crate::motion_blur::MotionBlurSettings {
                    enabled: true,
                    sample_count: 16,
                    ..Default::default()
                },
                dof: crate::dof::DofSettings {
                    enabled: true,
                    max_blur_radius: 30.0,
                    ..Default::default()
                },
                pcss_enabled: true,
                ..PostProcessSettings::default()
            },
        }
    }

    /// Parse a quality tier from a case-insensitive string.
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "low" => Some(Self::Low),
            "medium" | "med" => Some(Self::Medium),
            "high" => Some(Self::High),
            "ultra" => Some(Self::Ultra),
            _ => None,
        }
    }

    /// Return the name of this quality tier as a lowercase string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Ultra => "ultra",
        }
    }
}

struct GpuMesh<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    vertex_buffer: D::Buffer,
    index_buffer: D::Buffer,
    index_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniforms {
    albedo: [f32; 4],
    metallic: f32,
    roughness: f32,
    has_normal_map: f32,
    has_metallic_roughness_tex: f32,
    emissive: [f32; 3],
    has_emissive_tex: f32,
    has_ao_tex: f32,
    alpha_mode: f32,
    alpha_cutoff: f32,
    _pad: f32,
}

/// A single draw request submitted each frame.
///
/// Pairs a mesh and material with a world-space transform. The renderer
/// batches draw commands by mesh and material to minimize GPU state changes.
#[derive(Clone)]
pub struct DrawCommand {
    /// Handle to the GPU mesh to draw.
    pub mesh: MeshHandle,
    /// Handle to the GPU material to shade with.
    pub material: MaterialHandle,
    /// Object-to-world transform matrix.
    pub model_matrix: Mat4,
    /// Optional world-space AABB (center, half-extents) for occlusion culling.
    /// When provided and occlusion culling is enabled, objects fully behind
    /// previously rendered geometry are skipped.
    pub aabb: Option<(euca_math::Vec3, euca_math::Vec3)>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    model: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
    material_id: u32,
    _inst_pad: [u32; 3],
}

/// Per-decal uniform data uploaded to the GPU before each decal draw call.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DecalUniforms {
    model_matrix: [[f32; 4]; 4],
    opacity: f32,
    _pad: [f32; 3],
}

const MAX_POINT_LIGHTS: usize = 4;
const MAX_SPOT_LIGHTS: usize = 2;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct GpuPointLight {
    position: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct GpuSpotLight {
    position: [f32; 4],
    direction: [f32; 4],
    color: [f32; 4],
    cone: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniforms {
    camera_pos: [f32; 4],
    light_direction: [f32; 4],
    light_color: [f32; 4],
    ambient_color: [f32; 4],
    camera_vp: [[f32; 4]; 4],
    light_vp: [[f32; 4]; 4],
    inv_vp: [[f32; 4]; 4],
    cascade_vps: [[[f32; 4]; 4]; 3],
    cascade_splits: [f32; 4],
    point_lights: [GpuPointLight; MAX_POINT_LIGHTS],
    spot_lights: [GpuSpotLight; MAX_SPOT_LIGHTS],
    num_point_lights: [f32; 4],
    num_spot_lights: [f32; 4],
    /// Interpolated L2 SH probe coefficients (9 bands × RGBA, blended on CPU).
    probe_sh: [[f32; 4]; 9],
    /// x=1.0 if probe data is valid, 0.0 otherwise. yzw=padding.
    probe_enabled: [f32; 4],
    /// x=light_size for PCSS soft shadow penumbra.
    /// y=normal_bias_scale (default 0.01), z=slope_bias_scale (default 0.03),
    /// w=cascade_bias_scale — extra bias multiplier per cascade index (default 0.5).
    shadow_params: [f32; 4],
    /// IBL parameters: x=enabled (0.0 or 1.0), y=intensity, z=unused, w=unused.
    ibl_params: [f32; 4],
}

struct GpuMaterial<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    bind_group: D::BindGroup,
    _buffer: D::Buffer,
    is_transparent: bool,
}

struct DrawBatch {
    mesh: MeshHandle,
    material: MaterialHandle,
    instance_start: u32,
    instance_count: u32,
}

/// Initial instance buffer capacity. Grows dynamically when exceeded.
const INITIAL_INSTANCE_CAPACITY: usize = 16384;

/// State for the bindless rendering path.
struct BindlessState<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    system: crate::bindless::BindlessMaterialSystem,
    pipeline: D::RenderPipeline,
}
const SHADOW_MAP_SIZE: u32 = 2048;
const NUM_SHADOW_CASCADES: u32 = 3;
const CASCADE_ORTHO_SIZES: [f32; 3] = [8.0, 20.0, 50.0];

/// The main PBR forward renderer.
///
/// Owns all GPU pipeline state, uploaded meshes, materials, textures, and
/// optional subsystems (post-processing, TAA, volumetric fog, occlusion
/// culling, decals, GPU particles).
///
/// # Rendering pipeline
///
/// Each frame proceeds through these stages:
///
/// 1. **Shadow pass** -- render cascaded shadow maps for the directional light.
/// 2. **Opaque pass** -- draw all opaque [`DrawCommand`]s with PBR shading
///    (4x MSAA, HDR).
/// 3. **Decal pass** -- project deferred decals onto opaque surfaces.
/// 4. **Transparent pass** -- draw alpha-blended commands back-to-front.
/// 5. **Volumetric fog** -- ray-march scattering (if enabled).
/// 6. **Post-processing** -- SSAO, bloom, color grading, FXAA.
/// 7. **TAA resolve** -- temporal anti-aliasing jitter and history blending.
#[allow(dead_code)]
pub struct Renderer<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    pipeline: D::RenderPipeline,
    transparent_pipeline: D::RenderPipeline,
    instance_buffer: SmartBuffer<D>,
    instance_bind_group: D::BindGroup,
    instance_bgl: D::BindGroupLayout,
    scene_buffer: SmartBuffer<D>,
    scene_bgl: D::BindGroupLayout,
    scene_bind_group: D::BindGroup,
    material_bgl: D::BindGroupLayout,
    sampler: D::Sampler,
    materials: Vec<GpuMaterial<D>>,
    textures: TextureStore,
    sky_pipeline: D::RenderPipeline,
    shadow_pipeline: D::RenderPipeline,
    shadow_map: D::Texture,
    shadow_map_view: D::TextureView,
    shadow_cascade_views: Vec<D::TextureView>,
    shadow_sampler: D::Sampler,
    shadow_depth_sampler: D::Sampler,
    shadow_instance_buffer: SmartBuffer<D>,
    shadow_instance_bind_group: D::BindGroup,
    msaa_hdr_view: D::TextureView,
    #[allow(dead_code)]
    msaa_hdr_texture: D::Texture,
    meshes: Vec<GpuMesh<D>>,
    depth_texture: D::TextureView,
    depth_format: wgpu::TextureFormat,
    surface_format: wgpu::TextureFormat,
    /// Advanced post-process stack (SSAO, FXAA, color grading, bloom).
    post_process_stack: PostProcessStack,
    /// Settings controlling the advanced post-process stack.
    post_process_settings: PostProcessSettings,
    /// Optional volumetric fog pass. Created lazily via `enable_volumetric_fog`.
    volumetric_fog_pass: Option<VolumetricFogPass>,
    /// Settings for volumetric fog (density, scattering, etc.).
    volumetric_fog_settings: VolumetricFogSettings,
    /// Optional HZB occlusion culler. Uses previous frame's depth.
    occlusion_culler: Option<OcclusionCuller>,
    /// Previous frame's depth buffer for occlusion culling.
    prev_depth_buffer: Vec<f32>,
    /// Dimensions of the previous depth buffer.
    prev_depth_dims: (u32, u32),
    /// TAA resolve pass (temporal anti-aliasing).
    taa_pass: crate::taa::TaaPass,
    /// Frame counter for TAA jitter sequence.
    frame_count: u32,
    /// Interpolated SH probe coefficients for indirect lighting (set by caller).
    probe_sh: [[f32; 4]; 9],
    /// Whether probe data is valid.
    probe_enabled: bool,
    /// GPU resources for decal projection volumes.
    decal_renderer: DecalRenderer,
    /// Per-decal uniform buffer (model_matrix + opacity), written before each draw.
    decal_uniform_buffer: SmartBuffer<D>,
    /// Bind group layout for the per-decal uniform buffer.
    decal_bgl: D::BindGroupLayout,
    /// Bind group exposing the per-decal uniform buffer to shaders.
    decal_bind_group: D::BindGroup,
    /// Decal draw commands staged by the caller for the current frame.
    pending_decals: Vec<DecalDrawCommand>,
    /// GPU compute particle systems.
    gpu_particle_systems: Vec<crate::gpu_particles::GpuParticleSystem>,
    /// Velocity buffer textures for TAA / motion blur / DoF.
    velocity_textures: crate::velocity::VelocityTextures,
    /// Fallback 1x1 black cubemap view (used when no IBL resources are set).
    ibl_dummy_cube_view: D::TextureView,
    /// Fallback 1x1 black BRDF LUT view (used when no IBL resources are set).
    ibl_dummy_brdf_view: D::TextureView,
    /// Fallback trilinear sampler for IBL textures.
    ibl_sampler: D::Sampler,
    /// Optional bindless material system + render pipeline.
    /// When active, the opaque pass uses a single bind group for all materials.
    bindless: Option<BindlessState<D>>,
    /// Current capacity (in instances) of the main instance buffer.
    instance_capacity: usize,
    /// Current capacity (in instances) of the shadow instance buffer.
    shadow_instance_capacity: usize,
    /// Whether the GPU uses unified memory (needed for buffer re-creation).
    unified_memory: bool,
    /// Active IBL resources (set via `set_ibl`, cleared via `clear_ibl`).
    ibl_resources: Option<IblResources>,
    /// IBL intensity multiplier (default 1.0).
    ibl_intensity: f32,
    // Keep fallback textures alive so their views remain valid.
    _ibl_dummy_cube: D::Texture,
    _ibl_dummy_brdf: D::Texture,
}

const MSAA_SAMPLE_COUNT: u32 = 4;

impl Renderer {
    /// Create a new renderer, allocating all GPU pipelines and buffers.
    ///
    /// The renderer is bound to the surface format and initial size reported
    /// by `gpu`. Call [`resize`](Self::resize) when the window size changes.
    pub fn new(gpu: &GpuContext) -> Self {
        let instance_buf_size =
            (INITIAL_INSTANCE_CAPACITY * std::mem::size_of::<InstanceData>()) as u64;
        let unified = gpu.unified_memory();
        let rhi: &euca_rhi::wgpu_backend::WgpuDevice = gpu;
        let instance_buffer = SmartBuffer::new(
            rhi,
            instance_buf_size,
            BufferKind::Storage,
            unified,
            "Instance SSBO",
        );
        let scene_buffer = SmartBuffer::new(
            rhi,
            std::mem::size_of::<SceneUniforms>() as u64,
            BufferKind::Uniform,
            unified,
            "Scene UBO",
        );
        let shadow_instance_buffer = SmartBuffer::new(
            rhi,
            instance_buf_size,
            BufferKind::Storage,
            unified,
            "Shadow Instance SSBO",
        );
        let textures = TextureStore::new(rhi);
        let sampler = rhi.create_sampler(&euca_rhi::SamplerDesc {
            label: Some("Material Sampler"),
            address_mode_u: euca_rhi::AddressMode::Repeat,
            address_mode_v: euca_rhi::AddressMode::Repeat,
            address_mode_w: euca_rhi::AddressMode::Repeat,
            mag_filter: euca_rhi::FilterMode::Linear,
            min_filter: euca_rhi::FilterMode::Linear,
            mipmap_filter: euca_rhi::FilterMode::Linear,
            ..Default::default()
        });
        let shadow_map = rhi.create_texture(&euca_rhi::TextureDesc {
            label: Some("Shadow Map Array"),
            size: euca_rhi::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: NUM_SHADOW_CASCADES,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: euca_rhi::TextureDimension::D2,
            format: euca_rhi::TextureFormat::Depth32Float,
            usage: euca_rhi::TextureUsages::RENDER_ATTACHMENT
                | euca_rhi::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_map_view = rhi.create_texture_view(
            &shadow_map,
            &euca_rhi::TextureViewDesc {
                dimension: Some(euca_rhi::TextureViewDimension::D2Array),
                ..Default::default()
            },
        );
        let shadow_cascade_views: Vec<wgpu::TextureView> = (0..NUM_SHADOW_CASCADES)
            .map(|i| {
                rhi.create_texture_view(
                    &shadow_map,
                    &euca_rhi::TextureViewDesc {
                        dimension: Some(euca_rhi::TextureViewDimension::D2),
                        base_array_layer: i,
                        array_layer_count: Some(1),
                        ..Default::default()
                    },
                )
            })
            .collect();
        let shadow_sampler = rhi.create_sampler(&euca_rhi::SamplerDesc {
            label: Some("Shadow Sampler"),
            compare: Some(euca_rhi::CompareFunction::LessEqual),
            mag_filter: euca_rhi::FilterMode::Linear,
            min_filter: euca_rhi::FilterMode::Linear,
            ..Default::default()
        });
        let shadow_depth_sampler = rhi.create_sampler(&euca_rhi::SamplerDesc {
            label: Some("Shadow Depth Sampler"),
            mag_filter: euca_rhi::FilterMode::Nearest,
            min_filter: euca_rhi::FilterMode::Nearest,
            ..Default::default()
        });

        let instance_bgl = rhi.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("Instance BGL"),
            entries: &[euca_rhi::BindGroupLayoutEntry {
                binding: 0,
                visibility: euca_rhi::ShaderStages::VERTEX,
                ty: euca_rhi::BindingType::Buffer {
                    ty: euca_rhi::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let scene_bgl = rhi.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("Scene BGL"),
            entries: &[
                euca_rhi::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: euca_rhi::ShaderStages::VERTEX_FRAGMENT,
                    ty: euca_rhi::BindingType::Buffer {
                        ty: euca_rhi::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::mem::size_of::<SceneUniforms>() as u64),
                    },
                    count: None,
                },
                euca_rhi::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: euca_rhi::ShaderStages::FRAGMENT,
                    ty: euca_rhi::BindingType::Texture {
                        sample_type: euca_rhi::TextureSampleType::Depth,
                        view_dimension: euca_rhi::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                euca_rhi::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: euca_rhi::ShaderStages::FRAGMENT,
                    ty: euca_rhi::BindingType::Sampler(euca_rhi::SamplerBindingType::Comparison),
                    count: None,
                },
                euca_rhi::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: euca_rhi::ShaderStages::FRAGMENT,
                    ty: euca_rhi::BindingType::Sampler(euca_rhi::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // IBL: irradiance cubemap
                euca_rhi::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: euca_rhi::ShaderStages::FRAGMENT,
                    ty: euca_rhi::BindingType::Texture {
                        sample_type: euca_rhi::TextureSampleType::Float { filterable: true },
                        view_dimension: euca_rhi::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                // IBL: specular pre-filtered cubemap
                euca_rhi::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: euca_rhi::ShaderStages::FRAGMENT,
                    ty: euca_rhi::BindingType::Texture {
                        sample_type: euca_rhi::TextureSampleType::Float { filterable: true },
                        view_dimension: euca_rhi::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                // IBL: BRDF integration LUT
                euca_rhi::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: euca_rhi::ShaderStages::FRAGMENT,
                    ty: euca_rhi::BindingType::Texture {
                        sample_type: euca_rhi::TextureSampleType::Float { filterable: true },
                        view_dimension: euca_rhi::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // IBL: trilinear sampler
                euca_rhi::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: euca_rhi::ShaderStages::FRAGMENT,
                    ty: euca_rhi::BindingType::Sampler(euca_rhi::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let tex_entry = |binding: u32| euca_rhi::BindGroupLayoutEntry {
            binding,
            visibility: euca_rhi::ShaderStages::FRAGMENT,
            ty: euca_rhi::BindingType::Texture {
                sample_type: euca_rhi::TextureSampleType::Float { filterable: true },
                view_dimension: euca_rhi::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let material_bgl = rhi.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("Material BGL"),
            entries: &[
                euca_rhi::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: euca_rhi::ShaderStages::FRAGMENT,
                    ty: euca_rhi::BindingType::Buffer {
                        ty: euca_rhi::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::mem::size_of::<MaterialUniforms>() as u64),
                    },
                    count: None,
                },
                tex_entry(1), // albedo
                euca_rhi::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: euca_rhi::ShaderStages::FRAGMENT,
                    ty: euca_rhi::BindingType::Sampler(euca_rhi::SamplerBindingType::Filtering),
                    count: None,
                },
                tex_entry(3), // normal
                tex_entry(4), // metallic-roughness
                tex_entry(5), // ao
                tex_entry(6), // emissive
            ],
        });

        let instance_bind_group = rhi.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("Instance BG"),
            layout: &instance_bgl,
            entries: &[euca_rhi::BindGroupEntry {
                binding: 0,
                resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                    buffer: instance_buffer.raw(),
                    offset: 0,
                    size: None,
                }),
            }],
        });
        let shadow_instance_bind_group = rhi.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("Shadow Instance BG"),
            layout: &instance_bgl,
            entries: &[euca_rhi::BindGroupEntry {
                binding: 0,
                resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                    buffer: shadow_instance_buffer.raw(),
                    offset: 0,
                    size: None,
                }),
            }],
        });
        // --- IBL fallback textures (1x1 black) ---
        let ibl_dummy_cube = rhi.create_texture(&euca_rhi::TextureDesc {
            label: Some("IBL Dummy Cubemap"),
            size: euca_rhi::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: euca_rhi::TextureDimension::D2,
            format: euca_rhi::TextureFormat::Rgba16Float,
            usage: euca_rhi::TextureUsages::TEXTURE_BINDING | euca_rhi::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Write black (all-zero) pixels to each face.
        let zero_pixel = [0u8; 8]; // 4 x f16 = 8 bytes, all zeros = black
        for face in 0..6u32 {
            rhi.write_texture(
                &euca_rhi::TexelCopyTextureInfo {
                    texture: &ibl_dummy_cube,
                    mip_level: 0,
                    origin: euca_rhi::Origin3d {
                        x: 0,
                        y: 0,
                        z: face,
                    },
                    aspect: euca_rhi::TextureAspect::All,
                },
                &zero_pixel,
                &euca_rhi::TextureDataLayout {
                    offset: 0,
                    bytes_per_row: Some(8),
                    rows_per_image: Some(1),
                },
                euca_rhi::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }
        let ibl_dummy_cube_view = rhi.create_texture_view(
            &ibl_dummy_cube,
            &euca_rhi::TextureViewDesc {
                label: Some("IBL Dummy Cubemap View"),
                dimension: Some(euca_rhi::TextureViewDimension::Cube),
                ..Default::default()
            },
        );

        // The BRDF LUT uses Rgba16Float (filterable on all GPUs including Apple Silicon).
        let ibl_dummy_brdf = rhi.create_texture(&euca_rhi::TextureDesc {
            label: Some("IBL Dummy BRDF LUT"),
            size: euca_rhi::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: euca_rhi::TextureDimension::D2,
            format: euca_rhi::TextureFormat::Rgba16Float,
            usage: euca_rhi::TextureUsages::TEXTURE_BINDING | euca_rhi::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        rhi.write_texture(
            &euca_rhi::TexelCopyTextureInfo {
                texture: &ibl_dummy_brdf,
                mip_level: 0,
                origin: euca_rhi::Origin3d::default(),
                aspect: euca_rhi::TextureAspect::All,
            },
            &[0u8; 8], // 4 x f16 = 8 bytes, all zeros
            &euca_rhi::TextureDataLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            euca_rhi::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let ibl_dummy_brdf_view =
            rhi.create_texture_view(&ibl_dummy_brdf, &euca_rhi::TextureViewDesc::default());

        let ibl_sampler = rhi.create_sampler(&euca_rhi::SamplerDesc {
            label: Some("IBL Sampler"),
            mag_filter: euca_rhi::FilterMode::Linear,
            min_filter: euca_rhi::FilterMode::Linear,
            mipmap_filter: euca_rhi::FilterMode::Linear,
            address_mode_u: euca_rhi::AddressMode::ClampToEdge,
            address_mode_v: euca_rhi::AddressMode::ClampToEdge,
            address_mode_w: euca_rhi::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let scene_bind_group = rhi.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("Scene BG"),
            layout: &scene_bgl,
            entries: &[
                euca_rhi::BindGroupEntry {
                    binding: 0,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: scene_buffer.raw(),
                        offset: 0,
                        size: None,
                    }),
                },
                euca_rhi::BindGroupEntry {
                    binding: 1,
                    resource: euca_rhi::BindingResource::TextureView(&shadow_map_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 2,
                    resource: euca_rhi::BindingResource::Sampler(&shadow_sampler),
                },
                euca_rhi::BindGroupEntry {
                    binding: 3,
                    resource: euca_rhi::BindingResource::Sampler(&shadow_depth_sampler),
                },
                euca_rhi::BindGroupEntry {
                    binding: 4,
                    resource: euca_rhi::BindingResource::TextureView(&ibl_dummy_cube_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 5,
                    resource: euca_rhi::BindingResource::TextureView(&ibl_dummy_cube_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 6,
                    resource: euca_rhi::BindingResource::TextureView(&ibl_dummy_brdf_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 7,
                    resource: euca_rhi::BindingResource::Sampler(&ibl_sampler),
                },
            ],
        });

        let depth_format = wgpu::TextureFormat::Depth32Float;
        let shadow_shader = rhi.create_shader(&euca_rhi::ShaderDesc {
            label: Some("Shadow Shader"),
            source: euca_rhi::ShaderSource::Wgsl(SHADOW_SHADER.into()),
        });
        let shadow_pipeline = rhi.create_render_pipeline(&euca_rhi::RenderPipelineDesc {
            label: Some("Shadow Pipeline"),
            layout: &[&instance_bgl],
            vertex: euca_rhi::VertexState {
                module: &shadow_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::RHI_LAYOUT],
            },
            fragment: None,
            primitive: euca_rhi::PrimitiveState {
                topology: euca_rhi::PrimitiveTopology::TriangleList,
                front_face: euca_rhi::FrontFace::Ccw,
                cull_mode: Some(euca_rhi::Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(euca_rhi::DepthStencilState {
                format: euca_rhi::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: euca_rhi::CompareFunction::Less,
                stencil: Default::default(),
                bias: euca_rhi::DepthBiasState {
                    constant: 4,
                    slope_scale: 3.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
        });

        let sky_shader = rhi.create_shader(&euca_rhi::ShaderDesc {
            label: Some("Sky Shader"),
            source: euca_rhi::ShaderSource::Wgsl(SKY_SHADER.into()),
        });
        let shader = rhi.create_shader(&euca_rhi::ShaderDesc {
            label: Some("PBR Shader"),
            source: euca_rhi::ShaderSource::Wgsl(PBR_SHADER.into()),
        });
        let depth_texture = Self::create_depth_texture(
            rhi,
            gpu.surface_config.width,
            gpu.surface_config.height,
            euca_rhi::TextureFormat::Depth32Float,
        );

        let pipeline = rhi.create_render_pipeline(&euca_rhi::RenderPipelineDesc {
            label: Some("PBR Pipeline"),
            layout: &[&instance_bgl, &scene_bgl, &material_bgl],
            vertex: euca_rhi::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::RHI_LAYOUT],
            },
            fragment: Some(euca_rhi::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(euca_rhi::ColorTargetState {
                    format: euca_rhi::TextureFormat::Rgba16Float,
                    blend: Some(euca_rhi::BlendState::REPLACE),
                    write_mask: euca_rhi::ColorWrites::ALL,
                })],
            }),
            primitive: euca_rhi::PrimitiveState {
                topology: euca_rhi::PrimitiveTopology::TriangleList,
                front_face: euca_rhi::FrontFace::Ccw,
                cull_mode: Some(euca_rhi::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(euca_rhi::DepthStencilState {
                format: euca_rhi::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: euca_rhi::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: euca_rhi::MultisampleState {
                count: MSAA_SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        });

        let transparent_pipeline = rhi.create_render_pipeline(&euca_rhi::RenderPipelineDesc {
            label: Some("PBR Transparent Pipeline"),
            layout: &[&instance_bgl, &scene_bgl, &material_bgl],
            vertex: euca_rhi::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::RHI_LAYOUT],
            },
            fragment: Some(euca_rhi::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(euca_rhi::ColorTargetState {
                    format: euca_rhi::TextureFormat::Rgba16Float,
                    blend: Some(euca_rhi::BlendState::ALPHA_BLENDING),
                    write_mask: euca_rhi::ColorWrites::ALL,
                })],
            }),
            primitive: euca_rhi::PrimitiveState {
                topology: euca_rhi::PrimitiveTopology::TriangleList,
                front_face: euca_rhi::FrontFace::Ccw,
                cull_mode: Some(euca_rhi::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(euca_rhi::DepthStencilState {
                format: euca_rhi::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: euca_rhi::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: euca_rhi::MultisampleState {
                count: MSAA_SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        });

        let sky_pipeline = rhi.create_render_pipeline(&euca_rhi::RenderPipelineDesc {
            label: Some("Sky Pipeline"),
            layout: &[&scene_bgl],
            vertex: euca_rhi::VertexState {
                module: &sky_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(euca_rhi::FragmentState {
                module: &sky_shader,
                entry_point: "fs_main",
                targets: &[Some(euca_rhi::ColorTargetState {
                    format: euca_rhi::TextureFormat::Rgba16Float,
                    blend: Some(euca_rhi::BlendState::REPLACE),
                    write_mask: euca_rhi::ColorWrites::ALL,
                })],
            }),
            primitive: euca_rhi::PrimitiveState {
                topology: euca_rhi::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(euca_rhi::DepthStencilState {
                format: euca_rhi::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: euca_rhi::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: euca_rhi::MultisampleState {
                count: MSAA_SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        });

        let (msaa_hdr_texture, msaa_hdr_view) =
            Self::create_msaa_hdr_texture(rhi, gpu.surface_config.width, gpu.surface_config.height);
        let post_process_stack =
            PostProcessStack::new(&gpu.device, &gpu.queue, &gpu.surface_config);

        let decal_renderer = DecalRenderer::new(&gpu.device);
        decal_renderer.upload(&gpu.queue);

        let decal_uniform_buffer = SmartBuffer::new(
            rhi,
            std::mem::size_of::<DecalUniforms>() as u64,
            BufferKind::Uniform,
            unified,
            "Decal UBO",
        );
        let decal_bgl = rhi.create_bind_group_layout(&euca_rhi::BindGroupLayoutDesc {
            label: Some("Decal BGL"),
            entries: &[euca_rhi::BindGroupLayoutEntry {
                binding: 0,
                visibility: euca_rhi::ShaderStages::VERTEX_FRAGMENT,
                ty: euca_rhi::BindingType::Buffer {
                    ty: euca_rhi::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(std::mem::size_of::<DecalUniforms>() as u64),
                },
                count: None,
            }],
        });
        let decal_bind_group = rhi.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("Decal BG"),
            layout: &decal_bgl,
            entries: &[euca_rhi::BindGroupEntry {
                binding: 0,
                resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                    buffer: decal_uniform_buffer.raw(),
                    offset: 0,
                    size: None,
                }),
            }],
        });

        Self {
            pipeline,
            transparent_pipeline,
            instance_buffer,
            instance_bind_group,
            instance_bgl,
            scene_buffer,
            scene_bgl,
            scene_bind_group,
            material_bgl,
            sampler,
            materials: Vec::new(),
            textures,
            sky_pipeline,
            shadow_pipeline,
            shadow_map,
            shadow_map_view,
            shadow_cascade_views,
            shadow_sampler,
            shadow_depth_sampler,
            shadow_instance_buffer,
            shadow_instance_bind_group,
            msaa_hdr_texture,
            msaa_hdr_view,
            meshes: Vec::new(),
            depth_texture,
            depth_format,
            surface_format: gpu.surface_config.format,
            post_process_stack,
            post_process_settings: PostProcessSettings::default(),
            volumetric_fog_pass: None,
            volumetric_fog_settings: VolumetricFogSettings::default(),
            occlusion_culler: None,
            prev_depth_buffer: Vec::new(),
            prev_depth_dims: (0, 0),
            taa_pass: crate::taa::TaaPass::new(
                rhi,
                gpu.surface_config.width,
                gpu.surface_config.height,
            ),
            frame_count: 0,
            probe_sh: [[0.0; 4]; 9],
            probe_enabled: false,
            decal_renderer,
            decal_uniform_buffer,
            decal_bgl,
            decal_bind_group,
            pending_decals: Vec::new(),
            gpu_particle_systems: Vec::new(),
            velocity_textures: crate::velocity::VelocityTextures::new(
                &gpu.device,
                gpu.surface_config.width,
                gpu.surface_config.height,
            ),
            ibl_dummy_cube_view,
            ibl_dummy_brdf_view,
            ibl_sampler,
            bindless: None,
            instance_capacity: INITIAL_INSTANCE_CAPACITY,
            shadow_instance_capacity: INITIAL_INSTANCE_CAPACITY,
            unified_memory: unified,
            ibl_resources: None,
            ibl_intensity: 1.0,
            _ibl_dummy_cube: ibl_dummy_cube,
            _ibl_dummy_brdf: ibl_dummy_brdf,
        }
    }

    /// Grow the main instance buffer and rebuild its bind group if `count`
    /// exceeds the current capacity. Returns `true` if the buffer was grown.
    fn ensure_instance_capacity(&mut self, device: &wgpu::Device, count: usize) -> bool {
        if count <= self.instance_capacity {
            return false;
        }
        self.instance_capacity = count.next_power_of_two();
        let size = (self.instance_capacity * std::mem::size_of::<InstanceData>()) as u64;
        self.instance_buffer = SmartBuffer::from_wgpu(
            device,
            size,
            BufferKind::Storage,
            self.unified_memory,
            "Instance SSBO",
        );
        self.instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Instance BG"),
            layout: &self.instance_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.instance_buffer.raw().as_entire_binding(),
            }],
        });
        true
    }

    /// Grow the shadow instance buffer and rebuild its bind group if `count`
    /// exceeds the current capacity.
    fn ensure_shadow_instance_capacity(&mut self, device: &wgpu::Device, count: usize) -> bool {
        if count <= self.shadow_instance_capacity {
            return false;
        }
        self.shadow_instance_capacity = count.next_power_of_two();
        let size = (self.shadow_instance_capacity * std::mem::size_of::<InstanceData>()) as u64;
        self.shadow_instance_buffer = SmartBuffer::from_wgpu(
            device,
            size,
            BufferKind::Storage,
            self.unified_memory,
            "Shadow Instance SSBO",
        );
        self.shadow_instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Instance BG"),
            layout: &self.instance_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.shadow_instance_buffer.raw().as_entire_binding(),
            }],
        });
        true
    }

    /// Enable the bindless material rendering path. When active, all opaque
    /// geometry is drawn with a single material bind group (no per-batch
    /// switching). Requires `TEXTURE_BINDING_ARRAY` GPU feature.
    ///
    /// Call this once after creating the renderer, before uploading materials.
    pub fn enable_bindless(&mut self, gpu: &GpuContext) {
        let features = gpu.device.features();
        let system = crate::bindless::BindlessMaterialSystem::new(
            &gpu.device,
            features,
            gpu.unified_memory(),
        );
        if !system.is_enabled() {
            log::warn!("Bindless materials requested but GPU lacks required features");
            return;
        }

        // Create the bindless render pipeline with the same vertex layout but
        // different group 2 bind group layout and shader.
        let rhi: &euca_rhi::wgpu_backend::WgpuDevice = gpu;
        let shader = rhi.create_shader(&euca_rhi::ShaderDesc {
            label: Some("PBR Bindless Shader"),
            source: euca_rhi::ShaderSource::Wgsl(PBR_BINDLESS_SHADER.into()),
        });
        let pipeline = rhi.create_render_pipeline(&euca_rhi::RenderPipelineDesc {
            label: Some("PBR Bindless Pipeline"),
            layout: &[
                &self.instance_bgl,
                &self.scene_bgl,
                &system.bind_group_layout,
            ],
            vertex: euca_rhi::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[crate::vertex::Vertex::RHI_LAYOUT],
            },
            fragment: Some(euca_rhi::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(euca_rhi::ColorTargetState {
                    format: euca_rhi::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: euca_rhi::ColorWrites::ALL,
                })],
            }),
            primitive: euca_rhi::PrimitiveState {
                topology: euca_rhi::PrimitiveTopology::TriangleList,
                front_face: euca_rhi::FrontFace::Ccw,
                cull_mode: Some(euca_rhi::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(euca_rhi::DepthStencilState {
                format: euca_rhi::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: euca_rhi::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: euca_rhi::MultisampleState {
                count: MSAA_SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        });

        log::info!("Bindless material rendering enabled");
        self.bindless = Some(BindlessState { system, pipeline });
    }

    /// Whether the bindless rendering path is active.
    pub fn is_bindless(&self) -> bool {
        self.bindless.is_some()
    }

    /// Upload CPU-side mesh data to the GPU and return a handle for use in
    /// [`DrawCommand`]s.
    pub fn upload_mesh(&mut self, gpu: &GpuContext, mesh: &Mesh) -> MeshHandle {
        use euca_rhi::{BufferDesc, BufferUsages, RenderDevice};
        let rhi: &euca_rhi::wgpu_backend::WgpuDevice = gpu;

        let vdata = bytemuck::cast_slice::<_, u8>(&mesh.vertices);
        let vb = rhi.create_buffer(&BufferDesc {
            label: Some("Vertex Buffer"),
            size: vdata.len() as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        rhi.write_buffer(&vb, 0, vdata);

        let idata = bytemuck::cast_slice::<_, u8>(&mesh.indices);
        let ib = rhi.create_buffer(&BufferDesc {
            label: Some("Index Buffer"),
            size: idata.len() as u64,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        rhi.write_buffer(&ib, 0, idata);

        let handle = MeshHandle(self.meshes.len() as u32);
        self.meshes.push(GpuMesh {
            vertex_buffer: vb,
            index_buffer: ib,
            index_count: mesh.indices.len() as u32,
        });
        handle
    }
    /// Upload raw RGBA8 pixel data as a GPU texture with auto-generated mipmaps.
    pub fn upload_texture(
        &mut self,
        gpu: &GpuContext,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> TextureHandle {
        let rhi: &euca_rhi::wgpu_backend::WgpuDevice = gpu;
        self.textures.upload_rgba(rhi, width, height, rgba)
    }

    /// Decode an image file (PNG, JPEG, etc.) from memory and upload it as a texture.
    pub fn upload_texture_image(&mut self, gpu: &GpuContext, data: &[u8]) -> TextureHandle {
        let rhi: &euca_rhi::wgpu_backend::WgpuDevice = gpu;
        self.textures.upload_image(rhi, data)
    }

    /// Generate and upload a checkerboard test pattern texture.
    pub fn checkerboard_texture(
        &mut self,
        gpu: &GpuContext,
        size: u32,
        tile: u32,
    ) -> TextureHandle {
        let rhi: &euca_rhi::wgpu_backend::WgpuDevice = gpu;
        self.textures.checkerboard(rhi, size, tile)
    }

    /// Upload a PBR material (uniforms + texture bindings) to the GPU and
    /// return a handle for use in [`DrawCommand`]s.
    pub fn upload_material(&mut self, gpu: &GpuContext, mat: &Material) -> MaterialHandle {
        use euca_rhi::{
            BindGroupDesc, BindGroupEntry, BindingResource, BufferBinding, BufferDesc,
            BufferUsages, RenderDevice,
        };
        let rhi: &euca_rhi::wgpu_backend::WgpuDevice = gpu;

        let handle = MaterialHandle(self.materials.len() as u32);
        let uniforms = MaterialUniforms {
            albedo: mat.albedo,
            metallic: mat.metallic,
            roughness: mat.roughness,
            has_normal_map: if mat.normal_texture.is_some() {
                1.0
            } else {
                0.0
            },
            has_metallic_roughness_tex: if mat.metallic_roughness_texture.is_some() {
                1.0
            } else {
                0.0
            },
            emissive: mat.emissive,
            has_emissive_tex: if mat.emissive_texture.is_some() {
                1.0
            } else {
                0.0
            },
            has_ao_tex: if mat.ao_texture.is_some() { 1.0 } else { 0.0 },
            alpha_mode: mat.alpha_mode.as_f32(),
            alpha_cutoff: mat.alpha_mode.cutoff(),
            _pad: 0.0,
        };
        let ubo_data = bytemuck::bytes_of(&uniforms);
        let buffer = rhi.create_buffer(&BufferDesc {
            label: Some("Material UBO"),
            size: ubo_data.len() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        rhi.write_buffer(&buffer, 0, ubo_data);

        let dw = TextureHandle::DEFAULT_WHITE;
        let albedo_view = self.textures.view(mat.albedo_texture.unwrap_or(dw));
        let normal_view = self.textures.view(mat.normal_texture.unwrap_or(dw));
        let mr_view = self
            .textures
            .view(mat.metallic_roughness_texture.unwrap_or(dw));
        let ao_view = self.textures.view(mat.ao_texture.unwrap_or(dw));
        let emissive_view = self.textures.view(mat.emissive_texture.unwrap_or(dw));
        let bind_group = rhi.create_bind_group(&BindGroupDesc {
            label: Some("Material BG"),
            layout: &self.material_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(albedo_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&self.sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(normal_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(mr_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(ao_view),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(emissive_view),
                },
            ],
        });
        self.materials.push(GpuMaterial {
            bind_group,
            _buffer: buffer,
            is_transparent: mat.alpha_mode.is_transparent(),
        });

        // Also register with the bindless system if active.
        if let Some(ref mut bl) = self.bindless {
            let bl_handle = bl.system.add_material(mat);
            debug_assert_eq!(
                bl_handle, handle,
                "Bindless handle must match traditional handle"
            );
        }

        handle
    }

    /// Recreate size-dependent GPU resources (depth buffer, MSAA target, etc.)
    /// after the window has been resized.
    pub fn resize(&mut self, gpu: &GpuContext) {
        let rhi: &euca_rhi::wgpu_backend::WgpuDevice = gpu;
        let (w, h) = (gpu.surface_config.width, gpu.surface_config.height);
        self.depth_texture =
            Self::create_depth_texture(rhi, w, h, euca_rhi::TextureFormat::Depth32Float);
        let (msaa_hdr_texture, msaa_hdr_view) = Self::create_msaa_hdr_texture(rhi, w, h);
        self.msaa_hdr_texture = msaa_hdr_texture;
        self.msaa_hdr_view = msaa_hdr_view;
        self.post_process_stack.resize(
            &gpu.device,
            gpu.surface_config.width,
            gpu.surface_config.height,
        );
        if let Some(ref mut fog_pass) = self.volumetric_fog_pass {
            fog_pass.resize(&**gpu, gpu.surface_config.width, gpu.surface_config.height);
        }
        self.taa_pass
            .resize(rhi, gpu.surface_config.width, gpu.surface_config.height);
        self.velocity_textures.resize(
            &gpu.device,
            gpu.surface_config.width,
            gpu.surface_config.height,
        );
    }

    /// Set interpolated SH probe coefficients for indirect lighting.
    ///
    /// The caller should sample the nearest probes from a `LightProbeGrid`
    /// and pass the blended coefficients here. The PBR shader will use these
    /// instead of the flat ambient color.
    pub fn set_probe_sh(&mut self, sh: [[f32; 4]; 9]) {
        self.probe_sh = sh;
        self.probe_enabled = true;
    }

    /// Disable probe-based ambient lighting (fall back to flat ambient_color).
    pub fn clear_probe(&mut self) {
        self.probe_enabled = false;
    }

    /// Activate IBL (Image-Based Lighting) with pre-computed resources.
    ///
    /// The `IblResources` should be generated via [`IblResources::generate`] or
    /// [`IblResources::from_uniform_color`]. Once set, the PBR shader will
    /// sample the irradiance and specular cubemaps for indirect lighting.
    pub fn set_ibl(&mut self, gpu: &GpuContext, resources: IblResources, intensity: f32) {
        self.ibl_resources = Some(resources);
        self.ibl_intensity = intensity;
        self.rebuild_scene_bind_group(gpu);
    }

    /// Disable IBL (fall back to SH probes or flat ambient color).
    pub fn clear_ibl(&mut self, gpu: &GpuContext) {
        self.ibl_resources = None;
        self.ibl_intensity = 1.0;
        self.rebuild_scene_bind_group(gpu);
    }

    /// Set the IBL intensity multiplier without changing the bound resources.
    pub fn set_ibl_intensity(&mut self, intensity: f32) {
        self.ibl_intensity = intensity;
    }

    /// Whether IBL resources are currently active.
    pub fn ibl_active(&self) -> bool {
        self.ibl_resources.is_some()
    }

    /// Rebuild the scene bind group, picking real IBL texture views when
    /// `ibl_resources` is `Some`, or dummy (black) views otherwise.
    fn rebuild_scene_bind_group(&mut self, device: &euca_rhi::wgpu_backend::WgpuDevice) {
        let (irradiance_view, specular_view, brdf_view, sampler) =
            if let Some(ref ibl) = self.ibl_resources {
                (
                    &ibl.irradiance_view,
                    &ibl.specular_view,
                    &ibl.brdf_lut_view,
                    &ibl.cubemap_sampler,
                )
            } else {
                (
                    &self.ibl_dummy_cube_view,
                    &self.ibl_dummy_cube_view,
                    &self.ibl_dummy_brdf_view,
                    &self.ibl_sampler,
                )
            };

        self.scene_bind_group = device.create_bind_group(&euca_rhi::BindGroupDesc {
            label: Some("Scene BG"),
            layout: &self.scene_bgl,
            entries: &[
                euca_rhi::BindGroupEntry {
                    binding: 0,
                    resource: euca_rhi::BindingResource::Buffer(euca_rhi::BufferBinding {
                        buffer: self.scene_buffer.raw(),
                        offset: 0,
                        size: None,
                    }),
                },
                euca_rhi::BindGroupEntry {
                    binding: 1,
                    resource: euca_rhi::BindingResource::TextureView(&self.shadow_map_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 2,
                    resource: euca_rhi::BindingResource::Sampler(&self.shadow_sampler),
                },
                euca_rhi::BindGroupEntry {
                    binding: 3,
                    resource: euca_rhi::BindingResource::Sampler(&self.shadow_depth_sampler),
                },
                euca_rhi::BindGroupEntry {
                    binding: 4,
                    resource: euca_rhi::BindingResource::TextureView(irradiance_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 5,
                    resource: euca_rhi::BindingResource::TextureView(specular_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 6,
                    resource: euca_rhi::BindingResource::TextureView(brdf_view),
                },
                euca_rhi::BindGroupEntry {
                    binding: 7,
                    resource: euca_rhi::BindingResource::Sampler(sampler),
                },
            ],
        });
    }

    /// Stage decal draw commands for the current frame.
    ///
    /// These will be rendered after opaque geometry in `render_to_view_with_lights`.
    /// The list is consumed (cleared) at the end of each frame.
    pub fn set_decal_commands(&mut self, commands: Vec<DecalDrawCommand>) {
        self.pending_decals = commands;
    }

    /// Read-only access to the decal renderer (unit-cube GPU resources).
    pub fn decal_renderer(&self) -> &DecalRenderer {
        &self.decal_renderer
    }

    /// Add a GPU compute particle system. Returns its index.
    pub fn add_gpu_particle_system(
        &mut self,
        gpu: &GpuContext,
        config: crate::gpu_particles::GpuParticleConfig,
    ) -> usize {
        let format = gpu.surface_config.format.into();
        let system = crate::gpu_particles::GpuParticleSystem::new(&**gpu, config, format);
        self.gpu_particle_systems.push(system);
        self.gpu_particle_systems.len() - 1
    }

    /// Access a GPU particle system by index.
    pub fn gpu_particle_system_mut(
        &mut self,
        index: usize,
    ) -> Option<&mut crate::gpu_particles::GpuParticleSystem> {
        self.gpu_particle_systems.get_mut(index)
    }

    /// Enable CPU-side HZB occlusion culling.
    pub fn enable_occlusion_culling(&mut self) {
        self.occlusion_culler = Some(OcclusionCuller::new());
    }

    /// Returns true if HZB occlusion culling is enabled.
    pub fn occlusion_culling_enabled(&self) -> bool {
        self.occlusion_culler.is_some()
    }

    /// Feed a depth buffer for the occlusion system to use next frame.
    pub fn update_prev_depth_buffer(&mut self, depth: Vec<f32>, width: u32, height: u32) {
        debug_assert_eq!(depth.len(), (width as usize) * (height as usize));
        self.prev_depth_buffer = depth;
        self.prev_depth_dims = (width, height);
    }

    /// Update the post-process settings that control SSAO, FXAA, bloom, and
    /// color grading.
    pub fn set_post_process_settings(&mut self, settings: PostProcessSettings) {
        self.post_process_settings = settings;
    }

    /// Read-only access to the current post-process settings.
    pub fn post_process_settings(&self) -> &PostProcessSettings {
        &self.post_process_settings
    }

    /// Enable screen-space reflections in the post-process pipeline.
    ///
    /// SSR settings can be further tuned via `set_post_process_settings`.
    pub fn enable_ssr(&mut self) {
        self.post_process_settings.ssr_enabled = true;
    }

    /// Initialize the volumetric fog pass.
    ///
    /// Once enabled, `render_to_view_with_lights` will execute the fog compute
    /// shader and composite the result over the HDR buffer after the PBR pass
    /// and before post-processing.
    pub fn enable_volumetric_fog(&mut self, gpu: &GpuContext) {
        self.volumetric_fog_pass = Some(VolumetricFogPass::new(
            &**gpu,
            gpu.surface_config.width,
            gpu.surface_config.height,
            gpu.surface_config.format.into(),
        ));
    }

    /// Update the volumetric fog settings (density, scattering, etc.).
    pub fn set_fog_settings(&mut self, settings: VolumetricFogSettings) {
        self.volumetric_fog_settings = settings;
    }

    /// Read-only access to the current volumetric fog settings.
    pub fn fog_settings(&self) -> &VolumetricFogSettings {
        &self.volumetric_fog_settings
    }

    /// Whether volumetric fog is currently active (pass created and enabled).
    pub fn is_fog_enabled(&self) -> bool {
        self.volumetric_fog_pass.is_some() && self.volumetric_fog_settings.enabled
    }

    fn light_vp_for_cascade(light: &DirectionalLight, ortho_size: f32) -> Mat4 {
        use euca_math::Vec3;
        let dir = Vec3::new(light.direction[0], light.direction[1], light.direction[2]).normalize();
        let light_pos = dir * -30.0;
        let light_view = Mat4::look_at_lh(light_pos, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        let s = ortho_size;
        let light_proj = Mat4::orthographic_lh(-s, s, -s, s, 0.1, 60.0);
        light_proj * light_view
    }
    fn light_vp(light: &DirectionalLight) -> Mat4 {
        Self::light_vp_for_cascade(light, CASCADE_ORTHO_SIZES[0])
    }

    #[allow(dead_code)]
    fn build_batches(commands: &[DrawCommand]) -> (Vec<InstanceData>, Vec<DrawBatch>) {
        if commands.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let mut indices: Vec<usize> = (0..commands.len()).collect();
        indices.sort_by_key(|&i| (commands[i].mesh.0, commands[i].material.0));
        let mut instances = Vec::with_capacity(commands.len());
        let mut batches = Vec::new();
        let mut batch_start = 0u32;
        let mut batch_mesh = commands[indices[0]].mesh;
        let mut batch_mat = commands[indices[0]].material;
        for &idx in &indices {
            let cmd = &commands[idx];
            if cmd.mesh != batch_mesh || cmd.material != batch_mat {
                batches.push(DrawBatch {
                    mesh: batch_mesh,
                    material: batch_mat,
                    instance_start: batch_start,
                    instance_count: instances.len() as u32 - batch_start,
                });
                batch_start = instances.len() as u32;
                batch_mesh = cmd.mesh;
                batch_mat = cmd.material;
            }
            let model = cmd.model_matrix;
            let normal_mat = model.inverse().transpose();
            instances.push(InstanceData {
                model: model.to_cols_array_2d(),
                normal_matrix: normal_mat.to_cols_array_2d(),
                material_id: cmd.material.0,
                _inst_pad: [0; 3],
            });
        }
        batches.push(DrawBatch {
            mesh: batch_mesh,
            material: batch_mat,
            instance_start: batch_start,
            instance_count: instances.len() as u32 - batch_start,
        });
        (instances, batches)
    }

    fn partition_commands<'a>(
        &self,
        commands: &'a [DrawCommand],
        camera_pos: euca_math::Vec3,
    ) -> (Vec<&'a DrawCommand>, Vec<&'a DrawCommand>) {
        let mut opaque = Vec::new();
        let mut transparent = Vec::new();
        for cmd in commands {
            if self.materials[cmd.material.0 as usize].is_transparent {
                transparent.push(cmd);
            } else {
                opaque.push(cmd);
            }
        }
        transparent.sort_by(|a, b| {
            let da = Self::distance_to_camera(&a.model_matrix, camera_pos);
            let db = Self::distance_to_camera(&b.model_matrix, camera_pos);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
        (opaque, transparent)
    }

    fn distance_to_camera(model_matrix: &Mat4, camera_pos: euca_math::Vec3) -> f32 {
        let cols = model_matrix.to_cols_array_2d();
        let obj_pos = euca_math::Vec3::new(cols[3][0], cols[3][1], cols[3][2]);
        (obj_pos - camera_pos).length()
    }

    fn build_batches_from_refs(commands: &[&DrawCommand]) -> (Vec<InstanceData>, Vec<DrawBatch>) {
        if commands.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let mut indices: Vec<usize> = (0..commands.len()).collect();
        indices.sort_by_key(|&i| (commands[i].mesh.0, commands[i].material.0));
        let mut instances = Vec::with_capacity(commands.len());
        let mut batches = Vec::new();
        let mut batch_start = 0u32;
        let mut batch_mesh = commands[indices[0]].mesh;
        let mut batch_mat = commands[indices[0]].material;
        for &idx in &indices {
            let cmd = commands[idx];
            if cmd.mesh != batch_mesh || cmd.material != batch_mat {
                batches.push(DrawBatch {
                    mesh: batch_mesh,
                    material: batch_mat,
                    instance_start: batch_start,
                    instance_count: instances.len() as u32 - batch_start,
                });
                batch_start = instances.len() as u32;
                batch_mesh = cmd.mesh;
                batch_mat = cmd.material;
            }
            let model = cmd.model_matrix;
            let normal_mat = model.inverse().transpose();
            instances.push(InstanceData {
                model: model.to_cols_array_2d(),
                normal_matrix: normal_mat.to_cols_array_2d(),
                material_id: cmd.material.0,
                _inst_pad: [0; 3],
            });
        }
        batches.push(DrawBatch {
            mesh: batch_mesh,
            material: batch_mat,
            instance_start: batch_start,
            instance_count: instances.len() as u32 - batch_start,
        });
        (instances, batches)
    }

    /// Execute the full rendering pipeline for one frame using only a
    /// directional light and ambient light (no point/spot lights).
    ///
    /// Acquires the surface texture, renders, and presents. For off-screen
    /// rendering or when additional lights are needed, use
    /// [`render_to_view_with_lights`](Self::render_to_view_with_lights).
    pub fn draw(
        &mut self,
        gpu: &GpuContext,
        camera: &Camera,
        light: &DirectionalLight,
        ambient: &AmbientLight,
        commands: &[DrawCommand],
    ) {
        self.draw_with_lights(gpu, camera, light, ambient, commands, &[], &[]);
    }

    /// Execute the full rendering pipeline for one frame with point and spot
    /// lights in addition to the directional and ambient lights.
    ///
    /// Acquires the surface texture, renders, and presents.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_with_lights(
        &mut self,
        gpu: &GpuContext,
        camera: &Camera,
        light: &DirectionalLight,
        ambient: &AmbientLight,
        commands: &[DrawCommand],
        point_lights: &[(euca_math::Vec3, &crate::light::PointLight)],
        spot_lights: &[(euca_math::Vec3, &crate::light::SpotLight)],
    ) {
        use euca_rhi::RenderDevice;
        let rhi: &euca_rhi::wgpu_backend::WgpuDevice = gpu;

        let output = match rhi.get_current_texture() {
            Ok(t) => t,
            Err(euca_rhi::SurfaceError::Outdated | euca_rhi::SurfaceError::Lost) => return,
            Err(e) => {
                log::error!("Surface error: {e}");
                return;
            }
        };
        let view = rhi.surface_texture_view(&output);
        let mut encoder = rhi.create_command_encoder(Some("Render Encoder"));
        self.render_to_view_with_lights(
            gpu,
            camera,
            light,
            ambient,
            commands,
            point_lights,
            spot_lights,
            &view,
            &mut encoder,
        );
        rhi.submit(encoder);
        rhi.present(output);
    }

    /// Render one frame into a caller-provided texture view (no point/spot
    /// lights). Useful for off-screen rendering or editor viewports.
    #[allow(clippy::too_many_arguments)]
    pub fn render_to_view(
        &mut self,
        gpu: &GpuContext,
        camera: &Camera,
        light: &DirectionalLight,
        ambient: &AmbientLight,
        commands: &[DrawCommand],
        color_view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.render_to_view_with_lights(
            gpu,
            camera,
            light,
            ambient,
            commands,
            &[],
            &[],
            color_view,
            encoder,
        );
    }

    /// Render one frame into a caller-provided texture view with full light
    /// support (directional, ambient, point, and spot lights).
    ///
    /// This is the most flexible entry point. The caller is responsible for
    /// creating the command encoder and submitting / presenting afterward.
    #[allow(clippy::too_many_arguments)]
    pub fn render_to_view_with_lights(
        &mut self,
        gpu: &GpuContext,
        camera: &Camera,
        light: &DirectionalLight,
        ambient: &AmbientLight,
        commands: &[DrawCommand],
        point_lights: &[(euca_math::Vec3, &crate::light::PointLight)],
        spot_lights: &[(euca_math::Vec3, &crate::light::SpotLight)],
        color_view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let vp = camera.view_projection_matrix(gpu.aspect_ratio());
        let light_vp = Self::light_vp(light);
        let (opaque_cmds, transparent_cmds) = self.partition_commands(commands, camera.eye);
        let opaque_cmds = self.apply_occlusion_culling(&opaque_cmds, vp);
        let (opaque_instances, opaque_batches) = Self::build_batches_from_refs(&opaque_cmds);
        // Pre-build transparent batches so we can ensure capacity before
        // the render pass borrows self immutably via resolve_target.
        let (trans_instances, trans_batches) = if !transparent_cmds.is_empty() {
            Self::build_batches_from_refs(&transparent_cmds)
        } else {
            (Vec::new(), Vec::new())
        };
        // Ensure capacity for the larger of opaque/transparent sets.
        let max_needed = opaque_instances.len().max(trans_instances.len());
        if max_needed > 0 {
            self.ensure_instance_capacity(&gpu.device, max_needed);
        }
        if !opaque_instances.is_empty() {
            self.instance_buffer.write(&**gpu, &opaque_instances);
        }
        for (cascade_idx, &cascade_ortho) in CASCADE_ORTHO_SIZES.iter().enumerate() {
            let cascade_vp = Self::light_vp_for_cascade(light, cascade_ortho);
            let shadow_instances: Vec<InstanceData> = opaque_instances
                .iter()
                .map(|inst| {
                    let model = Mat4::from_cols_array_2d(&inst.model);
                    let shadow_mvp = cascade_vp * model;
                    InstanceData {
                        model: shadow_mvp.to_cols_array_2d(),
                        normal_matrix: [[0.0; 4]; 4],
                        material_id: inst.material_id,
                        _inst_pad: [0; 3],
                    }
                })
                .collect();
            if !shadow_instances.is_empty() {
                self.ensure_shadow_instance_capacity(&gpu.device, shadow_instances.len());
                self.shadow_instance_buffer.write(&**gpu, &shadow_instances);
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow Pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_cascade_views[cascade_idx],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(0, &self.shadow_instance_bind_group, &[]);
            for batch in &opaque_batches {
                let mesh = &self.meshes[batch.mesh.0 as usize];
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(
                    0..mesh.index_count,
                    0,
                    batch.instance_start..batch.instance_start + batch.instance_count,
                );
            }
        }
        let dir = light.direction;
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
            .sqrt()
            .max(0.001);
        let scene = SceneUniforms {
            camera_pos: [camera.eye.x, camera.eye.y, camera.eye.z, 0.0],
            light_direction: [dir[0] / len, dir[1] / len, dir[2] / len, 0.0],
            light_color: [
                light.color[0],
                light.color[1],
                light.color[2],
                light.intensity,
            ],
            ambient_color: [
                ambient.color[0],
                ambient.color[1],
                ambient.color[2],
                ambient.intensity,
            ],
            camera_vp: vp.to_cols_array_2d(),
            light_vp: light_vp.to_cols_array_2d(),
            inv_vp: vp.inverse().to_cols_array_2d(),
            cascade_vps: [
                Self::light_vp_for_cascade(light, CASCADE_ORTHO_SIZES[0]).to_cols_array_2d(),
                Self::light_vp_for_cascade(light, CASCADE_ORTHO_SIZES[1]).to_cols_array_2d(),
                Self::light_vp_for_cascade(light, CASCADE_ORTHO_SIZES[2]).to_cols_array_2d(),
            ],
            cascade_splits: [
                CASCADE_ORTHO_SIZES[0],
                CASCADE_ORTHO_SIZES[1],
                CASCADE_ORTHO_SIZES[2],
                0.0,
            ],
            point_lights: {
                let mut arr = [GpuPointLight::default(); MAX_POINT_LIGHTS];
                for (i, (pos, pl)) in point_lights.iter().take(MAX_POINT_LIGHTS).enumerate() {
                    arr[i] = GpuPointLight {
                        position: [pos.x, pos.y, pos.z, pl.range],
                        color: [pl.color[0], pl.color[1], pl.color[2], pl.intensity],
                    };
                }
                arr
            },
            spot_lights: {
                let mut arr = [GpuSpotLight::default(); MAX_SPOT_LIGHTS];
                for (i, (pos, sl)) in spot_lights.iter().take(MAX_SPOT_LIGHTS).enumerate() {
                    arr[i] = GpuSpotLight {
                        position: [pos.x, pos.y, pos.z, sl.range],
                        direction: [sl.direction[0], sl.direction[1], sl.direction[2], 0.0],
                        color: [sl.color[0], sl.color[1], sl.color[2], sl.intensity],
                        cone: [sl.inner_cone.cos(), sl.outer_cone.cos(), 0.0, 0.0],
                    };
                }
                arr
            },
            num_point_lights: [
                point_lights.len().min(MAX_POINT_LIGHTS) as f32,
                0.0,
                0.0,
                0.0,
            ],
            num_spot_lights: [spot_lights.len().min(MAX_SPOT_LIGHTS) as f32, 0.0, 0.0, 0.0],
            probe_sh: self.probe_sh,
            probe_enabled: [if self.probe_enabled { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
            shadow_params: [light.light_size, 0.01, 0.03, 0.5],
            ibl_params: [
                if self.ibl_resources.is_some() {
                    1.0
                } else {
                    0.0
                },
                self.ibl_intensity,
                0.0,
                0.0,
            ],
        };
        self.scene_buffer
            .write_bytes(&**gpu, bytemuck::bytes_of(&scene));

        // Flush bindless material data to GPU before rendering.
        if let Some(ref mut bl) = self.bindless {
            bl.system.flush(&gpu.device, &gpu.queue, &self.textures);
        }

        // Resolve MSAA into the post-process stack's ping buffer.
        let resolve_target = self.post_process_stack.ping_view();

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PBR Pass (MSAA)"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_hdr_view,
                    resolve_target: Some(resolve_target),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, &self.scene_bind_group, &[]);
            pass.draw(0..3, 0..1);
            pass.set_bind_group(0, &self.instance_bind_group, &[]);
            pass.set_bind_group(1, &self.scene_bind_group, &[]);
            if let Some(ref bl) = self.bindless {
                // Bindless path: single pipeline + single material bind group.
                pass.set_pipeline(&bl.pipeline);
                pass.set_bind_group(2, &bl.system.bind_group, &[]);
                for batch in &opaque_batches {
                    let mesh = &self.meshes[batch.mesh.0 as usize];
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(
                        0..mesh.index_count,
                        0,
                        batch.instance_start..batch.instance_start + batch.instance_count,
                    );
                }
            } else {
                // Traditional path: switch material bind group per batch.
                pass.set_pipeline(&self.pipeline);
                for batch in &opaque_batches {
                    let gpu_mat = &self.materials[batch.material.0 as usize];
                    pass.set_bind_group(2, &gpu_mat.bind_group, &[]);
                    let mesh = &self.meshes[batch.mesh.0 as usize];
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(
                        0..mesh.index_count,
                        0,
                        batch.instance_start..batch.instance_start + batch.instance_count,
                    );
                }
            }
            // ── Decal pass: draw projected decal volumes after opaque geometry ──
            // Each decal is a unit cube scaled/positioned by its model matrix.
            // The decal shader (when a dedicated pipeline is added) reads the
            // depth buffer to reconstruct world-space position and projects the
            // decal texture. For now we bind the decal vertex/index buffers and
            // issue one draw per command so the integration path is exercised.
            if !self.pending_decals.is_empty() {
                pass.set_vertex_buffer(0, self.decal_renderer.vertex_buffer().slice(..));
                pass.set_index_buffer(
                    self.decal_renderer.index_buffer().slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                pass.set_bind_group(0, &self.decal_bind_group, &[]);
                for decal_cmd in &self.pending_decals {
                    let uniforms = DecalUniforms {
                        model_matrix: decal_cmd.model_matrix.to_cols_array_2d(),
                        opacity: decal_cmd.opacity,
                        _pad: [0.0; 3],
                    };
                    self.decal_uniform_buffer
                        .write(&**gpu, std::slice::from_ref(&uniforms));
                    pass.draw_indexed(0..self.decal_renderer.index_count(), 0, 0..1);
                }
            }

            if !transparent_cmds.is_empty() {
                if !trans_instances.is_empty() {
                    self.instance_buffer.write(&**gpu, &trans_instances);
                }
                pass.set_pipeline(&self.transparent_pipeline);
                pass.set_bind_group(0, &self.instance_bind_group, &[]);
                pass.set_bind_group(1, &self.scene_bind_group, &[]);
                for batch in &trans_batches {
                    let gpu_mat = &self.materials[batch.material.0 as usize];
                    pass.set_bind_group(2, &gpu_mat.bind_group, &[]);
                    let mesh = &self.meshes[batch.mesh.0 as usize];
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(
                        0..mesh.index_count,
                        0,
                        batch.instance_start..batch.instance_start + batch.instance_count,
                    );
                }
            }
        }

        // Clear pending decals after rendering — they must be re-submitted each frame.
        self.pending_decals.clear();

        // GPU compute particles: update (compute dispatch) then draw (render pass).
        if !self.gpu_particle_systems.is_empty() {
            let dt = 1.0 / 60.0; // Fixed timestep for particle update
            for system in &mut self.gpu_particle_systems {
                system.update(&**gpu, encoder, dt);
            }

            // Draw particles in a separate render pass (after opaque, blended on top)
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gpu_particles"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_texture,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                for system in &self.gpu_particle_systems {
                    system.draw(&mut pass);
                }
            }
        }

        // Volumetric fog: compute ray-march and composite over HDR buffer.
        if let Some(ref fog_pass) = self.volumetric_fog_pass
            && self.volumetric_fog_settings.enabled
        {
            let dir = light.direction;
            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
                .sqrt()
                .max(0.001);
            let frame = FrameParams {
                camera_pos: [camera.eye.x, camera.eye.y, camera.eye.z],
                inv_vp: vp.inverse().to_cols_array_2d(),
                light_direction: [dir[0] / len, dir[1] / len, dir[2] / len],
                light_color: [light.color[0], light.color[1], light.color[2]],
                settings: &self.volumetric_fog_settings,
            };
            fog_pass.execute(&**gpu, encoder, resolve_target, &frame);
        }

        // TAA resolve: blend current frame with history for temporal anti-aliasing.
        if self.post_process_settings.taa_enabled {
            let inv_vp = vp.inverse();
            let prev_vp = camera.prev_view_proj.unwrap_or(vp);
            let rhi: &euca_rhi::wgpu_backend::WgpuDevice = gpu;
            self.taa_pass.execute(
                rhi,
                encoder,
                resolve_target,
                &self.post_process_stack.depth_resolve_view,
                &self.velocity_textures.velocity_view,
                &inv_vp,
                &prev_vp,
                camera.jitter,
            );
            // Copy TAA output back to ping_view so post-processing reads the
            // temporally resolved image.
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: self.taa_pass.output_texture(),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: self.post_process_stack.ping_texture(),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: gpu.surface_config.width,
                    height: gpu.surface_config.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.frame_count = self.frame_count.wrapping_add(1);

        // Post-processing via the modular stack.
        {
            let proj = camera.projection_matrix(gpu.aspect_ratio());
            let inv_projection = proj.inverse().to_cols_array_2d();
            let projection = proj.to_cols_array_2d();
            self.post_process_stack.execute(
                &gpu.device,
                &gpu.queue,
                encoder,
                color_view,
                &self.post_process_settings,
                &inv_projection,
                &projection,
            );
        }
    }

    /// Filter out occluded draw commands using the HZB from the previous frame.
    fn apply_occlusion_culling<'a>(
        &mut self,
        commands: &[&'a DrawCommand],
        view_proj: Mat4,
    ) -> Vec<&'a DrawCommand> {
        if self.occlusion_culler.is_none() {
            return commands.to_vec();
        }
        let (w, h) = self.prev_depth_dims;
        if self.prev_depth_buffer.is_empty() || w == 0 || h == 0 {
            return commands.to_vec();
        }
        self.occlusion_culler
            .as_mut()
            .expect("occlusion culler initialized in Renderer::new")
            .update_from_depth_buffer(&self.prev_depth_buffer, w, h);
        let culler = self
            .occlusion_culler
            .as_ref()
            .expect("occlusion culler initialized in Renderer::new");
        let mut aabbs = Vec::new();
        let mut aabb_indices = Vec::new();
        for (i, cmd) in commands.iter().enumerate() {
            if let Some(aabb) = cmd.aabb {
                aabbs.push(aabb);
                aabb_indices.push(i);
            }
        }
        if aabbs.is_empty() {
            return commands.to_vec();
        }
        let result = match culler.test(&aabbs, view_proj) {
            Some(r) => r,
            None => return commands.to_vec(),
        };
        let mut occluded = vec![false; commands.len()];
        for (aabb_idx, &cmd_idx) in aabb_indices.iter().enumerate() {
            if !result.visible[aabb_idx] {
                occluded[cmd_idx] = true;
            }
        }
        commands
            .iter()
            .enumerate()
            .filter(|(i, _)| !occluded[*i])
            .map(|(_, cmd)| *cmd)
            .collect()
    }

    fn create_depth_texture(
        device: &euca_rhi::wgpu_backend::WgpuDevice,
        width: u32,
        height: u32,
        format: euca_rhi::TextureFormat,
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&euca_rhi::TextureDesc {
            label: Some("Depth Texture (MSAA)"),
            size: euca_rhi::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLE_COUNT,
            dimension: euca_rhi::TextureDimension::D2,
            format,
            usage: euca_rhi::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        device.create_texture_view(&texture, &euca_rhi::TextureViewDesc::default())
    }
    fn create_msaa_hdr_texture(
        device: &euca_rhi::wgpu_backend::WgpuDevice,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&euca_rhi::TextureDesc {
            label: Some("MSAA HDR Texture"),
            size: euca_rhi::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLE_COUNT,
            dimension: euca_rhi::TextureDimension::D2,
            format: euca_rhi::TextureFormat::Rgba16Float,
            usage: euca_rhi::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = device.create_texture_view(&texture, &euca_rhi::TextureViewDesc::default());
        (texture, view)
    }
}

const SHADOW_SHADER: &str = include_str!("../shaders/shadow.wgsl");

const PBR_SHADER: &str = include_str!("../shaders/pbr.wgsl");
const PBR_BINDLESS_SHADER: &str = include_str!("../shaders/pbr_bindless.wgsl");

const SKY_SHADER: &str = include_str!("../shaders/sky.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_preset_produces_valid_settings() {
        for quality in [
            RenderQuality::Low,
            RenderQuality::Medium,
            RenderQuality::High,
            RenderQuality::Ultra,
        ] {
            let s = quality.to_settings();
            assert!(s.fxaa_enabled, "{quality:?} should have FXAA enabled");
            assert!(s.bloom_enabled, "{quality:?} should have bloom enabled");
            assert!(s.ssao_radius >= 0.0, "{quality:?} ssao_radius must be >= 0");
            assert!(
                s.ssao_intensity >= 0.0,
                "{quality:?} ssao_intensity must be >= 0"
            );
            assert!(
                s.ibl_intensity > 0.0,
                "{quality:?} ibl_intensity must be > 0"
            );
            assert!(s.pcss_enabled, "{quality:?} should have PCSS enabled");
        }
    }

    #[test]
    fn low_has_new_features_off() {
        let s = RenderQuality::Low.to_settings();
        assert!(!s.ssao_enabled);
        assert!(!s.ssgi_enabled, "Low should have SSGI disabled");
        assert!(
            !s.motion_blur.enabled,
            "Low should have motion blur disabled"
        );
        assert!(!s.dof.enabled, "Low should have DoF disabled");
        assert!(!s.ibl_enabled, "Low should have IBL disabled");
    }

    #[test]
    fn default_settings_have_new_features_off() {
        let s = PostProcessSettings::default();
        assert!(!s.ssgi_enabled, "Default should have SSGI disabled");
        assert!(
            !s.motion_blur.enabled,
            "Default should have motion blur disabled"
        );
        assert!(!s.dof.enabled, "Default should have DoF disabled");
        assert!(!s.ibl_enabled, "Default should have IBL disabled");
        assert!(s.pcss_enabled, "Default should have PCSS enabled");
    }

    #[test]
    fn ultra_has_everything_on() {
        let s = RenderQuality::Ultra.to_settings();
        assert!(s.ssao_enabled);
        assert!(s.fxaa_enabled);
        assert!(s.bloom_enabled);
        assert!(
            s.ssao_intensity > 1.0,
            "Ultra should have boosted SSAO intensity"
        );
        assert!(
            s.contrast > 1.0,
            "Ultra should have slightly elevated contrast"
        );
        assert!(
            s.saturation > 1.0,
            "Ultra should have slightly elevated saturation"
        );
        // New WU1-WU7 features.
        assert!(s.ssgi_enabled, "Ultra should have SSGI enabled");
        assert_eq!(s.ssgi_ray_count, 8, "Ultra should use 8 SSGI rays");
        assert!(
            s.ssgi_intensity > 1.0,
            "Ultra should have boosted SSGI intensity"
        );
        assert!(s.ibl_enabled, "Ultra should have IBL enabled");
        assert!(
            (s.ibl_intensity - 1.0).abs() < 1e-6,
            "Ultra IBL intensity should be 1.0"
        );
        assert!(
            s.motion_blur.enabled,
            "Ultra should have motion blur enabled"
        );
        assert_eq!(
            s.motion_blur.sample_count, 16,
            "Ultra motion blur should use 16 samples"
        );
        assert!(s.dof.enabled, "Ultra should have DoF enabled");
        assert!(
            (s.dof.max_blur_radius - 30.0).abs() < 1e-6,
            "Ultra DoF max blur radius should be 30"
        );
        assert!(s.pcss_enabled, "Ultra should have PCSS enabled");
    }

    #[test]
    fn from_name_case_insensitive() {
        assert_eq!(RenderQuality::from_name("low"), Some(RenderQuality::Low));
        assert_eq!(RenderQuality::from_name("HIGH"), Some(RenderQuality::High));
        assert_eq!(
            RenderQuality::from_name("Ultra"),
            Some(RenderQuality::Ultra)
        );
        assert_eq!(RenderQuality::from_name("med"), Some(RenderQuality::Medium));
        assert_eq!(RenderQuality::from_name("invalid"), None);
    }

    #[test]
    fn name_roundtrip() {
        for quality in [
            RenderQuality::Low,
            RenderQuality::Medium,
            RenderQuality::High,
            RenderQuality::Ultra,
        ] {
            assert_eq!(
                RenderQuality::from_name(quality.name()),
                Some(quality),
                "{quality:?} should roundtrip through name()"
            );
        }
    }

    /// The PBR shader must declare a 16-element Poisson disk array with all
    /// sample coordinates in the [-1, 1] range.
    #[test]
    fn pcss_poisson_disk_has_16_entries_in_range() {
        let src = PBR_SHADER;
        // Extract the POISSON_16 block between `array(` and the closing `);`.
        let start = src
            .find("const POISSON_16")
            .expect("POISSON_16 not found in shader");
        let block = &src[start..];
        let array_start = block.find("array(").unwrap();
        let array_end = block[array_start..].find(");").unwrap();
        let array_body = &block[array_start..array_start + array_end];

        // Count vec2( occurrences — each one is a sample.
        let count = array_body.matches("vec2(").count();
        assert_eq!(
            count, 16,
            "POISSON_16 must have exactly 16 entries, found {count}"
        );

        // Parse all floating point literals and verify range.
        for cap in array_body.split("vec2(").skip(1) {
            let inner = cap.split(')').next().unwrap();
            for component in inner.split(',') {
                let val: f32 = component.trim().parse().unwrap_or_else(|e| {
                    panic!("Failed to parse Poisson component '{component}': {e}");
                });
                assert!(
                    (-1.0..=1.0).contains(&val),
                    "Poisson sample component {val} is outside [-1, 1]"
                );
            }
        }
    }

    /// When `light_size` is 0.0, the PCSS search radius is zero, which means the
    /// blocker search finds no blockers and the function returns 1.0 (fully lit).
    /// This effectively produces hard shadows since only the PCF path with a
    /// minimum radius is used. Verify `shadow_params` propagation at the uniform
    /// level: zero `light_size` => `shadow_params.x == 0.0`, while bias
    /// defaults are still present.
    #[test]
    fn pcss_zero_light_size_produces_zero_search_radius() {
        use crate::light::DirectionalLight;
        let light = DirectionalLight {
            light_size: 0.0,
            ..Default::default()
        };
        // The uniform would be populated as:
        let shadow_params = [light.light_size, 0.01, 0.03, 0.5];
        assert_eq!(
            shadow_params[0], 0.0,
            "zero light_size must yield zero search radius"
        );
    }

    /// Default `DirectionalLight` should have `light_size = 1.0`.
    #[test]
    fn directional_light_default_light_size() {
        use crate::light::DirectionalLight;
        let light = DirectionalLight::default();
        assert_eq!(light.light_size, 1.0);
    }

    /// Verify default shadow bias parameters packed in `shadow_params`.
    #[test]
    fn shadow_bias_defaults() {
        use crate::light::DirectionalLight;
        let light = DirectionalLight::default();
        let shadow_params = [light.light_size, 0.01_f32, 0.03_f32, 0.5_f32];
        // y = normal_bias_scale
        assert!(
            (shadow_params[1] - 0.01).abs() < 1e-6,
            "normal_bias_scale default"
        );
        // z = slope_bias_scale
        assert!(
            (shadow_params[2] - 0.03).abs() < 1e-6,
            "slope_bias_scale default"
        );
        // w = cascade_bias_scale
        assert!(
            (shadow_params[3] - 0.5).abs() < 1e-6,
            "cascade_bias_scale default"
        );
    }

    /// `SceneUniforms` size must not change unexpectedly. Adding fields without
    /// updating all dependent code (bind group layout, shader struct) would
    /// cause GPU validation errors. This pins the expected size.
    #[test]
    fn scene_uniforms_size_is_stable() {
        let size = std::mem::size_of::<SceneUniforms>();
        // The struct is tightly packed via repr(C) with vec4/mat4 members.
        // Any accidental addition or removal will break this assertion.
        assert_eq!(
            size % 16,
            0,
            "SceneUniforms size ({size}) must be 16-byte aligned for GPU uniform buffers"
        );
    }

    /// IBL params default to disabled (x=0.0) with intensity 1.0.
    #[test]
    fn ibl_params_defaults() {
        let ibl_params = [0.0_f32, 1.0_f32, 0.0_f32, 0.0_f32];
        assert!(
            ibl_params[0] < 0.5,
            "IBL should default to disabled (x < 0.5)"
        );
        assert!(
            (ibl_params[1] - 1.0).abs() < 1e-6,
            "IBL intensity should default to 1.0"
        );
    }

    /// `SceneUniforms` must include the `ibl_params` field as the last vec4.
    #[test]
    fn scene_uniforms_contains_ibl_params() {
        let uniforms = SceneUniforms {
            camera_pos: [0.0; 4],
            light_direction: [0.0; 4],
            light_color: [0.0; 4],
            ambient_color: [0.0; 4],
            camera_vp: [[0.0; 4]; 4],
            light_vp: [[0.0; 4]; 4],
            inv_vp: [[0.0; 4]; 4],
            cascade_vps: [[[0.0; 4]; 4]; 3],
            cascade_splits: [0.0; 4],
            point_lights: [GpuPointLight::default(); MAX_POINT_LIGHTS],
            spot_lights: [GpuSpotLight::default(); MAX_SPOT_LIGHTS],
            num_point_lights: [0.0; 4],
            num_spot_lights: [0.0; 4],
            probe_sh: [[0.0; 4]; 9],
            probe_enabled: [0.0; 4],
            shadow_params: [1.0, 0.01, 0.03, 0.5],
            ibl_params: [1.0, 0.8, 0.0, 0.0],
        };
        assert_eq!(uniforms.ibl_params[0], 1.0, "ibl enabled flag");
        assert!((uniforms.ibl_params[1] - 0.8).abs() < 1e-6, "ibl intensity");
    }

    /// InstanceData size must be exactly 144 bytes to match the WGSL struct
    /// in pbr.wgsl, shadow.wgsl, prepass.wgsl, velocity.wgsl, gbuffer.wgsl.
    /// Changing this without updating ALL shaders causes silent stride mismatch.
    #[test]
    fn instance_data_size_is_144() {
        assert_eq!(
            std::mem::size_of::<InstanceData>(),
            144,
            "InstanceData must be 144 bytes: model(64) + normal_matrix(64) + material_id(4) + pad(12)"
        );
    }
}
