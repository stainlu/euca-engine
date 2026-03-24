//! Rendering subsystem for the Euca engine.
//!
//! Provides a PBR forward renderer built on `wgpu` with support for cascaded
//! shadow maps, MSAA, HDR post-processing (SSAO, FXAA, bloom, color grading),
//! TAA, volumetric fog, decals, and GPU-driven rendering.
//!
//! # Key types
//!
//! - [`Renderer`] -- main rendering pipeline; owns GPU resources and executes
//!   draw calls each frame.
//! - [`Camera`] / [`Frustum`] -- view and projection setup.
//! - [`Material`] / [`MaterialHandle`] -- PBR material definitions.
//! - [`Mesh`] / [`MeshHandle`] / [`MeshRenderer`] -- geometry data.
//! - [`TextureStore`] / [`TextureHandle`] -- GPU texture management.
//! - [`DrawCommand`] -- per-object draw parameters submitted each frame.
//! - [`Vertex`] -- interleaved vertex layout (position, normal, tangent, UV).

mod buffer;
mod camera;
/// Clustered light assignment for tiled/clustered forward shading.
pub mod clustered;
/// GPU compute pipeline management and indirect dispatch utilities.
pub mod compute;
/// Deferred decal projection volumes.
pub mod decal;
/// Deferred shading G-buffer and lighting pass.
pub mod deferred;
/// Depth of field (thin-lens bokeh blur).
pub mod dof;
/// Procedural foliage scattering and instanced rendering.
pub mod foliage;
mod gpu;
/// GPU-driven rendering with indirect draw and compute culling.
pub mod gpu_driven;
/// GPU compute particle systems.
pub mod gpu_particles;
mod hardware;
mod hlod;
/// Image-Based Lighting (IBL) resource generation pipeline.
pub mod ibl;
mod light;
/// Spherical-harmonics light probes for indirect lighting.
pub mod light_probe;
mod lod;
mod material;
mod mesh;
/// Apple Metal backend hints and render pass optimization.
pub mod metal_hints;
/// Per-pixel velocity-based motion blur.
pub mod motion_blur;
/// Hierarchical Z-buffer (HZB) occlusion culling.
pub mod occlusion;
mod plugin;
/// Post-processing stack (SSAO, FXAA, bloom, color grading, tone mapping).
pub mod post_process;
/// Depth and normal pre-pass for deferred techniques.
pub mod prepass;
mod renderer;
/// Screen-space global illumination (indirect diffuse via depth-buffer ray-march).
pub mod ssgi;
/// Screen-space reflections.
pub mod ssr;
/// Temporal anti-aliasing resolve pass.
pub mod taa;
mod texture;
/// Per-pixel velocity buffer for motion vectors (TAA, motion blur, temporal SSGI).
pub mod velocity;
mod vertex;
/// Volumetric fog (ray-marched scattering).
pub mod volumetric;

/// Re-export `wgpu` so downstream crates can use the same version.
pub use wgpu;

pub use buffer::{BufferKind, SmartBuffer};
pub use camera::{Camera, Frustum};
pub use clustered::{
    ClusterConfig, ClusteredLightGrid, GpuLightData, LightType, UpdateParams, cluster_index,
    sphere_aabb_intersect,
};
pub use compute::{
    ComputeManager, ComputePipeline, ComputePipelineDesc, CullParams, GpuAabb, GpuBuffer,
    GpuFrustumPlanes,
};
pub use decal::{
    Decal, DecalBlendMode, DecalDrawCommand, DecalProjection, DecalRenderer,
    collect_decal_draw_commands,
};
pub use deferred::{
    DeferredLightingUniforms, DeferredPipeline, GBuffer, GBufferFormats, RenderPath,
};
pub use dof::{DofPass, DofSettings};
pub use foliage::{
    FoliageDrawData, FoliageInstance, FoliageLayer, FoliageLayers, FoliageRenderer, scatter_foliage,
};
pub use gpu::GpuContext;
pub use gpu_driven::{
    DrawCommandGpu, DrawIndexedIndirectArgs, GpuCullParams, GpuDrivenPipeline, GpuFrustumData,
    IndirectDrawBuffer,
};
pub use gpu_particles::{GpuParticleConfig, GpuParticleSystem};
pub use hardware::{AdapterInfo, GpuVendor, HardwareSurvey, RenderBackend, SystemInfo};
pub use hlod::{
    HlodCluster, HlodClusterId, HlodRegistry, HlodVisibility, generate_hlod_cluster,
    hlod_select_system,
};
pub use ibl::IblResources;
pub use light::{AmbientLight, DirectionalLight, PointLight, SpotLight};
pub use light_probe::{LightProbe, LightProbeGrid, evaluate_sh};
pub use lod::{LodSettings, lod_select_system};
pub use material::{AlphaMode, Material, MaterialHandle, MaterialRef};
pub use mesh::{Mesh, MeshHandle, MeshRenderer};
pub use metal_hints::{
    AttachmentOps, ComputeOptimizer, ComputeTimingHint, DepthAttachmentOps, MetalRenderHints,
    RenderPassLayout, RenderPassOptimizer,
};
pub use motion_blur::{MotionBlurPass, MotionBlurSettings};
pub use occlusion::{HzbPyramid, OcclusionCuller, OcclusionResult};
pub use plugin::RenderPlugin;
pub use post_process::{PostProcessSettings, PostProcessStack};
pub use renderer::{DrawCommand, RenderQuality, Renderer};
pub use ssgi::{SsgiExecuteParams, SsgiPass, SsgiSettings, step_size as ssgi_step_size};
pub use ssr::{
    SsrExecuteParams, SsrPass, SsrSettings, compute_step_count, passes_roughness_filter,
};
pub use taa::TaaPass;
pub use texture::{TextureHandle, TextureStore};
pub use velocity::{VELOCITY_FORMAT, VelocityPipeline, VelocitySceneUniforms, VelocityTextures};
pub use vertex::Vertex;
pub use volumetric::{VolumetricFogPass, VolumetricFogSettings};
