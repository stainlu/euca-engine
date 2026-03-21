mod buffer;
mod camera;
pub mod compute;
pub mod decal;
pub mod deferred;
pub mod foliage;
mod gpu;
pub mod gpu_driven;
mod hardware;
mod hlod;
mod light;
mod lod;
mod material;
mod mesh;
pub mod occlusion;
mod plugin;
pub mod post_process;
mod renderer;
pub mod ssr;
mod texture;
mod vertex;
pub mod volumetric;

pub use wgpu;

pub use buffer::{BufferKind, SmartBuffer};
pub use camera::{Camera, Frustum};
pub use compute::{
    ComputeManager, ComputePipeline, ComputePipelineDesc, CullParams, GpuAabb, GpuBuffer,
    GpuFrustumPlanes,
};
pub use decal::{
    Decal, DecalBlendMode, DecalDrawCommand, DecalProjection, DecalRenderer,
    collect_decal_draw_commands,
};
pub use deferred::{GBuffer, GBufferFormats, RenderPath};
pub use foliage::{
    FoliageDrawData, FoliageInstance, FoliageLayer, FoliageRenderer, scatter_foliage,
};
pub use gpu::GpuContext;
pub use gpu_driven::{
    DrawCommandGpu, DrawIndexedIndirectArgs, GpuCullParams, GpuDrivenPipeline, GpuFrustumData,
    IndirectDrawBuffer,
};
pub use hardware::{AdapterInfo, GpuVendor, HardwareSurvey, RenderBackend, SystemInfo};
pub use hlod::{
    HlodCluster, HlodClusterId, HlodRegistry, HlodVisibility, generate_hlod_cluster,
    hlod_select_system,
};
pub use light::{AmbientLight, DirectionalLight, PointLight, SpotLight};
pub use lod::{LodSettings, lod_select_system};
pub use material::{AlphaMode, Material, MaterialHandle, MaterialRef};
pub use mesh::{Mesh, MeshHandle, MeshRenderer};
pub use occlusion::{HzbPyramid, OcclusionCuller, OcclusionResult};
pub use plugin::RenderPlugin;
pub use post_process::{PostProcessSettings, PostProcessStack};
pub use renderer::{DrawCommand, Renderer};
pub use ssr::{
    SsrExecuteParams, SsrPass, SsrSettings, compute_step_count, passes_roughness_filter,
};
pub use texture::{TextureHandle, TextureStore};
pub use vertex::Vertex;
pub use volumetric::{VolumetricFogPass, VolumetricFogSettings};
