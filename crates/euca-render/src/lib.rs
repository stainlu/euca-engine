mod camera;
pub mod compute;
pub mod decal;
pub mod deferred;
mod gpu;
mod hardware;
mod light;
mod lod;
mod material;
mod mesh;
pub mod metal_hints;
mod plugin;
pub mod post_process;
mod renderer;
mod texture;
mod vertex;

pub use wgpu;

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
pub use gpu::GpuContext;
pub use hardware::{AdapterInfo, GpuVendor, HardwareSurvey, RenderBackend, SystemInfo};
pub use light::{AmbientLight, DirectionalLight, PointLight, SpotLight};
pub use lod::{LodSettings, lod_select_system};
pub use material::{AlphaMode, Material, MaterialHandle, MaterialRef};
pub use mesh::{Mesh, MeshHandle, MeshRenderer};
pub use metal_hints::{
    AttachmentOps, DepthAttachmentOps, MetalRenderHints, RenderPassLayout, RenderPassOptimizer,
};
pub use plugin::RenderPlugin;
pub use post_process::{PostProcessSettings, PostProcessStack};
pub use renderer::{DrawCommand, Renderer};
pub use texture::{TextureHandle, TextureStore};
pub use vertex::Vertex;
