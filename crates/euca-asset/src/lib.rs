pub mod ai_gen;
pub mod animation;
pub mod cooked;
pub mod gltf_loader;
pub mod hot_reload;
pub mod loader;
pub mod lod;
pub mod mesh_opt;
pub mod skeleton;
pub mod systems;

pub use animation::{AnimationClipData, AnimationProperty};
pub use gltf_loader::{
    GltfImage, GltfMesh, GltfScene, MeshBounds, apply_texture_handles, load_gltf,
};
pub use hot_reload::FileWatcher;
pub use loader::{AssetHandle, AssetStore, LoadState};
pub use lod::{generate_lod_chain, simplify_mesh};
pub use mesh_opt::{compute_tangents, deduplicate_vertices, optimize_mesh, optimize_vertex_cache};
pub use skeleton::Skeleton;
pub use systems::{AnimationLibrary, BoneTransforms, SkeletalAnimator, skeletal_animation_system};

pub use ai_gen::service::{AssetGeneratedEvent, GenerationService, PendingAsset};
