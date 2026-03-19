pub mod animation;
mod gltf_loader;
pub mod skeleton;
pub mod systems;

pub use animation::{AnimationClipData, AnimationProperty};
pub use gltf_loader::{GltfMesh, GltfScene, load_gltf};
pub use skeleton::Skeleton;
pub use systems::{AnimationLibrary, BoneTransforms, SkeletalAnimator, skeletal_animation_system};
