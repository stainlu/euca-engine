//! Core math primitives for the Euca engine: vectors, matrices, quaternions, transforms, and AABBs.

/// Applies `#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "aarch64")))]`
/// to the annotated item. Reduces the 4-line cfg block to a single attribute.
macro_rules! cfg_simd {
    ($($item:item)*) => {
        $(
            #[cfg(all(
                feature = "simd",
                any(target_arch = "x86_64", target_arch = "aarch64")
            ))]
            $item
        )*
    };
}

/// Applies the negated SIMD cfg to the annotated item (scalar fallback path).
macro_rules! cfg_scalar {
    ($($item:item)*) => {
        $(
            #[cfg(not(all(
                feature = "simd",
                any(target_arch = "x86_64", target_arch = "aarch64")
            )))]
            $item
        )*
    };
}

// Make macros available to submodules within this crate.
pub(crate) use cfg_scalar;
pub(crate) use cfg_simd;

mod aabb;
mod mat;
mod quat;
cfg_simd! { mod simd; }
mod transform;
mod vec;

pub use self::aabb::Aabb;
pub use self::mat::Mat4;
pub use self::quat::Quat;
pub use self::transform::Transform;
pub use self::vec::{Vec2, Vec3, Vec4};
