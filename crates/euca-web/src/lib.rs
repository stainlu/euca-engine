//! Web/WASM entry point for Euca engine games.
//!
//! This crate provides the glue between the Euca engine and the browser
//! environment. It handles:
//!
//! - WASM initialization and panic hook setup
//! - Canvas element creation or lookup
//! - WebGPU surface binding via wgpu
//! - Event loop via winit's web support (`requestAnimationFrame`)
//! - Input forwarding from DOM events to [`euca_input::InputState`]
//!
//! On native targets this crate is intentionally empty — all browser-specific
//! code lives behind `#[cfg(target_arch = "wasm32")]`.

#[cfg(target_arch = "wasm32")]
mod web_runner;
#[cfg(target_arch = "wasm32")]
pub use web_runner::*;

/// Re-export wgpu for downstream crates that need the same version.
pub use wgpu;
/// Re-export winit for downstream crates that need the same version.
pub use winit;
