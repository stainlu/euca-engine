/// Capsule-based kinematic character controller with ground detection and coyote time.
pub mod character;
mod collision;
mod components;
/// Pre-allocated frame cache that eliminates per-tick heap allocations in the physics pipeline.
pub mod frame_cache;
/// GPU-accelerated AABB broadphase via compute shaders (requires `gpu-broadphase` feature).
#[cfg(feature = "gpu-broadphase")]
pub mod gpu_broadphase;
/// Joint constraints (distance, ball-and-socket, revolute) connecting physics bodies.
pub mod joints;
mod raycast;
mod systems;
/// Raycast-based vehicle physics: suspension, tire forces, and drivetrain.
pub mod vehicle;
mod world;

pub use character::{CharacterController, character_controller_system};
pub use collision::{CollisionPair, intersect_aabb};
pub use components::{
    Collider, ColliderShape, CollisionEvent, Gravity, Mass, PhysicsBody, RigidBodyType,
    SLEEP_THRESHOLD, Sleeping, Velocity, layers_interact,
};
pub use frame_cache::PhysicsFrameCache;
pub use joints::{Joint, JointKind};
pub use raycast::{
    OverlapHit, Ray, RayHit, SweepHit, WorldRayHit, overlap_sphere, raycast_aabb, raycast_collider,
    raycast_sphere, raycast_world, sweep_sphere,
};
pub use systems::{physics_step_system, physics_step_with_dt};
pub use vehicle::{
    EngineCurve, TorquePoint, Vehicle, VehicleInput, WheelConfig, WheelState,
    vehicle_physics_system,
};
pub use world::{Joints, PhysicsAccumulator, PhysicsConfig};
