//! Frame cache for physics system heap reuse.
//!
//! The physics system runs 60+ times per second. Every `Vec::new()` hits the
//! allocator. By retaining Vecs across frames (`.clear()` keeps capacity),
//! we eliminate per-tick allocator pressure entirely.

use euca_ecs::Entity;
use euca_math::Vec3;

use crate::components::{Collider, CollisionEvent};
use crate::joints::Joint;
use crate::systems::{Body, DeferredVelocityResponse, Island};

/// Pre-allocated storage retained across physics ticks.
///
/// Each tick calls `.clear_for_tick()` which zeroes lengths but preserves
/// heap capacity. The physics hot path then pushes into these Vecs instead
/// of allocating fresh ones.
#[derive(Default)]
pub struct PhysicsFrameCache {
    // ── apply_gravity ──
    pub(crate) gravity_entities: Vec<Entity>,

    // ── update_sleep_states ──
    pub(crate) sleep_candidates: Vec<(Entity, f32)>,

    // ── integrate_positions ──
    pub(crate) movers: Vec<(Entity, Vec3, Vec3, Vec3, f32)>,
    pub(crate) ccd_statics: Vec<(Entity, Vec3, Collider)>,

    // ── adaptive_cell_size ──
    pub(crate) extents_sample: Vec<f32>,

    // ── broadphase ──
    pub(crate) broadphase_pairs: Vec<(usize, usize)>,
    pub(crate) large_bodies: Vec<usize>,

    // ── resolve_collisions_with_joints ──
    pub(crate) bodies: Vec<Body>,
    pub(crate) events: Vec<CollisionEvent>,
    pub(crate) velocity_responses: Vec<DeferredVelocityResponse>,

    // ── build_islands ──
    pub(crate) islands: Vec<Island>,

    // ── parallel island results ──
    pub(crate) island_events: Vec<Vec<CollisionEvent>>,
    pub(crate) island_responses: Vec<Vec<DeferredVelocityResponse>>,

    // ── joints ──
    /// Cached joints snapshot. Only re-cloned when the Joints resource is
    /// marked dirty (or on first tick).
    pub(crate) cached_joints: Vec<Joint>,
    /// Generation counter from the last time joints were cloned.
    pub(crate) joints_generation: u64,
}

impl PhysicsFrameCache {
    /// Reset all Vecs for the next tick. Capacity is preserved.
    pub fn clear_for_tick(&mut self) {
        self.gravity_entities.clear();
        self.sleep_candidates.clear();
        self.movers.clear();
        self.ccd_statics.clear();
        self.extents_sample.clear();
        self.broadphase_pairs.clear();
        self.large_bodies.clear();
        self.bodies.clear();
        self.events.clear();
        self.velocity_responses.clear();
        // Note: islands are NOT cleared here. `build_islands_into` manages the
        // island vec lifecycle (clearing inner vecs, truncating) to preserve
        // the capacity of each Island's body_indices and pairs vecs.
        // island_events/island_responses are also cleared per-use since their
        // count varies with the number of islands each tick.
    }
}
