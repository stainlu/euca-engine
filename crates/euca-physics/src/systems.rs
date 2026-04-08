use euca_ecs::{Changed, Entity, Query, World};
use euca_math::Vec3;
use euca_scene::LocalTransform;

use crate::collision::intersect_shapes;
use crate::components::*;
use crate::world::PhysicsConfig;

/// Physics system with fixed-timestep accumulation.
///
/// Call with your frame's delta time. Accumulates time and runs fixed-dt
/// substeps as needed. Insert a `PhysicsAccumulator` resource to use this.
/// Falls back to single-step if accumulator is not present.
pub fn physics_step_with_dt(world: &mut World, frame_dt: f32) {
    let config = world
        .resource::<PhysicsConfig>()
        .cloned()
        .unwrap_or_default();

    let accumulator = world
        .resource::<crate::world::PhysicsAccumulator>()
        .map(|a| a.accumulator)
        .unwrap_or(0.0)
        + frame_dt;

    let mut remaining = accumulator;
    let mut steps = 0u32;
    while remaining >= config.fixed_dt && steps < config.max_substeps {
        physics_step_single(world, config.fixed_dt, config.gravity);
        remaining -= config.fixed_dt;
        steps += 1;
    }

    if let Some(acc) = world.resource_mut::<crate::world::PhysicsAccumulator>() {
        acc.accumulator = remaining;
    }
}

/// Main physics system: single fixed-dt step. Use `physics_step_with_dt` for accumulation.
pub fn physics_step_system(world: &mut World) {
    let config = world
        .resource::<PhysicsConfig>()
        .cloned()
        .unwrap_or_default();
    physics_step_single(world, config.fixed_dt, config.gravity);
}

fn physics_step_single(world: &mut World, dt: f32, gravity: Vec3) {
    sync_cached_shapes(world);
    apply_gravity(world, gravity, dt);
    apply_external_forces(world, dt);
    integrate_positions(world, dt);
    resolve_collisions_and_joints(world, dt);
    update_sleep_states(world);
}

/// Synchronise `CachedColliderShape` for every entity whose `Collider` has
/// changed (or was just added). This runs once at the start of each physics
/// step so the solver can read the cached shape without cloning per frame.
pub fn sync_cached_shapes(world: &mut World) {
    let updates: Vec<(Entity, ColliderShape)> = {
        let query = Query::<(Entity, &Collider), Changed<Collider>>::new(world);
        query
            .iter()
            .map(|(e, col)| (e, col.shape.clone()))
            .collect()
    };

    for (entity, shape) in updates {
        world.insert(entity, CachedColliderShape(shape));
    }
}

fn resolve_collisions_and_joints(world: &mut World, dt: f32) {
    // Collect joints (if any)
    let joints = world
        .resource::<crate::world::Joints>()
        .map(|j| j.joints.clone())
        .unwrap_or_default();

    resolve_collisions_with_joints(world, &joints, dt);
}

fn apply_gravity(world: &mut World, gravity: Vec3, dt: f32) {
    let entities: Vec<Entity> = {
        let query = Query::<(Entity, &PhysicsBody, &Velocity)>::new(world);
        query
            .iter()
            .filter(|(_, body, _)| body.body_type == RigidBodyType::Dynamic)
            .map(|(e, _, _)| e)
            .collect()
    };

    for entity in entities {
        // Skip sleeping bodies
        if world.get::<Sleeping>(entity).is_some() {
            continue;
        }
        let g = world.get::<Gravity>(entity).map(|g| g.0).unwrap_or(gravity);
        if let Some(vel) = world.get_mut::<Velocity>(entity) {
            vel.linear = vel.linear + g * dt;
        }
    }
}

/// Apply external forces and torques to dynamic bodies.
///
/// For each entity with `ExternalForce`, `Mass`, and `Velocity`:
///   - Static/kinematic bodies are skipped (their inverse mass is zero anyway).
///   - Sleeping bodies are skipped unless the force or torque is nonzero, in
///     which case the body is woken first.
///   - Linear:  `v += F * inverse_mass * dt`
///   - Angular: `w += T * inverse_inertia * dt`
///   - One-shot forces (`persistent == false`) are zeroed after application.
fn apply_external_forces(world: &mut World, dt: f32) {
    let entities: Vec<Entity> = {
        let query = Query::<(Entity, &PhysicsBody, &ExternalForce)>::new(world);
        query
            .iter()
            .filter(|(_, body, _)| body.body_type == RigidBodyType::Dynamic)
            .map(|(e, _, _)| e)
            .collect()
    };

    for entity in entities {
        // Single immutable read to extract all needed fields.
        let (force, torque, persistent) = match world.get::<ExternalForce>(entity) {
            Some(ef) => (ef.force, ef.torque, ef.persistent),
            None => continue,
        };

        let has_force = force.length_squared() > 0.0 || torque.length_squared() > 0.0;

        // Skip sleeping bodies unless external force/torque is nonzero.
        if world.get::<Sleeping>(entity).is_some() {
            if !has_force {
                continue;
            }
            world.remove::<Sleeping>(entity);
        }

        let (inv_mass, inv_inertia) = match world.get::<Mass>(entity) {
            Some(m) => (m.inverse_mass, m.inverse_inertia),
            None => continue,
        };

        if let Some(vel) = world.get_mut::<Velocity>(entity) {
            vel.linear = vel.linear + force * inv_mass * dt;
            vel.angular = vel.angular + torque * inv_inertia * dt;
        }

        if !persistent
            && let Some(ef) = world.get_mut::<ExternalForce>(entity)
        {
            ef.force = Vec3::ZERO;
            ef.torque = Vec3::ZERO;
        }
    }
}

/// Put slow bodies to sleep, wake bodies involved in collisions.
fn update_sleep_states(world: &mut World) {
    let candidates: Vec<(Entity, f32)> = {
        let query = Query::<(Entity, &PhysicsBody, &Velocity)>::new(world);
        query
            .iter()
            .filter(|(_, body, _)| body.body_type == RigidBodyType::Dynamic)
            .map(|(e, _, vel)| {
                (
                    e,
                    vel.linear.length_squared() + vel.angular.length_squared(),
                )
            })
            .collect()
    };

    for (entity, speed_sq) in candidates {
        if speed_sq < SLEEP_THRESHOLD * SLEEP_THRESHOLD {
            // Put to sleep if not already
            if world.get::<Sleeping>(entity).is_none() {
                world.insert(entity, Sleeping);
                // Zero out velocity to prevent drift
                if let Some(vel) = world.get_mut::<Velocity>(entity) {
                    vel.linear = Vec3::ZERO;
                    vel.angular = Vec3::ZERO;
                }
            }
        } else {
            // Wake up if sleeping
            world.remove::<Sleeping>(entity);
        }
    }
}

fn integrate_positions(world: &mut World, dt: f32) {
    use crate::raycast::{Ray, raycast_collider};

    // Collect movers: entity, old position, linear vel, angular vel, collider extent
    // Collider is optional — entities without colliders still move, just skip CCD.
    let movers: Vec<(Entity, Vec3, Vec3, Vec3, f32)> = {
        let query = Query::<(Entity, &PhysicsBody, &Velocity, &LocalTransform)>::new(world);
        query
            .iter()
            .filter(|(_, body, _, _)| body.body_type != RigidBodyType::Static)
            .map(|(e, _, vel, lt)| {
                let extent = world
                    .get::<Collider>(e)
                    .map(|c| shape_extent(&c.shape))
                    .unwrap_or(0.0);
                (e, lt.0.translation, vel.linear, vel.angular, extent)
            })
            .collect()
    };

    // Check if ANY mover needs CCD before building the expensive static grid.
    let any_needs_ccd = movers.iter().any(|(entity, _, linear_vel, _, extent)| {
        let speed = (*linear_vel * dt).length();
        let is_dynamic = world
            .get::<PhysicsBody>(*entity)
            .is_some_and(|b| b.body_type == RigidBodyType::Dynamic);
        is_dynamic && speed > extent * 0.5 && speed > 1e-6
    });

    // Only build the CCD spatial grid if at least one mover is fast enough.
    // This avoids O(cells) overhead from large static bodies (ground planes)
    // when all entities are moving slowly.
    let statics: Vec<(Entity, Vec3, Collider)>;
    let ccd_grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>>;
    let ccd_inv_cell: f32;

    if any_needs_ccd {
        statics = {
            let query = Query::<(Entity, &LocalTransform, &Collider, &PhysicsBody)>::new(world);
            query
                .iter()
                .filter(|(_, _, _, body)| body.body_type != RigidBodyType::Dynamic)
                .map(|(e, lt, col, _)| (e, lt.0.translation, col.clone()))
                .collect()
        };

        let ccd_cell_size = DEFAULT_CELL_SIZE;
        ccd_inv_cell = 1.0 / ccd_cell_size;
        let mut grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
            std::collections::HashMap::with_capacity(statics.len());
        for (idx, (_, pos, col)) in statics.iter().enumerate() {
            let (hx, hy, hz) = shape_half_extents(&col.shape);
            let min_x = ((pos.x - hx) / ccd_cell_size).floor() as i32;
            let max_x = ((pos.x + hx) / ccd_cell_size).floor() as i32;
            let min_y = ((pos.y - hy) / ccd_cell_size).floor() as i32;
            let max_y = ((pos.y + hy) / ccd_cell_size).floor() as i32;
            let min_z = ((pos.z - hz) / ccd_cell_size).floor() as i32;
            let max_z = ((pos.z + hz) / ccd_cell_size).floor() as i32;
            // Cap cells per body to avoid grid flooding from large statics.
            let span = (max_x - min_x + 1).max(1) as i64
                * (max_y - min_y + 1).max(1) as i64
                * (max_z - min_z + 1).max(1) as i64;
            if span > (MAX_CELLS_PER_BODY as i64).pow(3) {
                continue; // Skip; CCD will fall back to brute-force for this static
            }
            for cx in min_x..=max_x {
                for cy in min_y..=max_y {
                    for cz in min_z..=max_z {
                        grid.entry((cx, cy, cz)).or_default().push(idx);
                    }
                }
            }
        }
        ccd_grid = grid;
    } else {
        statics = Vec::new();
        ccd_grid = std::collections::HashMap::new();
        ccd_inv_cell = 1.0 / DEFAULT_CELL_SIZE;
    }

    for (entity, old_pos, linear_vel, angular_vel, extent) in movers {
        let displacement = linear_vel * dt;
        let mut new_pos = old_pos + displacement;

        // Apply angular velocity to rotation
        if angular_vel.length_squared() > 1e-8 {
            let angle = angular_vel.length() * dt;
            let axis = angular_vel * (1.0 / angular_vel.length());
            if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
                let delta_rot = euca_math::Quat::from_axis_angle(axis, angle);
                lt.0.rotation = delta_rot * lt.0.rotation;
            }
        }

        // CCD: only for Dynamic bodies (Kinematic skip collision entirely)
        let is_dynamic = world
            .get::<PhysicsBody>(entity)
            .is_some_and(|b| b.body_type == RigidBodyType::Dynamic);
        let speed = displacement.length();
        if is_dynamic && speed > extent * 0.5 && speed > 1e-6 {
            let ray = Ray::new(old_pos, displacement);
            let mut closest_t = 1.0_f32; // 1.0 = full displacement

            // Query only statics in cells overlapping the swept AABB
            // (from old_pos to new_pos, expanded by body extent).
            let swept_min = Vec3::new(
                old_pos.x.min(new_pos.x) - extent,
                old_pos.y.min(new_pos.y) - extent,
                old_pos.z.min(new_pos.z) - extent,
            );
            let swept_max = Vec3::new(
                old_pos.x.max(new_pos.x) + extent,
                old_pos.y.max(new_pos.y) + extent,
                old_pos.z.max(new_pos.z) + extent,
            );
            let cell_min_x = (swept_min.x * ccd_inv_cell).floor() as i32;
            let cell_max_x = (swept_max.x * ccd_inv_cell).floor() as i32;
            let cell_min_y = (swept_min.y * ccd_inv_cell).floor() as i32;
            let cell_max_y = (swept_max.y * ccd_inv_cell).floor() as i32;
            let cell_min_z = (swept_min.z * ccd_inv_cell).floor() as i32;
            let cell_max_z = (swept_max.z * ccd_inv_cell).floor() as i32;

            // Collect candidate static indices (deduplicate via a small set).
            let mut tested = std::collections::HashSet::new();
            for cx in cell_min_x..=cell_max_x {
                for cy in cell_min_y..=cell_max_y {
                    for cz in cell_min_z..=cell_max_z {
                        if let Some(indices) = ccd_grid.get(&(cx, cy, cz)) {
                            for &si in indices {
                                if !tested.insert(si) {
                                    continue;
                                }
                                let (static_e, static_pos, static_col) = &statics[si];
                                if *static_e == entity {
                                    continue;
                                }
                                if let Some(hit) = raycast_collider(&ray, *static_pos, static_col) {
                                    let t_normalized = hit.t / speed;
                                    if t_normalized < closest_t && t_normalized >= 0.0 {
                                        closest_t = t_normalized;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if closest_t < 1.0 {
                // Clamp position to just before the hit
                let safe_t = (closest_t - 0.01).max(0.0);
                new_pos = old_pos + displacement * safe_t;

                // Dampen velocity on impact
                if let Some(vel) = world.get_mut::<Velocity>(entity) {
                    vel.linear = vel.linear * 0.1;
                }
            }
        }

        if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
            lt.0.translation = new_pos;
        }
    }
}

/// Minimum approach speed for a bounce to occur. Below this, the object comes to rest.
const REST_VELOCITY_THRESHOLD: f32 = 0.5;

/// Default spatial hash cell size when there are too few bodies to compute
/// a meaningful adaptive size.
const DEFAULT_CELL_SIZE: f32 = 4.0;

/// Compute an adaptive broadphase cell size from the body population.
/// Uses 2x the approximate median body extent, clamped to [1.0, 32.0].
/// Samples every 64th body to keep cost O(1) relative to body count.
fn adaptive_cell_size(bodies: &[Body]) -> f32 {
    if bodies.len() < 20 {
        return DEFAULT_CELL_SIZE;
    }
    let step = (bodies.len() / 64).max(1);
    let mut extents: Vec<f32> = bodies
        .iter()
        .step_by(step)
        .map(|b| shape_extent(&b.shape))
        .collect();
    extents.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = extents[extents.len() / 2];
    (median * 2.0).clamp(1.0, 32.0)
}

/// Compute the max AABB half-extent for any collider shape.
fn shape_extent(shape: &ColliderShape) -> f32 {
    match shape {
        ColliderShape::Aabb { hx, hy, hz } => hx.max(*hy).max(*hz),
        ColliderShape::Sphere { radius } => *radius,
        ColliderShape::Capsule {
            radius,
            half_height,
        } => radius + half_height,
    }
}

/// Compute per-axis AABB half-extents for a collider shape.
fn shape_half_extents(shape: &ColliderShape) -> (f32, f32, f32) {
    match shape {
        ColliderShape::Aabb { hx, hy, hz } => (*hx, *hy, *hz),
        ColliderShape::Sphere { radius } => (*radius, *radius, *radius),
        ColliderShape::Capsule {
            radius,
            half_height,
        } => (*radius, radius + half_height, *radius),
    }
}

/// Collectable body data for broadphase + narrowphase.
struct Body {
    entity: Entity,
    pos: Vec3,
    shape: ColliderShape,
    body_type: RigidBodyType,
    restitution: f32,
    friction: f32,
    layer: u32,
    mask: u32,
    inverse_mass: f32,
    /// Diagonal of the inverse inertia tensor in body frame.
    inverse_inertia_tensor: Vec3,
}

/// Spatial hash broadphase: returns candidate pairs (indices into bodies slice).
/// Only pairs sharing at least one grid cell are returned. Eliminates most
/// non-colliding pairs for O(n * avg_neighbors) instead of O(n^2).
/// Maximum number of cells a single body can occupy in the spatial hash.
/// Bodies larger than this (e.g. terrain planes) are tracked separately and
/// tested against all other bodies, avoiding grid flooding.
const MAX_CELLS_PER_BODY: i32 = 8;

fn broadphase_spatial_hash(bodies: &[Body], cell_size: f32) -> Vec<(usize, usize)> {
    use std::collections::HashMap;

    if bodies.len() < 20 {
        let mut pairs = Vec::new();
        for i in 0..bodies.len() {
            for j in (i + 1)..bodies.len() {
                pairs.push((i, j));
            }
        }
        return pairs;
    }

    let inv_cell = 1.0 / cell_size;

    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::with_capacity(bodies.len());

    // Bodies whose AABB spans too many cells are tested against everyone.
    let mut large_bodies: Vec<usize> = Vec::new();

    for (idx, body) in bodies.iter().enumerate() {
        let ext = shape_extent(&body.shape);
        let min_x = ((body.pos.x - ext) * inv_cell).floor() as i32;
        let max_x = ((body.pos.x + ext) * inv_cell).floor() as i32;
        let min_y = ((body.pos.y - ext) * inv_cell).floor() as i32;
        let max_y = ((body.pos.y + ext) * inv_cell).floor() as i32;
        let min_z = ((body.pos.z - ext) * inv_cell).floor() as i32;
        let max_z = ((body.pos.z + ext) * inv_cell).floor() as i32;

        let span_x = max_x - min_x + 1;
        let span_y = max_y - min_y + 1;
        let span_z = max_z - min_z + 1;

        if span_x > MAX_CELLS_PER_BODY || span_y > MAX_CELLS_PER_BODY || span_z > MAX_CELLS_PER_BODY
        {
            large_bodies.push(idx);
            continue;
        }

        for cx in min_x..=max_x {
            for cy in min_y..=max_y {
                for cz in min_z..=max_z {
                    grid.entry((cx, cy, cz)).or_default().push(idx);
                }
            }
        }
    }

    // Collect unique pairs from grid cells.
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for cell_bodies in grid.values() {
        for i in 0..cell_bodies.len() {
            for j in (i + 1)..cell_bodies.len() {
                let a = cell_bodies[i];
                let b = cell_bodies[j];
                let pair = if a < b { (a, b) } else { (b, a) };
                pairs.push(pair);
            }
        }
    }

    // Large bodies generate pairs with every other body (per-axis AABB check).
    for &li in &large_bodies {
        let (lhx, lhy, lhz) = shape_half_extents(&bodies[li].shape);
        for (oi, other) in bodies.iter().enumerate() {
            if oi == li {
                continue;
            }
            let (ohx, ohy, ohz) = shape_half_extents(&other.shape);
            let dx = (bodies[li].pos.x - other.pos.x).abs();
            let dy = (bodies[li].pos.y - other.pos.y).abs();
            let dz = (bodies[li].pos.z - other.pos.z).abs();
            if dx <= lhx + ohx && dy <= lhy + ohy && dz <= lhz + ohz {
                let pair = if li < oi { (li, oi) } else { (oi, li) };
                pairs.push(pair);
            }
        }
    }

    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Number of constraint solver iterations. More = more stable stacking.
const SOLVER_ITERATIONS: usize = 4;

/// Minimum island size to justify rayon overhead. Smaller islands run inline.
const PARALLEL_ISLAND_THRESHOLD: usize = 64;

// ───────────────────────────── Union-Find ─────────────────────────────

/// Disjoint-set (Union-Find) with path compression and union-by-rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        // Union by rank
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }
}

// ───────────────────────────── Island ─────────────────────────────

/// A connected component of interacting bodies.
struct Island {
    /// Indices into the parent `bodies` slice.
    body_indices: Vec<usize>,
    /// Pairs as (local_i, local_j) into `body_indices`.
    pairs: Vec<(usize, usize)>,
}

/// Build islands from bodies and broadphase pairs using union-find.
fn build_islands(bodies: &[Body], candidate_pairs: &[(usize, usize)]) -> Vec<Island> {
    let n = bodies.len();
    if n == 0 {
        return Vec::new();
    }

    let mut uf = UnionFind::new(n);

    // Union only dynamic-dynamic (or kinematic) pairs. Static bodies are
    // immovable, so two dynamics touching the same static wall are independent
    // — they should NOT be merged into one island.
    for &(i, j) in candidate_pairs {
        let i_static = bodies[i].body_type == RigidBodyType::Static;
        let j_static = bodies[j].body_type == RigidBodyType::Static;
        if i_static || j_static {
            continue; // Don't union through statics
        }
        uf.union(i, j);
    }

    // Build islands from dynamic bodies only. Static bodies are referenced
    // via pairs but don't belong to islands themselves.
    let mut root_to_island: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    let mut islands: Vec<Island> = Vec::new();

    // Map: global body index → (island_index, local_index).
    // Static bodies get a sentinel value and are added to islands on demand.
    const NO_ISLAND: (usize, usize) = (usize::MAX, usize::MAX);
    let mut global_to_local: Vec<(usize, usize)> = vec![NO_ISLAND; n];

    // First pass: assign dynamic/kinematic bodies to islands.
    for idx in 0..n {
        if bodies[idx].body_type == RigidBodyType::Static {
            continue;
        }
        let root = uf.find(idx);
        let island_idx = if let Some(&existing) = root_to_island.get(&root) {
            existing
        } else {
            let new_idx = islands.len();
            root_to_island.insert(root, new_idx);
            islands.push(Island {
                body_indices: Vec::new(),
                pairs: Vec::new(),
            });
            new_idx
        };
        let local_idx = islands[island_idx].body_indices.len();
        islands[island_idx].body_indices.push(idx);
        global_to_local[idx] = (island_idx, local_idx);
    }

    // Distribute pairs to islands. For dynamic-static pairs, the pair goes
    // to the dynamic body's island, and the static body is added to that
    // island if not already present.
    for &(i, j) in candidate_pairs {
        let i_static = bodies[i].body_type == RigidBodyType::Static;
        let j_static = bodies[j].body_type == RigidBodyType::Static;
        if i_static && j_static {
            continue;
        }

        // Determine which island this pair belongs to (from the dynamic body).
        let island_idx = if !i_static {
            global_to_local[i].0
        } else {
            global_to_local[j].0
        };
        if island_idx == usize::MAX {
            continue; // Shouldn't happen, but guard against it.
        }

        // Ensure both bodies have local indices in this island.
        // Static bodies may appear in multiple islands (one per dynamic neighbor).
        let ensure_local =
            |idx: usize, islands: &mut Vec<Island>, g2l: &mut Vec<(usize, usize)>| -> usize {
                let (existing_island, existing_local) = g2l[idx];
                if existing_island == island_idx {
                    return existing_local;
                }
                // Add body to this island (static bodies can appear in multiple).
                let local = islands[island_idx].body_indices.len();
                islands[island_idx].body_indices.push(idx);
                if existing_island == usize::MAX {
                    // First island assignment for this body.
                    g2l[idx] = (island_idx, local);
                }
                // Note: for statics in multiple islands, g2l only tracks the first.
                // We return the local index directly.
                local
            };

        let local_i = ensure_local(i, &mut islands, &mut global_to_local);
        let local_j = ensure_local(j, &mut islands, &mut global_to_local);
        islands[island_idx].pairs.push((local_i, local_j));
    }

    // Remove empty islands (dynamics with no collision pairs).
    islands.retain(|island| !island.pairs.is_empty());

    islands
}

/// Per-island collision event: stores global body data needed for velocity response.
struct DeferredVelocityResponse {
    entity_a: Entity,
    type_a: RigidBodyType,
    entity_b: Entity,
    type_b: RigidBodyType,
    normal: Vec3,
    restitution: f32,
    friction: f32,
    inv_mass_a: f32,
    inv_mass_b: f32,
    /// Diagonal of inverse inertia tensor for body A.
    inv_inertia_a: Vec3,
    /// Diagonal of inverse inertia tensor for body B.
    inv_inertia_b: Vec3,
    /// World-space contact point.
    contact_point: Vec3,
    /// Center of body A at collision time.
    pos_a: Vec3,
    /// Center of body B at collision time.
    pos_b: Vec3,
}

/// Solve a single island's position constraints. Returns collision events and
/// deferred velocity responses (which need `&mut World` access).
fn solve_island(
    bodies: &mut [Body],
    island: &Island,
    events: &mut Vec<CollisionEvent>,
    velocity_responses: &mut Vec<DeferredVelocityResponse>,
) {
    for iteration in 0..SOLVER_ITERATIONS {
        for &(li, lj) in &island.pairs {
            let gi = island.body_indices[li];
            let gj = island.body_indices[lj];

            // Layer/mask filtering
            if !layers_interact(
                bodies[gi].layer,
                bodies[gi].mask,
                bodies[gj].layer,
                bodies[gj].mask,
            ) {
                continue;
            }

            if let Some((normal, depth, contact_point)) = intersect_shapes(
                bodies[gi].pos,
                &bodies[gi].shape,
                bodies[gj].pos,
                &bodies[gj].shape,
            ) {
                // Emit collision event on the first iteration only
                if iteration == 0 {
                    events.push(CollisionEvent {
                        entity_a: bodies[gi].entity,
                        entity_b: bodies[gj].entity,
                        normal,
                        penetration: depth,
                        contact_point,
                    });
                }

                // Mass-weighted position correction
                let inv_mass_a = bodies[gi].inverse_mass;
                let inv_mass_b = bodies[gj].inverse_mass;
                let total_inv_mass = inv_mass_a + inv_mass_b;

                if total_inv_mass > 0.0 {
                    let ratio_a = inv_mass_a / total_inv_mass;
                    let ratio_b = inv_mass_b / total_inv_mass;
                    bodies[gi].pos = bodies[gi].pos + normal * (-depth * ratio_a);
                    bodies[gj].pos = bodies[gj].pos + normal * (depth * ratio_b);
                }

                // Defer velocity response to after parallel solve (needs &mut World)
                if iteration == SOLVER_ITERATIONS - 1 {
                    velocity_responses.push(DeferredVelocityResponse {
                        entity_a: bodies[gi].entity,
                        type_a: bodies[gi].body_type,
                        entity_b: bodies[gj].entity,
                        type_b: bodies[gj].body_type,
                        normal,
                        restitution: bodies[gi].restitution * bodies[gj].restitution,
                        friction: (bodies[gi].friction * bodies[gj].friction).sqrt(),
                        inv_mass_a,
                        inv_mass_b,
                        inv_inertia_a: bodies[gi].inverse_inertia_tensor,
                        inv_inertia_b: bodies[gj].inverse_inertia_tensor,
                        contact_point,
                        pos_a: bodies[gi].pos,
                        pos_b: bodies[gj].pos,
                    });
                }
            }
        }
    }
}

fn resolve_collisions_with_joints(world: &mut World, joints: &[crate::joints::Joint], dt: f32) {
    // ── Iterative constraint solver ──
    // Collect bodies once, iterate position corrections in-place,
    // write back to world at the end.

    let mut bodies: Vec<Body> = {
        let query = Query::<(Entity, &LocalTransform, &Collider, &CachedColliderShape)>::new(world);
        query
            .iter()
            .filter_map(|(e, lt, col, cached)| {
                let body = world.get::<PhysicsBody>(e)?;
                // Skip sleeping bodies in the solver entirely
                if world.get::<Sleeping>(e).is_some() {
                    return None;
                }
                let mass_comp = world.get::<Mass>(e);
                let inv_mass = mass_comp.map(|m| m.inverse_mass).unwrap_or_else(|| {
                    if body.body_type == RigidBodyType::Dynamic {
                        1.0 // default: 1 kg
                    } else {
                        0.0 // static/kinematic: immovable
                    }
                });
                let inv_inertia_tensor = mass_comp
                    .map(|m| m.inverse_inertia_tensor)
                    .unwrap_or_else(|| {
                        if body.body_type == RigidBodyType::Dynamic {
                            // Default: unit sphere-like inertia (1/6 for a unit cube)
                            Vec3::new(6.0, 6.0, 6.0)
                        } else {
                            Vec3::ZERO
                        }
                    });
                Some(Body {
                    entity: e,
                    pos: lt.0.translation,
                    shape: cached.0.clone(),
                    body_type: body.body_type,
                    restitution: col.restitution,
                    friction: col.friction,
                    layer: col.layer,
                    mask: col.mask,
                    inverse_mass: inv_mass,
                    inverse_inertia_tensor: inv_inertia_tensor,
                })
            })
            .collect()
    };

    // Compute broadphase once.
    let cell_size = adaptive_cell_size(&bodies);
    let candidate_pairs = broadphase_spatial_hash(&bodies, cell_size);

    // Build islands (connected components of interacting bodies).
    let islands = build_islands(&bodies, &candidate_pairs);

    // Count total active bodies across all non-trivial islands to decide
    // whether parallel dispatch is worthwhile.
    let total_active: usize = islands.iter().map(|isl| isl.body_indices.len()).sum();

    // Solve islands — in parallel if enough work to justify it.
    let mut all_events: Vec<CollisionEvent> = Vec::new();
    let mut all_velocity_responses: Vec<DeferredVelocityResponse> = Vec::new();

    if total_active >= PARALLEL_ISLAND_THRESHOLD && islands.len() > 1 {
        // Solve islands concurrently using rayon::in_place_scope.
        // Safety: each island operates on disjoint body indices, so no two
        // islands write the same body position.
        /// Wrapper that makes a `*mut [Body]` fat pointer `Send + Sync`.
        /// Safety: caller ensures no two threads write the same index.
        struct SendSlice {
            ptr: *mut Body,
            len: usize,
        }
        unsafe impl Send for SendSlice {}
        unsafe impl Sync for SendSlice {}

        impl SendSlice {
            /// Reconstruct the mutable slice. Caller must ensure exclusive access
            /// to the indices they touch. Multiple callers may hold overlapping
            /// slices as long as they access disjoint index ranges (island solver).
            #[allow(clippy::mut_from_ref)]
            unsafe fn get(&self) -> &mut [Body] {
                unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
            }
        }

        let shared = SendSlice {
            ptr: bodies.as_mut_ptr(),
            len: bodies.len(),
        };

        // Pre-allocate per-island result storage.
        let mut island_results: Vec<(Vec<CollisionEvent>, Vec<DeferredVelocityResponse>)> =
            islands.iter().map(|_| (Vec::new(), Vec::new())).collect();

        rayon::in_place_scope(|s| {
            for (island, result) in islands.iter().zip(island_results.iter_mut()) {
                let (ref mut events, ref mut responses) = *result;
                s.spawn(|_| {
                    // Safety: islands have disjoint body_indices, so no two
                    // tasks write the same Body entry.
                    let bodies_slice = unsafe { shared.get() };
                    solve_island(bodies_slice, island, events, responses);
                });
            }
        });

        for (events, responses) in island_results {
            all_events.extend(events);
            all_velocity_responses.extend(responses);
        }
    } else {
        // Sequential: solve all islands inline.
        for island in &islands {
            solve_island(
                &mut bodies,
                island,
                &mut all_events,
                &mut all_velocity_responses,
            );
        }
    }

    // Apply deferred velocity responses (needs &mut World).
    for resp in &all_velocity_responses {
        apply_velocity_response(world, resp);
    }

    // ── Solve joint constraints (using body positions from the solver) ──
    if !joints.is_empty() {
        // Build entity -> body index map for fast lookup
        let entity_to_idx: std::collections::HashMap<Entity, usize> = bodies
            .iter()
            .enumerate()
            .map(|(i, b)| (b.entity, i))
            .collect();

        // Read current rotations for joint entities (needed for angle limits).
        let entity_rotations: std::collections::HashMap<Entity, euca_math::Quat> = joints
            .iter()
            .flat_map(|j| [j.entity_a, j.entity_b])
            .filter_map(|e| world.get::<LocalTransform>(e).map(|lt| (e, lt.0.rotation)))
            .collect();

        // Position solving iterations.
        for _iter in 0..SOLVER_ITERATIONS {
            for joint in joints {
                let idx_a = entity_to_idx.get(&joint.entity_a).copied();
                let idx_b = entity_to_idx.get(&joint.entity_b).copied();

                let (pos_a, is_a_dyn) = match idx_a {
                    Some(i) => (bodies[i].pos, bodies[i].body_type == RigidBodyType::Dynamic),
                    None => continue,
                };
                let (pos_b, is_b_dyn) = match idx_b {
                    Some(i) => (bodies[i].pos, bodies[i].body_type == RigidBodyType::Dynamic),
                    None => continue,
                };

                let rot_a = entity_rotations
                    .get(&joint.entity_a)
                    .copied()
                    .unwrap_or(euca_math::Quat::IDENTITY);
                let rot_b = entity_rotations
                    .get(&joint.entity_b)
                    .copied()
                    .unwrap_or(euca_math::Quat::IDENTITY);

                let (ca, cb) = joint.solve(pos_a, pos_b, rot_a, rot_b, is_a_dyn, is_b_dyn);

                if let Some(i) = idx_a {
                    bodies[i].pos = bodies[i].pos + ca;
                }
                if let Some(i) = idx_b {
                    bodies[i].pos = bodies[i].pos + cb;
                }
            }
        }

        // Velocity solving: apply motor impulses (skip joints without motors).
        for joint in joints.iter().filter(|j| j.has_motor()) {
            let idx_a = entity_to_idx.get(&joint.entity_a).copied();
            let idx_b = entity_to_idx.get(&joint.entity_b).copied();

            let is_a_dyn = idx_a
                .map(|i| bodies[i].body_type == RigidBodyType::Dynamic)
                .unwrap_or(false);
            let is_b_dyn = idx_b
                .map(|i| bodies[i].body_type == RigidBodyType::Dynamic)
                .unwrap_or(false);

            let ang_vel_a = world
                .get::<Velocity>(joint.entity_a)
                .map(|v| v.angular)
                .unwrap_or(Vec3::ZERO);
            let ang_vel_b = world
                .get::<Velocity>(joint.entity_b)
                .map(|v| v.angular)
                .unwrap_or(Vec3::ZERO);

            let (imp_a, imp_b) = joint.solve_velocity(ang_vel_a, ang_vel_b, is_a_dyn, is_b_dyn, dt);

            if imp_a.length_squared() > 1e-12
                && let Some(vel) = world.get_mut::<Velocity>(joint.entity_a)
            {
                vel.angular = vel.angular + imp_a;
            }
            if imp_b.length_squared() > 1e-12
                && let Some(vel) = world.get_mut::<Velocity>(joint.entity_b)
            {
                vel.angular = vel.angular + imp_b;
            }
        }
    }

    // Write solved positions back to world
    for body in &bodies {
        if body.body_type == RigidBodyType::Dynamic
            && let Some(lt) = world.get_mut::<LocalTransform>(body.entity)
        {
            lt.0.translation = body.pos;
        }
    }

    // Emit collision events
    for event in all_events {
        world.send_event(event);
    }
}

/// Component-wise multiply: `Vec3(a.x*b.x, a.y*b.y, a.z*b.z)`.
/// Used for applying diagonal inverse inertia tensors.
fn vec3_comp_mul(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x * b.x, a.y * b.y, a.z * b.z)
}

/// Apply impulse-based velocity response between two colliding bodies.
///
/// Uses the standard rigid-body contact impulse formula including angular
/// terms: off-center contacts generate torque, and the effective mass
/// denominator accounts for rotational inertia.
fn apply_velocity_response(world: &mut World, resp: &DeferredVelocityResponse) {
    let inv_mass_a = resp.inv_mass_a;
    let inv_mass_b = resp.inv_mass_b;
    let total_inv_mass = inv_mass_a + inv_mass_b;
    if total_inv_mass < 1e-12 {
        return; // Both immovable
    }

    let normal = resp.normal;
    let inv_i_a = resp.inv_inertia_a;
    let inv_i_b = resp.inv_inertia_b;

    // Lever arms from body centers to contact point
    let r_a = resp.contact_point - resp.pos_a;
    let r_b = resp.contact_point - resp.pos_b;

    // Read full velocities (linear + angular)
    let (lin_a, ang_a) = if resp.type_a == RigidBodyType::Dynamic {
        world
            .get::<Velocity>(resp.entity_a)
            .map(|v| (v.linear, v.angular))
            .unwrap_or((Vec3::ZERO, Vec3::ZERO))
    } else {
        (Vec3::ZERO, Vec3::ZERO)
    };
    let (lin_b, ang_b) = if resp.type_b == RigidBodyType::Dynamic {
        world
            .get::<Velocity>(resp.entity_b)
            .map(|v| (v.linear, v.angular))
            .unwrap_or((Vec3::ZERO, Vec3::ZERO))
    } else {
        (Vec3::ZERO, Vec3::ZERO)
    };

    // Velocity at contact point = linear + angular x r
    let v_a_contact = lin_a + ang_a.cross(r_a);
    let v_b_contact = lin_b + ang_b.cross(r_b);
    let v_rel = v_b_contact - v_a_contact;
    let v_rel_n = v_rel.dot(normal);

    // Only resolve if bodies are approaching at the contact
    if v_rel_n >= 0.0 {
        return;
    }

    let approach_speed = -v_rel_n;
    let bounce_factor = if approach_speed < REST_VELOCITY_THRESHOLD {
        0.0
    } else {
        resp.restitution
    };

    // Impulse denominator includes rotational terms:
    //   denom = 1/m_a + 1/m_b + n . ((I_a^-1 * (r_a x n)) x r_a)
    //                               + n . ((I_b^-1 * (r_b x n)) x r_b)
    let rn_a = r_a.cross(normal);
    let rn_b = r_b.cross(normal);
    let denom = total_inv_mass
        + normal.dot(vec3_comp_mul(inv_i_a, rn_a).cross(r_a))
        + normal.dot(vec3_comp_mul(inv_i_b, rn_b).cross(r_b));

    if denom < 1e-12 {
        return;
    }

    // Normal impulse magnitude
    let j = -(1.0 + bounce_factor) * v_rel_n / denom;
    let impulse = normal * j;

    // Friction impulse (tangential)
    let v_tangent = v_rel - normal * v_rel_n;
    let tangent_speed_sq = v_tangent.length_squared();
    let friction_impulse = if tangent_speed_sq > 1e-12 {
        let tangent = v_tangent * (1.0 / tangent_speed_sq.sqrt());
        let jt = (-v_rel.dot(tangent) / denom).clamp(-j * resp.friction, j * resp.friction);
        tangent * jt
    } else {
        Vec3::ZERO
    };

    // Total impulse (normal + friction) applied once for both linear and angular.
    let total_impulse = impulse + friction_impulse;

    if resp.type_a == RigidBodyType::Dynamic
        && let Some(vel) = world.get_mut::<Velocity>(resp.entity_a)
    {
        vel.linear = vel.linear - total_impulse * inv_mass_a;
        let torque_a = r_a.cross(total_impulse);
        vel.angular = vel.angular - vec3_comp_mul(inv_i_a, torque_a);
    }
    if resp.type_b == RigidBodyType::Dynamic
        && let Some(vel) = world.get_mut::<Velocity>(resp.entity_b)
    {
        vel.linear = vel.linear + total_impulse * inv_mass_b;
        let torque_b = r_b.cross(total_impulse);
        vel.angular = vel.angular + vec3_comp_mul(inv_i_b, torque_b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Transform;
    use euca_scene::GlobalTransform;

    #[test]
    fn gravity_moves_dynamic_body() {
        let mut world = World::new();
        world.insert_resource(PhysicsConfig::new());

        let entity = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, 10.0, 0.0,
        ))));
        world.insert(entity, GlobalTransform::default());
        world.insert(entity, PhysicsBody::dynamic());
        world.insert(entity, Velocity::default());
        world.insert(entity, Collider::aabb(0.5, 0.5, 0.5));

        for _ in 0..120 {
            physics_step_system(&mut world);
        }

        let lt = world.get::<LocalTransform>(entity).unwrap();
        assert!(
            lt.0.translation.y < 0.0,
            "Body should have fallen past origin, y={}",
            lt.0.translation.y
        );
    }

    #[test]
    fn static_body_does_not_move() {
        let mut world = World::new();
        world.insert_resource(PhysicsConfig::new());

        let entity = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(entity, GlobalTransform::default());
        world.insert(entity, PhysicsBody::fixed());
        world.insert(entity, Collider::aabb(10.0, 0.5, 10.0));

        for _ in 0..60 {
            physics_step_system(&mut world);
        }

        let lt = world.get::<LocalTransform>(entity).unwrap();
        assert!((lt.0.translation.y).abs() < 0.01);
    }

    #[test]
    fn dynamic_body_lands_on_static() {
        let mut world = World::new();
        world.insert_resource(PhysicsConfig::new());

        // Ground at y=0
        let ground = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(ground, GlobalTransform::default());
        world.insert(ground, PhysicsBody::fixed());
        world.insert(ground, Collider::aabb(10.0, 0.5, 10.0));

        // Cube at y=5
        let cube = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, 5.0, 0.0,
        ))));
        world.insert(cube, GlobalTransform::default());
        world.insert(cube, PhysicsBody::dynamic());
        world.insert(cube, Velocity::default());
        world.insert(cube, Collider::aabb(0.5, 0.5, 0.5));

        for _ in 0..300 {
            physics_step_system(&mut world);
        }

        let lt = world.get::<LocalTransform>(cube).unwrap();
        assert!(
            lt.0.translation.y > -1.0,
            "Cube should be near ground, y={}",
            lt.0.translation.y
        );
        assert!(
            lt.0.translation.y < 5.0,
            "Cube should have fallen, y={}",
            lt.0.translation.y
        );
    }

    #[test]
    fn stacking_stability() {
        // Three cubes stacked on a static ground. With iterative solver,
        // they should settle without exploding or falling through.
        let mut world = World::new();
        world.insert_resource(PhysicsConfig::new());

        // Ground at y=0
        let ground = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(ground, GlobalTransform::default());
        world.insert(ground, PhysicsBody::fixed());
        world.insert(ground, Collider::aabb(10.0, 0.5, 10.0));

        // Stack: cube1 at y=1, cube2 at y=2, cube3 at y=3
        let mut cubes = Vec::new();
        for i in 1..=3 {
            let e = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
                0.0, i as f32, 0.0,
            ))));
            world.insert(e, GlobalTransform::default());
            world.insert(e, PhysicsBody::dynamic());
            world.insert(e, Velocity::default());
            world.insert(e, Collider::aabb(0.5, 0.5, 0.5));
            cubes.push(e);
        }

        // Run simulation for a while
        for _ in 0..300 {
            physics_step_system(&mut world);
        }

        // All cubes should be above ground (y > -0.5) and below starting height
        for (i, &cube) in cubes.iter().enumerate() {
            let y = world.get::<LocalTransform>(cube).unwrap().0.translation.y;
            assert!(y > -1.0, "Cube {} fell through ground, y={}", i, y);
            assert!(y < 5.0, "Cube {} exploded upward, y={}", i, y);
        }
    }

    #[test]
    fn ccd_prevents_tunneling() {
        // Fast bullet (speed >> collider size) aimed at a thin wall.
        // Without CCD, bullet would pass through. With CCD, it stops before.
        let mut world = World::new();
        world.insert_resource(PhysicsConfig {
            gravity: Vec3::ZERO, // no gravity for this test
            fixed_dt: 1.0 / 60.0,
            max_substeps: 8,
        });

        // Thin wall at x=10 (AABB half-extent 0.1 in X)
        let wall = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            10.0, 0.0, 0.0,
        ))));
        world.insert(wall, GlobalTransform::default());
        world.insert(wall, PhysicsBody::fixed());
        world.insert(wall, Collider::aabb(0.1, 2.0, 2.0));

        // Bullet at x=0, moving at 600 m/s (10 units per frame at 60fps)
        // Bullet size is 0.1 — displacement per frame (10) >> size (0.1)
        let bullet = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(bullet, GlobalTransform::default());
        world.insert(bullet, PhysicsBody::dynamic());
        world.insert(
            bullet,
            Velocity {
                linear: Vec3::new(600.0, 0.0, 0.0),
                angular: Vec3::ZERO,
            },
        );
        world.insert(bullet, Collider::sphere(0.1));

        // Run one physics step
        physics_step_system(&mut world);

        let lt = world.get::<LocalTransform>(bullet).unwrap();
        // Dynamic body with CCD should stop before the wall
        assert!(
            lt.0.translation.x < 10.0,
            "Bullet should not tunnel through wall, x={}",
            lt.0.translation.x
        );
    }

    #[test]
    fn collision_layers_prevent_interaction() {
        // Two dynamic bodies on different layers should not collide.
        let mut world = World::new();
        world.insert_resource(PhysicsConfig {
            gravity: Vec3::ZERO,
            fixed_dt: 1.0 / 60.0,
            max_substeps: 1,
        });

        // Body A on layer 1, mask = layer 1 only
        let a = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(a, GlobalTransform::default());
        world.insert(a, PhysicsBody::dynamic());
        world.insert(a, Velocity::default());
        world.insert(a, Collider::sphere(1.0).with_layer(1).with_mask(1));

        // Body B on layer 2, mask = layer 2 only — should NOT collide with A
        let b = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.5, 0.0, 0.0,
        ))));
        world.insert(b, GlobalTransform::default());
        world.insert(b, PhysicsBody::dynamic());
        world.insert(b, Velocity::default());
        world.insert(b, Collider::sphere(1.0).with_layer(2).with_mask(2));

        // They overlap geometrically but layers don't interact
        physics_step_system(&mut world);

        // Positions should be unchanged (no separation applied)
        let pos_a = world.get::<LocalTransform>(a).unwrap().0.translation;
        let pos_b = world.get::<LocalTransform>(b).unwrap().0.translation;
        assert!(
            (pos_a.x).abs() < 0.01,
            "A should not have moved, x={}",
            pos_a.x
        );
        assert!(
            (pos_b.x - 0.5).abs() < 0.01,
            "B should not have moved, x={}",
            pos_b.x
        );
    }

    #[test]
    fn collision_layers_allow_interaction() {
        // Two dynamic bodies on matching layers should collide normally.
        let mut world = World::new();
        world.insert_resource(PhysicsConfig {
            gravity: Vec3::ZERO,
            fixed_dt: 1.0 / 60.0,
            max_substeps: 1,
        });

        // Body A on layer 1, mask = all
        let a = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(a, GlobalTransform::default());
        world.insert(a, PhysicsBody::dynamic());
        world.insert(a, Velocity::default());
        world.insert(a, Collider::sphere(1.0).with_layer(1).with_mask(u32::MAX));

        // Body B on layer 1, mask = all — should collide with A
        let b = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.5, 0.0, 0.0,
        ))));
        world.insert(b, GlobalTransform::default());
        world.insert(b, PhysicsBody::dynamic());
        world.insert(b, Velocity::default());
        world.insert(b, Collider::sphere(1.0).with_layer(1).with_mask(u32::MAX));

        physics_step_system(&mut world);

        // Bodies should have been pushed apart
        let pos_a = world.get::<LocalTransform>(a).unwrap().0.translation;
        let pos_b = world.get::<LocalTransform>(b).unwrap().0.translation;
        let dist = (pos_b.x - pos_a.x).abs();
        assert!(
            dist > 0.5,
            "Bodies should have been separated, dist={}",
            dist
        );
    }

    #[test]
    fn mass_weighted_collision_response() {
        // A heavy body and a light body collide. The light body should move more.
        let mut world = World::new();
        world.insert_resource(PhysicsConfig {
            gravity: Vec3::ZERO,
            fixed_dt: 1.0 / 60.0,
            max_substeps: 1,
        });

        // Heavy body (mass=10) at origin
        let heavy = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(heavy, GlobalTransform::default());
        world.insert(heavy, PhysicsBody::dynamic());
        world.insert(heavy, Mass::new(10.0, 1.0));
        world.insert(heavy, Velocity::default());
        world.insert(heavy, Collider::sphere(1.0));

        // Light body (mass=1) overlapping
        let light = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            1.0, 0.0, 0.0,
        ))));
        world.insert(light, GlobalTransform::default());
        world.insert(light, PhysicsBody::dynamic());
        world.insert(light, Mass::new(1.0, 0.1));
        world.insert(light, Velocity::default());
        world.insert(light, Collider::sphere(1.0));

        physics_step_system(&mut world);

        let pos_heavy = world.get::<LocalTransform>(heavy).unwrap().0.translation;
        let pos_light = world.get::<LocalTransform>(light).unwrap().0.translation;

        // Light body should have moved further from origin than heavy body
        let heavy_displacement = pos_heavy.x.abs();
        let light_displacement = (pos_light.x - 1.0).abs();
        assert!(
            light_displacement > heavy_displacement,
            "Light body should move more: light_d={}, heavy_d={}",
            light_displacement,
            heavy_displacement
        );
    }

    #[test]
    fn friction_decelerates_sliding() {
        // A dynamic body sliding on a static surface should be slowed by friction,
        // not accelerated. The friction impulse must oppose the sliding direction.
        let mut world = World::new();
        world.insert_resource(PhysicsConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            fixed_dt: 1.0 / 60.0,
            max_substeps: 1,
        });

        // Static ground at y=0
        let ground = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(ground, GlobalTransform::default());
        world.insert(ground, PhysicsBody::fixed());
        world.insert(
            ground,
            Collider::aabb(10.0, 0.5, 10.0)
                .with_friction(0.5)
                .with_restitution(0.0),
        );

        // Sliding box at y=0.5 moving in +X
        let slider = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, 0.5, 0.0,
        ))));
        world.insert(slider, GlobalTransform::default());
        world.insert(slider, PhysicsBody::dynamic());
        world.insert(
            slider,
            Velocity {
                linear: Vec3::new(10.0, 0.0, 0.0),
                angular: Vec3::ZERO,
            },
        );
        world.insert(
            slider,
            Collider::aabb(0.5, 0.5, 0.5)
                .with_friction(0.5)
                .with_restitution(0.0),
        );

        // Step physics several times
        for _ in 0..10 {
            physics_step_system(&mut world);
        }

        let vel = world.get::<Velocity>(slider).unwrap();
        // Friction should have reduced the X velocity, not increased it
        assert!(
            vel.linear.x < 10.0,
            "Friction should decelerate sliding body, vx={}",
            vel.linear.x
        );
        assert!(
            vel.linear.x >= 0.0,
            "Friction should not reverse direction, vx={}",
            vel.linear.x
        );
    }

    #[test]
    fn collision_events_are_emitted() {
        let mut world = World::new();
        world.insert_resource(PhysicsConfig {
            gravity: Vec3::ZERO,
            fixed_dt: 1.0 / 60.0,
            max_substeps: 1,
        });

        let a = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(a, GlobalTransform::default());
        world.insert(a, PhysicsBody::dynamic());
        world.insert(a, Velocity::default());
        world.insert(a, Collider::sphere(1.0));

        let b = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            1.0, 0.0, 0.0,
        ))));
        world.insert(b, GlobalTransform::default());
        world.insert(b, PhysicsBody::dynamic());
        world.insert(b, Velocity::default());
        world.insert(b, Collider::sphere(1.0));

        physics_step_system(&mut world);

        let events: Vec<&CollisionEvent> = world.read_events::<CollisionEvent>().collect();
        assert!(
            !events.is_empty(),
            "Should have emitted at least one collision event"
        );
        assert!(events[0].penetration > 0.0);
    }

    // ---- ExternalForce tests ------------------------------------------------

    /// Helper: create a zero-gravity world with a single dynamic body that has
    /// the given mass/inertia and an `ExternalForce` component. Returns the
    /// entity.
    fn spawn_forced_body(world: &mut World, mass: f32, inertia: f32) -> Entity {
        world.insert_resource(PhysicsConfig {
            gravity: Vec3::ZERO,
            fixed_dt: 1.0 / 60.0,
            max_substeps: 1,
        });
        let entity = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(entity, GlobalTransform::default());
        world.insert(entity, PhysicsBody::dynamic());
        world.insert(entity, Mass::new(mass, inertia));
        world.insert(entity, Velocity::default());
        world.insert(entity, ExternalForce::default());
        entity
    }

    #[test]
    fn constant_force_linear_velocity_grows_linearly() {
        let mut world = World::new();
        let entity = spawn_forced_body(&mut world, 1.0, 1.0);

        // Persistent force of 10 N in +X.
        world.insert(
            entity,
            ExternalForce {
                force: Vec3::new(10.0, 0.0, 0.0),
                torque: Vec3::ZERO,
                persistent: true,
            },
        );

        let dt = 1.0 / 60.0;
        let n = 10;
        for _ in 0..n {
            physics_step_system(&mut world);
        }

        let vel = world.get::<Velocity>(entity).unwrap();
        let expected = 10.0 * dt * n as f32; // F * inv_mass(=1) * dt * steps
        assert!(
            (vel.linear.x - expected).abs() < 1e-4,
            "Expected vx ~ {expected}, got {}",
            vel.linear.x
        );
    }

    #[test]
    fn constant_torque_angular_velocity_grows_linearly() {
        let mut world = World::new();
        let entity = spawn_forced_body(&mut world, 1.0, 2.0);

        world.insert(
            entity,
            ExternalForce {
                force: Vec3::ZERO,
                torque: Vec3::new(0.0, 6.0, 0.0),
                persistent: true,
            },
        );

        let dt = 1.0 / 60.0;
        let inv_inertia = 1.0 / 2.0;
        let n = 10;
        for _ in 0..n {
            physics_step_system(&mut world);
        }

        let vel = world.get::<Velocity>(entity).unwrap();
        let expected = 6.0 * inv_inertia * dt * n as f32;
        assert!(
            (vel.angular.y - expected).abs() < 1e-4,
            "Expected wy ~ {expected}, got {}",
            vel.angular.y
        );
    }

    #[test]
    fn oneshot_force_cleared_after_one_step() {
        let mut world = World::new();
        let entity = spawn_forced_body(&mut world, 1.0, 1.0);

        world.insert(
            entity,
            ExternalForce {
                force: Vec3::new(60.0, 0.0, 0.0),
                torque: Vec3::ZERO,
                persistent: false,
            },
        );

        // First step applies the force.
        physics_step_system(&mut world);
        let vel_after_1 = world.get::<Velocity>(entity).unwrap().linear.x;
        assert!(vel_after_1 > 0.0, "Force should have been applied");

        // Second step: force was cleared, so velocity should not increase.
        physics_step_system(&mut world);
        let vel_after_2 = world.get::<Velocity>(entity).unwrap().linear.x;
        assert!(
            (vel_after_2 - vel_after_1).abs() < 1e-6,
            "One-shot force should not re-apply: v1={vel_after_1}, v2={vel_after_2}"
        );
    }

    #[test]
    fn persistent_force_persists_across_steps() {
        let mut world = World::new();
        let entity = spawn_forced_body(&mut world, 1.0, 1.0);

        world.insert(
            entity,
            ExternalForce {
                force: Vec3::new(10.0, 0.0, 0.0),
                torque: Vec3::ZERO,
                persistent: true,
            },
        );

        physics_step_system(&mut world);
        let v1 = world.get::<Velocity>(entity).unwrap().linear.x;

        physics_step_system(&mut world);
        let v2 = world.get::<Velocity>(entity).unwrap().linear.x;

        // Velocity should keep growing each step.
        assert!(
            v2 > v1,
            "Persistent force should keep accelerating: v1={v1}, v2={v2}"
        );
        // And force component should still be set.
        let ef = world.get::<ExternalForce>(entity).unwrap();
        assert!(
            ef.force.length_squared() > 0.0,
            "Persistent force should not be cleared"
        );
    }

    #[test]
    fn force_on_sleeping_body_wakes_it() {
        let mut world = World::new();
        let entity = spawn_forced_body(&mut world, 1.0, 1.0);
        world.insert(entity, Sleeping);

        world.insert(
            entity,
            ExternalForce {
                force: Vec3::new(5.0, 0.0, 0.0),
                torque: Vec3::ZERO,
                persistent: false,
            },
        );

        physics_step_system(&mut world);

        assert!(
            world.get::<Sleeping>(entity).is_none(),
            "Body should have been woken by external force"
        );
        let vel = world.get::<Velocity>(entity).unwrap();
        assert!(vel.linear.x > 0.0, "Force should have been applied");
    }

    #[test]
    fn force_on_static_body_is_ignored() {
        let mut world = World::new();
        world.insert_resource(PhysicsConfig {
            gravity: Vec3::ZERO,
            fixed_dt: 1.0 / 60.0,
            max_substeps: 1,
        });

        let entity = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(entity, GlobalTransform::default());
        world.insert(entity, PhysicsBody::fixed());
        world.insert(entity, Mass::infinite());
        world.insert(entity, Velocity::default());
        world.insert(
            entity,
            ExternalForce {
                force: Vec3::new(100.0, 0.0, 0.0),
                torque: Vec3::ZERO,
                persistent: true,
            },
        );

        physics_step_system(&mut world);

        let vel = world.get::<Velocity>(entity).unwrap();
        assert!(
            vel.linear.x.abs() < 1e-8,
            "Static body should not be affected by external force, vx={}",
            vel.linear.x
        );
    }

    // ── Angular impulse tests ──

    #[test]
    fn sphere_head_on_no_angular_velocity() {
        // Two spheres colliding head-on along the X axis.
        // The contact point lies on the line between centers, so r x n == 0
        // and no angular velocity should be generated.
        let mut world = World::new();
        world.insert_resource(PhysicsConfig {
            gravity: Vec3::ZERO,
            fixed_dt: 1.0 / 60.0,
            max_substeps: 1,
        });

        let a = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(a, GlobalTransform::default());
        world.insert(a, PhysicsBody::dynamic());
        world.insert(a, Mass::from_sphere(1.0, 1.0));
        world.insert(
            a,
            Velocity {
                linear: Vec3::new(5.0, 0.0, 0.0),
                angular: Vec3::ZERO,
            },
        );
        world.insert(a, Collider::sphere(1.0).with_restitution(0.5));

        let b = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            1.5, 0.0, 0.0,
        ))));
        world.insert(b, GlobalTransform::default());
        world.insert(b, PhysicsBody::dynamic());
        world.insert(b, Mass::from_sphere(1.0, 1.0));
        world.insert(
            b,
            Velocity {
                linear: Vec3::new(-5.0, 0.0, 0.0),
                angular: Vec3::ZERO,
            },
        );
        world.insert(b, Collider::sphere(1.0).with_restitution(0.5));

        physics_step_system(&mut world);

        let vel_a = world.get::<Velocity>(a).unwrap();
        let vel_b = world.get::<Velocity>(b).unwrap();

        // Angular velocity should remain ~zero for head-on sphere collision
        assert!(
            vel_a.angular.length() < 0.1,
            "Sphere A should have no angular vel, got {:?}",
            vel_a.angular
        );
        assert!(
            vel_b.angular.length() < 0.1,
            "Sphere B should have no angular vel, got {:?}",
            vel_b.angular
        );
    }

    #[test]
    fn zero_force_on_sleeping_body_does_not_wake_it() {
        let mut world = World::new();
        let entity = spawn_forced_body(&mut world, 1.0, 1.0);
        world.insert(entity, Sleeping);
        // ExternalForce is default (zero), so the body should stay asleep.

        physics_step_system(&mut world);

        assert!(
            world.get::<Sleeping>(entity).is_some(),
            "Zero external force should not wake a sleeping body"
        );
    }

    #[test]
    fn sphere_hits_aabb_edge_gains_angular_velocity() {
        // A sphere moving in +X hits the corner/edge of an AABB offset in Y.
        // The off-center contact should produce angular velocity on the AABB.
        let mut world = World::new();
        world.insert_resource(PhysicsConfig {
            gravity: Vec3::ZERO,
            fixed_dt: 1.0 / 60.0,
            max_substeps: 1,
        });

        // Static wall (AABB) at x=2
        let wall = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            2.0, 0.0, 0.0,
        ))));
        world.insert(wall, GlobalTransform::default());
        world.insert(wall, PhysicsBody::fixed());
        world.insert(wall, Collider::aabb(0.5, 2.0, 2.0).with_restitution(0.5));

        // Dynamic AABB approaching the wall, offset in Y so contact is off-center
        let box_e = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, 0.5, 0.0,
        ))));
        world.insert(box_e, GlobalTransform::default());
        world.insert(box_e, PhysicsBody::dynamic());
        world.insert(box_e, Mass::from_aabb(1.0, 0.5, 0.5, 0.5));
        world.insert(
            box_e,
            Velocity {
                linear: Vec3::new(10.0, 0.0, 0.0),
                angular: Vec3::ZERO,
            },
        );
        world.insert(box_e, Collider::aabb(0.5, 0.5, 0.5).with_restitution(0.5));

        // Run enough steps for the box to reach and collide with the wall
        for _ in 0..20 {
            physics_step_system(&mut world);
        }

        let vel = world.get::<Velocity>(box_e).unwrap();
        // The box should have gained some angular velocity from the off-center
        // collision. The exact axis depends on contact geometry, but it shouldn't
        // be zero.
        let ang_speed = vel.angular.length();
        assert!(
            ang_speed > 0.01,
            "Off-center AABB-wall collision should produce angular velocity, got {}",
            ang_speed
        );
    }

    #[test]
    fn off_center_box_wall_collision_bounces_and_spins() {
        // A box hits a static floor off-center. It should both bounce (positive
        // Y velocity after collision) and spin (non-zero angular velocity).
        let mut world = World::new();
        world.insert_resource(PhysicsConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            fixed_dt: 1.0 / 60.0,
            max_substeps: 1,
        });

        // Static ground at y=0
        let ground = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(ground, GlobalTransform::default());
        world.insert(ground, PhysicsBody::fixed());
        world.insert(
            ground,
            Collider::aabb(10.0, 0.5, 10.0).with_restitution(0.8),
        );

        // Box falling with offset X velocity (so contact is off-center)
        let box_e = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, 3.0, 0.0,
        ))));
        world.insert(box_e, GlobalTransform::default());
        world.insert(box_e, PhysicsBody::dynamic());
        world.insert(box_e, Mass::from_aabb(1.0, 0.5, 0.5, 0.5));
        world.insert(
            box_e,
            Velocity {
                linear: Vec3::new(5.0, 0.0, 0.0),
                angular: Vec3::ZERO,
            },
        );
        world.insert(box_e, Collider::aabb(0.5, 0.5, 0.5).with_restitution(0.8));

        // Let the box fall and hit the ground
        for _ in 0..120 {
            physics_step_system(&mut world);
        }

        let pos = world.get::<LocalTransform>(box_e).unwrap().0.translation;
        // Box should not have fallen through the ground
        assert!(pos.y > -1.0, "Box fell through ground, y={}", pos.y);
    }
}
