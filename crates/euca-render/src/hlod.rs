//! Hierarchical Level of Detail (HLOD) system.
//!
//! HLOD groups clusters of static entities into a single merged representation
//! that is rendered when the camera is far away. When the camera is close,
//! individual entities are rendered normally and the merged mesh is hidden.
//!
//! This is complementary to per-entity LOD (`lod.rs`): LOD swaps meshes for a
//! single entity, while HLOD replaces entire groups of entities with one mesh.

use crate::camera::Camera;
use crate::mesh::MeshHandle;
use euca_ecs::{Entity, World};
use euca_math::{Aabb, Vec3};
use euca_scene::GlobalTransform;

/// Identifies an HLOD cluster within the [`HlodRegistry`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HlodClusterId(pub u32);

/// A group of static entities that can be replaced by a single merged mesh
/// when viewed from a distance exceeding [`transition_distance`](HlodCluster::transition_distance).
#[derive(Clone, Debug)]
pub struct HlodCluster {
    /// The individual entities that belong to this cluster.
    pub entities: Vec<Entity>,
    /// World-space bounding box enclosing all entities in the cluster.
    pub bounds: Aabb,
    /// Optional pre-merged mesh to render when the cluster is far away.
    /// If `None`, the cluster simply hides its entities beyond the transition
    /// distance (useful for culling groups that are too small to matter).
    pub merged_mesh: Option<MeshHandle>,
    /// Distance from the camera beyond which the merged mesh replaces
    /// individual entities. Computed from the cluster's spatial extent.
    pub transition_distance: f32,
}

/// World resource that owns all HLOD clusters and provides lookup.
#[derive(Clone, Debug, Default)]
pub struct HlodRegistry {
    clusters: Vec<HlodCluster>,
}

impl HlodRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            clusters: Vec::new(),
        }
    }

    /// Add a cluster and return its ID.
    pub fn add(&mut self, cluster: HlodCluster) -> HlodClusterId {
        let id = HlodClusterId(self.clusters.len() as u32);
        self.clusters.push(cluster);
        id
    }

    /// Look up a cluster by ID.
    pub fn get(&self, id: HlodClusterId) -> Option<&HlodCluster> {
        self.clusters.get(id.0 as usize)
    }

    /// Mutable access to a cluster by ID.
    pub fn get_mut(&mut self, id: HlodClusterId) -> Option<&mut HlodCluster> {
        self.clusters.get_mut(id.0 as usize)
    }

    /// Iterate over all clusters with their IDs.
    pub fn iter(&self) -> impl Iterator<Item = (HlodClusterId, &HlodCluster)> {
        self.clusters
            .iter()
            .enumerate()
            .map(|(i, c)| (HlodClusterId(i as u32), c))
    }

    /// Number of registered clusters.
    pub fn len(&self) -> usize {
        self.clusters.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.clusters.is_empty()
    }
}

/// Component attached to entities that participate in an HLOD cluster.
///
/// The HLOD selection system writes this each frame to indicate whether the
/// entity should be rendered individually or is hidden because its cluster's
/// merged mesh is being shown instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HlodVisibility {
    /// The camera is close enough: render this entity individually.
    Individual,
    /// The camera is far away: this entity is hidden; the cluster's merged
    /// mesh is shown instead.
    ClusterMerged,
}

/// Multiplier applied to the cluster's maximum extent to derive the
/// transition distance. Larger clusters transition at proportionally
/// greater distances so they remain plausible stand-ins longer.
const TRANSITION_DISTANCE_FACTOR: f32 = 10.0;

/// Build an [`HlodCluster`] from a set of entities currently in the world.
///
/// Computes the axis-aligned bounding box from entity positions and derives
/// a transition distance proportional to the cluster's spatial extent.
///
/// Returns `None` if `entities` is empty or none of them have a
/// [`GlobalTransform`].
pub fn generate_hlod_cluster(entities: &[Entity], world: &World) -> Option<HlodCluster> {
    if entities.is_empty() {
        return None;
    }

    // Collect world-space positions of valid entities.
    let positions: Vec<Vec3> = entities
        .iter()
        .filter_map(|&e| world.get::<GlobalTransform>(e).map(|gt| gt.0.translation))
        .collect();

    let bounds = Aabb::from_points(positions)?;
    let size = bounds.size();
    let max_extent = size.x.max(size.y).max(size.z);
    let transition_distance = max_extent * TRANSITION_DISTANCE_FACTOR;

    Some(HlodCluster {
        entities: entities.to_vec(),
        bounds,
        merged_mesh: None,
        transition_distance,
    })
}

/// Per-frame system that evaluates each HLOD cluster against the camera
/// distance and sets [`HlodVisibility`] on member entities accordingly.
///
/// - **Close** (distance <= transition_distance): entities get
///   `HlodVisibility::Individual`.
/// - **Far** (distance > transition_distance): entities get
///   `HlodVisibility::ClusterMerged`.
///
/// Run this system after transform propagation and before draw-command
/// collection so that the renderer can respect the visibility state.
pub fn hlod_select_system(world: &mut World) {
    let camera = match world.resource::<Camera>() {
        Some(c) => c.clone(),
        None => return,
    };

    let registry = match world.resource::<HlodRegistry>() {
        Some(r) => r.clone(),
        None => return,
    };

    for (_id, cluster) in registry.iter() {
        let center = cluster.bounds.center();
        let distance = (center - camera.eye).length();

        let visibility = if distance > cluster.transition_distance {
            HlodVisibility::ClusterMerged
        } else {
            HlodVisibility::Individual
        };

        for &entity in &cluster.entities {
            if let Some(existing) = world.get_mut::<HlodVisibility>(entity) {
                *existing = visibility;
            } else {
                world.insert(entity, visibility);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Transform;

    /// Helper: spawn an entity with a GlobalTransform at the given position.
    fn spawn_at(world: &mut World, pos: Vec3) -> Entity {
        world.spawn(GlobalTransform(Transform::from_translation(pos)))
    }

    // ── Test 1: Cluster bounds are computed correctly ──

    #[test]
    fn cluster_bounds_enclose_all_entities() {
        let mut world = World::new();

        let e1 = spawn_at(&mut world, Vec3::new(-5.0, 0.0, 0.0));
        let e2 = spawn_at(&mut world, Vec3::new(5.0, 10.0, 3.0));
        let e3 = spawn_at(&mut world, Vec3::new(0.0, -2.0, 8.0));

        let cluster =
            generate_hlod_cluster(&[e1, e2, e3], &world).expect("should produce a cluster");

        assert_eq!(cluster.bounds.min, Vec3::new(-5.0, -2.0, 0.0));
        assert_eq!(cluster.bounds.max, Vec3::new(5.0, 10.0, 8.0));
        // Transition distance = max_extent(10, 12, 8) * 10 = 120
        assert!((cluster.transition_distance - 120.0).abs() < 1e-3);
    }

    // ── Test 2: Distance-based transition toggles visibility ──

    #[test]
    fn visibility_toggles_by_camera_distance() {
        let mut world = World::new();

        // Cluster centered at (100, 0, 0) with extent 10 -> transition = 100
        let e1 = spawn_at(&mut world, Vec3::new(95.0, 0.0, 0.0));
        let e2 = spawn_at(&mut world, Vec3::new(105.0, 0.0, 0.0));

        let cluster =
            generate_hlod_cluster(&[e1, e2], &world).expect("should produce a cluster");
        assert!((cluster.transition_distance - 100.0).abs() < 1e-3);

        let mut registry = HlodRegistry::new();
        registry.add(cluster);
        world.insert_resource(registry);

        // Camera close to the cluster -> Individual
        let camera = Camera::new(Vec3::new(100.0, 0.0, 0.0), Vec3::new(100.0, 0.0, 1.0));
        world.insert_resource(camera);
        hlod_select_system(&mut world);

        assert_eq!(
            *world.get::<HlodVisibility>(e1).unwrap(),
            HlodVisibility::Individual
        );
        assert_eq!(
            *world.get::<HlodVisibility>(e2).unwrap(),
            HlodVisibility::Individual
        );

        // Move camera far away -> ClusterMerged
        let camera_far = Camera::new(Vec3::new(-500.0, 0.0, 0.0), Vec3::new(100.0, 0.0, 0.0));
        world.insert_resource(camera_far);
        hlod_select_system(&mut world);

        assert_eq!(
            *world.get::<HlodVisibility>(e1).unwrap(),
            HlodVisibility::ClusterMerged
        );
        assert_eq!(
            *world.get::<HlodVisibility>(e2).unwrap(),
            HlodVisibility::ClusterMerged
        );
    }

    // ── Test 3: Visibility is updated when camera moves ──

    #[test]
    fn visibility_updates_on_camera_movement() {
        let mut world = World::new();

        let e = spawn_at(&mut world, Vec3::new(0.0, 0.0, 50.0));
        let cluster = generate_hlod_cluster(&[e], &world).expect("should produce a cluster");
        // Single point -> bounds are degenerate (size=0), transition_distance = 0
        // So any non-zero distance should trigger ClusterMerged
        assert_eq!(cluster.transition_distance, 0.0);

        let mut registry = HlodRegistry::new();
        registry.add(cluster);
        world.insert_resource(registry);

        // Camera at origin, entity at z=50 -> distance=50 > 0 -> ClusterMerged
        let camera = Camera::new(Vec3::ZERO, Vec3::Z);
        world.insert_resource(camera);
        hlod_select_system(&mut world);
        assert_eq!(
            *world.get::<HlodVisibility>(e).unwrap(),
            HlodVisibility::ClusterMerged
        );

        // Move camera exactly to entity position -> distance=0 -> Individual
        let camera_close = Camera::new(Vec3::new(0.0, 0.0, 50.0), Vec3::Z);
        world.insert_resource(camera_close);
        hlod_select_system(&mut world);
        assert_eq!(
            *world.get::<HlodVisibility>(e).unwrap(),
            HlodVisibility::Individual
        );
    }

    // ── Test 4: Empty entity list produces None ──

    #[test]
    fn empty_cluster_returns_none() {
        let world = World::new();
        assert!(generate_hlod_cluster(&[], &world).is_none());
    }

    // ── Test 5: Nested / multiple clusters operate independently ──

    #[test]
    fn multiple_clusters_select_independently() {
        let mut world = World::new();

        // Cluster A: near origin, small extent
        let a1 = spawn_at(&mut world, Vec3::new(-1.0, 0.0, 0.0));
        let a2 = spawn_at(&mut world, Vec3::new(1.0, 0.0, 0.0));
        let cluster_a = generate_hlod_cluster(&[a1, a2], &world).expect("cluster A");
        // extent = 2, transition = 20

        // Cluster B: far away, large extent
        let b1 = spawn_at(&mut world, Vec3::new(500.0, 0.0, 0.0));
        let b2 = spawn_at(&mut world, Vec3::new(600.0, 0.0, 0.0));
        let cluster_b = generate_hlod_cluster(&[b1, b2], &world).expect("cluster B");
        // extent = 100, transition = 1000

        let mut registry = HlodRegistry::new();
        registry.add(cluster_a);
        registry.add(cluster_b);
        world.insert_resource(registry);

        // Camera at origin: close to A (distance ~0 < 20 -> Individual),
        // far from B (distance ~550 < 1000 -> Individual)
        let camera = Camera::new(Vec3::ZERO, Vec3::Z);
        world.insert_resource(camera);
        hlod_select_system(&mut world);

        assert_eq!(
            *world.get::<HlodVisibility>(a1).unwrap(),
            HlodVisibility::Individual
        );
        assert_eq!(
            *world.get::<HlodVisibility>(b1).unwrap(),
            HlodVisibility::Individual
        );

        // Move camera to (0, 0, -5000): far from both clusters
        // A center=(0,0,0), distance=5000 > 20 -> ClusterMerged
        // B center=(550,0,0), distance=~5030 > 1000 -> ClusterMerged
        let camera_far = Camera::new(Vec3::new(0.0, 0.0, -5000.0), Vec3::Z);
        world.insert_resource(camera_far);
        hlod_select_system(&mut world);

        assert_eq!(
            *world.get::<HlodVisibility>(a1).unwrap(),
            HlodVisibility::ClusterMerged
        );
        assert_eq!(
            *world.get::<HlodVisibility>(a2).unwrap(),
            HlodVisibility::ClusterMerged
        );
        assert_eq!(
            *world.get::<HlodVisibility>(b1).unwrap(),
            HlodVisibility::ClusterMerged
        );
        assert_eq!(
            *world.get::<HlodVisibility>(b2).unwrap(),
            HlodVisibility::ClusterMerged
        );
    }

    // ── Test 6: Registry operations ──

    #[test]
    fn registry_add_and_lookup() {
        let mut registry = HlodRegistry::new();
        assert!(registry.is_empty());

        let cluster = HlodCluster {
            entities: vec![],
            bounds: Aabb::new(Vec3::ZERO, Vec3::ONE),
            merged_mesh: Some(MeshHandle(42)),
            transition_distance: 100.0,
        };
        let id = registry.add(cluster);
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());

        let fetched = registry.get(id).unwrap();
        assert_eq!(fetched.merged_mesh, Some(MeshHandle(42)));
        assert!((fetched.transition_distance - 100.0).abs() < 1e-5);
    }

    // ── Test 7: System is no-op without camera or registry ──

    #[test]
    fn system_noop_without_camera() {
        let mut world = World::new();
        let e = spawn_at(&mut world, Vec3::ZERO);

        let cluster = generate_hlod_cluster(&[e], &world).unwrap();
        let mut registry = HlodRegistry::new();
        registry.add(cluster);
        world.insert_resource(registry);
        // No camera inserted

        hlod_select_system(&mut world);

        // No HlodVisibility should be written
        assert!(world.get::<HlodVisibility>(e).is_none());
    }

    #[test]
    fn system_noop_without_registry() {
        let mut world = World::new();
        let camera = Camera::new(Vec3::ZERO, Vec3::Z);
        world.insert_resource(camera);
        // No registry inserted

        hlod_select_system(&mut world);
        // Should not panic or crash
    }
}
