#![allow(dead_code)]
use crate::camera::Camera;
use crate::mesh::MeshHandle;
use euca_ecs::{Query, World};
use euca_math::Vec3;
use euca_scene::GlobalTransform;

/// A single level of detail within an LOD group.
///
/// `max_screen_percentage` defines the upper bound of screen-space size for which
/// this level is selected. Levels are sorted by quality: index 0 is the highest
/// detail (closest to camera), and the last index is the lowest detail (farthest).
#[derive(Clone, Debug)]
pub struct LodLevel {
    /// Mesh to render at this LOD.
    pub mesh: MeshHandle,
    /// This LOD is used when the object's screen-space size (as a fraction of
    /// screen height) is at or below this value. Range: 0.0..=1.0.
    pub max_screen_percentage: f32,
}

/// Component that assigns multiple levels of detail to an entity.
///
/// The LOD selection system evaluates screen-space size each frame and picks
/// the appropriate level, storing the result in [`CurrentLod`].
///
/// Levels must be sorted from highest detail (index 0, largest
/// `max_screen_percentage`) to lowest detail (last index, smallest
/// `max_screen_percentage`).
#[derive(Clone, Debug)]
pub struct LodGroup {
    pub levels: Vec<LodLevel>,
    /// Bounding sphere radius in local space. Used to compute screen-space size.
    /// If zero, the entity is treated as a point and LOD 0 is always selected.
    pub radius: f32,
}

/// Component written by `lod_select_system` to indicate which mesh the
/// renderer should use for this frame. When present, draw-command collection
/// should prefer this mesh over `MeshRenderer.mesh`.
///
/// If the entity is too far away (screen size below all LOD thresholds), the
/// component is set to `None`, meaning the entity should be culled.
#[derive(Clone, Copy, Debug)]
pub struct CurrentLod {
    /// The selected mesh handle, or `None` if the entity should be culled.
    pub mesh: Option<MeshHandle>,
}

/// Global resource controlling LOD behavior.
///
/// Insert into the world via `world.insert_resource(LodSettings { ... })`.
#[derive(Clone, Debug)]
pub struct LodSettings {
    /// Multiplier applied to the computed screen-space size before threshold
    /// comparison. Values > 1.0 push selection toward higher-detail LODs
    /// (objects appear "closer"), values < 1.0 push toward lower-detail LODs.
    pub bias: f32,
}

impl Default for LodSettings {
    fn default() -> Self {
        Self { bias: 1.0 }
    }
}

/// Compute the screen-space height fraction of a bounding sphere.
///
/// Returns a value in roughly 0.0..1.0 representing what fraction of the
/// screen height the object's bounding sphere covers.
///
/// Formula: `screen_size = (diameter) / (visible_height_at_distance)`
///   where `visible_height_at_distance = 2 * distance * tan(fov_y / 2)`
///
/// For orthographic cameras, the formula is `diameter / (2 * ortho_size)`.
fn screen_space_size(camera: &Camera, object_position: Vec3, radius: f32) -> f32 {
    if radius <= 0.0 {
        return 1.0; // Treat zero-radius as always full-detail
    }

    if camera.orthographic {
        // Orthographic: screen fraction is independent of distance
        let visible_height = 2.0 * camera.ortho_size;
        if visible_height <= 0.0 {
            return 1.0;
        }
        return (radius * 2.0) / visible_height;
    }

    let offset = object_position - camera.eye;
    let distance = offset.length();

    if distance < 1e-6 {
        return 1.0; // Object at camera position: full detail
    }

    let half_fov_tan = (camera.fov_y * 0.5).tan();
    let visible_height = 2.0 * distance * half_fov_tan;

    if visible_height <= 0.0 {
        return 1.0;
    }

    (radius * 2.0) / visible_height
}

/// Select the appropriate LOD level for each entity that has a [`LodGroup`]
/// and a [`GlobalTransform`].
///
/// This system reads the active [`Camera`] resource and optional [`LodSettings`]
/// resource, computes screen-space size for each LOD entity, and writes a
/// [`CurrentLod`] component with the selected mesh.
///
/// Run this system **before** draw-command collection so that the renderer
/// can use the LOD-selected mesh.
pub fn lod_select_system(world: &mut World) {
    // Read camera and settings from resources
    let camera = match world.resource::<Camera>() {
        Some(c) => c.clone(),
        None => return, // No camera: nothing to do
    };

    let bias = world
        .resource::<LodSettings>()
        .map(|s| s.bias)
        .unwrap_or(1.0);

    // Collect entities that need LOD selection
    let selections: Vec<(euca_ecs::Entity, CurrentLod)> = {
        let query = Query::<(euca_ecs::Entity, &LodGroup, &GlobalTransform)>::new(world);
        query
            .iter()
            .map(|(entity, lod_group, global_transform)| {
                let position = global_transform.0.translation;
                let screen_size = screen_space_size(&camera, position, lod_group.radius) * bias;
                let selected = select_level(&lod_group.levels, screen_size);
                (entity, CurrentLod { mesh: selected })
            })
            .collect()
    };

    // Write results back
    for (entity, current_lod) in selections {
        if let Some(existing) = world.get_mut::<CurrentLod>(entity) {
            *existing = current_lod;
        } else {
            world.insert(entity, current_lod);
        }
    }
}

/// Pick the best LOD level for the given screen-space size.
///
/// Levels are iterated from highest detail (index 0) to lowest. The first
/// level whose `max_screen_percentage` is >= `screen_size` is selected.
/// If the screen size is larger than all thresholds, level 0 (highest
/// detail) is used. If smaller than all thresholds, `None` is returned
/// (the object should be culled).
fn select_level(levels: &[LodLevel], screen_size: f32) -> Option<MeshHandle> {
    if levels.is_empty() {
        return None;
    }

    // Levels are sorted from highest detail (largest max_screen_percentage)
    // to lowest detail (smallest max_screen_percentage).
    //
    // We walk from highest to lowest. For each level, if the object's
    // screen size is within this level's threshold, use it. Otherwise,
    // try the next (lower-detail) level.
    //
    // Example with 3 levels:
    //   LOD 0: max_screen_percentage = 1.0  (used when screen_size > 0.5)
    //   LOD 1: max_screen_percentage = 0.5  (used when screen_size > 0.1)
    //   LOD 2: max_screen_percentage = 0.1  (used when screen_size > 0.0)
    //   Below 0.0 (impossible) -> cull
    //
    // The rule: select level i if screen_size <= levels[i].max_screen_percentage
    // AND (i is the last level OR screen_size > levels[i+1].max_screen_percentage).

    for i in 0..levels.len() {
        let threshold = levels[i].max_screen_percentage;
        if screen_size <= threshold {
            // Check if a lower-detail level is a better fit
            if i + 1 < levels.len() {
                let next_threshold = levels[i + 1].max_screen_percentage;
                if screen_size <= next_threshold {
                    continue; // A lower-detail level still covers this size
                }
            }
            return Some(levels[i].mesh);
        }
    }

    // Screen size exceeds all thresholds — use highest detail (level 0)
    Some(levels[0].mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Transform;

    // ── select_level tests ──

    #[test]
    fn select_level_empty_returns_none() {
        assert!(select_level(&[], 0.5).is_none());
    }

    #[test]
    fn select_level_single_level() {
        let levels = vec![LodLevel {
            mesh: MeshHandle(0),
            max_screen_percentage: 1.0,
        }];
        // Any screen size should return this level
        assert_eq!(select_level(&levels, 0.5), Some(MeshHandle(0)));
        assert_eq!(select_level(&levels, 0.01), Some(MeshHandle(0)));
        assert_eq!(select_level(&levels, 2.0), Some(MeshHandle(0)));
    }

    #[test]
    fn select_level_three_levels() {
        let levels = vec![
            LodLevel {
                mesh: MeshHandle(0),
                max_screen_percentage: 1.0,
            },
            LodLevel {
                mesh: MeshHandle(1),
                max_screen_percentage: 0.5,
            },
            LodLevel {
                mesh: MeshHandle(2),
                max_screen_percentage: 0.1,
            },
        ];

        // Large object (screen_size > 0.5) -> LOD 0
        assert_eq!(select_level(&levels, 0.8), Some(MeshHandle(0)));
        assert_eq!(select_level(&levels, 0.6), Some(MeshHandle(0)));

        // Medium object (0.1 < screen_size <= 0.5) -> LOD 1
        assert_eq!(select_level(&levels, 0.5), Some(MeshHandle(1)));
        assert_eq!(select_level(&levels, 0.3), Some(MeshHandle(1)));
        assert_eq!(select_level(&levels, 0.11), Some(MeshHandle(1)));

        // Small object (screen_size <= 0.1) -> LOD 2
        assert_eq!(select_level(&levels, 0.1), Some(MeshHandle(2)));
        assert_eq!(select_level(&levels, 0.05), Some(MeshHandle(2)));

        // Very large object -> LOD 0
        assert_eq!(select_level(&levels, 1.5), Some(MeshHandle(0)));
    }

    #[test]
    fn select_level_at_exact_thresholds() {
        let levels = vec![
            LodLevel {
                mesh: MeshHandle(0),
                max_screen_percentage: 0.8,
            },
            LodLevel {
                mesh: MeshHandle(1),
                max_screen_percentage: 0.3,
            },
        ];

        // At 0.8 boundary: should go to LOD 1 (since 0.8 <= 0.8 and 0.8 > 0.3)
        assert_eq!(select_level(&levels, 0.8), Some(MeshHandle(0)));
        // At 0.3 boundary: should go to LOD 1
        assert_eq!(select_level(&levels, 0.3), Some(MeshHandle(1)));
        // Above 0.8: LOD 0
        assert_eq!(select_level(&levels, 0.9), Some(MeshHandle(0)));
        // Below 0.3: LOD 1 (last level, so it catches everything below)
        assert_eq!(select_level(&levels, 0.1), Some(MeshHandle(1)));
    }

    // ── screen_space_size tests ──

    #[test]
    fn screen_size_at_known_distance() {
        let camera = Camera::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0));
        // Camera at origin looking along +Z with fov_y = PI/4 (45 deg)
        // At distance 10, visible height = 2 * 10 * tan(22.5 deg) ≈ 8.284
        // Sphere radius 1 -> diameter 2 -> screen_size ≈ 2 / 8.284 ≈ 0.2414

        let size = screen_space_size(&camera, Vec3::new(0.0, 0.0, 10.0), 1.0);
        let expected = 2.0 / (2.0 * 10.0 * (std::f32::consts::FRAC_PI_4 * 0.5).tan());
        assert!(
            (size - expected).abs() < 1e-4,
            "Expected {expected}, got {size}"
        );
    }

    #[test]
    fn screen_size_zero_radius_returns_full() {
        let camera = Camera::default();
        let size = screen_space_size(&camera, Vec3::new(0.0, 0.0, 10.0), 0.0);
        assert_eq!(size, 1.0);
    }

    #[test]
    fn screen_size_at_camera_returns_full() {
        let camera = Camera::new(Vec3::ZERO, Vec3::Z);
        let size = screen_space_size(&camera, Vec3::ZERO, 1.0);
        assert_eq!(size, 1.0);
    }

    #[test]
    fn screen_size_decreases_with_distance() {
        let camera = Camera::new(Vec3::ZERO, Vec3::Z);
        let near = screen_space_size(&camera, Vec3::new(0.0, 0.0, 5.0), 1.0);
        let far = screen_space_size(&camera, Vec3::new(0.0, 0.0, 50.0), 1.0);
        assert!(near > far, "Closer objects should have larger screen size");
    }

    #[test]
    fn screen_size_orthographic() {
        let mut camera = Camera::default();
        camera.orthographic = true;
        camera.ortho_size = 10.0;
        // Visible height = 2 * 10 = 20, diameter = 2 * 1 = 2 -> screen_size = 0.1
        let size = screen_space_size(&camera, Vec3::new(0.0, 0.0, 50.0), 1.0);
        assert!((size - 0.1).abs() < 1e-5, "Expected 0.1, got {size}");
    }

    // ── lod_select_system integration tests ──

    #[test]
    fn system_selects_lod_based_on_distance() {
        let mut world = World::new();

        // Insert camera at origin looking along +Z
        let camera = Camera::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0));
        world.insert_resource(camera);

        // Spawn entity with LOD group at distance 10
        let entity = world.spawn(LodGroup {
            levels: vec![
                LodLevel {
                    mesh: MeshHandle(0),
                    max_screen_percentage: 1.0,
                },
                LodLevel {
                    mesh: MeshHandle(1),
                    max_screen_percentage: 0.3,
                },
                LodLevel {
                    mesh: MeshHandle(2),
                    max_screen_percentage: 0.05,
                },
            ],
            radius: 1.0,
        });
        world.insert(
            entity,
            GlobalTransform(Transform::from_translation(Vec3::new(0.0, 0.0, 10.0))),
        );

        lod_select_system(&mut world);

        let current = world
            .get::<CurrentLod>(entity)
            .expect("CurrentLod should exist");
        // At distance 10, screen_size ≈ 0.24 -> between 0.05 and 0.3 -> LOD 1
        assert_eq!(
            current.mesh,
            Some(MeshHandle(1)),
            "Should select LOD 1 at moderate distance"
        );
    }

    #[test]
    fn system_selects_highest_detail_when_close() {
        let mut world = World::new();

        let camera = Camera::new(Vec3::ZERO, Vec3::Z);
        world.insert_resource(camera);

        let entity = world.spawn(LodGroup {
            levels: vec![
                LodLevel {
                    mesh: MeshHandle(10),
                    max_screen_percentage: 1.0,
                },
                LodLevel {
                    mesh: MeshHandle(11),
                    max_screen_percentage: 0.1,
                },
            ],
            radius: 5.0,
        });
        world.insert(
            entity,
            GlobalTransform(Transform::from_translation(Vec3::new(0.0, 0.0, 2.0))),
        );

        lod_select_system(&mut world);

        let current = world.get::<CurrentLod>(entity).unwrap();
        assert_eq!(current.mesh, Some(MeshHandle(10)));
    }

    #[test]
    fn system_respects_bias() {
        let mut world = World::new();

        let camera = Camera::new(Vec3::ZERO, Vec3::Z);
        world.insert_resource(camera);
        // Bias > 1 makes objects appear closer (higher detail)
        world.insert_resource(LodSettings { bias: 10.0 });

        let entity = world.spawn(LodGroup {
            levels: vec![
                LodLevel {
                    mesh: MeshHandle(0),
                    max_screen_percentage: 1.0,
                },
                LodLevel {
                    mesh: MeshHandle(1),
                    max_screen_percentage: 0.1,
                },
            ],
            radius: 1.0,
        });
        world.insert(
            entity,
            GlobalTransform(Transform::from_translation(Vec3::new(0.0, 0.0, 100.0))),
        );

        lod_select_system(&mut world);

        let current = world.get::<CurrentLod>(entity).unwrap();
        // Without bias, screen_size at distance 100 is very small -> LOD 1
        // With bias 10x, it gets pushed into LOD 0 range
        assert_eq!(current.mesh, Some(MeshHandle(0)));
    }

    #[test]
    fn system_no_camera_is_noop() {
        let mut world = World::new();
        // No camera inserted
        let entity = world.spawn(LodGroup {
            levels: vec![LodLevel {
                mesh: MeshHandle(0),
                max_screen_percentage: 1.0,
            }],
            radius: 1.0,
        });
        world.insert(
            entity,
            GlobalTransform(Transform::from_translation(Vec3::Z)),
        );

        lod_select_system(&mut world);

        // No CurrentLod should be written because no camera exists
        assert!(world.get::<CurrentLod>(entity).is_none());
    }

    #[test]
    fn entities_without_lod_group_unaffected() {
        let mut world = World::new();

        let camera = Camera::new(Vec3::ZERO, Vec3::Z);
        world.insert_resource(camera);

        // Spawn entity with only GlobalTransform (no LodGroup)
        let entity = world.spawn(GlobalTransform(Transform::from_translation(Vec3::Z)));

        lod_select_system(&mut world);

        // No CurrentLod should be added
        assert!(world.get::<CurrentLod>(entity).is_none());
    }

    #[test]
    fn system_updates_existing_current_lod() {
        let mut world = World::new();

        let camera = Camera::new(Vec3::ZERO, Vec3::Z);
        world.insert_resource(camera);

        let entity = world.spawn(LodGroup {
            levels: vec![
                LodLevel {
                    mesh: MeshHandle(0),
                    max_screen_percentage: 1.0,
                },
                LodLevel {
                    mesh: MeshHandle(1),
                    max_screen_percentage: 0.1,
                },
            ],
            radius: 1.0,
        });
        world.insert(
            entity,
            GlobalTransform(Transform::from_translation(Vec3::new(0.0, 0.0, 2.0))),
        );

        // First run
        lod_select_system(&mut world);
        let first = world.get::<CurrentLod>(entity).unwrap().mesh;

        // Move entity far away
        world
            .get_mut::<GlobalTransform>(entity)
            .unwrap()
            .0
            .translation = Vec3::new(0.0, 0.0, 500.0);

        // Second run should update the existing CurrentLod
        lod_select_system(&mut world);
        let second = world.get::<CurrentLod>(entity).unwrap().mesh;

        assert_ne!(first, second, "LOD should change when entity moves far");
    }
}
