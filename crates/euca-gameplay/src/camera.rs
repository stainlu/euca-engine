//! MOBA-style camera — follows a hero entity with edge-pan and scroll zoom.
//!
//! Resource: [`MobaCamera`] — configuration and runtime state.
//! System: [`moba_camera_system`] — reads input, updates the render camera each frame.

use euca_ecs::{Entity, World};
use euca_math::Vec3;

/// Screen dimensions resource. Insert into the world so the camera system
/// knows how large the viewport is (needed for edge-pan margin detection).
#[derive(Clone, Debug)]
pub struct ScreenSize {
    pub width: f32,
    pub height: f32,
}

impl Default for ScreenSize {
    fn default() -> Self {
        Self {
            width: 1920.0,
            height: 1080.0,
        }
    }
}

/// MOBA camera configuration and runtime state.
///
/// Insert as a resource into the [`World`]. The [`moba_camera_system`] reads
/// this every frame to position the render [`Camera`](euca_render::Camera).
#[derive(Clone, Debug)]
pub struct MobaCamera {
    /// The entity the camera follows. `None` disables follow.
    pub follow_entity: Option<Entity>,
    /// Base offset from the hero position to the camera eye (world units).
    /// Default: `(0, 12, 8)` — isometric top-down view.
    pub offset: Vec3,
    /// Offset applied to the look-at target relative to the hero position.
    pub look_at_offset: Vec3,
    /// Zoom multiplier applied to the offset distance. 1.0 = default distance.
    pub zoom: f32,
    /// Minimum zoom (closest the camera can get).
    pub min_zoom: f32,
    /// Maximum zoom (farthest the camera can pull out).
    pub max_zoom: f32,
    /// Edge-pan speed in world units per second.
    pub edge_pan_speed: f32,
    /// Pixel margin from screen edge that triggers panning.
    pub edge_pan_margin: f32,
    /// When `true`, the camera is always centered on the hero (no edge pan).
    pub locked: bool,
    /// Accumulated displacement from edge panning (world XZ plane).
    pub pan_offset: Vec3,
}

impl Default for MobaCamera {
    fn default() -> Self {
        Self {
            follow_entity: None,
            offset: Vec3::new(0.0, 8.0, 5.0),
            look_at_offset: Vec3::ZERO,
            zoom: 1.0,
            min_zoom: 0.5,
            max_zoom: 3.0,
            edge_pan_speed: 15.0,
            edge_pan_margin: 50.0,
            locked: true,
            pan_offset: Vec3::ZERO,
        }
    }
}

/// Update the render camera to follow the hero with edge-pan and scroll zoom.
///
/// Reads:
/// - [`MobaCamera`] resource — configuration and state
/// - [`euca_core::Time`] resource — frame delta for speed-based panning
/// - [`euca_input::InputState`] resource — mouse position and scroll delta
/// - [`ScreenSize`] resource — viewport dimensions for edge detection
/// - [`euca_scene::GlobalTransform`] component on the follow entity
///
/// Writes:
/// - [`euca_render::Camera`] resource — eye and target positions
/// - [`MobaCamera`] resource — zoom and pan_offset
///
/// Gracefully degrades when optional resources are missing: without `InputState`
/// or `ScreenSize` the camera still follows the entity but does not pan or zoom.
pub fn moba_camera_system(world: &mut World) {
    // ── Read immutable inputs first ─────────────────────────────────────

    let dt = world
        .resource::<euca_core::Time>()
        .map(|t| t.delta)
        .unwrap_or(1.0 / 60.0);

    let (mouse_x, mouse_y, scroll) = world
        .resource::<euca_input::InputState>()
        .map(|input| {
            (
                input.mouse_position[0],
                input.mouse_position[1],
                input.scroll_delta,
            )
        })
        .unwrap_or((f32::NAN, f32::NAN, 0.0));

    let (screen_w, screen_h) = world
        .resource::<ScreenSize>()
        .map(|s| (s.width, s.height))
        .unwrap_or((f32::NAN, f32::NAN));

    // ── Read camera config (snapshot to avoid borrow) ───────────────────

    let cam = match world.resource::<MobaCamera>() {
        Some(c) => c.clone(),
        None => return,
    };

    // ── Resolve follow entity position ──────────────────────────────────

    let hero_pos = cam.follow_entity.and_then(|entity| {
        world
            .get::<euca_scene::GlobalTransform>(entity)
            .map(|gt| gt.0.translation)
    });

    let hero_pos = match hero_pos {
        Some(p) => p,
        None => return, // nothing to follow
    };

    // ── Compute zoom ────────────────────────────────────────────────────

    let mut new_zoom = cam.zoom;
    if scroll.abs() > f32::EPSILON {
        // Scroll up (positive) zooms in (smaller multiplier), scroll down zooms out.
        new_zoom -= scroll * 0.1;
        new_zoom = new_zoom.clamp(cam.min_zoom, cam.max_zoom);
    }

    // ── Compute edge-pan delta ──────────────────────────────────────────

    let mut new_pan_offset = cam.pan_offset;

    if cam.locked {
        // Locked mode: always reset pan offset
        new_pan_offset = Vec3::ZERO;
    } else if !mouse_x.is_nan() && !screen_w.is_nan() {
        let margin = cam.edge_pan_margin;
        let speed = cam.edge_pan_speed * dt;

        // Horizontal panning (world X axis)
        if mouse_x < margin {
            new_pan_offset.x -= speed;
        } else if mouse_x > screen_w - margin {
            new_pan_offset.x += speed;
        }

        // Vertical panning (world Z axis) — screen top = forward (-Z or +Z
        // depending on convention). In a typical top-down MOBA:
        // screen-top → move camera forward (−Z), screen-bottom → backward (+Z).
        if mouse_y < margin {
            new_pan_offset.z -= speed;
        } else if mouse_y > screen_h - margin {
            new_pan_offset.z += speed;
        }
    }

    // ── Compute final camera position ───────────────────────────────────

    let eye = hero_pos + cam.offset * new_zoom + new_pan_offset;
    let target = hero_pos + cam.look_at_offset + new_pan_offset;

    // ── Write back ──────────────────────────────────────────────────────

    if let Some(moba) = world.resource_mut::<MobaCamera>() {
        moba.zoom = new_zoom;
        moba.pan_offset = new_pan_offset;
    }

    if let Some(camera) = world.resource_mut::<euca_render::Camera>() {
        camera.eye = eye;
        camera.target = target;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::{Transform, Vec3};
    use euca_scene::GlobalTransform;

    /// Helper: set up a world with a hero entity and the required resources.
    fn setup_world() -> (World, Entity) {
        let mut world = World::new();

        // Spawn hero at (10, 0, 5) with GlobalTransform
        let hero = world.spawn(GlobalTransform(Transform::from_translation(Vec3::new(
            10.0, 0.0, 5.0,
        ))));

        // Insert resources
        world.insert_resource(MobaCamera {
            follow_entity: Some(hero),
            ..MobaCamera::default()
        });
        world.insert_resource(euca_render::Camera::default());

        (world, hero)
    }

    #[test]
    fn camera_follows_entity() {
        let (mut world, _hero) = setup_world();

        moba_camera_system(&mut world);

        let cam = world.resource::<euca_render::Camera>().unwrap();

        // Eye = hero_pos + offset * zoom = (10,0,5) + (0,8,5)*1.0 = (10,8,10)
        let expected_eye = Vec3::new(10.0, 8.0, 10.0);
        assert!(
            (cam.eye.x - expected_eye.x).abs() < 1e-5,
            "eye.x: expected {}, got {}",
            expected_eye.x,
            cam.eye.x
        );
        assert!(
            (cam.eye.y - expected_eye.y).abs() < 1e-5,
            "eye.y: expected {}, got {}",
            expected_eye.y,
            cam.eye.y
        );
        assert!(
            (cam.eye.z - expected_eye.z).abs() < 1e-5,
            "eye.z: expected {}, got {}",
            expected_eye.z,
            cam.eye.z
        );

        // Target = hero_pos + look_at_offset = (10,0,5)
        assert!(
            (cam.target.x - 10.0).abs() < 1e-5,
            "target.x: expected 10.0, got {}",
            cam.target.x
        );
        assert!(
            (cam.target.y - 0.0).abs() < 1e-5,
            "target.y: expected 0.0, got {}",
            cam.target.y
        );
        assert!(
            (cam.target.z - 5.0).abs() < 1e-5,
            "target.z: expected 5.0, got {}",
            cam.target.z
        );

        // Pan offset should remain zero (locked = true by default)
        let moba = world.resource::<MobaCamera>().unwrap();
        assert_eq!(moba.pan_offset, Vec3::ZERO);
    }

    #[test]
    fn zoom_clamps_to_bounds() {
        let (mut world, _hero) = setup_world();

        // Large negative scroll to zoom out (increases zoom multiplier)
        {
            let mut input = euca_input::InputState::new();
            input.scroll_delta = -1000.0;
            world.insert_resource(input);
        }

        moba_camera_system(&mut world);

        let moba = world.resource::<MobaCamera>().unwrap();
        assert!(
            (moba.zoom - moba.max_zoom).abs() < 1e-5,
            "zoom should clamp to max_zoom ({}), got {}",
            moba.max_zoom,
            moba.zoom
        );

        // Now scroll the other way to go past min
        {
            let mut input = euca_input::InputState::new();
            input.scroll_delta = 1000.0;
            world.insert_resource(input);
        }

        moba_camera_system(&mut world);

        let moba = world.resource::<MobaCamera>().unwrap();
        assert!(
            (moba.zoom - moba.min_zoom).abs() < 1e-5,
            "zoom should clamp to min_zoom ({}), got {}",
            moba.min_zoom,
            moba.zoom
        );
    }

    #[test]
    fn locked_resets_pan_offset() {
        let (mut world, _hero) = setup_world();

        // Manually set a non-zero pan offset, then run with locked = true
        if let Some(moba) = world.resource_mut::<MobaCamera>() {
            moba.locked = true;
            moba.pan_offset = Vec3::new(100.0, 0.0, 200.0);
        }

        moba_camera_system(&mut world);

        let moba = world.resource::<MobaCamera>().unwrap();
        assert_eq!(
            moba.pan_offset,
            Vec3::ZERO,
            "locked mode should reset pan_offset to zero"
        );
    }

    #[test]
    fn edge_pan_moves_offset_when_unlocked() {
        let (mut world, _hero) = setup_world();

        // Unlock camera and place mouse in the left margin
        if let Some(moba) = world.resource_mut::<MobaCamera>() {
            moba.locked = false;
        }

        let mut input = euca_input::InputState::new();
        input.set_mouse_position(10.0, 500.0); // x < margin (50)
        world.insert_resource(input);
        world.insert_resource(ScreenSize {
            width: 1920.0,
            height: 1080.0,
        });

        // Time::new() starts with delta=0; set a known non-zero value.
        let mut time = euca_core::Time::new();
        time.delta = 1.0 / 60.0;
        world.insert_resource(time);

        moba_camera_system(&mut world);

        let moba = world.resource::<MobaCamera>().unwrap();
        // Pan offset X should have decreased (panned left)
        assert!(
            moba.pan_offset.x < 0.0,
            "edge pan left should decrease pan_offset.x, got {}",
            moba.pan_offset.x
        );
        // Z should be unchanged (mouse not in top/bottom margin)
        assert!(
            moba.pan_offset.z.abs() < 1e-5,
            "pan_offset.z should remain ~0, got {}",
            moba.pan_offset.z
        );
    }

    #[test]
    fn no_follow_entity_is_noop() {
        let mut world = World::new();
        world.insert_resource(MobaCamera::default());
        let original_cam = euca_render::Camera::default();
        let eye_before = original_cam.eye;
        let target_before = original_cam.target;
        world.insert_resource(original_cam);

        moba_camera_system(&mut world);

        let cam = world.resource::<euca_render::Camera>().unwrap();
        assert_eq!(cam.eye, eye_before);
        assert_eq!(cam.target, target_before);
    }

    #[test]
    fn graceful_without_input_state() {
        // System should still follow without InputState — just no panning/zoom
        let (mut world, _hero) = setup_world();

        // No InputState inserted — should not panic
        moba_camera_system(&mut world);

        let cam = world.resource::<euca_render::Camera>().unwrap();
        // Should still have moved to follow hero
        let expected_eye = Vec3::new(10.0, 8.0, 10.0);
        assert!((cam.eye.x - expected_eye.x).abs() < 1e-5);
        assert!((cam.eye.y - expected_eye.y).abs() < 1e-5);
        assert!((cam.eye.z - expected_eye.z).abs() < 1e-5);
    }
}
