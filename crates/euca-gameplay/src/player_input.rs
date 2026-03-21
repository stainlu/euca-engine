//! Player input -> command translation for MOBA-style click-to-move / attack-target.
//!
//! Reads [`InputState`](euca_input::InputState) + [`Camera`](euca_render::Camera) +
//! [`ViewportSize`] resources, finds the [`PlayerHero`] entity, and converts
//! mouse clicks and key presses into [`PlayerCommand`]s queued on the entity's
//! [`PlayerCommandQueue`].
//!
//! Components: [`PlayerHero`] (marker), [`PlayerCommandQueue`].
//! Types: [`PlayerCommand`], [`ViewportSize`].
//! System: [`player_input_system`].

use euca_ecs::{Entity, Query, World};
use euca_input::InputState;
use euca_math::Vec3;
use euca_render::Camera;
use euca_scene::LocalTransform;

use crate::abilities::AbilitySlot;
use crate::health::Health;
use crate::teams::Team;

// ── Components & types ─────────────────────────────────────────────────────

/// Marker: this entity is the locally-controlled player hero.
///
/// Exactly one entity in the world should have this component. Systems like
/// [`player_input_system`] use it to route input to the correct entity.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerHero;

/// A single command issued by the player.
#[derive(Clone, Debug, PartialEq)]
pub enum PlayerCommand {
    /// Move to a world-space position (right-click on ground).
    MoveTo(Vec3),
    /// Attack a specific enemy entity (right-click on enemy).
    AttackTarget(Entity),
    /// Stop all movement and actions (S key).
    Stop,
    /// Self-cast an ability in the given slot (Q/W/E/R keys).
    /// Targeted casting is a future enhancement.
    UseAbility { slot: AbilitySlot },
}

/// Queue of player commands on the hero entity, consumed by downstream systems.
#[derive(Clone, Debug, Default)]
pub struct PlayerCommandQueue {
    pub commands: Vec<PlayerCommand>,
}

impl PlayerCommandQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, command: PlayerCommand) {
        self.commands.push(command);
    }

    pub fn drain(&mut self) -> Vec<PlayerCommand> {
        std::mem::take(&mut self.commands)
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

// ── Resource ───────────────────────────────────────────────────────────────

/// Viewport dimensions resource. Insert into the world so the input system
/// knows the screen size for screen-to-ray conversion.
#[derive(Clone, Debug)]
pub struct ViewportSize {
    pub width: f32,
    pub height: f32,
}

impl ViewportSize {
    pub fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }
}

// ── Constants ──────────────────────────────────────────────────────────────

/// Radius used for click-target detection (world units).
const CLICK_HIT_RADIUS: f32 = 1.0;

// ── Ground intersection ────────────────────────────────────────────────────

/// Intersect a ray with the Y=0 ground plane.
///
/// Returns the intersection point if the ray points toward the plane
/// (i.e. the intersection is in front of the ray origin).
pub fn ray_ground_intersection(origin: Vec3, direction: Vec3) -> Option<Vec3> {
    if direction.y.abs() < 1e-6 {
        return None;
    }
    let t = -origin.y / direction.y;
    if t < 0.0 {
        return None;
    }
    Some(origin + direction * t)
}

// ── System ─────────────────────────────────────────────────────────────────

/// Translate player input into [`PlayerCommand`]s on the [`PlayerHero`] entity.
///
/// Each frame:
/// - Right mouse button just pressed:
///   1. Build a world-space ray from the mouse position via
///      [`Camera::screen_to_ray`].
///   2. Raycast against all entities with `Health` + `Team` (enemies on a
///      different team). If hit, queue `AttackTarget`.
///   3. If no enemy hit, intersect the ray with the Y=0 ground plane and
///      queue `MoveTo`.
/// - S key just pressed: queue `Stop`.
/// - Q/W/E/R just pressed: queue `UseAbility { slot }`.
pub fn player_input_system(world: &mut World) {
    // --- Find the player hero entity ---
    let player_entity: Option<Entity> = {
        let query = Query::<(Entity, &PlayerHero)>::new(world);
        query.iter().next().map(|(e, _)| e)
    };
    let player_entity = match player_entity {
        Some(e) => e,
        None => return,
    };

    // --- Snapshot input state ---
    let (mouse_pos, right_click, s_pressed, q_pressed, w_pressed, e_pressed, r_pressed) = {
        let input = match world.resource::<InputState>() {
            Some(i) => i,
            None => return,
        };
        (
            input.mouse_position,
            input.is_just_pressed(&euca_input::InputKey::MouseRight),
            input.is_just_pressed(&euca_input::InputKey::Key("S".into())),
            input.is_just_pressed(&euca_input::InputKey::Key("Q".into())),
            input.is_just_pressed(&euca_input::InputKey::Key("W".into())),
            input.is_just_pressed(&euca_input::InputKey::Key("E".into())),
            input.is_just_pressed(&euca_input::InputKey::Key("R".into())),
        )
    };

    let mut commands: Vec<PlayerCommand> = Vec::new();

    // --- Right-click: attack-target or move-to ---
    if right_click {
        let ray = {
            let camera = world.resource::<Camera>();
            let viewport = world.resource::<ViewportSize>();
            match (camera, viewport) {
                (Some(cam), Some(vp)) => {
                    Some(cam.screen_to_ray(mouse_pos[0], mouse_pos[1], vp.width, vp.height))
                }
                _ => None,
            }
        };

        if let Some((ray_origin, ray_dir)) = ray {
            let player_team = world.get::<Team>(player_entity).map(|t| t.0);

            let hit_enemy = find_enemy_under_ray(world, ray_origin, ray_dir, player_team);

            if let Some(enemy) = hit_enemy {
                commands.push(PlayerCommand::AttackTarget(enemy));
            } else if let Some(ground_point) = ray_ground_intersection(ray_origin, ray_dir) {
                commands.push(PlayerCommand::MoveTo(ground_point));
            }
        }
    }

    // --- S key: stop ---
    if s_pressed {
        commands.push(PlayerCommand::Stop);
    }

    // --- Ability keys: Q/W/E/R (self-cast for now) ---
    if q_pressed {
        commands.push(PlayerCommand::UseAbility {
            slot: AbilitySlot::Q,
        });
    }
    if w_pressed {
        commands.push(PlayerCommand::UseAbility {
            slot: AbilitySlot::W,
        });
    }
    if e_pressed {
        commands.push(PlayerCommand::UseAbility {
            slot: AbilitySlot::E,
        });
    }
    if r_pressed {
        commands.push(PlayerCommand::UseAbility {
            slot: AbilitySlot::R,
        });
    }

    // --- Queue commands on the player entity ---
    if !commands.is_empty()
        && let Some(queue) = world.get_mut::<PlayerCommandQueue>(player_entity)
    {
        for cmd in commands {
            queue.push(cmd);
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Find the closest enemy entity whose bounding sphere intersects the ray.
///
/// Returns the closest hit enemy, or `None` if no enemy is under the cursor.
fn find_enemy_under_ray(
    world: &World,
    ray_origin: Vec3,
    ray_dir: Vec3,
    player_team: Option<u8>,
) -> Option<Entity> {
    let candidates: Vec<(Entity, Vec3)> = {
        let query = Query::<(Entity, &LocalTransform, &Health, &Team)>::new(world);
        query
            .iter()
            .filter(|(_, _, health, team)| {
                !health.is_dead() && player_team.is_some_and(|pt| pt != team.0)
            })
            .map(|(e, lt, _, _)| (e, lt.0.translation))
            .collect()
    };

    let mut best: Option<(Entity, f32)> = None;

    for (entity, pos) in &candidates {
        if let Some(dist) = ray_sphere_distance(ray_origin, ray_dir, *pos, CLICK_HIT_RADIUS) {
            let closer = best.is_none_or(|(_, best_dist)| dist < best_dist);
            if closer {
                best = Some((*entity, dist));
            }
        }
    }

    best.map(|(e, _)| e)
}

/// Distance from ray origin to the closest intersection with a sphere.
///
/// Returns `None` if the ray misses the sphere entirely.
fn ray_sphere_distance(
    ray_origin: Vec3,
    ray_dir: Vec3,
    sphere_center: Vec3,
    sphere_radius: f32,
) -> Option<f32> {
    let oc = ray_origin - sphere_center;
    let a = ray_dir.dot(ray_dir);
    let b = 2.0 * oc.dot(ray_dir);
    let c = oc.dot(oc) - sphere_radius * sphere_radius;
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return None;
    }

    let sqrt_d = discriminant.sqrt();
    let t1 = (-b - sqrt_d) / (2.0 * a);
    let t2 = (-b + sqrt_d) / (2.0 * a);

    if t1 >= 0.0 {
        Some(t1)
    } else if t2 >= 0.0 {
        Some(t2)
    } else {
        None // Sphere is behind the ray.
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use euca_ecs::Events;
    use euca_math::Transform;

    // ── Ground intersection math ──

    #[test]
    fn ground_intersection_straight_down() {
        let origin = Vec3::new(0.0, 10.0, 0.0);
        let direction = Vec3::new(0.0, -1.0, 0.0);
        let hit = ray_ground_intersection(origin, direction).expect("should hit ground");
        assert!((hit.x).abs() < 1e-4);
        assert!((hit.y).abs() < 1e-4);
        assert!((hit.z).abs() < 1e-4);
    }

    #[test]
    fn ground_intersection_angled() {
        let origin = Vec3::new(0.0, 10.0, 0.0);
        let direction = Vec3::new(1.0, -1.0, 0.0).normalize();
        let hit = ray_ground_intersection(origin, direction).expect("should hit ground");
        assert!((hit.x - 10.0).abs() < 1e-4, "x should be 10, got {}", hit.x);
        assert!((hit.y).abs() < 1e-4);
    }

    #[test]
    fn ground_intersection_parallel_misses() {
        let origin = Vec3::new(0.0, 5.0, 0.0);
        let direction = Vec3::new(1.0, 0.0, 0.0);
        assert!(ray_ground_intersection(origin, direction).is_none());
    }

    #[test]
    fn ground_intersection_pointing_away() {
        let origin = Vec3::new(0.0, 5.0, 0.0);
        let direction = Vec3::new(0.0, 1.0, 0.0);
        assert!(ray_ground_intersection(origin, direction).is_none());
    }

    // ── Enemy detection logic ──

    #[test]
    fn find_enemy_under_ray_hits_closest() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let near_enemy = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, 0.0, 5.0,
        ))));
        world.insert(near_enemy, Health::new(100.0));
        world.insert(near_enemy, Team(2));

        let far_enemy = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, 0.0, 20.0,
        ))));
        world.insert(far_enemy, Health::new(100.0));
        world.insert(far_enemy, Team(2));

        let hit = find_enemy_under_ray(&world, Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), Some(1));
        assert_eq!(
            hit.unwrap().index(),
            near_enemy.index(),
            "should hit the closer enemy"
        );
    }

    #[test]
    fn find_enemy_ignores_same_team() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let ally = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, 0.0, 5.0,
        ))));
        world.insert(ally, Health::new(100.0));
        world.insert(ally, Team(1));

        let hit = find_enemy_under_ray(&world, Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), Some(1));
        assert!(hit.is_none(), "should not target same-team entity");
    }

    #[test]
    fn find_enemy_ignores_dead() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let dead_enemy = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, 0.0, 5.0,
        ))));
        world.insert(
            dead_enemy,
            Health {
                current: 0.0,
                max: 100.0,
            },
        );
        world.insert(dead_enemy, Team(2));

        let hit = find_enemy_under_ray(&world, Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), Some(1));
        assert!(hit.is_none(), "should not target dead entity");
    }

    // ── Stop command ──

    #[test]
    fn stop_command_on_s_key() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let mut input = InputState::new();
        input.press(euca_input::InputKey::Key("S".into()));
        world.insert_resource(input);

        let hero = world.spawn(PlayerHero);
        world.insert(hero, PlayerCommandQueue::new());

        player_input_system(&mut world);

        let queue = world
            .get::<PlayerCommandQueue>(hero)
            .expect("hero should have command queue");
        assert!(
            queue
                .commands
                .iter()
                .any(|c| matches!(c, PlayerCommand::Stop)),
            "S key should queue a Stop command"
        );
    }

    // ── Ability keys ──

    #[test]
    fn ability_keys_queue_commands() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let mut input = InputState::new();
        input.press(euca_input::InputKey::Key("Q".into()));
        input.press(euca_input::InputKey::Key("R".into()));
        world.insert_resource(input);

        let hero = world.spawn(PlayerHero);
        world.insert(hero, PlayerCommandQueue::new());

        player_input_system(&mut world);

        let queue = world.get::<PlayerCommandQueue>(hero).unwrap();
        assert!(queue.commands.iter().any(|c| matches!(
            c,
            PlayerCommand::UseAbility {
                slot: AbilitySlot::Q
            }
        )));
        assert!(queue.commands.iter().any(|c| matches!(
            c,
            PlayerCommand::UseAbility {
                slot: AbilitySlot::R
            }
        )));
    }

    // ── Right-click ground move ──

    #[test]
    fn right_click_ground_queues_move_to() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        // Orthographic camera looking down from above.
        let mut camera = Camera::new(Vec3::new(0.0, 20.0, 0.001), Vec3::ZERO);
        camera.orthographic = true;
        camera.ortho_size = 10.0;
        world.insert_resource(camera);
        world.insert_resource(ViewportSize::new(800.0, 600.0));

        // Right-click at screen center.
        let mut input = InputState::new();
        input.set_mouse_position(400.0, 300.0);
        input.press(euca_input::InputKey::MouseRight);
        world.insert_resource(input);

        let hero = world.spawn(PlayerHero);
        world.insert(hero, PlayerCommandQueue::new());
        world.insert(hero, Team(1));

        player_input_system(&mut world);

        let queue = world.get::<PlayerCommandQueue>(hero).unwrap();
        assert_eq!(queue.commands.len(), 1);
        match &queue.commands[0] {
            PlayerCommand::MoveTo(pos) => {
                assert!(
                    pos.y.abs() < 0.1,
                    "move-to Y should be near ground, got {}",
                    pos.y
                );
            }
            other => panic!("expected MoveTo, got {:?}", other),
        }
    }

    // ── No hero -> no-op ──

    #[test]
    fn no_player_hero_is_noop() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let mut input = InputState::new();
        input.press(euca_input::InputKey::Key("S".into()));
        world.insert_resource(input);

        // No PlayerHero -- must not panic.
        player_input_system(&mut world);
    }

    // ── Queue drain ──

    #[test]
    fn command_queue_drain() {
        let mut queue = PlayerCommandQueue::new();
        queue.push(PlayerCommand::Stop);
        queue.push(PlayerCommand::MoveTo(Vec3::ZERO));
        assert!(!queue.is_empty());

        let drained = queue.drain();
        assert_eq!(drained.len(), 2);
        assert!(queue.is_empty());
    }
}
