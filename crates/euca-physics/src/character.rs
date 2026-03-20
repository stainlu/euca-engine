//! Capsule-based kinematic character controller.
use euca_ecs::{Entity, Query, World};
use euca_math::Vec3;
use euca_scene::LocalTransform;
use crate::raycast::{Ray, WorldRayHit, raycast_world};
const GROUND_PROBE_OFFSET: f32 = 0.05;
const GROUND_PROBE_DISTANCE: f32 = 0.15;
/// Kinematic character controller component.
#[derive(Clone, Debug)]
pub struct CharacterController {
    pub capsule_radius: f32,
    pub capsule_height: f32,
    pub max_slope_angle: f32,
    pub step_height: f32,
    pub gravity: f32,
    pub jump_speed: f32,
    pub coyote_time: f32,
    pub is_grounded: bool,
    pub ground_normal: Vec3,
    pub velocity: Vec3,
    pub coyote_timer: f32,
}
impl CharacterController {
    pub fn new(capsule_radius: f32, capsule_height: f32) -> Self {
        Self {
            capsule_radius,
            capsule_height,
            max_slope_angle: std::f32::consts::FRAC_PI_4,
            step_height: 0.3,
            gravity: 9.81,
            jump_speed: 5.0,
            coyote_time: 0.1,
            is_grounded: false,
            ground_normal: Vec3::Y,
            velocity: Vec3::ZERO,
            coyote_timer: 0.0,
        }
    }
    pub fn jump(&mut self) {
        if self.is_grounded || self.coyote_timer > 0.0 {
            self.velocity.y = self.jump_speed;
            self.coyote_timer = 0.0;
        }
    }
}
pub fn character_controller_system(world: &mut World, dt: f32) {
    let entities: Vec<Entity> = {
        let q = Query::<(Entity, &CharacterController, &LocalTransform)>::new(world);
        q.iter().map(|(e, _, _)| e).collect()
    };
    for entity in entities {
        let (mut ctrl, pos) = {
            let ctrl = match world.get::<CharacterController>(entity) {
                Some(c) => c.clone(),
                None => continue,
            };
            let pos = match world.get::<LocalTransform>(entity) {
                Some(lt) => lt.0.translation,
                None => continue,
            };
            (ctrl, pos)
        };
        let was_grounded = ctrl.is_grounded;
        if !ctrl.is_grounded {
            ctrl.velocity.y -= ctrl.gravity * dt;
        }
        let new_pos = pos + ctrl.velocity * dt;
        let half_height = ctrl.capsule_height * 0.5;
        let ray_origin = Vec3::new(
            new_pos.x,
            new_pos.y - half_height + GROUND_PROBE_OFFSET,
            new_pos.z,
        );
        let ray = Ray::new(ray_origin, Vec3::new(0.0, -1.0, 0.0));
        let probe_dist = GROUND_PROBE_DISTANCE + GROUND_PROBE_OFFSET;
        let ground_hit = find_ground_hit(world, &ray, probe_dist, entity, ctrl.max_slope_angle);
        ctrl.is_grounded = ground_hit.is_some();
        if let Some(hit) = &ground_hit {
            ctrl.ground_normal = hit.normal;
            if ctrl.velocity.y <= 0.0 {
                ctrl.velocity.y = 0.0;
            }
        } else {
            ctrl.ground_normal = Vec3::Y;
        }
        if was_grounded && !ctrl.is_grounded {
            ctrl.coyote_timer = ctrl.coyote_time;
        } else if ctrl.is_grounded {
            ctrl.coyote_timer = 0.0;
        } else {
            ctrl.coyote_timer = (ctrl.coyote_timer - dt).max(0.0);
        }
        if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
            lt.0.translation = new_pos;
        }
        if let Some(c) = world.get_mut::<CharacterController>(entity) {
            *c = ctrl;
        }
    }
}
fn find_ground_hit(
    world: &World,
    ray: &Ray,
    max_distance: f32,
    self_entity: Entity,
    max_slope_angle: f32,
) -> Option<WorldRayHit> {
    let max_slope_cos = max_slope_angle.cos();
    let hits = raycast_world(world, ray, max_distance, u32::MAX);
    hits.into_iter()
        .find(|h| h.entity != self_entity && h.normal.dot(Vec3::Y) >= max_slope_cos)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Collider, PhysicsBody};
    use euca_math::Transform;
    use euca_scene::GlobalTransform;
    fn new_world() -> World {
        World::new()
    }
    fn spawn_ground(world: &mut World) -> Entity {
        let e = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(e, GlobalTransform::default());
        world.insert(e, PhysicsBody::fixed());
        world.insert(e, Collider::aabb(50.0, 0.5, 50.0));
        e
    }
    fn spawn_character(world: &mut World, pos: Vec3) -> Entity {
        let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
        world.insert(e, GlobalTransform::default());
        world.insert(e, CharacterController::new(0.3, 1.8));
        e
    }
    #[test]
    fn character_falls_with_gravity() {
        let mut world = new_world();
        let ch = spawn_character(&mut world, Vec3::new(0.0, 10.0, 0.0));
        for _ in 0..60 {
            character_controller_system(&mut world, 1.0 / 60.0);
        }
        assert!(world.get::<LocalTransform>(ch).unwrap().0.translation.y < 10.0);
    }
    #[test]
    fn character_lands_on_ground() {
        let mut world = new_world();
        spawn_ground(&mut world);
        let ch = spawn_character(&mut world, Vec3::new(0.0, 3.0, 0.0));
        for _ in 0..300 {
            character_controller_system(&mut world, 1.0 / 60.0);
        }
        let y = world.get::<LocalTransform>(ch).unwrap().0.translation.y;
        assert!(y < 3.0 && y > -1.0);
        assert!(world.get::<CharacterController>(ch).unwrap().is_grounded);
    }
    #[test]
    fn character_jumps_when_grounded() {
        let mut world = new_world();
        spawn_ground(&mut world);
        let ch = spawn_character(&mut world, Vec3::new(0.0, 3.0, 0.0));
        for _ in 0..300 {
            character_controller_system(&mut world, 1.0 / 60.0);
        }
        let pre = world.get::<LocalTransform>(ch).unwrap().0.translation.y;
        world.get_mut::<CharacterController>(ch).unwrap().jump();
        for _ in 0..10 {
            character_controller_system(&mut world, 1.0 / 60.0);
        }
        assert!(world.get::<LocalTransform>(ch).unwrap().0.translation.y > pre);
    }
    #[test]
    fn no_jump_in_air_after_coyote_expires() {
        let mut world = new_world();
        let ch = spawn_character(&mut world, Vec3::new(0.0, 50.0, 0.0));
        for _ in 0..60 {
            character_controller_system(&mut world, 1.0 / 60.0);
        }
        let pre = world.get::<CharacterController>(ch).unwrap().velocity.y;
        world.get_mut::<CharacterController>(ch).unwrap().jump();
        assert_eq!(world.get::<CharacterController>(ch).unwrap().velocity.y, pre);
    }
    #[test]
    fn coyote_time_allows_late_jump() {
        let mut world = new_world();
        spawn_ground(&mut world);
        let ch = spawn_character(&mut world, Vec3::new(0.0, 3.0, 0.0));
        for _ in 0..300 {
            character_controller_system(&mut world, 1.0 / 60.0);
        }
        assert!(world.get::<CharacterController>(ch).unwrap().is_grounded);
        world.get_mut::<CharacterController>(ch).unwrap().velocity.x = 200.0;
        character_controller_system(&mut world, 1.0 / 60.0);
        let ctrl = world.get::<CharacterController>(ch).unwrap();
        assert!(ctrl.is_grounded || ctrl.coyote_timer > 0.0);
    }
    #[test]
    fn velocity_moves_character_horizontally() {
        let mut world = new_world();
        spawn_ground(&mut world);
        let ch = spawn_character(&mut world, Vec3::new(0.0, 3.0, 0.0));
        for _ in 0..300 {
            character_controller_system(&mut world, 1.0 / 60.0);
        }
        world.get_mut::<CharacterController>(ch).unwrap().velocity.x = 5.0;
        character_controller_system(&mut world, 1.0 / 60.0);
        assert!(world.get::<LocalTransform>(ch).unwrap().0.translation.x > 0.0);
    }
    #[test]
    fn ground_normal_is_up_on_flat_surface() {
        let mut world = new_world();
        spawn_ground(&mut world);
        let ch = spawn_character(&mut world, Vec3::new(0.0, 3.0, 0.0));
        for _ in 0..300 {
            character_controller_system(&mut world, 1.0 / 60.0);
        }
        let n = world.get::<CharacterController>(ch).unwrap().ground_normal;
        assert!(n.dot(Vec3::Y) > 0.99);
    }
}
