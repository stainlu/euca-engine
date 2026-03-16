//! Arena game: top-down multiplayer, projectiles, health, last player standing.

use euca_ecs::{Entity, Query, World};
use euca_math::Vec3;
use euca_net::NetworkId;
use euca_physics::{Collider, Velocity};
use euca_scene::{GlobalTransform, LocalTransform};

/// Player health component.
#[derive(Clone, Copy, Debug)]
pub struct Health {
    pub current: i32,
    pub max: i32,
}

impl Health {
    pub fn new(max: i32) -> Self {
        Self { current: max, max }
    }

    pub fn is_dead(&self) -> bool {
        self.current <= 0
    }
}

/// Marks an entity as a projectile.
#[derive(Clone, Copy, Debug)]
pub struct Projectile {
    /// Who fired this (NetworkId of the shooter).
    pub owner: u64,
    /// Ticks remaining before auto-despawn.
    pub lifetime: u32,
    /// Damage dealt on hit.
    pub damage: i32,
}

/// Game state resource.
#[derive(Clone, Debug)]
pub struct ArenaState {
    pub round_active: bool,
    pub winner: Option<u64>, // NetworkId of winner
}

impl ArenaState {
    pub fn new() -> Self {
        Self {
            round_active: true,
            winner: None,
        }
    }
}

impl Default for ArenaState {
    fn default() -> Self {
        Self::new()
    }
}

/// Projectile speed (world units per tick at 60Hz).
const PROJECTILE_SPEED: f32 = 15.0;
/// Projectile lifetime in ticks.
const PROJECTILE_LIFETIME: u32 = 120; // 2 seconds at 60Hz
/// Projectile damage.
const PROJECTILE_DAMAGE: i32 = 1;
/// Projectile collider radius.
const PROJECTILE_RADIUS: f32 = 0.15;

/// Spawn a projectile from a player position in a given direction.
pub fn spawn_projectile(world: &mut World, owner: u64, origin: Vec3, direction: Vec3) -> Entity {
    let dir = if direction.length_squared() > 0.001 {
        direction.normalize()
    } else {
        Vec3::Z // default forward
    };

    let spawn_pos = origin + dir * 0.8; // offset from player center
    let e = world.spawn(LocalTransform(euca_math::Transform::from_translation(
        spawn_pos,
    )));
    world.insert(e, GlobalTransform::default());
    world.insert(
        e,
        Velocity {
            linear: dir * PROJECTILE_SPEED,
            angular: Vec3::ZERO,
        },
    );
    world.insert(e, Collider::sphere(PROJECTILE_RADIUS));
    world.insert(
        e,
        Projectile {
            owner,
            lifetime: PROJECTILE_LIFETIME,
            damage: PROJECTILE_DAMAGE,
        },
    );
    e
}

/// System: move projectiles, check collisions with players, apply damage.
pub fn projectile_system(world: &mut World) {
    // Move projectiles (velocity already handled by physics, just tick lifetime)
    let projectiles: Vec<(Entity, Projectile, Vec3)> = {
        let query = Query::<(Entity, &Projectile, &GlobalTransform)>::new(world);
        query
            .iter()
            .map(|(e, p, gt)| (e, *p, gt.0.translation))
            .collect()
    };

    let players: Vec<(Entity, u64, Vec3, f32)> = {
        let query = Query::<(Entity, &NetworkId, &GlobalTransform)>::new(world);
        query
            .iter()
            .filter_map(|(e, nid, gt)| {
                // Only players (with Health) — not projectiles
                world.get::<Health>(e)?;
                let radius = world
                    .get::<Collider>(e)
                    .map(|c| match &c.shape {
                        euca_physics::ColliderShape::Sphere { radius } => *radius,
                        euca_physics::ColliderShape::Aabb { hx, .. } => *hx,
                    })
                    .unwrap_or(0.5);
                Some((e, nid.0, gt.0.translation, radius))
            })
            .collect()
    };

    let mut despawn_list = Vec::new();
    let mut damage_list: Vec<(Entity, i32)> = Vec::new();

    for (proj_entity, proj, proj_pos) in &projectiles {
        // Check lifetime
        let mut proj_copy = *proj;
        proj_copy.lifetime = proj_copy.lifetime.saturating_sub(1);
        if proj_copy.lifetime == 0 {
            despawn_list.push(*proj_entity);
            continue;
        }

        // Update lifetime
        if let Some(p) = world.get_mut::<Projectile>(*proj_entity) {
            p.lifetime = proj_copy.lifetime;
        }

        // Check collision with players
        for (player_entity, player_nid, player_pos, player_radius) in &players {
            if *player_nid == proj.owner {
                continue; // Don't hit yourself
            }
            let dist = (*proj_pos - *player_pos).length();
            if dist < PROJECTILE_RADIUS + player_radius {
                damage_list.push((*player_entity, proj.damage));
                despawn_list.push(*proj_entity);
                break;
            }
        }
    }

    // Apply damage
    for (entity, damage) in &damage_list {
        if let Some(health) = world.get_mut::<Health>(*entity) {
            health.current -= damage;
        }
    }

    // Despawn dead projectiles
    for entity in &despawn_list {
        if world.is_alive(*entity) {
            world.despawn(*entity);
        }
    }
}

/// System: check for eliminated players and determine winner.
pub fn elimination_system(world: &mut World) {
    let alive_players: Vec<(Entity, u64)> = {
        let query = Query::<(Entity, &NetworkId, &Health)>::new(world);
        query
            .iter()
            .filter(|(_, _, h)| !h.is_dead())
            .map(|(e, nid, _)| (e, nid.0))
            .collect()
    };

    // Check win condition
    let Some(state) = world.resource_mut::<ArenaState>() else {
        return;
    };
    if state.round_active && alive_players.len() <= 1 {
        state.round_active = false;
        state.winner = alive_players.first().map(|(_, nid)| *nid);
        if let Some(winner) = state.winner {
            log::info!("Round over! Winner: NetworkId {winner}");
        } else {
            log::info!("Round over! No survivors.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Transform;

    #[test]
    fn health_system() {
        let mut h = Health::new(3);
        assert!(!h.is_dead());
        h.current -= 3;
        assert!(h.is_dead());
    }

    #[test]
    fn spawn_projectile_creates_entity() {
        let mut world = World::new();
        let e = spawn_projectile(&mut world, 1, Vec3::ZERO, Vec3::Z);
        assert!(world.is_alive(e));
        assert!(world.get::<Projectile>(e).is_some());
        assert!(world.get::<Velocity>(e).is_some());
    }

    #[test]
    fn projectile_despawns_after_lifetime() {
        let mut world = World::new();
        let e = spawn_projectile(&mut world, 1, Vec3::ZERO, Vec3::Z);
        // Set lifetime to 1
        if let Some(p) = world.get_mut::<Projectile>(e) {
            p.lifetime = 1;
        }
        euca_scene::transform_propagation_system(&mut world);
        projectile_system(&mut world);
        assert!(!world.is_alive(e));
    }

    #[test]
    fn arena_state_default() {
        let state = ArenaState::new();
        assert!(state.round_active);
        assert!(state.winner.is_none());
    }
}
