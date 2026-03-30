//! ECS systems for the building module — backdoor protection, fortification,
//! and barracks-death creep upgrades.
//!
//! These systems read/write the pure-data structs from [`crate::building`] as
//! ECS components and resources, bridging the data layer to the game world.

use euca_ecs::{Entity, Events, Query, World};
use euca_math::Vec3;
use euca_scene::LocalTransform;

use crate::building::{
    BackdoorProtection, BuildingStats, BuildingType, Fortification, Lane, TowerAggro,
    backdoor_damage_modifier, is_building_invulnerable, tick_fortification,
    update_backdoor_protection, update_tower_aggro,
};
use crate::combat::EntityRole;
use crate::health::{Dead, DeathEvent, Health};
use crate::teams::Team;

// ---------------------------------------------------------------------------
// Team fortification resource
// ---------------------------------------------------------------------------

/// Per-team fortification state, keyed by team ID.
///
/// Stored as a world resource. Each team gets its own independent glyph.
#[derive(Debug, Clone, Default)]
pub struct TeamFortifications {
    pub teams: std::collections::HashMap<u32, Fortification>,
}

impl TeamFortifications {
    /// Get or create the fortification state for a team.
    pub fn get_or_insert(&mut self, team: u32) -> &mut Fortification {
        self.teams.entry(team).or_default()
    }

    /// Get the fortification state for a team (read-only).
    pub fn get(&self, team: u32) -> Option<&Fortification> {
        self.teams.get(&team)
    }
}

// ---------------------------------------------------------------------------
// Super creep flags
// ---------------------------------------------------------------------------

/// Tracks which barracks have been destroyed, per team.
///
/// When a barracks is destroyed, the corresponding flag is set. The creep
/// spawner reads these flags to upgrade lane creeps to super/mega creeps.
#[derive(Debug, Clone, Default)]
pub struct DestroyedBarracks {
    /// Key: (team_that_lost_barracks, lane), Value: (melee_destroyed, ranged_destroyed).
    pub flags: std::collections::HashMap<(u32, Lane), (bool, bool)>,
}

impl DestroyedBarracks {
    /// Record that a barracks was destroyed.
    pub fn mark_destroyed(&mut self, team: u32, lane: Lane, building_type: BuildingType) {
        let entry = self.flags.entry((team, lane)).or_insert((false, false));
        match building_type {
            BuildingType::MeleeBarracks => entry.0 = true,
            BuildingType::RangedBarracks => entry.1 = true,
            _ => {}
        }
    }

    /// Check if a lane has super melee creeps (enemy melee barracks destroyed).
    pub fn has_super_melee(&self, team: u32, lane: Lane) -> bool {
        self.flags
            .get(&(team, lane))
            .map(|(m, _)| *m)
            .unwrap_or(false)
    }

    /// Check if a lane has super ranged creeps (enemy ranged barracks destroyed).
    pub fn has_super_ranged(&self, team: u32, lane: Lane) -> bool {
        self.flags
            .get(&(team, lane))
            .map(|(_, r)| *r)
            .unwrap_or(false)
    }

    /// Check if ALL barracks for a team are destroyed (mega creeps).
    pub fn all_destroyed(&self, team: u32) -> bool {
        let lanes = [Lane::Top, Lane::Mid, Lane::Bot];
        lanes.iter().all(|lane| {
            self.flags
                .get(&(team, *lane))
                .map(|(m, r)| *m && *r)
                .unwrap_or(false)
        })
    }
}

// ---------------------------------------------------------------------------
// Backdoor protection system
// ---------------------------------------------------------------------------

/// Update backdoor protection for all buildings based on nearby enemy creeps.
///
/// For each entity with `BackdoorProtection` + `BuildingStats`:
/// - Count enemy creeps (entities with `EntityRole::Minion`) within the
///   protection's `check_radius`.
/// - If no enemy creeps nearby: activate protection (regen HP).
/// - If enemy creeps nearby: deactivate protection.
///
/// Also applies HP regeneration when protection is active.
pub fn backdoor_protection_system(world: &mut World, dt: f32) {
    // Collect buildings with backdoor protection.
    let buildings: Vec<(Entity, Vec3, u32)> = {
        let query = Query::<(Entity, &LocalTransform, &BuildingStats)>::new(world);
        query
            .iter()
            .filter(|(e, _, bs)| {
                bs.is_alive
                    && world.get::<BackdoorProtection>(*e).is_some()
                    && world.get::<Dead>(*e).is_none()
            })
            .map(|(e, lt, bs)| (e, lt.0.translation, bs.team))
            .collect()
    };

    // Collect enemy creeps (minions) with their positions and teams.
    let creeps: Vec<(Vec3, u8)> = {
        let query = Query::<(Entity, &LocalTransform, &Team, &EntityRole)>::new(world);
        query
            .iter()
            .filter(|(e, _, _, role)| {
                **role == EntityRole::Minion && world.get::<Dead>(*e).is_none()
            })
            .map(|(_, lt, team, _)| (lt.0.translation, team.0))
            .collect()
    };

    for (building_entity, building_pos, building_team) in &buildings {
        let check_radius = world
            .get::<BackdoorProtection>(*building_entity)
            .map(|bp| bp.check_radius)
            .unwrap_or(900.0);

        // Count enemy creeps within radius.
        let enemy_creeps_in_range = creeps.iter().any(|(creep_pos, creep_team)| {
            *creep_team as u32 != *building_team
                && (*creep_pos - *building_pos).length() <= check_radius
        });

        // Update protection state.
        if let Some(protection) = world.get_mut::<BackdoorProtection>(*building_entity) {
            update_backdoor_protection(protection, enemy_creeps_in_range);
        }

        // Apply HP regen when protection is active.
        let (active, regen) = world
            .get::<BackdoorProtection>(*building_entity)
            .map(|bp| (bp.active, bp.hp_regen_per_sec))
            .unwrap_or((false, 0.0));

        if active && regen > 0.0 {
            if let Some(bs) = world.get_mut::<BuildingStats>(*building_entity) {
                bs.current_hp = (bs.current_hp + regen * dt).min(bs.max_hp);
            }
            // Also sync with Health component if present.
            if let Some(health) = world.get_mut::<Health>(*building_entity) {
                health.current = (health.current + regen * dt).min(health.max);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fortification tick system
// ---------------------------------------------------------------------------

/// Tick all team fortification timers each frame.
pub fn fortification_tick_system(world: &mut World, dt: f32) {
    if let Some(forts) = world.resource_mut::<TeamFortifications>() {
        for fort in forts.teams.values_mut() {
            tick_fortification(fort, dt);
        }
    }
}

// ---------------------------------------------------------------------------
// Building damage modifier
// ---------------------------------------------------------------------------

/// Returns the damage multiplier for a building target, accounting for
/// backdoor protection and fortification invulnerability.
///
/// - Fortification active -> 0.0 (invulnerable).
/// - Backdoor active -> 0.25 (75% reduction).
/// - Neither -> 1.0 (full damage).
pub fn building_damage_multiplier(world: &World, target: Entity) -> f32 {
    // Check fortification first (team-wide invulnerability).
    if let Some(bs) = world.get::<BuildingStats>(target)
        && let Some(forts) = world.resource::<TeamFortifications>()
        && let Some(fort) = forts.get(bs.team)
        && is_building_invulnerable(fort)
    {
        return 0.0;
    }

    // Check backdoor protection.
    if let Some(protection) = world.get::<BackdoorProtection>(target) {
        return backdoor_damage_modifier(protection);
    }

    1.0
}

// ---------------------------------------------------------------------------
// Tower aggro integration
// ---------------------------------------------------------------------------

/// Update `TowerAggro` for all tower entities that have `BuildingStats`.
///
/// For each tower with `BuildingStats` + `TowerAggro`:
/// - Find the closest enemy within aggro range.
/// - Check if any enemy hero is attacking an allied hero nearby.
/// - Call `update_tower_aggro()` to decide the target.
pub fn building_tower_aggro_system(world: &mut World) {
    // Collect towers with TowerAggro.
    let towers: Vec<(Entity, Vec3, u32, f32)> = {
        let query = Query::<(Entity, &LocalTransform, &BuildingStats, &TowerAggro)>::new(world);
        query
            .iter()
            .filter(|(e, _, bs, _)| {
                bs.is_alive && bs.attack_damage.is_some() && world.get::<Dead>(*e).is_none()
            })
            .map(|(e, lt, bs, ta)| (e, lt.0.translation, bs.team, ta.aggro_range))
            .collect()
    };

    // Collect all living combat entities for target selection.
    let entities: Vec<(Entity, Vec3, u8, EntityRole)> = {
        let query = Query::<(Entity, &LocalTransform, &Team, &EntityRole)>::new(world);
        query
            .iter()
            .filter(|(e, _, _, _)| world.get::<Dead>(*e).is_none())
            .map(|(e, lt, team, role)| (e, lt.0.translation, team.0, *role))
            .collect()
    };

    // Check for heroes attacking allied heroes (for priority targeting).
    let attacked_allies: Vec<(u8, Entity)> = {
        let query = Query::<(Entity, &Team, &EntityRole, &crate::health::LastAttacker)>::new(world);
        query
            .iter()
            .filter(|(e, _, role, _)| **role == EntityRole::Hero && world.get::<Dead>(*e).is_none())
            .filter_map(|(_, team, _, la)| la.0.map(|attacker| (team.0, attacker)))
            .collect()
    };

    for (tower_entity, tower_pos, tower_team, aggro_range) in &towers {
        // Find closest enemy within aggro range.
        let mut closest_enemy: Option<(u64, f32)> = None;
        for (e, pos, team, _) in &entities {
            if *team as u32 == *tower_team {
                continue;
            }
            let dist = (*pos - *tower_pos).length();
            if dist <= *aggro_range {
                let better = closest_enemy
                    .map(|(_, best_dist)| dist < best_dist)
                    .unwrap_or(true);
                if better {
                    closest_enemy = Some((e.index() as u64, dist));
                }
            }
        }

        // Find enemy hero attacking an allied hero within range.
        let mut hero_attacking_ally: Option<u64> = None;
        for (ally_team, attacker) in &attacked_allies {
            if *ally_team as u32 != *tower_team {
                continue;
            }
            // Verify the attacker is an enemy hero within range.
            if let Some((_, attacker_pos, attacker_team, attacker_role)) =
                entities.iter().find(|(e, _, _, _)| *e == *attacker)
                && *attacker_role == EntityRole::Hero
                && *attacker_team as u32 != *tower_team
                && (*attacker_pos - *tower_pos).length() <= *aggro_range
            {
                hero_attacking_ally = Some(attacker.index() as u64);
                break;
            }
        }

        // Update the tower's aggro state.
        if let Some(aggro) = world.get_mut::<TowerAggro>(*tower_entity) {
            update_tower_aggro(aggro, hero_attacking_ally, closest_enemy.map(|(id, _)| id));
        }
    }
}

// ---------------------------------------------------------------------------
// Barracks death system
// ---------------------------------------------------------------------------

/// When a barracks dies, record it in `DestroyedBarracks` so future creep
/// waves on that lane produce super creeps.
pub fn barracks_death_system(world: &mut World) {
    let death_events: Vec<DeathEvent> = world
        .resource::<Events>()
        .map(|e| e.read::<DeathEvent>().cloned().collect())
        .unwrap_or_default();

    for event in death_events {
        let Some(bs) = world.get::<BuildingStats>(event.entity) else {
            continue;
        };

        let building_type = bs.building_type;
        let team = bs.team;
        let lane = bs.lane;

        // Only process barracks deaths.
        if !matches!(
            building_type,
            BuildingType::MeleeBarracks | BuildingType::RangedBarracks
        ) {
            continue;
        }

        let Some(lane) = lane else {
            continue;
        };

        // Mark the building as dead.
        if let Some(bs_mut) = world.get_mut::<BuildingStats>(event.entity) {
            bs_mut.is_alive = false;
        }

        // Record in the destroyed barracks tracker.
        if let Some(destroyed) = world.resource_mut::<DestroyedBarracks>() {
            destroyed.mark_destroyed(team, lane, building_type);
            log::info!("Barracks destroyed: team={team}, lane={lane:?}, type={building_type:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building::{self, building_stats};
    use euca_ecs::Events;
    use euca_math::Transform;

    fn setup_world() -> World {
        let mut world = World::new();
        world.insert_resource(Events::default());
        world.insert_resource(TeamFortifications::default());
        world.insert_resource(DestroyedBarracks::default());
        world
    }

    fn spawn_building(
        world: &mut World,
        pos: Vec3,
        building_type: BuildingType,
        team: u32,
        lane: Option<Lane>,
    ) -> Entity {
        let bs = building_stats(building_type, team, lane);
        let entity = world.spawn(LocalTransform(Transform::from_translation(pos)));
        world.insert(entity, Health::new(bs.max_hp));
        world.insert(entity, Team(team as u8));
        world.insert(entity, bs);
        world.insert(entity, BackdoorProtection::default());
        entity
    }

    fn spawn_creep(world: &mut World, pos: Vec3, team: u8) -> Entity {
        let entity = world.spawn(LocalTransform(Transform::from_translation(pos)));
        world.insert(entity, Health::new(300.0));
        world.insert(entity, Team(team));
        world.insert(entity, EntityRole::Minion);
        entity
    }

    // ── Backdoor protection ──

    #[test]
    fn backdoor_active_when_no_enemy_creeps() {
        let mut world = setup_world();
        let building = spawn_building(
            &mut world,
            Vec3::ZERO,
            BuildingType::Tier2Tower,
            1,
            Some(Lane::Mid),
        );

        // No enemy creeps -- protection should stay active.
        backdoor_protection_system(&mut world, 1.0);

        let bp = world.get::<BackdoorProtection>(building).unwrap();
        assert!(bp.active);
    }

    #[test]
    fn backdoor_deactivated_when_enemy_creeps_nearby() {
        let mut world = setup_world();
        let building = spawn_building(
            &mut world,
            Vec3::ZERO,
            BuildingType::Tier2Tower,
            1,
            Some(Lane::Mid),
        );

        // Enemy creep within 900 range.
        spawn_creep(&mut world, Vec3::new(5.0, 0.0, 0.0), 2);

        backdoor_protection_system(&mut world, 1.0);

        let bp = world.get::<BackdoorProtection>(building).unwrap();
        assert!(
            !bp.active,
            "protection should deactivate when enemy creeps nearby"
        );
    }

    #[test]
    fn backdoor_regen_heals_building() {
        let mut world = setup_world();
        let building = spawn_building(
            &mut world,
            Vec3::ZERO,
            BuildingType::Tier2Tower,
            1,
            Some(Lane::Mid),
        );

        // Damage the building.
        if let Some(bs) = world.get_mut::<BuildingStats>(building) {
            bs.current_hp = 1000.0;
        }
        if let Some(h) = world.get_mut::<Health>(building) {
            h.current = 1000.0;
        }

        // No enemy creeps -> protection active -> should regen.
        backdoor_protection_system(&mut world, 1.0);

        let h = world.get::<Health>(building).unwrap();
        // 1000 + 90 * 1.0 = 1090
        assert!(
            (h.current - 1090.0).abs() < 1.0,
            "should regen 90 HP/s, got {}",
            h.current
        );
    }

    // ── Fortification ──

    #[test]
    fn fortification_makes_buildings_invulnerable() {
        let world = setup_world();
        let building = {
            let mut w = world;
            let b = spawn_building(
                &mut w,
                Vec3::ZERO,
                BuildingType::Tier1Tower,
                1,
                Some(Lane::Top),
            );

            // Activate fortification for team 1.
            if let Some(forts) = w.resource_mut::<TeamFortifications>() {
                let fort = forts.get_or_insert(1);
                building::activate_fortification(fort).unwrap();
            }

            let mult = building_damage_multiplier(&w, b);
            assert_eq!(
                mult, 0.0,
                "building should be invulnerable during fortification"
            );
            (w, b)
        };
        let _ = building;
    }

    #[test]
    fn fortification_tick_expires() {
        let mut world = setup_world();

        if let Some(forts) = world.resource_mut::<TeamFortifications>() {
            let fort = forts.get_or_insert(1);
            building::activate_fortification(fort).unwrap();
        }

        // Tick past the 5s duration.
        fortification_tick_system(&mut world, 6.0);

        let forts = world.resource::<TeamFortifications>().unwrap();
        let fort = forts.get(1).unwrap();
        assert!(
            !is_building_invulnerable(fort),
            "fortification should expire after duration"
        );
    }

    // ── Damage multiplier ──

    #[test]
    fn damage_multiplier_with_backdoor_active() {
        let mut world = setup_world();
        let building = spawn_building(
            &mut world,
            Vec3::ZERO,
            BuildingType::Tier2Tower,
            1,
            Some(Lane::Mid),
        );

        let mult = building_damage_multiplier(&world, building);
        assert!(
            (mult - 0.25).abs() < f32::EPSILON,
            "backdoor active should give 0.25 multiplier"
        );
    }

    #[test]
    fn damage_multiplier_without_building_stats() {
        let mut world = setup_world();
        let entity = world.spawn(Health::new(100.0));

        let mult = building_damage_multiplier(&world, entity);
        assert_eq!(mult, 1.0, "non-building entity should have 1.0 multiplier");
    }

    // ── Barracks death ──

    #[test]
    fn barracks_death_sets_flag() {
        let mut world = setup_world();
        let barracks = spawn_building(
            &mut world,
            Vec3::ZERO,
            BuildingType::MeleeBarracks,
            1,
            Some(Lane::Bot),
        );

        // Simulate death.
        if let Some(events) = world.resource_mut::<Events>() {
            events.send(DeathEvent {
                entity: barracks,
                killer: None,
            });
        }

        barracks_death_system(&mut world);

        let destroyed = world.resource::<DestroyedBarracks>().unwrap();
        assert!(
            destroyed.has_super_melee(1, Lane::Bot),
            "melee barracks destruction should set super_melee flag"
        );
    }

    // ── Destroyed barracks tracking ──

    #[test]
    fn all_destroyed_detects_mega_creeps() {
        let mut destroyed = DestroyedBarracks::default();
        for lane in [Lane::Top, Lane::Mid, Lane::Bot] {
            destroyed.mark_destroyed(2, lane, BuildingType::MeleeBarracks);
            destroyed.mark_destroyed(2, lane, BuildingType::RangedBarracks);
        }
        assert!(
            destroyed.all_destroyed(2),
            "all 6 barracks destroyed should trigger mega creeps"
        );
    }

    #[test]
    fn partial_destruction_not_mega() {
        let mut destroyed = DestroyedBarracks::default();
        destroyed.mark_destroyed(2, Lane::Top, BuildingType::MeleeBarracks);
        destroyed.mark_destroyed(2, Lane::Mid, BuildingType::MeleeBarracks);
        assert!(
            !destroyed.all_destroyed(2),
            "partial destruction should not trigger mega creeps"
        );
    }
}
