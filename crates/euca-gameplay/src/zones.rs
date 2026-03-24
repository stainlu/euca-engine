//! Zone/area system — spatial regions that apply continuous effects to entities inside.
//!
//! Zones are entities with a [`Zone`] component and a [`LocalTransform`] for position.
//! Each tick, the [`zone_system`] finds entities within the zone's shape and applies
//! effects (damage, healing, status effects, stat modifications).
//!
//! For zones that change over time (e.g. a shrinking battle royale circle), attach a
//! [`ZoneDynamic`] component. The [`zone_dynamic_system`] handles shrinking and applies
//! damage to entities outside the zone boundary.
//!
//! Zones compose with the existing status effects and health systems — they don't
//! duplicate any logic. Move, scale, or despawn zones like any other entity.

use euca_ecs::{Entity, Events, Query, World};
use euca_math::Vec3;

/// Snapshot of a zone's data, collected before mutating the world.
type ZoneSnapshot = (Entity, Vec3, ZoneShape, Vec<ZoneEffect>, Option<u8>);
use euca_scene::LocalTransform;

use crate::health::{DamageEvent, heal};
use crate::status_effects::{
    ModifierOp, StackPolicy, StatModifier, StatusEffect, apply_status_effect,
};
use crate::teams::Team;

// ── Data types ──

/// The spatial shape of a zone, used for containment tests.
#[derive(Clone, Debug)]
pub enum ZoneShape {
    /// A circle centered on the zone's position, tested on the XZ plane.
    Circle { radius: f32 },
    /// An axis-aligned rectangle centered on the zone's position, tested on the XZ plane.
    /// `half_extents[0]` is the half-width (X), `half_extents[1]` is the half-depth (Z).
    Rectangle { half_extents: [f32; 2] },
}

/// An effect applied to entities inside a zone each tick.
#[derive(Clone, Debug)]
pub enum ZoneEffect {
    /// Apply a status effect with the given tag, modifiers, and duration.
    /// Re-applied each tick while inside — the status effect's stack policy
    /// determines whether it refreshes or stacks.
    ApplyStatusEffect {
        tag: String,
        modifiers: Vec<(String, ModifierOp, f64)>,
        duration: f32,
    },
    /// Deal damage per second (scaled by dt).
    DamagePerSecond(f32),
    /// Heal per second (scaled by dt).
    HealPerSecond(f32),
    /// Set a stat via a modifier on the entity's status effects.
    SetStat {
        stat: String,
        op: ModifierOp,
        value: f64,
    },
}

/// Component: a spatial zone that applies effects to entities within its shape.
///
/// Attach to an entity that also has a [`LocalTransform`] for position.
#[derive(Clone, Debug)]
pub struct Zone {
    /// The shape used for containment testing.
    pub shape: ZoneShape,
    /// Effects applied each tick to entities inside the zone.
    pub effects: Vec<ZoneEffect>,
    /// If set, only affects entities on this team.
    pub team_filter: Option<u8>,
    /// Whether the zone is currently active. Inactive zones are skipped.
    pub active: bool,
}

impl Zone {
    /// Create an active zone with the given shape and effects, no team filter.
    pub fn new(shape: ZoneShape, effects: Vec<ZoneEffect>) -> Self {
        Self {
            shape,
            effects,
            team_filter: None,
            active: true,
        }
    }
}

/// Component: optional modifier for zones that change over time.
///
/// Attached alongside [`Zone`] on the same entity. The [`zone_dynamic_system`]
/// shrinks the zone's circle radius toward `target_radius` and damages entities
/// outside the zone boundary.
#[derive(Clone, Debug)]
pub struct ZoneDynamic {
    /// Rate at which the circle radius shrinks per second.
    pub shrink_rate: f32,
    /// The radius the zone is shrinking toward (will not shrink past this).
    pub target_radius: f32,
    /// Damage per second applied to entities outside the zone boundary.
    pub damage_outside_dps: f32,
}

// ── Containment helpers ──

/// Test whether a point (on the XZ plane) is inside a zone shape centered at `zone_pos`.
fn contains(shape: &ZoneShape, zone_pos: &Vec3, point: &Vec3) -> bool {
    let dx = point.x - zone_pos.x;
    let dz = point.z - zone_pos.z;
    match shape {
        ZoneShape::Circle { radius } => dx * dx + dz * dz <= radius * radius,
        ZoneShape::Rectangle { half_extents } => {
            dx.abs() <= half_extents[0] && dz.abs() <= half_extents[1]
        }
    }
}

// ── Systems ──

/// Each tick, find entities within each active zone's shape and apply effects.
///
/// For `DamagePerSecond` and `HealPerSecond`, the amount is scaled by `dt`.
/// For `ApplyStatusEffect`, the effect is applied via [`apply_status_effect`],
/// which respects the status effect's stack policy.
pub fn zone_system(world: &mut World, dt: f32) {
    // Phase 1: collect zone data (avoids holding a borrow on World).
    let zones: Vec<ZoneSnapshot> = {
        let query = Query::<(Entity, &Zone, &LocalTransform)>::new(world);
        query
            .iter()
            .filter(|(_, z, _)| z.active)
            .map(|(e, z, lt)| {
                (
                    e,
                    lt.0.translation,
                    z.shape.clone(),
                    z.effects.clone(),
                    z.team_filter,
                )
            })
            .collect()
    };

    // Phase 2: collect all positioned entities (potential targets).
    let entities: Vec<(Entity, Vec3)> = {
        let query = Query::<(Entity, &LocalTransform)>::new(world);
        query.iter().map(|(e, lt)| (e, lt.0.translation)).collect()
    };

    // Phase 3: determine which effects to apply to which entities.
    let mut damage_events: Vec<DamageEvent> = Vec::new();
    let mut heals: Vec<(Entity, f32)> = Vec::new();
    let mut status_applications: Vec<(Entity, StatusEffect)> = Vec::new();

    for (zone_entity, zone_pos, shape, effects, team_filter) in &zones {
        for (entity, entity_pos) in &entities {
            if entity == zone_entity {
                continue;
            }

            if !contains(shape, zone_pos, entity_pos) {
                continue;
            }

            if let Some(required_team) = team_filter {
                let entity_team = world.get::<Team>(*entity).map(|t| t.0);
                if entity_team != Some(*required_team) {
                    continue;
                }
            }

            for effect in effects {
                match effect {
                    ZoneEffect::DamagePerSecond(dps) => {
                        damage_events.push(DamageEvent::new(*entity, dps * dt, Some(*zone_entity)));
                    }
                    ZoneEffect::HealPerSecond(hps) => {
                        heals.push((*entity, hps * dt));
                    }
                    ZoneEffect::ApplyStatusEffect {
                        tag,
                        modifiers,
                        duration,
                    } => {
                        let mods = modifiers
                            .iter()
                            .map(|(stat, op, value)| StatModifier {
                                stat: stat.clone(),
                                op: op.clone(),
                                value: *value,
                            })
                            .collect();
                        status_applications.push((
                            *entity,
                            StatusEffect {
                                tag: tag.clone(),
                                modifiers: mods,
                                duration: *duration,
                                remaining: *duration,
                                source: Some(*zone_entity),
                                stack_policy: StackPolicy::Replace,
                                tick_effect: None,
                            },
                        ));
                    }
                    ZoneEffect::SetStat { stat, op, value } => {
                        // Apply as a short-lived status effect that refreshes each tick.
                        let tag = format!("zone_stat_{stat}");
                        status_applications.push((
                            *entity,
                            StatusEffect {
                                tag,
                                modifiers: vec![StatModifier {
                                    stat: stat.clone(),
                                    op: op.clone(),
                                    value: *value,
                                }],
                                // Lasts slightly longer than one tick so it persists
                                // until the next zone_system tick refreshes it.
                                // If the entity leaves the zone, it expires naturally.
                                duration: dt * 2.0,
                                remaining: dt * 2.0,
                                source: Some(*zone_entity),
                                stack_policy: StackPolicy::Replace,
                                tick_effect: None,
                            },
                        ));
                    }
                }
            }
        }
    }

    // Phase 4: apply collected effects (world is no longer borrowed).
    if let Some(events) = world.resource_mut::<Events>() {
        for dmg in damage_events {
            events.send(dmg);
        }
    }

    for (entity, amount) in heals {
        heal(world, entity, amount);
    }

    for (entity, effect) in status_applications {
        apply_status_effect(world, entity, effect);
    }
}

/// Tick dynamic zones: shrink circle radius toward target, apply outside damage.
///
/// Only operates on zones with both [`Zone`] and [`ZoneDynamic`] components.
/// The zone must use [`ZoneShape::Circle`] for shrinking to apply.
pub fn zone_dynamic_system(world: &mut World, dt: f32) {
    // Phase 1: collect dynamic zone data and perform shrinking.
    struct DynamicZoneInfo {
        zone_entity: Entity,
        zone_pos: Vec3,
        radius: f32,
        damage_outside_dps: f32,
    }

    let mut dynamic_zones: Vec<DynamicZoneInfo> = Vec::new();

    {
        let query = Query::<(Entity, &mut Zone, &ZoneDynamic, &LocalTransform)>::new(world);
        for (entity, zone, dynamic, lt) in query.iter() {
            if !zone.active {
                continue;
            }

            if let ZoneShape::Circle { ref mut radius } = zone.shape {
                if *radius > dynamic.target_radius {
                    *radius = (*radius - dynamic.shrink_rate * dt).max(dynamic.target_radius);
                }
                dynamic_zones.push(DynamicZoneInfo {
                    zone_entity: entity,
                    zone_pos: lt.0.translation,
                    radius: *radius,
                    damage_outside_dps: dynamic.damage_outside_dps,
                });
            }
        }
    }

    // Phase 2: find entities outside each dynamic zone and apply damage.
    let entities: Vec<(Entity, Vec3)> = {
        let query = Query::<(Entity, &LocalTransform)>::new(world);
        query.iter().map(|(e, lt)| (e, lt.0.translation)).collect()
    };

    let mut damage_events: Vec<DamageEvent> = Vec::new();

    for info in &dynamic_zones {
        let shape = ZoneShape::Circle {
            radius: info.radius,
        };
        for (entity, entity_pos) in &entities {
            if *entity == info.zone_entity {
                continue;
            }
            if !contains(&shape, &info.zone_pos, entity_pos) {
                damage_events.push(DamageEvent::new(
                    *entity,
                    info.damage_outside_dps * dt,
                    Some(info.zone_entity),
                ));
            }
        }
    }

    // Phase 3: send damage events.
    if let Some(events) = world.resource_mut::<Events>() {
        for dmg in damage_events {
            events.send(dmg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::Health;
    use crate::status_effects::StatusEffects;
    use euca_math::{Transform, Vec3};

    fn test_world() -> World {
        let mut world = World::new();
        world.insert_resource(Events::default());
        world
    }

    fn spawn_at(world: &mut World, pos: Vec3) -> Entity {
        world.spawn(LocalTransform(Transform::from_translation(pos)))
    }

    // ── Shape containment ──

    #[test]
    fn circle_contains_point_inside() {
        let shape = ZoneShape::Circle { radius: 5.0 };
        let center = Vec3::ZERO;
        let inside = Vec3::new(3.0, 0.0, 4.0); // distance = 5.0, on boundary
        assert!(contains(&shape, &center, &inside));
    }

    #[test]
    fn circle_excludes_point_outside() {
        let shape = ZoneShape::Circle { radius: 5.0 };
        let center = Vec3::ZERO;
        let outside = Vec3::new(3.1, 0.0, 4.1); // distance > 5.0
        assert!(!contains(&shape, &center, &outside));
    }

    #[test]
    fn circle_ignores_y_axis() {
        let shape = ZoneShape::Circle { radius: 5.0 };
        let center = Vec3::ZERO;
        // X=1, Z=1 is inside the circle even with a large Y offset.
        let point = Vec3::new(1.0, 100.0, 1.0);
        assert!(contains(&shape, &center, &point));
    }

    #[test]
    fn rectangle_contains_point_inside() {
        let shape = ZoneShape::Rectangle {
            half_extents: [3.0, 4.0],
        };
        let center = Vec3::new(10.0, 0.0, 10.0);
        let inside = Vec3::new(12.0, 0.0, 13.0); // dx=2, dz=3
        assert!(contains(&shape, &center, &inside));
    }

    #[test]
    fn rectangle_excludes_point_outside() {
        let shape = ZoneShape::Rectangle {
            half_extents: [3.0, 4.0],
        };
        let center = Vec3::new(10.0, 0.0, 10.0);
        let outside = Vec3::new(14.0, 0.0, 10.0); // dx=4 > 3
        assert!(!contains(&shape, &center, &outside));
    }

    #[test]
    fn rectangle_boundary_is_inclusive() {
        let shape = ZoneShape::Rectangle {
            half_extents: [3.0, 4.0],
        };
        let center = Vec3::ZERO;
        let on_edge = Vec3::new(3.0, 0.0, 4.0);
        assert!(contains(&shape, &center, &on_edge));
    }

    // ── DPS effect ──

    #[test]
    fn zone_applies_damage_per_second() {
        let mut world = test_world();

        let zone_entity = spawn_at(&mut world, Vec3::ZERO);
        world.insert(
            zone_entity,
            Zone::new(
                ZoneShape::Circle { radius: 10.0 },
                vec![ZoneEffect::DamagePerSecond(20.0)],
            ),
        );

        let target = spawn_at(&mut world, Vec3::new(3.0, 0.0, 0.0));
        world.insert(target, Health::new(100.0));

        zone_system(&mut world, 0.5);

        let events = world.resource::<Events>().unwrap();
        let damages: Vec<_> = events.read::<DamageEvent>().collect();
        assert_eq!(damages.len(), 1);
        assert_eq!(damages[0].target, target);
        assert_eq!(damages[0].amount, 10.0); // 20 DPS * 0.5s
        assert_eq!(damages[0].source, Some(zone_entity));
    }

    // ── HPS effect ──

    #[test]
    fn zone_applies_heal_per_second() {
        let mut world = test_world();

        let zone_entity = spawn_at(&mut world, Vec3::ZERO);
        world.insert(
            zone_entity,
            Zone::new(
                ZoneShape::Circle { radius: 10.0 },
                vec![ZoneEffect::HealPerSecond(30.0)],
            ),
        );

        let target = spawn_at(&mut world, Vec3::new(1.0, 0.0, 0.0));
        world.insert(
            target,
            Health {
                current: 50.0,
                max: 100.0,
            },
        );

        zone_system(&mut world, 1.0);

        let health = world.get::<Health>(target).unwrap();
        assert_eq!(health.current, 80.0); // 50 + 30*1.0
    }

    // ── Status effect application ──

    #[test]
    fn zone_applies_status_effect() {
        let mut world = test_world();

        let zone_entity = spawn_at(&mut world, Vec3::ZERO);
        world.insert(
            zone_entity,
            Zone::new(
                ZoneShape::Circle { radius: 10.0 },
                vec![ZoneEffect::ApplyStatusEffect {
                    tag: "slow".to_string(),
                    modifiers: vec![("speed".to_string(), ModifierOp::Multiply, 0.5)],
                    duration: 2.0,
                }],
            ),
        );

        let target = spawn_at(&mut world, Vec3::new(1.0, 0.0, 0.0));
        world.insert(target, Health::new(100.0));

        zone_system(&mut world, 0.016);

        let effects = world.get::<StatusEffects>(target).unwrap();
        assert_eq!(effects.effects.len(), 1);
        assert_eq!(effects.effects[0].tag, "slow");
        assert_eq!(effects.effects[0].modifiers[0].stat, "speed");
        assert_eq!(effects.effects[0].source, Some(zone_entity));
    }

    // ── Team filtering ──

    #[test]
    fn zone_team_filter_only_affects_matching_team() {
        let mut world = test_world();

        let zone_entity = spawn_at(&mut world, Vec3::ZERO);
        let mut zone = Zone::new(
            ZoneShape::Circle { radius: 10.0 },
            vec![ZoneEffect::DamagePerSecond(10.0)],
        );
        zone.team_filter = Some(2);
        world.insert(zone_entity, zone);

        // Entity on team 2 (should be affected)
        let target_team2 = spawn_at(&mut world, Vec3::new(1.0, 0.0, 0.0));
        world.insert(target_team2, Health::new(100.0));
        world.insert(target_team2, Team(2));

        // Entity on team 1 (should NOT be affected)
        let target_team1 = spawn_at(&mut world, Vec3::new(2.0, 0.0, 0.0));
        world.insert(target_team1, Health::new(100.0));
        world.insert(target_team1, Team(1));

        // Entity with no team (should NOT be affected)
        let target_no_team = spawn_at(&mut world, Vec3::new(3.0, 0.0, 0.0));
        world.insert(target_no_team, Health::new(100.0));

        zone_system(&mut world, 1.0);

        let events = world.resource::<Events>().unwrap();
        let damages: Vec<_> = events.read::<DamageEvent>().collect();
        assert_eq!(damages.len(), 1);
        assert_eq!(damages[0].target, target_team2);
    }

    // ── Inactive zone ──

    #[test]
    fn inactive_zone_has_no_effect() {
        let mut world = test_world();

        let zone_entity = spawn_at(&mut world, Vec3::ZERO);
        let mut zone = Zone::new(
            ZoneShape::Circle { radius: 10.0 },
            vec![ZoneEffect::DamagePerSecond(10.0)],
        );
        zone.active = false;
        world.insert(zone_entity, zone);

        let target = spawn_at(&mut world, Vec3::new(1.0, 0.0, 0.0));
        world.insert(target, Health::new(100.0));

        zone_system(&mut world, 1.0);

        let events = world.resource::<Events>().unwrap();
        assert_eq!(events.read::<DamageEvent>().count(), 0);
    }

    // ── Entity outside zone ──

    #[test]
    fn zone_ignores_entity_outside() {
        let mut world = test_world();

        let zone_entity = spawn_at(&mut world, Vec3::ZERO);
        world.insert(
            zone_entity,
            Zone::new(
                ZoneShape::Circle { radius: 5.0 },
                vec![ZoneEffect::DamagePerSecond(10.0)],
            ),
        );

        let target = spawn_at(&mut world, Vec3::new(100.0, 0.0, 0.0));
        world.insert(target, Health::new(100.0));

        zone_system(&mut world, 1.0);

        let events = world.resource::<Events>().unwrap();
        assert_eq!(events.read::<DamageEvent>().count(), 0);
    }

    // ── Dynamic shrinking ──

    #[test]
    fn dynamic_zone_shrinks_toward_target() {
        let mut world = test_world();

        let zone_entity = spawn_at(&mut world, Vec3::ZERO);
        world.insert(
            zone_entity,
            Zone::new(ZoneShape::Circle { radius: 100.0 }, vec![]),
        );
        world.insert(
            zone_entity,
            ZoneDynamic {
                shrink_rate: 10.0,
                target_radius: 50.0,
                damage_outside_dps: 0.0,
            },
        );

        zone_dynamic_system(&mut world, 1.0);

        let zone = world.get::<Zone>(zone_entity).unwrap();
        match zone.shape {
            ZoneShape::Circle { radius } => assert_eq!(radius, 90.0), // 100 - 10*1.0
            _ => panic!("expected circle shape"),
        }
    }

    #[test]
    fn dynamic_zone_does_not_shrink_past_target() {
        let mut world = test_world();

        let zone_entity = spawn_at(&mut world, Vec3::ZERO);
        world.insert(
            zone_entity,
            Zone::new(ZoneShape::Circle { radius: 55.0 }, vec![]),
        );
        world.insert(
            zone_entity,
            ZoneDynamic {
                shrink_rate: 10.0,
                target_radius: 50.0,
                damage_outside_dps: 0.0,
            },
        );

        // Shrink by 10 would go to 45, but target is 50.
        zone_dynamic_system(&mut world, 1.0);

        let zone = world.get::<Zone>(zone_entity).unwrap();
        match zone.shape {
            ZoneShape::Circle { radius } => assert_eq!(radius, 50.0),
            _ => panic!("expected circle shape"),
        }
    }

    #[test]
    fn dynamic_zone_damages_entities_outside() {
        let mut world = test_world();

        let zone_entity = spawn_at(&mut world, Vec3::ZERO);
        world.insert(
            zone_entity,
            Zone::new(ZoneShape::Circle { radius: 10.0 }, vec![]),
        );
        world.insert(
            zone_entity,
            ZoneDynamic {
                shrink_rate: 0.0,
                target_radius: 10.0,
                damage_outside_dps: 5.0,
            },
        );

        // Entity inside — should NOT take damage.
        let inside = spawn_at(&mut world, Vec3::new(3.0, 0.0, 0.0));
        world.insert(inside, Health::new(100.0));

        // Entity outside — should take damage.
        let outside = spawn_at(&mut world, Vec3::new(20.0, 0.0, 0.0));
        world.insert(outside, Health::new(100.0));

        zone_dynamic_system(&mut world, 1.0);

        let events = world.resource::<Events>().unwrap();
        let damages: Vec<_> = events.read::<DamageEvent>().collect();
        assert_eq!(damages.len(), 1);
        assert_eq!(damages[0].target, outside);
        assert_eq!(damages[0].amount, 5.0); // 5 DPS * 1.0s
    }

    // ── Zone as entity lifecycle ──

    #[test]
    fn zone_entity_lifecycle() {
        let mut world = test_world();

        // Create zone entity.
        let zone_entity = spawn_at(&mut world, Vec3::ZERO);
        world.insert(
            zone_entity,
            Zone::new(
                ZoneShape::Circle { radius: 10.0 },
                vec![ZoneEffect::DamagePerSecond(10.0)],
            ),
        );

        let target = spawn_at(&mut world, Vec3::new(1.0, 0.0, 0.0));
        world.insert(target, Health::new(100.0));

        // Zone is active — should produce damage.
        zone_system(&mut world, 1.0);
        let count = world
            .resource::<Events>()
            .unwrap()
            .read::<DamageEvent>()
            .count();
        assert_eq!(count, 1);

        // Deactivate the zone — should stop producing damage.
        world.get_mut::<Zone>(zone_entity).unwrap().active = false;

        // Reset events for the next tick.
        world.insert_resource(Events::default());
        zone_system(&mut world, 1.0);
        let count = world
            .resource::<Events>()
            .unwrap()
            .read::<DamageEvent>()
            .count();
        assert_eq!(count, 0);

        // Reactivate and move the zone away — target should no longer be inside.
        world.get_mut::<Zone>(zone_entity).unwrap().active = true;
        world
            .get_mut::<LocalTransform>(zone_entity)
            .unwrap()
            .0
            .translation = Vec3::new(1000.0, 0.0, 1000.0);

        world.insert_resource(Events::default());
        zone_system(&mut world, 1.0);
        let count = world
            .resource::<Events>()
            .unwrap()
            .read::<DamageEvent>()
            .count();
        assert_eq!(count, 0);
    }

    // ── Rectangle zone ──

    #[test]
    fn rectangle_zone_applies_effects() {
        let mut world = test_world();

        let zone_entity = spawn_at(&mut world, Vec3::new(5.0, 0.0, 5.0));
        world.insert(
            zone_entity,
            Zone::new(
                ZoneShape::Rectangle {
                    half_extents: [3.0, 3.0],
                },
                vec![ZoneEffect::HealPerSecond(10.0)],
            ),
        );

        // Inside the rectangle (5+2=7, which is within [2,8]).
        let inside = spawn_at(&mut world, Vec3::new(7.0, 0.0, 7.0));
        world.insert(
            inside,
            Health {
                current: 50.0,
                max: 100.0,
            },
        );

        // Outside the rectangle.
        let outside = spawn_at(&mut world, Vec3::new(20.0, 0.0, 20.0));
        world.insert(
            outside,
            Health {
                current: 50.0,
                max: 100.0,
            },
        );

        zone_system(&mut world, 1.0);

        assert_eq!(world.get::<Health>(inside).unwrap().current, 60.0);
        assert_eq!(world.get::<Health>(outside).unwrap().current, 50.0);
    }

    // ── SetStat effect ──

    #[test]
    fn zone_set_stat_applies_modifier() {
        let mut world = test_world();

        let zone_entity = spawn_at(&mut world, Vec3::ZERO);
        world.insert(
            zone_entity,
            Zone::new(
                ZoneShape::Circle { radius: 10.0 },
                vec![ZoneEffect::SetStat {
                    stat: "armor".to_string(),
                    op: ModifierOp::Add,
                    value: 50.0,
                }],
            ),
        );

        let target = spawn_at(&mut world, Vec3::new(1.0, 0.0, 0.0));

        zone_system(&mut world, 0.016);

        let effects = world.get::<StatusEffects>(target).unwrap();
        assert_eq!(effects.effects.len(), 1);
        assert_eq!(effects.effects[0].tag, "zone_stat_armor");
        assert_eq!(effects.effects[0].modifiers[0].stat, "armor");
    }

    // ── Multiple effects ──

    #[test]
    fn zone_with_multiple_effects() {
        let mut world = test_world();

        let zone_entity = spawn_at(&mut world, Vec3::ZERO);
        world.insert(
            zone_entity,
            Zone::new(
                ZoneShape::Circle { radius: 10.0 },
                vec![
                    ZoneEffect::DamagePerSecond(5.0),
                    ZoneEffect::ApplyStatusEffect {
                        tag: "burn".to_string(),
                        modifiers: vec![("fire_resist".to_string(), ModifierOp::Add, -10.0)],
                        duration: 3.0,
                    },
                ],
            ),
        );

        let target = spawn_at(&mut world, Vec3::new(1.0, 0.0, 0.0));
        world.insert(target, Health::new(100.0));

        zone_system(&mut world, 1.0);

        // Should have both a damage event and a status effect.
        let events = world.resource::<Events>().unwrap();
        assert_eq!(events.read::<DamageEvent>().count(), 1);

        let effects = world.get::<StatusEffects>(target).unwrap();
        assert_eq!(effects.effects.len(), 1);
        assert_eq!(effects.effects[0].tag, "burn");
    }
}
