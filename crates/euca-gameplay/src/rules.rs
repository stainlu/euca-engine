//! Data-driven game rules — agents define "when X happens, do Y" without code.
//!
//! Rules ARE entities with condition + action components. A single system
//! evaluates all rules each tick. The agent composes data, never writes code.
//!
//! Condition components: `OnDeathRule`, `TimerRule`, `HealthBelowRule`
//! Action execution: `execute_action()` applies GameAction to the world.

use euca_ecs::{Entity, Events, Query, World};
use euca_math::Vec3;
use euca_scene::LocalTransform;
use serde::{Deserialize, Serialize};

use crate::health::{DamageEvent, DeathEvent, Health};
use crate::teams::Team;

/// Event emitted when a rule spawns an entity that needs visual components.
/// The rendering layer listens for these and attaches MeshRenderer + MaterialRef.
#[derive(Clone, Debug)]
pub struct RuleSpawnEvent {
    pub entity: Entity,
    pub mesh: String,
    pub color: Option<String>,
    pub scale: Option<[f32; 3]>,
}

// ── Actions ──

/// What to do when a rule fires.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "action")]
pub enum GameAction {
    #[serde(rename = "spawn")]
    Spawn {
        mesh: String,
        position: [f32; 3],
        #[serde(default)]
        color: Option<String>,
        #[serde(default)]
        health: Option<f32>,
        #[serde(default)]
        team: Option<u8>,
        #[serde(default)]
        combat: Option<bool>,
        #[serde(default)]
        speed: Option<f32>,
        /// Patrol waypoints as Vec of [x,y,z].
        #[serde(default)]
        waypoints: Option<Vec<[f32; 3]>>,
        #[serde(default)]
        scale: Option<[f32; 3]>,
        #[serde(default)]
        gold_bounty: Option<i32>,
        #[serde(default)]
        xp_bounty: Option<u32>,
        /// "hero", "minion", "tower", "structure"
        #[serde(default)]
        role: Option<String>,
    },
    #[serde(rename = "damage")]
    Damage { target: ActionTarget, amount: f32 },
    #[serde(rename = "heal")]
    Heal { target: ActionTarget, amount: f32 },
    #[serde(rename = "score")]
    Score { target: ActionTarget, points: i32 },
    #[serde(rename = "despawn")]
    Despawn { target: ActionTarget },
    #[serde(rename = "teleport")]
    Teleport {
        target: ActionTarget,
        position: [f32; 3],
    },
    #[serde(rename = "color")]
    SetColor { target: ActionTarget, color: String },
    #[serde(rename = "text")]
    ShowText {
        text: String,
        x: f32,
        y: f32,
        #[serde(default = "default_text_size")]
        size: f32,
        #[serde(default = "default_text_color")]
        color: String,
    },
}

fn default_text_size() -> f32 {
    20.0
}
fn default_text_color() -> String {
    "white".to_string()
}

/// Who the action targets.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActionTarget {
    /// The entity that triggered the condition.
    This,
    /// The entity that caused the event (e.g. killer).
    Source,
    /// A specific entity by index.
    Entity(u32),
}

/// Which entities a rule condition watches.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RuleFilter {
    Any,
    Entity(u32),
    Team(u8),
}

impl RuleFilter {
    pub fn matches(&self, entity: Entity, world: &World) -> bool {
        match self {
            RuleFilter::Any => true,
            RuleFilter::Entity(id) => entity.index() == *id,
            RuleFilter::Team(t) => world.get::<Team>(entity).is_some_and(|team| team.0 == *t),
        }
    }
}

// ── Rule conditions (each is a component) ──

/// Fire actions when an entity matching the filter dies.
#[derive(Clone, Debug)]
pub struct OnDeathRule {
    pub filter: RuleFilter,
    pub actions: Vec<GameAction>,
}

/// Fire actions periodically.
#[derive(Clone, Debug)]
pub struct TimerRule {
    pub interval: f32,
    pub elapsed: f32,
    pub repeat: bool,
    pub actions: Vec<GameAction>,
}

/// Fire actions once when a watched entity's health drops below threshold.
#[derive(Clone, Debug)]
pub struct HealthBelowRule {
    pub filter: RuleFilter,
    pub threshold: f32,
    pub triggered_entities: std::collections::HashSet<u32>,
    pub actions: Vec<GameAction>,
}

/// Fire actions when any player's score reaches threshold.
#[derive(Clone, Debug)]
pub struct OnScoreRule {
    pub score_threshold: i32,
    pub triggered: bool,
    pub actions: Vec<GameAction>,
}

/// Fire actions when game phase changes to a specific phase.
#[derive(Clone, Debug)]
pub struct OnPhaseRule {
    pub phase: String, // "playing", "post_match", "lobby"
    pub triggered: bool,
    pub actions: Vec<GameAction>,
}

// ── Rule systems ──

/// Process OnScoreRule: when any player reaches score threshold, execute actions.
pub fn on_score_rule_system(world: &mut World) {
    let scores: Vec<(u32, i32)> = world
        .resource::<crate::game_state::GameState>()
        .map(|s| s.scoreboard())
        .unwrap_or_default();

    let rules: Vec<(Entity, i32, bool, Vec<GameAction>)> = {
        let query = Query::<(Entity, &OnScoreRule)>::new(world);
        query
            .iter()
            .map(|(e, r)| (e, r.score_threshold, r.triggered, r.actions.clone()))
            .collect()
    };

    let dummy = Entity::from_raw(0, 0);
    for (rule_entity, threshold, triggered, actions) in &rules {
        if *triggered {
            continue;
        }
        if scores.iter().any(|(_, score)| *score >= *threshold) {
            for action in actions {
                execute_action(world, action, dummy, None);
            }
            if let Some(rule) = world.get_mut::<OnScoreRule>(*rule_entity) {
                rule.triggered = true;
            }
        }
    }
}

/// Process OnPhaseRule: when game phase matches, execute actions.
pub fn on_phase_rule_system(world: &mut World) {
    let current_phase: Option<String> = world.resource::<crate::game_state::GameState>().map(|s| {
        match &s.phase {
            crate::game_state::GamePhase::Lobby => "lobby",
            crate::game_state::GamePhase::Countdown { .. } => "countdown",
            crate::game_state::GamePhase::Playing => "playing",
            crate::game_state::GamePhase::PostMatch { .. } => "post_match",
        }
        .to_string()
    });

    let phase = match current_phase {
        Some(p) => p,
        None => return,
    };

    let rules: Vec<(Entity, String, bool, Vec<GameAction>)> = {
        let query = Query::<(Entity, &OnPhaseRule)>::new(world);
        query
            .iter()
            .map(|(e, r)| (e, r.phase.clone(), r.triggered, r.actions.clone()))
            .collect()
    };

    let dummy = Entity::from_raw(0, 0);
    for (rule_entity, target_phase, triggered, actions) in &rules {
        if *triggered {
            continue;
        }
        if phase == *target_phase {
            for action in actions {
                execute_action(world, action, dummy, None);
            }
            if let Some(rule) = world.get_mut::<OnPhaseRule>(*rule_entity) {
                rule.triggered = true;
            }
        }
    }
}

/// Process OnDeathRule: when DeathEvent matches filter, execute actions.
pub fn on_death_rule_system(world: &mut World) {
    let deaths: Vec<(Entity, Option<Entity>)> = world
        .resource::<Events>()
        .map(|e| {
            e.read::<DeathEvent>()
                .map(|d| (d.entity, d.killer))
                .collect()
        })
        .unwrap_or_default();

    if deaths.is_empty() {
        return;
    }

    // Collect all OnDeathRule entities
    let rules: Vec<(RuleFilter, Vec<GameAction>)> = {
        let query = Query::<&OnDeathRule>::new(world);
        query
            .iter()
            .map(|r| (r.filter.clone(), r.actions.clone()))
            .collect()
    };

    for (dead_entity, killer) in &deaths {
        for (filter, actions) in &rules {
            if filter.matches(*dead_entity, world) {
                for action in actions {
                    execute_action(world, action, *dead_entity, *killer);
                }
            }
        }
    }
}

/// Process TimerRule: tick elapsed, fire when ready.
pub fn timer_rule_system(world: &mut World, dt: f32) {
    // Collect timers that fired
    let fired: Vec<(Entity, Vec<GameAction>)> = {
        let query = Query::<(Entity, &TimerRule)>::new(world);
        query
            .iter()
            .filter(|(_, t)| t.elapsed + dt >= t.interval)
            .map(|(e, t)| (e, t.actions.clone()))
            .collect()
    };

    // Update all timer elapsed
    {
        let query = Query::<(Entity, &mut TimerRule)>::new(world);
        for (_, timer) in query.iter() {
            timer.elapsed += dt;
        }
    }

    // Execute fired timers and reset
    let dummy = Entity::from_raw(0, 0);
    for (entity, actions) in &fired {
        for action in actions {
            execute_action(world, action, dummy, None);
        }
        if let Some(timer) = world.get_mut::<TimerRule>(*entity)
            && timer.repeat
        {
            timer.elapsed = 0.0;
        }
    }

    // Despawn non-repeating fired timers
    for (entity, _) in &fired {
        if let Some(timer) = world.get::<TimerRule>(*entity)
            && !timer.repeat
        {
            world.despawn(*entity);
        }
    }
}

/// Process HealthBelowRule: check health vs threshold, fire once per entity.
pub fn health_below_rule_system(world: &mut World) {
    // Collect entities with low health
    let low_health: Vec<(Entity, f32)> = {
        let query = Query::<(Entity, &Health)>::new(world);
        query
            .iter()
            .filter(|(_, h)| !h.is_dead())
            .map(|(e, h)| (e, h.current))
            .collect()
    };

    // Collect rules
    #[allow(clippy::type_complexity)]
    let rules: Vec<(
        Entity,
        RuleFilter,
        f32,
        std::collections::HashSet<u32>,
        Vec<GameAction>,
    )> = {
        let query = Query::<(Entity, &HealthBelowRule)>::new(world);
        query
            .iter()
            .map(|(e, r)| {
                (
                    e,
                    r.filter.clone(),
                    r.threshold,
                    r.triggered_entities.clone(),
                    r.actions.clone(),
                )
            })
            .collect()
    };

    for (rule_entity, filter, threshold, triggered, actions) in &rules {
        for (entity, current_health) in &low_health {
            if *current_health < *threshold
                && !triggered.contains(&entity.index())
                && filter.matches(*entity, world)
            {
                for action in actions {
                    execute_action(world, action, *entity, None);
                }
                // Mark as triggered
                if let Some(rule) = world.get_mut::<HealthBelowRule>(*rule_entity) {
                    rule.triggered_entities.insert(entity.index());
                }
            }
        }
    }
}

// ── Action execution ──

/// Execute a single GameAction in the world.
pub fn execute_action(
    world: &mut World,
    action: &GameAction,
    trigger_entity: Entity,
    source: Option<Entity>,
) {
    match action {
        GameAction::Spawn {
            mesh: _mesh,
            position,
            color: _color,
            health,
            team,
            combat,
            speed,
            waypoints,
            scale,
            gold_bounty,
            xp_bounty,
            role,
        } => {
            let mut transform = euca_math::Transform::from_translation(Vec3::new(
                position[0],
                position[1],
                position[2],
            ));
            if let Some(s) = scale {
                transform.scale = Vec3::new(s[0], s[1], s[2]);
            }
            let entity = world.spawn(LocalTransform(transform));
            world.insert(entity, euca_scene::GlobalTransform::default());
            if let Some(h) = health {
                world.insert(entity, Health::new(*h));
            }
            if let Some(t) = team {
                world.insert(entity, Team(*t));
            }
            if *combat == Some(true) {
                let mut ac = crate::combat::AutoCombat::new();
                if let Some(s) = speed {
                    ac.speed = *s;
                }
                world.insert(entity, ac);
                // Combat entities need Velocity + Kinematic PhysicsBody for movement.
                // Kinematic = gameplay-driven movement (no gravity, no collision blocking).
                world.insert(entity, euca_physics::Velocity::default());
                world.insert(
                    entity,
                    euca_physics::PhysicsBody {
                        body_type: euca_physics::RigidBodyType::Kinematic,
                    },
                );
            }
            if let Some(wps) = waypoints {
                let wp_vecs: Vec<Vec3> = wps.iter().map(|w| Vec3::new(w[0], w[1], w[2])).collect();
                let patrol_speed = speed.unwrap_or(3.0);
                world.insert(entity, crate::ai::AiGoal::patrol(wp_vecs, patrol_speed));
            }
            // Economy + role
            if let Some(b) = gold_bounty {
                world.insert(entity, crate::economy::GoldBounty(*b));
            }
            if let Some(xp) = xp_bounty {
                world.insert(entity, crate::leveling::XpBounty(*xp));
            }
            if let Some(r) = role {
                let entity_role = match r.as_str() {
                    "hero" => crate::combat::EntityRole::Hero,
                    "tower" => crate::combat::EntityRole::Tower,
                    "structure" => crate::combat::EntityRole::Structure,
                    _ => crate::combat::EntityRole::Minion,
                };
                world.insert(entity, entity_role);
            }

            // Emit event so the rendering layer can attach MeshRenderer + MaterialRef
            if let Some(events) = world.resource_mut::<Events>() {
                events.send(RuleSpawnEvent {
                    entity,
                    mesh: _mesh.clone(),
                    color: _color.clone(),
                    scale: *scale,
                });
            }
            log::info!(
                "Rule spawned entity {} at ({}, {}, {})",
                entity.index(),
                position[0],
                position[1],
                position[2]
            );
        }
        GameAction::Damage { target, amount } => {
            if let Some(entity) = resolve_target(target, trigger_entity, source)
                && let Some(events) = world.resource_mut::<Events>()
            {
                events.send(DamageEvent {
                    target: entity,
                    amount: *amount,
                    source: None,
                });
            }
        }
        GameAction::Heal { target, amount } => {
            if let Some(entity) = resolve_target(target, trigger_entity, source) {
                crate::health::heal(world, entity, *amount);
            }
        }
        GameAction::Score { target, points } => {
            if let Some(entity) = resolve_target(target, trigger_entity, source)
                && let Some(state) = world.resource_mut::<crate::game_state::GameState>()
            {
                state.add_score(entity.index(), *points);
            }
        }
        GameAction::Despawn { target } => {
            if let Some(entity) = resolve_target(target, trigger_entity, source) {
                world.despawn(entity);
            }
        }
        GameAction::Teleport { target, position } => {
            if let Some(entity) = resolve_target(target, trigger_entity, source)
                && let Some(lt) = world.get_mut::<LocalTransform>(entity)
            {
                lt.0.translation = Vec3::new(position[0], position[1], position[2]);
            }
        }
        GameAction::SetColor { .. } => {
            // Color change requires render system access (MaterialRef).
            // Would need DefaultAssets resource — skip for now.
            log::info!("Rule SetColor: not yet implemented (needs render access)");
        }
        GameAction::ShowText {
            text,
            x,
            y,
            size,
            color,
        } => {
            // Add to HudCanvas if available
            // HudCanvas is in euca-agent crate, not accessible from euca-gameplay.
            // For now, log the text. The HTTP layer can bridge this.
            log::info!("Rule ShowText: '{text}' at ({x}, {y}) size={size} color={color}");
        }
    }
}

fn resolve_target(
    target: &ActionTarget,
    trigger_entity: Entity,
    source: Option<Entity>,
) -> Option<Entity> {
    match target {
        ActionTarget::This => Some(trigger_entity),
        ActionTarget::Source => source,
        ActionTarget::Entity(id) => Some(Entity::from_raw(*id, 0)),
    }
}

// ── Action string parsing ──

/// Parse a simple action string like "spawn cube 0,5,0 red" into a GameAction.
pub fn parse_action(s: &str) -> Option<GameAction> {
    let parts: Vec<&str> = s.splitn(2, ' ').collect();
    if parts.is_empty() {
        return None;
    }

    match parts[0] {
        // spawn mesh x,y,z [color] [health] [team] [combat:true] [wp1:wp2:wp3]
        "spawn" => {
            let args: Vec<&str> = parts.get(1)?.split_whitespace().collect();
            let mesh = args.first()?.to_string();
            let pos = parse_vec3(args.get(1)?)?;
            let color = args.get(2).map(|s| s.to_string());
            let health = args.get(3).and_then(|s| s.parse::<f32>().ok());
            let team = args.get(4).and_then(|s| s.parse::<u8>().ok());
            let combat = args.get(5).map(|s| *s == "true");
            // Waypoints: colon-separated "x,y,z:x,y,z:x,y,z"
            let waypoints = args
                .get(6)
                .map(|s| s.split(':').filter_map(parse_vec3).collect::<Vec<_>>());
            let speed = args.get(7).and_then(|s| s.parse::<f32>().ok());
            let scale = args.get(8).and_then(|s| parse_vec3(s));
            let gold_bounty = args.get(9).and_then(|s| s.parse::<i32>().ok());
            let xp_bounty = args.get(10).and_then(|s| s.parse::<u32>().ok());
            let role = args.get(11).map(|s| s.to_string());
            Some(GameAction::Spawn {
                mesh,
                position: pos,
                color,
                health,
                team,
                combat,
                speed,
                waypoints,
                scale,
                gold_bounty,
                xp_bounty,
                role,
            })
        }
        "damage" => {
            let args: Vec<&str> = parts.get(1)?.split_whitespace().collect();
            let target = parse_target(args.first()?)?;
            let amount = args.get(1)?.parse().ok()?;
            Some(GameAction::Damage { target, amount })
        }
        "heal" => {
            let args: Vec<&str> = parts.get(1)?.split_whitespace().collect();
            let target = parse_target(args.first()?)?;
            let amount = args.get(1)?.parse().ok()?;
            Some(GameAction::Heal { target, amount })
        }
        "score" => {
            let args: Vec<&str> = parts.get(1)?.split_whitespace().collect();
            let target = parse_target(args.first()?)?;
            let points = args.get(1)?.parse().ok()?;
            Some(GameAction::Score { target, points })
        }
        "despawn" => {
            let target = parse_target(parts.get(1)?)?;
            Some(GameAction::Despawn { target })
        }
        "teleport" => {
            let args: Vec<&str> = parts.get(1)?.split_whitespace().collect();
            let target = parse_target(args.first()?)?;
            let pos = parse_vec3(args.get(1)?)?;
            Some(GameAction::Teleport {
                target,
                position: pos,
            })
        }
        "color" => {
            let args: Vec<&str> = parts.get(1)?.split_whitespace().collect();
            let target = parse_target(args.first()?)?;
            let color = args.get(1)?.to_string();
            Some(GameAction::SetColor { target, color })
        }
        "text" => {
            let rest = parts.get(1)?;
            // Simple parse: "text 'message' x,y size color"
            // For now, just use the whole string as text
            Some(GameAction::ShowText {
                text: rest.to_string(),
                x: 0.5,
                y: 0.1,
                size: 20.0,
                color: "white".to_string(),
            })
        }
        _ => None,
    }
}

fn parse_target(s: &str) -> Option<ActionTarget> {
    match s {
        "this" => Some(ActionTarget::This),
        "source" => Some(ActionTarget::Source),
        s if s.starts_with("entity:") => {
            let id = s.strip_prefix("entity:")?.parse().ok()?;
            Some(ActionTarget::Entity(id))
        }
        _ => None,
    }
}

fn parse_vec3(s: &str) -> Option<[f32; 3]> {
    let parts: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
    if parts.len() == 3 {
        Some([parts[0], parts[1], parts[2]])
    } else {
        None
    }
}

/// Parse a "when" condition string.
pub fn parse_when(s: &str) -> Option<RuleCondition> {
    if s == "death" {
        Some(RuleCondition::Death)
    } else if let Some(rest) = s.strip_prefix("timer:") {
        let interval: f32 = rest.parse().ok()?;
        Some(RuleCondition::Timer(interval))
    } else if let Some(rest) = s.strip_prefix("health-below:") {
        let threshold: f32 = rest.parse().ok()?;
        Some(RuleCondition::HealthBelow(threshold))
    } else if let Some(rest) = s.strip_prefix("score:") {
        let threshold: i32 = rest.parse().ok()?;
        Some(RuleCondition::Score(threshold))
    } else {
        s.strip_prefix("phase:")
            .map(|rest| RuleCondition::Phase(rest.to_string()))
    }
}

/// Parse a "filter" string.
pub fn parse_filter(s: &str) -> Option<RuleFilter> {
    if s == "any" {
        Some(RuleFilter::Any)
    } else if let Some(rest) = s.strip_prefix("entity:") {
        let id: u32 = rest.parse().ok()?;
        Some(RuleFilter::Entity(id))
    } else if let Some(rest) = s.strip_prefix("team:") {
        let t: u8 = rest.parse().ok()?;
        Some(RuleFilter::Team(t))
    } else {
        None
    }
}

/// Parsed condition type (used by HTTP/CLI to create rule entities).
#[derive(Clone, Debug)]
pub enum RuleCondition {
    Death,
    Timer(f32),
    HealthBelow(f32),
    Score(i32),
    Phase(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Transform;

    #[test]
    fn parse_action_spawn() {
        let action = parse_action("spawn cube 1,2,3 red").unwrap();
        match action {
            GameAction::Spawn {
                mesh,
                position,
                color,
                ..
            } => {
                assert_eq!(mesh, "cube");
                assert_eq!(position, [1.0, 2.0, 3.0]);
                assert_eq!(color, Some("red".to_string()));
            }
            _ => panic!("Expected Spawn"),
        }
    }

    #[test]
    fn parse_action_damage() {
        let action = parse_action("damage this 25").unwrap();
        match action {
            GameAction::Damage { amount, .. } => assert_eq!(amount, 25.0),
            _ => panic!("Expected Damage"),
        }
    }

    #[test]
    fn parse_action_score() {
        let action = parse_action("score source +1").unwrap();
        match action {
            GameAction::Score { points, .. } => assert_eq!(points, 1),
            _ => panic!("Expected Score"),
        }
    }

    #[test]
    fn parse_filter_team() {
        let f = parse_filter("team:2").unwrap();
        match f {
            RuleFilter::Team(2) => {}
            _ => panic!("Expected Team(2)"),
        }
    }

    #[test]
    fn parse_when_death() {
        let c = parse_when("death").unwrap();
        assert!(matches!(c, RuleCondition::Death));
    }

    #[test]
    fn parse_when_timer() {
        let c = parse_when("timer:10").unwrap();
        match c {
            RuleCondition::Timer(t) => assert_eq!(t, 10.0),
            _ => panic!("Expected Timer"),
        }
    }

    #[test]
    fn on_death_rule_fires() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        // Create entity that will die
        let victim = world.spawn(Health {
            current: 0.0,
            max: 100.0,
        });
        world.insert(victim, Team(2));
        world.insert(victim, crate::health::Dead);
        world.insert(
            victim,
            LocalTransform(Transform::from_translation(Vec3::ZERO)),
        );

        // Send death event
        world.resource_mut::<Events>().unwrap().send(DeathEvent {
            entity: victim,
            killer: None,
        });

        // Create rule: when team 2 dies, spawn at 0,5,0
        let rule = world.spawn(OnDeathRule {
            filter: RuleFilter::Team(2),
            actions: vec![GameAction::Spawn {
                mesh: "cube".to_string(),
                position: [0.0, 5.0, 0.0],
                color: None,
                health: Some(50.0),
                team: Some(2),
                combat: None,
                speed: None,
                waypoints: None,
                scale: None,
                gold_bounty: None,
                xp_bounty: None,
                role: None,
            }],
        });
        let _ = rule;

        let count_before = world.entity_count();
        on_death_rule_system(&mut world);
        let count_after = world.entity_count();

        // A new entity should have been spawned
        assert!(count_after > count_before);
    }

    #[test]
    fn timer_rule_fires_on_interval() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let _rule = world.spawn(TimerRule {
            interval: 1.0,
            elapsed: 0.0,
            repeat: true,
            actions: vec![GameAction::Spawn {
                mesh: "sphere".to_string(),
                position: [0.0, 3.0, 0.0],
                color: None,
                health: None,
                team: None,
                combat: None,
                speed: None,
                waypoints: None,
                scale: None,
                gold_bounty: None,
                xp_bounty: None,
                role: None,
            }],
        });

        let count_before = world.entity_count();

        // Not enough time
        timer_rule_system(&mut world, 0.5);
        assert_eq!(world.entity_count(), count_before);

        // Enough time
        timer_rule_system(&mut world, 0.6);
        assert!(world.entity_count() > count_before);
    }

    #[test]
    fn health_below_rule_fires_once() {
        let mut world = World::new();
        world.insert_resource(Events::default());

        let entity = world.spawn(Health {
            current: 20.0,
            max: 100.0,
        });
        world.insert(entity, Team(1));
        world.insert(
            entity,
            LocalTransform(Transform::from_translation(Vec3::ZERO)),
        );

        let _rule = world.spawn(HealthBelowRule {
            filter: RuleFilter::Any,
            threshold: 30.0,
            triggered_entities: std::collections::HashSet::new(),
            actions: vec![GameAction::Heal {
                target: ActionTarget::This,
                amount: 50.0,
            }],
        });

        health_below_rule_system(&mut world);
        // Should have healed
        assert_eq!(world.get::<Health>(entity).unwrap().current, 70.0);

        // Second call — should NOT heal again (already triggered)
        health_below_rule_system(&mut world);
        assert_eq!(world.get::<Health>(entity).unwrap().current, 70.0);
    }
}
