//! DotA-style MOBA client — playable single-binary demo.
//!
//! Opens a window, loads the DotA map from `levels/dota.json`, sets up items,
//! heroes, the MOBA camera, and a full gameplay loop with click-to-move, QWER
//! abilities, and shop access.
//!
//! Run: `cargo run -p euca-game --example dota_client`

use std::collections::{HashMap, HashSet};

use euca_core::Time;
use euca_ecs::{Entity, Events, Query, World};
use euca_gameplay::camera::{MobaCamera, ScreenSize};
use euca_gameplay::combat_math::DamageType;
use euca_gameplay::creep_wave::{Lane, LaneConfig, LaneWaypoints, WaveSpawner};
use euca_gameplay::player_input::ViewportSize;
use euca_gameplay::{
    AbilityDef, AbilityEffect, AbilitySlot, AttrGrowth, BaseAttributes, DayNightCycle,
    Fortification, GameState, HeroDef, HeroName, HeroRegistry, HeroTimings, ItemDef, ItemRegistry,
    ItemState, PrimaryAttribute, Roshan, VisionMap, VisionSource, WardStock,
};
use euca_math::{Mat4, Quat, Transform, Vec3};
use euca_physics::PhysicsConfig;
use euca_render::*;
use euca_scene::{GlobalTransform, LocalTransform};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{WindowAttributes, WindowId};

// ── Visual Effects ───────────────────────────────────────────────────────────

/// What kind of visual effect to display.
#[derive(Clone, Debug)]
enum VfxKind {
    /// A sphere that lerps from `from` to `to` over its lifetime, then despawns.
    Projectile {
        from: Vec3,
        to: Vec3,
        color: [f32; 4],
    },
    /// A flat disc that expands from 0 to `max_radius` over its lifetime, then fades.
    AreaCircle {
        center: Vec3,
        max_radius: f32,
        color: [f32; 4],
    },
    /// A small sphere that rises upward and fades.
    FloatingRise { origin: Vec3, color: [f32; 4] },
    /// A brief flash at a position (melee hit).
    MeleeSlash { position: Vec3 },
}

/// Component attached to VFX entities to drive their animation.
#[derive(Clone, Debug)]
struct VisualEffect {
    kind: VfxKind,
    /// Total time this effect should live (seconds).
    lifetime: f32,
    /// Time elapsed since spawn.
    elapsed: f32,
}

/// Pre-uploaded GPU handles for VFX meshes and materials.
#[derive(Clone)]
struct VfxAssets {
    sphere_mesh: MeshHandle,
    disc_mesh: MeshHandle,
    mat_white: MaterialHandle,
    mat_yellow: MaterialHandle,
    mat_red: MaterialHandle,
    mat_green: MaterialHandle,
    mat_blue: MaterialHandle,
    mat_cyan: MaterialHandle,
}

// ── MOBA gameplay state (fog, items, roshan, wards, waves) ──────────────────

/// Bundles all DotA-specific gameplay subsystems that are pure data + logic
/// (not ECS systems). Stored as a single World resource and driven each tick.
struct DotaMobaState {
    /// Per-team fog of war vision grids.
    vision_t1: VisionMap,
    vision_t2: VisionMap,
    /// Day/night cycle controlling vision radii and lighting.
    day_night: DayNightCycle,
    /// Ward stock per team (restock timers, counts).
    ward_stock_t1: WardStock,
    ward_stock_t2: WardStock,
    /// Placed wards on the map.
    wards: Vec<euca_gameplay::Ward>,
    /// Roshan boss state.
    roshan: Roshan,
    /// Aegis (if dropped and not yet consumed/expired).
    aegis: Option<euca_gameplay::Aegis>,
    /// Per-team fortification (glyph).
    fort_t1: Fortification,
    fort_t2: Fortification,
    /// Creep wave spawner (3-lane).
    wave_spawner: WaveSpawner,
    /// Per-hero item active state, keyed by entity index.
    item_states: HashMap<u32, ItemState>,
}

// ── Loading phase state machine ──────────────────────────────────────────────

/// Tracks the application lifecycle so the window stays responsive during
/// level loading. Each phase renders a different screen.
enum AppPhase {
    /// Window just opened. Render one frame so the OS shows the window, then
    /// start loading on the next frame.
    WaitingToLoad,
    /// Level JSON parsed and entities created (GLB files loaded from disk).
    /// GPU mesh uploads happen one per frame so the progress bar animates.
    Loading { total: usize, loaded: usize },
    /// All meshes uploaded and post-load setup complete. Normal gameplay.
    Playing,
}

// ── Constants ───────────────────────────────────────────────────────────────

const WINDOW_WIDTH: u32 = 1280;
const WINDOW_HEIGHT: u32 = 720;

/// Fixed timestep for gameplay systems (60 Hz).
const DT: f32 = 1.0 / 60.0;

// NOTE: No hardcoded entity indices. All entities are found by their
// ECS components (PlayerHero, Team, EntityRole, etc.), not by creation order.

// ── Death / respawn visual transition components ─────────────────────────────

/// Drives the death animation: entity tilts forward, scales down, and fades out.
struct DeathAnimation {
    elapsed: f32,
    duration: f32,
}

/// Drives the respawn animation: entity scales up from zero with a golden tint.
struct RespawnAnimation {
    elapsed: f32,
    duration: f32,
}

impl DotaMobaState {
    fn new() -> Self {
        // 128x128 grid, 1 world-unit per cell — covers the -35..35 map range.
        let vision_t1 = VisionMap::new(1, 128, 128, 1.0);
        let vision_t2 = VisionMap::new(2, 128, 128, 1.0);

        // Six-lane wave spawner: 3 L-shaped lanes x 2 teams (Radiant + Dire).
        // Radiant base = bottom-left (-28,-28), Dire base = top-right (28,28).
        // Top lane: UP left edge, then RIGHT along top.
        // Mid lane: diagonal from base to base.
        // Bot lane: RIGHT along bottom, then UP right edge.
        let lanes = vec![
            // Radiant lanes (team 1) — L-shaped paths from bottom-left base.
            LaneConfig {
                lane: Lane::Top,
                waypoints: LaneWaypoints {
                    lane: Lane::Top,
                    points: vec![
                        Vec3::new(-28.0, 0.0, -25.0),
                        Vec3::new(-28.0, 0.0, 25.0),
                        Vec3::new(28.0, 0.0, 25.0),
                    ],
                },
                barracks_destroyed: false,
                team: 1,
                mesh: "assets/generated/radiant_minion.glb".to_string(),
                color: "cyan".to_string(),
            },
            LaneConfig {
                lane: Lane::Mid,
                waypoints: LaneWaypoints {
                    lane: Lane::Mid,
                    points: vec![
                        Vec3::new(-25.0, 0.0, -25.0),
                        Vec3::new(0.0, 0.0, 0.0),
                        Vec3::new(25.0, 0.0, 25.0),
                    ],
                },
                barracks_destroyed: false,
                team: 1,
                mesh: "assets/generated/radiant_minion.glb".to_string(),
                color: "cyan".to_string(),
            },
            LaneConfig {
                lane: Lane::Bot,
                waypoints: LaneWaypoints {
                    lane: Lane::Bot,
                    points: vec![
                        Vec3::new(-28.0, 0.0, -25.0),
                        Vec3::new(25.0, 0.0, -28.0),
                        Vec3::new(28.0, 0.0, 25.0),
                    ],
                },
                barracks_destroyed: false,
                team: 1,
                mesh: "assets/generated/radiant_minion.glb".to_string(),
                color: "cyan".to_string(),
            },
            // Dire lanes (team 2) — L-shaped paths from top-right base (reverse of Radiant).
            LaneConfig {
                lane: Lane::Top,
                waypoints: LaneWaypoints {
                    lane: Lane::Top,
                    points: vec![
                        Vec3::new(28.0, 0.0, 25.0),
                        Vec3::new(-28.0, 0.0, 25.0),
                        Vec3::new(-28.0, 0.0, -25.0),
                    ],
                },
                barracks_destroyed: false,
                team: 2,
                mesh: "assets/generated/dire_minion.glb".to_string(),
                color: "red".to_string(),
            },
            LaneConfig {
                lane: Lane::Mid,
                waypoints: LaneWaypoints {
                    lane: Lane::Mid,
                    points: vec![
                        Vec3::new(25.0, 0.0, 25.0),
                        Vec3::new(0.0, 0.0, 0.0),
                        Vec3::new(-25.0, 0.0, -25.0),
                    ],
                },
                barracks_destroyed: false,
                team: 2,
                mesh: "assets/generated/dire_minion.glb".to_string(),
                color: "red".to_string(),
            },
            LaneConfig {
                lane: Lane::Bot,
                waypoints: LaneWaypoints {
                    lane: Lane::Bot,
                    points: vec![
                        Vec3::new(28.0, 0.0, 25.0),
                        Vec3::new(25.0, 0.0, -28.0),
                        Vec3::new(-28.0, 0.0, -25.0),
                    ],
                },
                barracks_destroyed: false,
                team: 2,
                mesh: "assets/generated/dire_minion.glb".to_string(),
                color: "red".to_string(),
            },
        ];

        Self {
            vision_t1,
            vision_t2,
            day_night: DayNightCycle::new(),
            ward_stock_t1: WardStock::new(),
            ward_stock_t2: WardStock::new(),
            wards: Vec::new(),
            roshan: euca_gameplay::spawn_roshan(0.0),
            aegis: None,
            fort_t1: Fortification::default(),
            fort_t2: Fortification::default(),
            wave_spawner: WaveSpawner::new(lanes),
            item_states: HashMap::new(),
        }
    }
}

// ── Floating combat text & kill feed ─────────────────────────────────────────

/// A single floating text indicator (damage number, gold/XP gain).
struct FloatingText {
    /// Screen position (pixels) at spawn time.
    screen_x: f32,
    screen_y: f32,
    /// Width of the damage bar (proportional to the amount).
    bar_width: f32,
    /// RGBA color including alpha.
    color: [f32; 4],
    /// Total lifetime in seconds.
    lifetime: f32,
    /// Elapsed time since spawn.
    elapsed: f32,
}

/// World resource: all active floating text indicators.
struct FloatingTexts {
    entries: Vec<FloatingText>,
}

impl FloatingTexts {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

/// A single kill feed entry (top-right corner).
struct KillFeedEntry {
    /// Bar widths representing killer and victim (sized by role importance).
    killer_color: [f32; 4],
    victim_color: [f32; 4],
    /// Total lifetime in seconds.
    lifetime: f32,
    /// Elapsed time since spawn.
    elapsed: f32,
}

/// World resource: kill feed displayed in the top-right corner.
struct KillFeed {
    entries: Vec<KillFeedEntry>,
}

impl KillFeed {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

/// Tracks previous frame gold/XP values to detect gains for popup display.
struct GoldXpTracker {
    prev_gold: i32,
    prev_xp: u32,
}

impl GoldXpTracker {
    fn new() -> Self {
        Self {
            prev_gold: 0,
            prev_xp: 0,
        }
    }
}

// ── Item definitions (same as dota.sh) ──────────────────────────────────────

fn define_items() -> ItemRegistry {
    let mut registry = ItemRegistry::new();

    let items = [
        (
            1,
            "Iron Branch",
            &[("cost", 50.0), ("health", 15.0), ("damage", 1.0)] as &[_],
        ),
        (2, "Healing Salve", &[("cost", 100.0), ("heal", 400.0)]),
        (3, "Boots of Speed", &[("cost", 500.0), ("speed", 2.0)]),
        (4, "Broadsword", &[("cost", 1000.0), ("damage", 18.0)]),
        (5, "Platemail", &[("cost", 1400.0), ("armor", 10.0)]),
        (
            6,
            "Power Treads",
            &[("cost", 1400.0), ("speed", 3.0), ("damage", 10.0)],
        ),
        (
            7,
            "Black King Bar",
            &[("cost", 4050.0), ("health", 200.0), ("damage", 24.0)],
        ),
        (
            8,
            "Daedalus",
            &[("cost", 5150.0), ("damage", 88.0), ("crit_chance", 30.0)],
        ),
    ];

    for (id, name, props) in items {
        let properties: HashMap<String, f64> =
            props.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        registry.register(ItemDef {
            id,
            name: name.to_string(),
            properties,
        });
    }

    registry
}

// ── Hero definitions (same as dota.sh) ──────────────────────────────────────

fn define_heroes() -> HeroRegistry {
    let mut registry = HeroRegistry::new();

    // Juggernaut — melee carry with Blade Fury and Omnislash
    // STR primary, base 20+2.2 / AGI 16+2.4 / INT 14+1.4
    registry.register(HeroDef {
        name: "Juggernaut".into(),
        health: 620.0,
        mana: 290.0,
        gold: 625,
        damage: 52.0,
        range: 1.5,
        base_stats: [
            ("max_health".into(), 620.0),
            ("attack_damage".into(), 52.0),
            ("armor".into(), 2.0),
            ("move_speed".into(), 5.0),
        ]
        .into_iter()
        .collect(),
        growth: [
            ("max_health".into(), 85.0),
            ("attack_damage".into(), 3.0),
            ("armor".into(), 0.3),
            ("move_speed".into(), 0.0),
        ]
        .into_iter()
        .collect(),
        abilities: vec![
            AbilityDef {
                slot: AbilitySlot::Q,
                name: "Blade Fury".into(),
                cooldown: 12.0,
                mana_cost: 110.0,
                effect: AbilityEffect::AreaDamage {
                    radius: 3.0,
                    damage: 120.0,
                },
            },
            AbilityDef {
                slot: AbilitySlot::W,
                name: "Healing Ward".into(),
                cooldown: 30.0,
                mana_cost: 120.0,
                effect: AbilityEffect::Heal { amount: 200.0 },
            },
            AbilityDef {
                slot: AbilitySlot::R,
                name: "Omnislash".into(),
                cooldown: 80.0,
                mana_cost: 200.0,
                effect: AbilityEffect::Chain(vec![
                    AbilityEffect::Dash { distance: 5.0 },
                    AbilityEffect::AreaDamage {
                        radius: 2.0,
                        damage: 250.0,
                    },
                ]),
            },
        ],
        primary_attribute: Some(PrimaryAttribute::Strength),
        base_attributes: Some(BaseAttributes {
            strength: 20.0,
            agility: 16.0,
            intelligence: 14.0,
        }),
        attribute_growth: Some(AttrGrowth {
            strength: 2.2,
            agility: 2.4,
            intelligence: 1.4,
        }),
        hero_timings: Some(HeroTimings {
            attack_point: 0.33,
            attack_backswing: 0.84,
            base_attack_time: 1.4,
            movement_speed: 300.0,
            attack_range: 150.0,
            ..HeroTimings::default()
        }),
    });

    // Crystal Maiden — ranged support
    // INT primary, base STR 18+2.2 / AGI 16+1.6 / INT 16+3.3
    registry.register(HeroDef {
        name: "Crystal Maiden".into(),
        health: 480.0,
        mana: 400.0,
        gold: 625,
        damage: 35.0,
        range: 6.0,
        base_stats: [
            ("max_health".into(), 480.0),
            ("attack_damage".into(), 35.0),
            ("armor".into(), 1.0),
            ("move_speed".into(), 4.0),
        ]
        .into_iter()
        .collect(),
        growth: [
            ("max_health".into(), 60.0),
            ("attack_damage".into(), 1.5),
            ("armor".into(), 0.1),
            ("move_speed".into(), 0.0),
        ]
        .into_iter()
        .collect(),
        abilities: vec![
            AbilityDef {
                slot: AbilitySlot::Q,
                name: "Crystal Nova".into(),
                cooldown: 10.0,
                mana_cost: 130.0,
                effect: AbilityEffect::AreaDamage {
                    radius: 5.0,
                    damage: 100.0,
                },
            },
            AbilityDef {
                slot: AbilitySlot::W,
                name: "Frostbite".into(),
                cooldown: 9.0,
                mana_cost: 115.0,
                effect: AbilityEffect::Chain(vec![
                    AbilityEffect::Damage {
                        amount: 150.0,
                        category: "magical".into(),
                    },
                    AbilityEffect::AreaEffect {
                        radius: 6.0,
                        effect: Box::new(AbilityEffect::ApplyCc {
                            cc_type: euca_gameplay::CcType::Root,
                            duration: 1.5,
                            dispel: euca_gameplay::DispelType::BasicDispel,
                        }),
                    },
                ]),
            },
            AbilityDef {
                slot: AbilitySlot::R,
                name: "Freezing Field".into(),
                cooldown: 90.0,
                mana_cost: 300.0,
                effect: AbilityEffect::AreaDamage {
                    radius: 8.0,
                    damage: 400.0,
                },
            },
        ],
        primary_attribute: Some(PrimaryAttribute::Intelligence),
        base_attributes: Some(BaseAttributes {
            strength: 18.0,
            agility: 16.0,
            intelligence: 16.0,
        }),
        attribute_growth: Some(AttrGrowth {
            strength: 2.2,
            agility: 1.6,
            intelligence: 3.3,
        }),
        hero_timings: Some(HeroTimings {
            attack_point: 0.45,
            attack_backswing: 0.0,
            base_attack_time: 1.7,
            movement_speed: 280.0,
            attack_range: 600.0,
            projectile_speed: 900.0,
            ..HeroTimings::default()
        }),
    });

    // Sven — melee strength carry
    // STR primary, base STR 22+3.2 / AGI 21+2.0 / INT 16+1.3
    registry.register(HeroDef {
        name: "Sven".into(),
        health: 700.0,
        mana: 250.0,
        gold: 625,
        damage: 63.0,
        range: 1.5,
        base_stats: [
            ("max_health".into(), 700.0),
            ("attack_damage".into(), 63.0),
            ("armor".into(), 3.0),
            ("move_speed".into(), 5.0),
        ]
        .into_iter()
        .collect(),
        growth: [
            ("max_health".into(), 95.0),
            ("attack_damage".into(), 3.5),
            ("armor".into(), 0.4),
            ("move_speed".into(), 0.0),
        ]
        .into_iter()
        .collect(),
        abilities: vec![
            AbilityDef {
                slot: AbilitySlot::Q,
                name: "Storm Hammer".into(),
                cooldown: 13.0,
                mana_cost: 140.0,
                effect: AbilityEffect::Chain(vec![
                    AbilityEffect::SpawnProjectile {
                        speed: 12.0,
                        range: 8.0,
                        width: 0.5,
                        damage: 100.0,
                        category: "magical".into(),
                    },
                    AbilityEffect::AreaEffect {
                        radius: 3.0,
                        effect: Box::new(AbilityEffect::ApplyCc {
                            cc_type: euca_gameplay::CcType::Stun,
                            duration: 1.4,
                            dispel: euca_gameplay::DispelType::StrongDispel,
                        }),
                    },
                ]),
            },
            AbilityDef {
                slot: AbilitySlot::W,
                name: "Warcry".into(),
                cooldown: 20.0,
                mana_cost: 60.0,
                effect: AbilityEffect::ApplyEffect {
                    tag: "warcry".into(),
                    modifiers: vec![
                        ("armor".into(), "add".into(), 10.0),
                        ("move_speed".into(), "add".into(), 2.0),
                    ],
                    duration: 8.0,
                },
            },
            AbilityDef {
                slot: AbilitySlot::R,
                name: "Gods Strength".into(),
                cooldown: 80.0,
                mana_cost: 100.0,
                effect: AbilityEffect::ApplyEffect {
                    tag: "gods_strength".into(),
                    modifiers: vec![("attack_damage".into(), "multiply".into(), 2.0)],
                    duration: 25.0,
                },
            },
        ],
        primary_attribute: Some(PrimaryAttribute::Strength),
        base_attributes: Some(BaseAttributes {
            strength: 22.0,
            agility: 21.0,
            intelligence: 16.0,
        }),
        attribute_growth: Some(AttrGrowth {
            strength: 3.2,
            agility: 2.0,
            intelligence: 1.3,
        }),
        hero_timings: Some(HeroTimings {
            attack_point: 0.4,
            attack_backswing: 0.3,
            base_attack_time: 1.8,
            movement_speed: 325.0,
            attack_range: 150.0,
            ..HeroTimings::default()
        }),
    });

    registry
}

// ── Apply hero template to an existing entity ───────────────────────────────

/// Find an entity by its index and apply a hero definition to it, adding all
/// hero-specific components (Health, Mana, Gold, abilities, stats, etc.).
fn apply_hero_template(world: &mut World, entity: Entity, hero_name: &str) {
    let def = {
        let registry = match world.resource::<HeroRegistry>() {
            Some(r) => r.clone(),
            None => {
                log::error!("No HeroRegistry resource — cannot apply hero template");
                return;
            }
        };
        match registry.get(hero_name) {
            Some(d) => d.clone(),
            None => {
                log::error!("Hero '{hero_name}' not found in registry");
                return;
            }
        }
    };

    world.insert(entity, HeroName(hero_name.to_string()));
    world.insert(entity, euca_gameplay::Health::new(def.health));
    world.insert(entity, euca_gameplay::Mana::new(def.mana, 5.0));
    world.insert(entity, euca_gameplay::Gold::new(def.gold));
    world.insert(entity, euca_gameplay::HeroEconomy::new());
    world.insert(entity, euca_gameplay::Level::new(1));
    world.insert(entity, euca_gameplay::BaseStats(def.base_stats.clone()));
    world.insert(entity, euca_gameplay::StatGrowth(def.growth.clone()));
    world.insert(entity, euca_gameplay::EntityRole::Hero);

    let mut combat = euca_gameplay::AutoCombat::new();
    combat.damage = def.damage;
    combat.range = def.range;
    world.insert(entity, combat);

    let mut ability_set = euca_gameplay::AbilitySet::new();
    for ability_def in &def.abilities {
        ability_set.add(
            ability_def.slot,
            euca_gameplay::Ability {
                name: ability_def.name.clone(),
                cooldown: ability_def.cooldown,
                cooldown_remaining: 0.0,
                mana_cost: ability_def.mana_cost,
                effect: ability_def.effect.clone(),
                ..Default::default()
            },
        );
    }
    world.insert(entity, ability_set);

    // If the definition has Dota 2 attribute data, attach HeroAttributes.
    if let (Some(primary), Some(base), Some(growth)) = (
        def.primary_attribute,
        def.base_attributes,
        def.attribute_growth,
    ) {
        world.insert(
            entity,
            euca_gameplay::HeroAttributes {
                primary,
                base,
                growth,
                timings: def.hero_timings.unwrap_or_default(),
            },
        );
    }
}

// ── DefaultAssets setup ─────────────────────────────────────────────────────

fn setup_default_assets(world: &mut World, gpu: &GpuContext, renderer: &mut Renderer) {
    let plane = renderer.upload_mesh(gpu, &Mesh::plane(40.0));
    let cube = renderer.upload_mesh(gpu, &Mesh::cube());
    let sphere = renderer.upload_mesh(gpu, &Mesh::sphere(0.5, 16, 32));

    let palette: &[(&str, Material)] = &[
        ("blue", Material::blue_plastic()),
        ("red", Material::red_plastic()),
        ("green", Material::green()),
        ("gold", Material::gold()),
        ("silver", Material::silver()),
        ("gray", Material::gray()),
        ("white", Material::new([1.0, 1.0, 1.0, 1.0], 0.0, 0.5)),
        ("black", Material::new([0.05, 0.05, 0.05, 1.0], 0.0, 0.5)),
        ("yellow", Material::new([1.0, 1.0, 0.0, 1.0], 0.0, 0.4)),
        ("cyan", Material::new([0.0, 0.9, 0.9, 1.0], 0.0, 0.4)),
        ("magenta", Material::new([0.9, 0.0, 0.9, 1.0], 0.0, 0.4)),
        ("orange", Material::new([1.0, 0.5, 0.0, 1.0], 0.0, 0.4)),
        ("purple", Material::new([0.5, 0.0, 0.8, 1.0], 0.0, 0.4)),
    ];

    let mut materials = HashMap::new();
    let mut blue = None;
    for (name, mat) in palette {
        let h = renderer.upload_material(gpu, mat);
        if *name == "blue" {
            blue = Some(h);
        }
        materials.insert((*name).to_string(), h);
    }
    let blue = blue.expect("blue material");

    // Tree mesh: cylinder trunk + sphere canopy in one draw call.
    let tree_mesh = renderer.upload_mesh(gpu, &Mesh::tree_default());
    let tree_mat = renderer.upload_material(
        gpu,
        &Material::new([0.15, 0.5, 0.1, 1.0], 0.0, 0.6), // dark green, matte
    );

    // Procedural creature meshes — used as fallback when GLB assets are missing.
    let roshan_mesh = renderer.upload_mesh(gpu, &Mesh::roshan());
    let wolf_mesh = renderer.upload_mesh(gpu, &Mesh::wolf());
    let troll_mesh = renderer.upload_mesh(gpu, &Mesh::troll());

    // Creature materials
    let roshan_mat = renderer.upload_material(
        gpu,
        &Material::new([0.35, 0.2, 0.4, 1.0], 0.0, 0.5), // dark purple-gray
    );
    let wolf_mat = renderer.upload_material(
        gpu,
        &Material::new([0.3, 0.35, 0.25, 1.0], 0.0, 0.6), // dark gray-green
    );
    let troll_mat = renderer.upload_material(
        gpu,
        &Material::new([0.35, 0.3, 0.2, 1.0], 0.0, 0.6), // brown-green
    );
    materials.insert("roshan".to_string(), roshan_mat);
    materials.insert("wolf".to_string(), wolf_mat);
    materials.insert("troll".to_string(), troll_mat);

    let mut meshes = HashMap::new();
    meshes.insert("cube".to_string(), cube);
    meshes.insert("sphere".to_string(), sphere);
    meshes.insert("plane".to_string(), plane);
    meshes.insert("roshan_boss".to_string(), roshan_mesh);
    meshes.insert("neutral_wolf".to_string(), wolf_mesh);
    meshes.insert("neutral_troll".to_string(), troll_mesh);

    // Pre-generate procedural creep meshes: 3 types x 2 teams = 6 meshes.
    // Each type has a distinct silhouette; both teams share geometry but
    // differ in material color (assigned at spawn time).
    let melee_mesh = renderer.upload_mesh(gpu, &euca_render::melee_creep_mesh());
    let ranged_mesh = renderer.upload_mesh(gpu, &euca_render::ranged_creep_mesh());
    let siege_mesh = renderer.upload_mesh(gpu, &euca_render::siege_creep_mesh());

    for team in [1u8, 2] {
        meshes.insert(euca_render::creep_mesh_name("melee", team), melee_mesh);
        meshes.insert(euca_render::creep_mesh_name("ranged", team), ranged_mesh);
        meshes.insert(euca_render::creep_mesh_name("siege", team), siege_mesh);
        // Super creeps reuse the melee mesh (they are upgraded melee creeps).
        meshes.insert(euca_render::creep_mesh_name("super", team), melee_mesh);
    }

    world.insert_resource(euca_agent::routes::DefaultAssets {
        meshes,
        materials,
        default_material: blue,
    });

    // ── MOBA terrain ────────────────────────────────────────────────────────
    // Multiple flat quads at different heights with distinct materials form
    // the DotA 2 map geography: grass base, diagonal river, L-shaped lanes,
    // and two base areas.
    spawn_moba_terrain(world, gpu, renderer);

    // Directional light — neutral white sun for the DotA arena
    world.spawn(DirectionalLight {
        direction: [0.4, -0.9, 0.25],
        color: [1.0, 1.0, 1.0],
        intensity: 2.0,
        ..Default::default()
    });

    // Trees in jungle areas between L-shaped lanes — defines MOBA map geography
    spawn_tree_lines(world, tree_mesh, tree_mat);

    // VFX assets — small sphere for projectiles, flat disc for area effects.
    // Materials are bright emissive-style (low roughness, full saturation).
    let vfx_sphere = renderer.upload_mesh(gpu, &Mesh::sphere(0.2, 8, 16));
    let vfx_disc = renderer.upload_mesh(gpu, &Mesh::disc(1.0, 24));
    let mut vfx_mat =
        |color: [f32; 4]| renderer.upload_material(gpu, &Material::new(color, 0.0, 0.2));

    world.insert_resource(VfxAssets {
        sphere_mesh: vfx_sphere,
        disc_mesh: vfx_disc,
        mat_white: vfx_mat([1.0, 1.0, 1.0, 1.0]),
        mat_yellow: vfx_mat([1.0, 0.9, 0.2, 1.0]),
        mat_red: vfx_mat([1.0, 0.2, 0.1, 1.0]),
        mat_green: vfx_mat([0.2, 1.0, 0.3, 1.0]),
        mat_blue: vfx_mat([0.3, 0.5, 1.0, 1.0]),
        mat_cyan: vfx_mat([0.2, 0.9, 1.0, 1.0]),
    });
}

// ── MOBA terrain generation ────────────────────────────────────────────────

/// Spawn a single flat terrain quad entity with the given transform and material.
fn spawn_terrain_quad(
    world: &mut World,
    mesh: MeshHandle,
    material: MaterialHandle,
    transform: Transform,
) -> Entity {
    let e = world.spawn(LocalTransform(transform));
    world.insert(e, GlobalTransform::default());
    world.insert(e, MeshRenderer { mesh });
    world.insert(e, MaterialRef { handle: material });
    e
}

/// Build the full MOBA terrain: grass base, diagonal river, three L-shaped
/// lanes, and two base areas. Each zone is a separate entity with its own
/// material so colours distinguish the different map regions.
fn spawn_moba_terrain(world: &mut World, gpu: &GpuContext, renderer: &mut Renderer) {
    // Shared unit plane mesh -- each entity scales/rotates/translates it.
    let unit_plane = renderer.upload_mesh(gpu, &Mesh::plane(1.0));

    // ── Materials ──────────────────────────────────────────────────────────
    let grass_mat = renderer.upload_material(
        gpu,
        &Material::new([0.2, 0.5, 0.15, 1.0], 0.0, 0.95), // bright green
    );
    let river_mat = renderer.upload_material(
        gpu,
        &Material::new([0.15, 0.3, 0.6, 0.9], 0.5, 0.15), // blue, slightly metallic
    );
    let lane_mat = renderer.upload_material(
        gpu,
        &Material::new([0.4, 0.35, 0.2, 1.0], 0.0, 0.85), // brown / tan
    );
    let base_mat = renderer.upload_material(
        gpu,
        &Material::new([0.35, 0.35, 0.3, 1.0], 0.0, 0.8), // stone gray
    );
    let jungle_mat = renderer.upload_material(
        gpu,
        &Material::new([0.12, 0.3, 0.1, 1.0], 0.0, 0.95), // darker green
    );

    // Helper: build a Transform that positions a unit plane as an axis-aligned
    // rectangle covering [cx - hw, cx + hw] x [cz - hd, cz + hd] at height y.
    let axis_rect = |cx: f32, cz: f32, half_w: f32, half_d: f32, y: f32| -> Transform {
        Transform {
            translation: Vec3::new(cx, y, cz),
            rotation: Quat::IDENTITY,
            // plane(1.0) spans -0.5..0.5, so scale by 2*half to get the desired size.
            scale: Vec3::new(half_w * 2.0, 1.0, half_d * 2.0),
        }
    };

    // Helper: diagonal strip centered at (cx, cz), rotated 45 degrees around Y,
    // with given width and length (measured along the diagonal).
    let diag_rect = |cx: f32, cz: f32, length: f32, width: f32, y: f32| -> Transform {
        Transform {
            translation: Vec3::new(cx, y, cz),
            rotation: Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), std::f32::consts::FRAC_PI_4),
            scale: Vec3::new(length, 1.0, width),
        }
    };

    // ── 1. Grass base plane ───────────────────────────────────────────────
    // Covers the full map (-35..35) at y=0. Also carries the physics collider
    // so click-to-move raycasts can hit the ground.
    let grass = spawn_terrain_quad(
        world,
        unit_plane,
        grass_mat,
        axis_rect(0.0, 0.0, 35.0, 35.0, 0.0),
    );
    world.insert(grass, euca_physics::PhysicsBody::fixed());
    world.insert(grass, euca_physics::Collider::aabb(40.0, 0.01, 40.0));

    // ── 2. River ──────────────────────────────────────────────────────────
    // Diagonal band from bottom-left to top-right, slightly below ground.
    // Length ~100 (diagonal of 70x70), width 6 units.
    spawn_terrain_quad(
        world,
        unit_plane,
        river_mat,
        diag_rect(0.0, 0.0, 100.0, 6.0, -0.05),
    );

    // ── 3. Lane paths ─────────────────────────────────────────────────────
    let lane_w = 3.0; // half-width of each lane strip

    // Top lane: vertical segment along the left edge, then horizontal along the top.
    // Vertical: x=-28, z from -25 to +25
    spawn_terrain_quad(
        world,
        unit_plane,
        lane_mat,
        axis_rect(-28.0, 0.0, lane_w, 25.0, 0.01),
    );
    // Horizontal: z=+25, x from -28 to +28
    spawn_terrain_quad(
        world,
        unit_plane,
        lane_mat,
        axis_rect(0.0, 25.0, 28.0, lane_w, 0.01),
    );

    // Mid lane: diagonal from bottom-left base to top-right base.
    // Length ~71 (diagonal of ~50x50), width 5 units.
    spawn_terrain_quad(
        world,
        unit_plane,
        lane_mat,
        diag_rect(0.0, 0.0, 71.0, 5.0, 0.01),
    );

    // Bot lane: horizontal along the bottom edge, then vertical along the right.
    // Horizontal: z=-25, x from -28 to +28
    spawn_terrain_quad(
        world,
        unit_plane,
        lane_mat,
        axis_rect(0.0, -25.0, 28.0, lane_w, 0.01),
    );
    // Vertical: x=+28, z from -25 to +25
    spawn_terrain_quad(
        world,
        unit_plane,
        lane_mat,
        axis_rect(28.0, 0.0, lane_w, 25.0, 0.01),
    );

    // ── 4. Base areas ─────────────────────────────────────────────────────
    // Radiant base: bottom-left corner
    spawn_terrain_quad(
        world,
        unit_plane,
        base_mat,
        axis_rect(-28.0, -28.0, 7.0, 7.0, 0.01),
    );
    // Dire base: top-right corner
    spawn_terrain_quad(
        world,
        unit_plane,
        base_mat,
        axis_rect(28.0, 28.0, 7.0, 7.0, 0.01),
    );

    // ── 5. Jungle zones ──────────────────────────────────────────────────
    // Darker green patches between lanes to distinguish jungle from grass.
    // Upper-left jungle (Radiant side, between top lane and mid lane)
    spawn_terrain_quad(
        world,
        unit_plane,
        jungle_mat,
        axis_rect(-12.0, 14.0, 13.0, 8.0, 0.005),
    );
    // Lower-right jungle (Dire side, between mid lane and bot lane)
    spawn_terrain_quad(
        world,
        unit_plane,
        jungle_mat,
        axis_rect(12.0, -14.0, 13.0, 8.0, 0.005),
    );
}

/// Spawn tree entities in the jungle areas between L-shaped lanes.
/// DotA 2 layout: top lane (left edge + top edge), mid (diagonal from
/// bottom-left to top-right), bot lane (bottom edge + right edge).
/// Trees fill the interior jungle between lanes.
fn spawn_tree_lines(world: &mut World, mesh: MeshHandle, material: MaterialHandle) {
    // Fill the full map area with trees, but skip lane corridors.
    // The jungle zones are the two triangular interior areas between lanes.
    let zones: &[(f32, f32, f32, f32)] = &[
        // Upper-left jungle (between top lane and mid lane, Radiant side)
        (-24.0, 8.0, 4.0, 20.0),
        // Lower-right jungle (between mid lane and bot lane, Dire side)
        (-8.0, 24.0, -20.0, -4.0),
        // Far top-right corner (behind Dire base)
        (20.0, 28.0, 24.0, 28.0),
        // Far bottom-left corner (behind Radiant base)
        (-28.0, -20.0, -28.0, -24.0),
    ];

    let spacing = 3.0f32;
    let lane_half_width = 3.0f32;
    let mut seed: u32 = 42;

    for &(x_min, x_max, z_min, z_max) in zones {
        let mut x = x_min;
        while x <= x_max {
            let mut z = z_min;
            while z <= z_max {
                // Pseudo-random offset for natural look
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let ox = ((seed >> 16) as f32 / 65536.0 - 0.5) * 1.5;
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let oz = ((seed >> 16) as f32 / 65536.0 - 0.5) * 1.5;
                // Random uniform scale (0.8 – 1.2)
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let scale = 0.8 + ((seed >> 16) as f32 / 65536.0) * 0.4;
                // Random Y-rotation (0 – 2pi)
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let y_rot = ((seed >> 16) as f32 / 65536.0) * 2.0 * std::f32::consts::PI;

                let px = x + ox;
                let pz = z + oz;

                // Skip trees on L-shaped lane paths:
                // Top lane: left edge (x=-28, z from -25 to 25) + top edge (z=25, x from -28 to 28)
                let on_top = ((px + 28.0).abs() < lane_half_width && pz > -25.0)
                    || ((pz - 25.0).abs() < lane_half_width && px > -28.0);
                // Mid lane: diagonal from (-25,-25) to (25,25) — slope = 1
                let mid_z_at_x = px;
                let on_mid = (pz - mid_z_at_x).abs() < lane_half_width;
                // Bot lane: bottom edge (z=-28, x from -28 to 25) + right edge (x=28, z from -28 to 25)
                let on_bot = ((pz + 28.0).abs() < lane_half_width && px < 25.0)
                    || ((px - 28.0).abs() < lane_half_width && pz < 25.0);

                if !on_top && !on_mid && !on_bot {
                    // Tree mesh has its bottom at y=0; place at ground level.
                    let xform = Transform {
                        translation: Vec3::new(px, 0.0, pz),
                        rotation: Quat::from_axis_angle(Vec3::Y, y_rot),
                        scale: Vec3::new(scale, scale, scale),
                    };
                    let t = world.spawn(LocalTransform(xform));
                    world.insert(t, GlobalTransform::default());
                    world.insert(t, MeshRenderer { mesh });
                    world.insert(t, MaterialRef { handle: material });
                    world.insert(
                        t,
                        euca_physics::PhysicsBody {
                            body_type: euca_physics::RigidBodyType::Static,
                        },
                    );
                    // Collider sized to the canopy radius (0.6) scaled.
                    let canopy_r = 0.6 * scale;
                    let tree_h = 2.4 * scale; // trunk 1.5 + canopy top 0.6 above centre
                    world.insert(
                        t,
                        euca_physics::Collider::aabb(canopy_r, tree_h * 0.5, canopy_r),
                    );
                }

                z += spacing;
            }
            x += spacing;
        }
    }
}

// ── Application ─────────────────────────────────────────────────────────────

struct DotaClientApp {
    world: World,
    initialized: bool,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    ui_overlay: Option<UiOverlayRenderer>,
    window_attrs: WindowAttributes,
    phase: AppPhase,
    /// Entities that had the `Dead` marker last frame — used to detect respawns.
    previously_dead: HashSet<Entity>,
}

impl DotaClientApp {
    fn new() -> Self {
        let mut world = World::new();
        world.insert_resource(Time::new());
        world.insert_resource(Camera::new(
            Vec3::new(-28.0, 40.0, -10.0),
            Vec3::new(-28.0, 0.0, -28.0),
        ));
        world.insert_resource(PhysicsConfig::new());
        world.insert_resource(AmbientLight {
            color: [1.0, 1.0, 1.0],
            intensity: 0.2,
        });
        world.insert_resource(Events::default());
        world.insert_resource(euca_input::InputState::new());
        world.insert_resource(euca_input::InputContextStack::new());
        // Start locked to prevent edge-pan drift during the 30s level load.
        // Unlocked after level loading completes (in load_level).
        world.insert_resource(MobaCamera {
            locked: true,
            ..MobaCamera::default()
        });
        world.insert_resource(ViewportSize::new(WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32));
        world.insert_resource(ScreenSize {
            width: WINDOW_WIDTH as f32,
            height: WINDOW_HEIGHT as f32,
        });
        // Use Medium quality but disable SSAO (causes diamond artifacts on flat terrain)
        let mut pps = euca_render::RenderQuality::Medium.to_settings();
        pps.ssao_enabled = false;
        world.insert_resource(pps);

        // Register items and heroes
        world.insert_resource(define_items());
        world.insert_resource(define_heroes());

        // Initialize DotA MOBA gameplay state (fog, wards, roshan, waves, items)
        world.insert_resource(DotaMobaState::new());

        // Floating combat text and kill feed
        world.insert_resource(FloatingTexts::new());
        world.insert_resource(KillFeed::new());
        world.insert_resource(GoldXpTracker::new());

        Self {
            world,
            initialized: false,
            gpu: None,
            renderer: None,
            ui_overlay: None,
            window_attrs: WindowAttributes::default()
                .with_title("Euca Engine — DotA Client")
                .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT)),
            phase: AppPhase::WaitingToLoad,
            previously_dead: HashSet::new(),
        }
    }

    /// Phase 1: Parse the level JSON and create all entities. GLB files are
    /// loaded from disk during `spawn_entity` (the blocking part), and queued
    /// into `PendingMeshUpload` for incremental GPU upload. Returns the number
    /// of pending mesh uploads.
    fn start_loading(&mut self) -> usize {
        let path = "levels/dota.json";
        match std::fs::read_to_string(path) {
            Ok(data) => match serde_json::from_str::<serde_json::Value>(&data) {
                Ok(level) => {
                    let count = euca_agent::load_level_into_world(&mut self.world, &level);
                    log::info!("Level loaded: {count} entities from {path}");
                }
                Err(e) => {
                    log::error!("Invalid level JSON in {path}: {e}");
                    return 0;
                }
            },
            Err(e) => {
                log::error!("Cannot read level file {path}: {e}");
                return 0;
            }
        }

        // Return pending mesh count so the loading phase knows the total.
        self.world
            .resource::<euca_agent::routes::PendingMeshUpload>()
            .map(|p| p.queue.len())
            .unwrap_or(0)
    }

    /// Phase 2: Called once all pending mesh uploads are complete. Sets up
    /// hero template, Roshan, navmesh, camera, and starts the game.
    fn finish_loading(&mut self) {
        // Initialize Roshan manager — find the Roshan entity loaded from the level
        // (team 0 structure with combat, at the pit location).
        {
            let mut mgr = euca_gameplay::RoshanManager::new(0.0);
            let roshan_entity = {
                let q = Query::<(
                    Entity,
                    &euca_gameplay::Team,
                    &euca_gameplay::EntityRole,
                    &euca_gameplay::Health,
                )>::new(&self.world);
                q.iter()
                    .find(|(_, t, r, _)| t.0 == 0 && **r == euca_gameplay::EntityRole::Structure)
                    .map(|(e, _, _, _)| e)
            };
            mgr.entity = roshan_entity;
            self.world.insert_resource(mgr);
            if let Some(e) = roshan_entity {
                log::info!("RoshanManager initialized (entity {})", e.index());
            }
        }

        // Find the player hero by its PlayerHero marker component — never
        // hardcode entity indices (creation order varies between client/server).
        let hero_entity = {
            let q = Query::<(Entity, &euca_gameplay::player::PlayerHero)>::new(&self.world);
            q.iter().map(|(e, _)| e).next()
        };

        if let Some(hero) = hero_entity {
            apply_hero_template(&mut self.world, hero, "Juggernaut");
            log::info!(
                "Applied Juggernaut template to player hero (entity {})",
                hero.index()
            );

            // Initialize item active state for this hero (6 main inventory slots).
            if let Some(moba) = self.world.resource_mut::<DotaMobaState>() {
                moba.item_states.insert(
                    hero.index(),
                    ItemState {
                        actives: vec![None; 6],
                        charges: vec![None; 6],
                        ..Default::default()
                    },
                );
            }

            // Read the hero's position for camera initialization.
            // Use LocalTransform (source of truth from level JSON), NOT GlobalTransform
            // which is still at default (0,0,0) because transform_propagation hasn't run yet.
            let hero_world_pos = self
                .world
                .get::<LocalTransform>(hero)
                .map(|lt| lt.0.translation)
                .unwrap_or(Vec3::ZERO);

            let (offset, zoom, look_at_offset) =
                if let Some(cam) = self.world.resource_mut::<MobaCamera>() {
                    cam.follow_entity = Some(hero);
                    cam.locked = true;
                    cam.center = hero_world_pos;
                    cam.follow_key = Some(euca_input::InputKey::Key("1".into()));
                    cam.toggle_lock_key = Some(euca_input::InputKey::Key("Y".into()));
                    (cam.offset, cam.zoom, cam.look_at_offset)
                } else {
                    (Vec3::new(0.0, 12.0, 8.0), 1.0, Vec3::ZERO)
                };

            // Sync render Camera immediately so the first frame shows
            // the hero, not sky. Without this, Camera stays at its init
            // position until moba_camera_system runs.
            if let Some(render_cam) = self.world.resource_mut::<Camera>() {
                render_cam.eye = hero_world_pos + offset * zoom;
                render_cam.target = hero_world_pos + look_at_offset;
            }
        } else {
            log::error!("No PlayerHero entity found in level — check dota.json has 'player': true");
        }

        // Start the game
        if let Some(state) = self.world.resource_mut::<GameState>() {
            state.start();
            log::info!("Match started");
        }

        // Initialize building system resources (fortification + barracks tracking).
        if self
            .world
            .resource::<euca_gameplay::TeamFortifications>()
            .is_none()
        {
            self.world
                .insert_resource(euca_gameplay::TeamFortifications::default());
        }
        if self
            .world
            .resource::<euca_gameplay::DestroyedBarracks>()
            .is_none()
        {
            self.world
                .insert_resource(euca_gameplay::DestroyedBarracks::default());
        }

        // Build navmesh for pathfinding
        if self.world.resource::<euca_nav::NavMesh>().is_none() {
            let config = euca_nav::GridConfig {
                min: [-35.0, -35.0],
                max: [35.0, 35.0],
                cell_size: 0.5,
                ground_y: 0.0,
            };
            let mesh = euca_nav::build_navmesh_from_world_with_radius(&self.world, config, 0.5);
            self.world.insert_resource(mesh);
            log::info!("Navmesh built for DotA arena");
        }

        // Add point lights on towers and ancients for atmospheric glow
        add_structure_lights(&mut self.world);

        // Reset the Time resource so the first frame after loading doesn't
        // have a massive delta (30+ seconds of GLB loading). Without this,
        // edge-pan speed * huge_delta drifts the camera hundreds of units.
        self.world.resource_mut::<Time>().unwrap().update();

        // Set camera to player hero position and unlock for edge-pan.
        let hero_pos = {
            let q = Query::<(Entity, &euca_gameplay::player::PlayerHero)>::new(&self.world);
            q.iter().next().and_then(|(e, _)| {
                self.world
                    .get::<euca_scene::LocalTransform>(e)
                    .map(|lt| lt.0.translation)
            })
        };
        log::info!("Hero position after load: {:?}", hero_pos);
        if let Some(cam) = self.world.resource_mut::<MobaCamera>() {
            if let Some(pos) = hero_pos {
                cam.center = pos;
            }
            cam.locked = false;
            log::info!("Camera centered on hero, unlocked for edge-pan");
        }
    }

    fn render_frame(&mut self) {
        // Take current phase to allow mutation during the match.
        let phase = std::mem::replace(&mut self.phase, AppPhase::Playing);
        match phase {
            AppPhase::WaitingToLoad => {
                // First frame: render a loading screen so the window is visible,
                // then parse the level JSON and load GLB files from disk.
                // The GLB I/O blocks (~48s) but at least the window shows
                // "Loading..." before the freeze.
                self.render_loading_screen(0, 1);
                log::info!("Loading screen displayed, starting level load...");
                let total = self.start_loading();
                log::info!("Level entities created, {total} meshes pending GPU upload");
                self.phase = AppPhase::Loading { total, loaded: 0 };
            }
            AppPhase::Loading { total, mut loaded } => {
                // Upload one pending mesh per frame so the progress bar animates.
                let gpu = self.gpu.as_ref().unwrap();
                let renderer = self.renderer.as_mut().unwrap();
                let did_upload = euca_agent::routes::drain_one_pending_mesh_upload(
                    &mut self.world,
                    renderer,
                    gpu,
                );
                if did_upload {
                    loaded += 1;
                }
                self.render_loading_screen(loaded, total);

                // Check if all done.
                let pending_empty = self
                    .world
                    .resource::<euca_agent::routes::PendingMeshUpload>()
                    .map(|p| p.queue.is_empty())
                    .unwrap_or(true);
                if pending_empty {
                    log::info!("All meshes uploaded, finishing level setup...");
                    self.finish_loading();
                    self.phase = AppPhase::Playing;
                } else {
                    self.phase = AppPhase::Loading { total, loaded };
                }
            }
            AppPhase::Playing => {
                self.gameplay_frame();
                self.phase = AppPhase::Playing;
            }
        }
    }

    /// Render the loading screen: dark background with a centered progress bar.
    fn render_loading_screen(&mut self, loaded: usize, total: usize) {
        let gpu = self.gpu.as_ref().unwrap();
        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("loading frame"),
            });

        // Clear to dark background using the 3D renderer (no draw commands).
        let camera = self.world.resource::<Camera>().unwrap().clone();
        let light = DirectionalLight::default();
        let ambient = AmbientLight {
            color: [0.0, 0.0, 0.0],
            intensity: 0.0,
        };
        let renderer = self.renderer.as_mut().unwrap();
        renderer.render_to_view_with_lights(
            gpu,
            &camera,
            &light,
            &ambient,
            &[], // no draw commands
            &[], // no point lights
            &[], // no spot lights
            &view,
            &mut encoder,
        );

        // Build progress bar UI quads.
        let vw = gpu.surface_config.width as f32;
        let vh = gpu.surface_config.height as f32;
        let quads = build_loading_screen_quads(loaded, total, vw, vh);
        if let Some(ui) = self.ui_overlay.as_mut() {
            ui.render(&*gpu, &mut encoder, &view, &quads, vw, vh);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    /// Full gameplay frame: run all ECS systems, then render the 3D scene
    /// with the HUD overlay.
    fn gameplay_frame(&mut self) {
        self.world.resource_mut::<Time>().unwrap().update();

        let dt = self.world.resource::<Time>().map(|t| t.delta).unwrap_or(DT);

        // ── Gameplay systems (same order as euca-game main.rs) ──────────

        // Physics & transforms
        euca_physics::physics_step_system(&mut self.world);
        euca_physics::character_controller_system(&mut self.world, dt);

        // Player input -> commands -> execution
        euca_gameplay::player_input_system(&mut self.world);
        euca_gameplay::cc_tick_system(&mut self.world, dt);
        euca_gameplay::player::player_command_system(&mut self.world, dt);

        // Stat pipeline
        euca_gameplay::equipment_stat_system(&mut self.world);
        euca_gameplay::zone_system(&mut self.world, dt);
        euca_gameplay::zone_dynamic_system(&mut self.world, dt);
        euca_gameplay::status_effect_tick_system(&mut self.world, dt);
        euca_gameplay::stat_resolution_system(&mut self.world);
        euca_gameplay::attribute_update_system(&mut self.world);

        // Building systems (before damage so protection state is current)
        euca_gameplay::backdoor_protection_system(&mut self.world, dt);
        euca_gameplay::fortification_tick_system(&mut self.world, dt);

        // Core gameplay
        euca_gameplay::apply_damage_system(&mut self.world);
        euca_gameplay::death_check_system(&mut self.world);
        euca_gameplay::barracks_death_system(&mut self.world);
        euca_gameplay::projectile_system(&mut self.world, dt);
        euca_gameplay::trigger_system(&mut self.world);
        euca_gameplay::ai_system(&mut self.world, dt);
        euca_gameplay::tower_aggro_system(&mut self.world);
        euca_gameplay::auto_combat_system(&mut self.world, dt);
        euca_gameplay::neutral_camp_system(&mut self.world, dt);

        // Game state & scoring
        euca_gameplay::game_state_system(&mut self.world, dt);
        euca_gameplay::on_death_rule_system(&mut self.world);
        euca_gameplay::timer_rule_system(&mut self.world, dt);
        euca_gameplay::health_below_rule_system(&mut self.world);
        euca_gameplay::on_score_rule_system(&mut self.world);
        euca_gameplay::on_phase_rule_system(&mut self.world);

        // Respawn & cleanup
        let respawn_delay = self
            .world
            .resource::<GameState>()
            .map(|s| s.config.respawn_delay);
        if let Some(delay) = respawn_delay {
            euca_gameplay::start_respawn_on_death(&mut self.world, delay);
        }
        euca_gameplay::respawn_system(&mut self.world, dt);
        euca_gameplay::corpse_cleanup_system(&mut self.world, dt);

        // Death/respawn animations
        spawn_death_animations(&mut self.world);
        spawn_respawn_animations(&mut self.world, &self.previously_dead);
        tick_death_animations(&mut self.world, dt);
        tick_respawn_animations(&mut self.world, dt);
        self.previously_dead = collect_dead_entities(&self.world);

        // Roshan lifecycle + Aegis resurrection
        euca_gameplay::roshan_system(&mut self.world, dt);
        euca_gameplay::aegis_system(&mut self.world, dt);

        // Attach visuals to rule-spawned entities (minion waves etc.)
        let spawn_events: Vec<euca_gameplay::RuleSpawnEvent> = self
            .world
            .resource::<Events>()
            .map(|e| e.read::<euca_gameplay::RuleSpawnEvent>().cloned().collect())
            .unwrap_or_default();
        if let Some(assets) = self
            .world
            .resource::<euca_agent::routes::DefaultAssets>()
            .cloned()
        {
            for ev in spawn_events {
                if let Some(mesh_handle) = assets.mesh(&ev.mesh) {
                    self.world
                        .insert(ev.entity, MeshRenderer { mesh: mesh_handle });
                    let mat = ev
                        .color
                        .as_deref()
                        .and_then(|c| assets.material(c))
                        .unwrap_or(assets.default_material);
                    self.world.insert(ev.entity, MaterialRef { handle: mat });
                }
            }
        }

        // Economy & abilities
        euca_gameplay::gold_on_kill_system(&mut self.world);
        euca_gameplay::economy_death_system(&mut self.world);
        euca_gameplay::passive_income_system(&mut self.world, dt);
        euca_gameplay::buyback_cooldown_system(&mut self.world, dt);
        euca_gameplay::xp_on_kill_system(&mut self.world);
        euca_gameplay::ability_tick_system(&mut self.world, dt);
        euca_gameplay::use_ability_system(&mut self.world);

        // Visual effects: spawn VFX from combat/ability events, then animate.
        vfx_spawn_system(&mut self.world);
        vfx_tick_system(&mut self.world, dt);

        // ── MOBA subsystems (fog, CC, items, roshan, wards, waves) ───────
        moba_subsystems_tick(&mut self.world, dt);

        // Navigation
        euca_nav::pathfinding_system(&mut self.world);
        euca_nav::steering_system(&mut self.world, dt);

        // Visibility
        euca_gameplay::visibility_system(&mut self.world);

        // Harvest combat events for floating text and kill feed (before events.update clears them)
        harvest_damage_events(&mut self.world);
        harvest_death_events(&mut self.world);
        harvest_gold_xp_changes(&mut self.world);
        tick_floating_texts(&mut self.world, dt);
        tick_kill_feed(&mut self.world, dt);

        // Tick events and world
        if let Some(events) = self.world.resource_mut::<Events>() {
            events.update();
        }
        self.world.tick();

        // Input clear (after gameplay consumed it)
        if let Some(input) = self.world.resource_mut::<euca_input::InputState>() {
            input.begin_frame();
        }

        // Day/night cycle + Radiant/Dire color grading
        day_night_system(&mut self.world, dt);

        // Transform propagation
        euca_scene::transform_propagation_system(&mut self.world);

        // MOBA camera follow
        euca_gameplay::camera::moba_camera_system(&mut self.world);

        // Upload GLB meshes that were loaded by the spawn handler.
        {
            let gpu = self.gpu.as_ref().unwrap();
            let renderer = self.renderer.as_mut().unwrap();
            euca_agent::routes::drain_pending_mesh_uploads(&mut self.world, renderer, gpu);
        }

        // ── Render ──────────────────────────────────────────────────────

        let gpu = self.gpu.as_ref().unwrap();
        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dota frame"),
            });

        let draw_commands = collect_draw_commands(&self.world);
        let light = {
            let query = Query::<&DirectionalLight>::new(&self.world);
            query.iter().next().cloned().unwrap_or_default()
        };
        let ambient = self
            .world
            .resource::<AmbientLight>()
            .cloned()
            .unwrap_or_default();
        let camera = self.world.resource::<Camera>().unwrap().clone();

        let renderer = self.renderer.as_mut().unwrap();

        // Sync post-process settings from world resource to renderer
        if let Some(pps) = self.world.resource::<PostProcessSettings>().cloned() {
            renderer.set_post_process_settings(pps);
        }

        // Collect point lights from the world
        let point_lights: Vec<(euca_math::Vec3, PointLight)> = {
            let query = Query::<(&GlobalTransform, &PointLight)>::new(&self.world);
            query
                .iter()
                .map(|(gt, pl)| (gt.0.translation, pl.clone()))
                .collect()
        };
        let pl_refs: Vec<(euca_math::Vec3, &PointLight)> =
            point_lights.iter().map(|(pos, pl)| (*pos, pl)).collect();

        renderer.render_to_view_with_lights(
            gpu,
            &camera,
            &light,
            &ambient,
            &draw_commands,
            &pl_refs,
            &[],
            &view,
            &mut encoder,
        );

        // UI overlay: health bars above entities + HUD
        {
            let vp = camera.view_projection_matrix(gpu.aspect_ratio());
            let vw = gpu.surface_config.width as f32;
            let vh = gpu.surface_config.height as f32;
            let mut ui_quads = build_health_bar_quads(&self.world, &vp, vw, vh);
            ui_quads.extend(build_hud_quads(&self.world, vw, vh));
            ui_quads.extend(build_top_bar_quads(&self.world, vw, vh));
            ui_quads.extend(build_minimap_quads(&self.world, vw, vh));
            ui_quads.extend(build_floating_text_quads(&self.world));
            ui_quads.extend(build_kill_feed_quads(&self.world, vw));
            if let Some(ui) = self.ui_overlay.as_mut() {
                ui.render(&*gpu, &mut encoder, &view, &ui_quads, vw, vh);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

// ── Loading screen ──────────────────────────────────────────────────────────

/// Build UI quads for the loading screen: a centered progress bar that fills
/// from left to right as meshes are uploaded to the GPU.
fn build_loading_screen_quads(loaded: usize, total: usize, vw: f32, vh: f32) -> Vec<UiQuad> {
    let mut quads = Vec::new();

    // Full-screen dark background.
    quads.push(UiQuad {
        x: 0.0,
        y: 0.0,
        w: vw,
        h: vh,
        color: [0.05, 0.05, 0.08, 1.0],
    });

    // Title bar — a colored accent stripe near the top-center.
    let title_w = 300.0;
    let title_h = 6.0;
    let title_x = (vw - title_w) * 0.5;
    let title_y = vh * 0.35;
    quads.push(UiQuad {
        x: title_x,
        y: title_y,
        w: title_w,
        h: title_h,
        color: [0.3, 0.6, 1.0, 0.9],
    });

    // Progress bar background (dark gray).
    let bar_w = vw * 0.5;
    let bar_h = 24.0;
    let bar_x = (vw - bar_w) * 0.5;
    let bar_y = vh * 0.5 - bar_h * 0.5;
    quads.push(UiQuad {
        x: bar_x,
        y: bar_y,
        w: bar_w,
        h: bar_h,
        color: [0.15, 0.15, 0.18, 1.0],
    });

    // Progress bar fill (green, proportional to loaded/total).
    let progress = if total > 0 {
        loaded as f32 / total as f32
    } else {
        0.0
    };
    let fill_w = bar_w * progress;
    if fill_w > 0.0 {
        quads.push(UiQuad {
            x: bar_x,
            y: bar_y,
            w: fill_w,
            h: bar_h,
            color: [0.2, 0.8, 0.3, 1.0],
        });
    }

    // Progress bar border (thin outline around the bar).
    let border = 2.0;
    // Top edge
    quads.push(UiQuad {
        x: bar_x - border,
        y: bar_y - border,
        w: bar_w + border * 2.0,
        h: border,
        color: [0.3, 0.3, 0.35, 1.0],
    });
    // Bottom edge
    quads.push(UiQuad {
        x: bar_x - border,
        y: bar_y + bar_h,
        w: bar_w + border * 2.0,
        h: border,
        color: [0.3, 0.3, 0.35, 1.0],
    });
    // Left edge
    quads.push(UiQuad {
        x: bar_x - border,
        y: bar_y,
        w: border,
        h: bar_h,
        color: [0.3, 0.3, 0.35, 1.0],
    });
    // Right edge
    quads.push(UiQuad {
        x: bar_x + bar_w,
        y: bar_y,
        w: border,
        h: bar_h,
        color: [0.3, 0.3, 0.35, 1.0],
    });

    // Small "counter" bar below the progress bar — width proportional to count,
    // giving a visual hint of how many items are done even without text.
    let counter_h = 8.0;
    let counter_y = bar_y + bar_h + 12.0;
    let max_dot_w = 6.0;
    let dot_gap = 2.0;
    let dots_to_show = loaded.min(50); // cap visual dots at 50
    let total_dots_w = dots_to_show as f32 * (max_dot_w + dot_gap);
    let dots_start_x = (vw - total_dots_w) * 0.5;
    for i in 0..dots_to_show {
        quads.push(UiQuad {
            x: dots_start_x + i as f32 * (max_dot_w + dot_gap),
            y: counter_y,
            w: max_dot_w,
            h: counter_h,
            color: [0.3, 0.7, 0.4, 0.6],
        });
    }

    quads
}

/// Map a `CreepType` to the string tag used in procedural mesh names.
fn creep_type_tag(ct: euca_gameplay::CreepType) -> &'static str {
    match ct {
        euca_gameplay::CreepType::Melee => "melee",
        euca_gameplay::CreepType::Ranged => "ranged",
        euca_gameplay::CreepType::Siege => "siege",
        euca_gameplay::CreepType::Super => "super",
    }
}

/// Tick all DotA MOBA subsystems that are driven by pure data + logic
/// (not ECS-native systems). Reads/writes ECS components as needed.
fn moba_subsystems_tick(world: &mut World, dt: f32) {
    // Borrow the moba state. We take it out temporarily to avoid holding
    // a mutable borrow on World while we also need to query components.
    let Some(mut moba) = world.remove_resource::<DotaMobaState>() else {
        return;
    };

    // ── 1. Day/night cycle ───────────────────────────────────────────
    moba.day_night.tick(dt);

    // ── 2. Crowd control tick — expire CC durations on all entities ──
    {
        let entities_with_cc: Vec<Entity> = {
            let q = Query::<(Entity, &euca_gameplay::CcState)>::new(world);
            q.iter().map(|(e, _)| e).collect()
        };
        for entity in entities_with_cc {
            if let Some(cc) = world.get_mut::<euca_gameplay::CcState>(entity) {
                cc.remove_expired(dt);
            }
        }
    }

    // ── 3. Fog of war — collect vision sources and update maps ───────
    {
        let vision_mult = moba.day_night.vision_multiplier();

        let mut sources_t1 = Vec::new();
        let mut sources_t2 = Vec::new();

        // Heroes and structures provide vision.
        let query_data: Vec<(Vec3, u8)> = {
            let q = Query::<(&GlobalTransform, &euca_gameplay::Team)>::new(world);
            q.iter().map(|(gt, t)| (gt.0.translation, t.0)).collect()
        };
        for (pos, team) in &query_data {
            // Base vision radius 12 units, modulated by day/night.
            let radius = 12.0 * vision_mult;
            // Vision map uses 2D (x, z) mapped to positive grid coordinates.
            // Offset by 35 so that world x=-35 maps to grid x=0 (128 cells cover 0..128).
            let src = VisionSource {
                team: *team as u32,
                position: [pos.x + 64.0, pos.z + 64.0],
                radius,
                provides_true_sight: false,
            };
            match team {
                1 => sources_t1.push(src),
                2 => sources_t2.push(src),
                _ => {}
            }
        }

        // Wards provide vision.
        for ward in &moba.wards {
            let src = VisionSource {
                team: ward.team,
                position: [ward.position[0] + 64.0, ward.position[1] + 64.0],
                radius: ward.vision_radius * vision_mult,
                provides_true_sight: ward.true_sight_radius > 0.0,
            };
            match ward.team {
                1 => sources_t1.push(src),
                2 => sources_t2.push(src),
                _ => {}
            }
        }

        euca_gameplay::update_vision(&mut moba.vision_t1, &sources_t1);
        euca_gameplay::update_vision(&mut moba.vision_t2, &sources_t2);
    }

    // ── 4. Ward tick — count down durations, remove expired ──────────
    euca_gameplay::tick_wards(&mut moba.wards, dt);
    euca_gameplay::tick_ward_stock(&mut moba.ward_stock_t1, dt);
    euca_gameplay::tick_ward_stock(&mut moba.ward_stock_t2, dt);

    // ── 5. Item active cooldowns and charges — tick per hero ─────────
    for item_state in moba.item_states.values_mut() {
        euca_gameplay::tick_cooldowns(item_state, dt);
        euca_gameplay::tick_charges(item_state, dt);
    }

    // ── 6. Roshan tick — respawn timer ───────────────────────────────
    let game_elapsed = world
        .resource::<GameState>()
        .map(|gs| gs.elapsed)
        .unwrap_or(0.0);
    if euca_gameplay::tick_roshan(&mut moba.roshan, dt) {
        euca_gameplay::respawn_roshan(&mut moba.roshan, game_elapsed / 60.0);
        log::info!("Roshan has respawned!");
    }

    // ── 7. Aegis tick — expire if 5 minutes elapsed ─────────────────
    if let Some(aegis) = &mut moba.aegis {
        if euca_gameplay::tick_aegis(aegis, dt) {
            log::info!("Aegis has expired");
            moba.aegis = None;
        }
    }

    // ── 8. Fortification tick ────────────────────────────────────────
    euca_gameplay::tick_fortification(&mut moba.fort_t1, dt);
    euca_gameplay::tick_fortification(&mut moba.fort_t2, dt);

    // ── 9. Creep wave spawner — spawn entities from wave events ──────
    let game_time_minutes = world
        .resource::<GameState>()
        .map(|gs| gs.elapsed / 60.0)
        .unwrap_or(0.0);

    let wave_events = moba.wave_spawner.tick(dt);
    for event in &wave_events {
        let spawn_pos = event.waypoints.first().copied().unwrap_or(Vec3::ZERO);

        // March direction: from first waypoint toward last.
        let march_dir = if event.waypoints.len() >= 2 {
            let last = event.waypoints.last().unwrap();
            (*last - spawn_pos).normalize()
        } else if event.team == 1 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(-1.0, 0.0, 0.0)
        };

        let creep_scale = Vec3::new(0.4, 0.4, 0.4);
        let z_spacing = 1.0_f32;
        let z_offset_base = -z_spacing * (event.composition.len() as f32 - 1.0) / 2.0;

        for (i, &creep_type) in event.composition.iter().enumerate() {
            let stats = euca_gameplay::creep_stats(creep_type);
            let bounty = euca_gameplay::creep_bounty(creep_type, game_time_minutes);
            let z_offset = z_offset_base + z_spacing * i as f32;

            let mut transform = Transform::from_translation(Vec3::new(
                spawn_pos.x,
                spawn_pos.y,
                spawn_pos.z + z_offset,
            ));
            transform.scale = creep_scale;

            let entity = world.spawn(LocalTransform(transform));
            world.insert(entity, GlobalTransform::default());
            world.insert(entity, euca_gameplay::Health::new(stats.hp));
            world.insert(entity, euca_gameplay::Team(event.team));
            world.insert(entity, euca_gameplay::EntityRole::Minion);
            world.insert(entity, euca_gameplay::GoldBounty(bounty as i32));

            let mut combat = euca_gameplay::AutoCombat::new();
            combat.damage = stats.damage;
            combat.speed = 3.0;
            world.insert(entity, combat);

            world.insert(entity, euca_physics::Velocity::default());
            world.insert(
                entity,
                euca_physics::PhysicsBody {
                    body_type: euca_physics::RigidBodyType::Kinematic,
                },
            );
            world.insert(entity, euca_gameplay::MarchDirection(march_dir));

            // Emit RuleSpawnEvent with a per-creep-type procedural mesh name.
            // The rendering layer resolves this to the pre-generated mesh handle.
            let mesh_name = euca_render::creep_mesh_name(creep_type_tag(creep_type), event.team);
            if let Some(events) = world.resource_mut::<Events>() {
                events.send(euca_gameplay::RuleSpawnEvent {
                    entity,
                    mesh: mesh_name,
                    color: Some(event.color.clone()),
                    scale: Some([creep_scale.x, creep_scale.y, creep_scale.z]),
                });
            }
        }

        if !event.composition.is_empty() {
            log::info!(
                "Wave {} spawned {} creeps for {:?} lane (team {})",
                event.wave_number,
                event.composition.len(),
                event.lane,
                event.team
            );
        }
    }

    // Return the state to the world.
    world.insert_resource(moba);
}

// ── Death / respawn animation systems ────────────────────────────────────────

/// Attach `DeathAnimation` to entities that just received the `Dead` marker.
///
/// We detect "just died" by looking for entities that have `Dead` but do
/// not yet have a `DeathAnimation` component.
fn spawn_death_animations(world: &mut World) {
    let newly_dead: Vec<Entity> = {
        let query = Query::<(Entity, &euca_gameplay::Dead)>::new(world);
        query
            .iter()
            .filter(|(e, _)| world.get::<DeathAnimation>(*e).is_none())
            .map(|(e, _)| e)
            .collect()
    };
    for entity in newly_dead {
        world.insert(
            entity,
            DeathAnimation {
                elapsed: 0.0,
                duration: 1.5,
            },
        );
    }
}

/// Attach `RespawnAnimation` to entities that were dead last frame but are
/// no longer dead (i.e. the `Dead` component was removed by `respawn_system`).
fn spawn_respawn_animations(world: &mut World, previously_dead: &HashSet<Entity>) {
    for &entity in previously_dead {
        if !world.is_alive(entity) {
            continue;
        }
        // Entity was dead last frame but no longer has Dead => just respawned.
        if world.get::<euca_gameplay::Dead>(entity).is_none() {
            // Clean up any leftover death animation component.
            world.remove::<DeathAnimation>(entity);
            world.insert(
                entity,
                RespawnAnimation {
                    elapsed: 0.0,
                    duration: 1.0,
                },
            );
        }
    }
}

/// Advance death animation timers each frame.
fn tick_death_animations(world: &mut World, dt: f32) {
    let entities: Vec<Entity> = {
        let query = Query::<(Entity, &DeathAnimation)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };
    for entity in entities {
        if let Some(anim) = world.get_mut::<DeathAnimation>(entity) {
            anim.elapsed += dt;
        }
    }
}

/// Advance respawn animation timers and remove the component when complete.
fn tick_respawn_animations(world: &mut World, dt: f32) {
    let entities: Vec<Entity> = {
        let query = Query::<(Entity, &RespawnAnimation)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };
    let mut finished = Vec::new();
    for entity in entities {
        if let Some(anim) = world.get_mut::<RespawnAnimation>(entity) {
            anim.elapsed += dt;
            if anim.elapsed >= anim.duration {
                finished.push(entity);
            }
        }
    }
    for entity in finished {
        world.remove::<RespawnAnimation>(entity);
    }
}

/// Snapshot the set of currently-dead entities for next-frame respawn detection.
fn collect_dead_entities(world: &World) -> HashSet<Entity> {
    let query = Query::<(Entity, &euca_gameplay::Dead)>::new(world);
    query.iter().map(|(e, _)| e).collect()
}

// ── Visual Effect Systems ────────────────────────────────────────────────────

/// Scan for combat `DamageEvent`s and `UseAbilityEvent`s emitted this frame
/// and spawn corresponding `VisualEffect` entities with meshes and materials.
fn vfx_spawn_system(world: &mut World) {
    let Some(vfx) = world.resource::<VfxAssets>().cloned() else {
        return;
    };

    // ── 1. Combat damage VFX (auto-attack projectiles / melee flashes) ──────

    let damage_events: Vec<euca_gameplay::DamageEvent> = world
        .resource::<Events>()
        .map(|e| e.read::<euca_gameplay::DamageEvent>().cloned().collect())
        .unwrap_or_default();

    for dmg in &damage_events {
        let Some(source) = dmg.source else {
            continue;
        };
        // Determine source and target positions.
        let source_pos = world
            .get::<LocalTransform>(source)
            .map(|lt| lt.0.translation);
        let target_pos = world
            .get::<LocalTransform>(dmg.target)
            .map(|lt| lt.0.translation);
        let (Some(src_pos), Some(tgt_pos)) = (source_pos, target_pos) else {
            continue;
        };

        // Decide VFX based on attacker properties.
        let attack_style = world
            .get::<euca_gameplay::AutoCombat>(source)
            .map(|ac| ac.attack_style);
        let attack_range = world
            .get::<euca_gameplay::AutoCombat>(source)
            .map(|ac| ac.range)
            .unwrap_or(1.5);

        let is_ranged =
            attack_style == Some(euca_gameplay::AttackStyle::Stationary) || attack_range > 3.0;

        if is_ranged {
            // Ranged attack or tower: flying projectile sphere.
            let team = world
                .get::<euca_gameplay::Team>(source)
                .map(|t| t.0)
                .unwrap_or(0);
            let color = if team == 1 {
                [0.2, 0.9, 1.0, 1.0] // Radiant: cyan
            } else {
                [1.0, 0.3, 0.1, 1.0] // Dire: red-orange
            };

            let dist = (tgt_pos - src_pos).length().max(0.1);
            let speed = 20.0; // units/sec — fast enough to feel snappy
            let lifetime = dist / speed;

            spawn_vfx_entity(
                world,
                &vfx,
                VisualEffect {
                    kind: VfxKind::Projectile {
                        from: Vec3::new(src_pos.x, src_pos.y + 1.0, src_pos.z),
                        to: Vec3::new(tgt_pos.x, tgt_pos.y + 0.5, tgt_pos.z),
                        color,
                    },
                    lifetime,
                    elapsed: 0.0,
                },
            );
        } else {
            // Melee attack: brief flash at target.
            spawn_vfx_entity(
                world,
                &vfx,
                VisualEffect {
                    kind: VfxKind::MeleeSlash { position: tgt_pos },
                    lifetime: 0.2,
                    elapsed: 0.0,
                },
            );
        }
    }

    // ── 2. Ability VFX ──────────────────────────────────────────────────────

    let ability_events: Vec<euca_gameplay::UseAbilityEvent> = world
        .resource::<Events>()
        .map(|e| {
            e.read::<euca_gameplay::UseAbilityEvent>()
                .cloned()
                .collect()
        })
        .unwrap_or_default();

    for evt in &ability_events {
        let caster_pos = world
            .get::<LocalTransform>(evt.entity)
            .map(|lt| lt.0.translation)
            .unwrap_or(Vec3::ZERO);
        let caster_rotation = world
            .get::<LocalTransform>(evt.entity)
            .map(|lt| lt.0.rotation)
            .unwrap_or(euca_math::Quat::IDENTITY);

        // Look up the ability effect for this slot.
        let effect = world
            .get::<euca_gameplay::AbilitySet>(evt.entity)
            .and_then(|set| set.get(evt.slot).map(|a| a.effect.clone()));
        let Some(effect) = effect else { continue };

        // Walk the effect tree and spawn VFX for visual-worthy effects.
        spawn_ability_vfx(world, &vfx, &effect, caster_pos, caster_rotation);
    }
}

/// Recursively walk an `AbilityEffect` tree and spawn VFX for visual effects.
fn spawn_ability_vfx(
    world: &mut World,
    vfx: &VfxAssets,
    effect: &AbilityEffect,
    caster_pos: Vec3,
    caster_rotation: euca_math::Quat,
) {
    match effect {
        AbilityEffect::AreaDamage { radius, .. } => {
            spawn_vfx_entity(
                world,
                vfx,
                VisualEffect {
                    kind: VfxKind::AreaCircle {
                        center: caster_pos,
                        max_radius: *radius,
                        color: [1.0, 0.4, 0.1, 0.7], // fiery orange
                    },
                    lifetime: 0.5,
                    elapsed: 0.0,
                },
            );
        }
        AbilityEffect::Heal { .. } => {
            spawn_vfx_entity(
                world,
                vfx,
                VisualEffect {
                    kind: VfxKind::FloatingRise {
                        origin: caster_pos,
                        color: [0.2, 1.0, 0.3, 1.0], // green
                    },
                    lifetime: 0.8,
                    elapsed: 0.0,
                },
            );
        }
        AbilityEffect::SpawnProjectile { speed, range, .. } => {
            // Compute facing direction from rotation.
            let forward = caster_rotation * Vec3::new(0.0, 0.0, 1.0);
            let from = Vec3::new(caster_pos.x, caster_pos.y + 1.0, caster_pos.z);
            let to = from + forward * *range;
            let lifetime = if *speed > 0.0 { *range / *speed } else { 1.0 };
            spawn_vfx_entity(
                world,
                vfx,
                VisualEffect {
                    kind: VfxKind::Projectile {
                        from,
                        to,
                        color: [0.3, 0.5, 1.0, 1.0], // blue ability projectile
                    },
                    lifetime,
                    elapsed: 0.0,
                },
            );
        }
        AbilityEffect::Chain(effects) => {
            for sub in effects {
                spawn_ability_vfx(world, vfx, sub, caster_pos, caster_rotation);
            }
        }
        // Other effects (Dash, ApplyEffect, ApplyCc, etc.) don't need VFX.
        _ => {}
    }
}

/// Spawn a VFX entity with the correct mesh, material, and transform.
fn spawn_vfx_entity(world: &mut World, vfx: &VfxAssets, effect: VisualEffect) {
    let (initial_pos, mesh, material, initial_scale) = match &effect.kind {
        VfxKind::Projectile { from, color, .. } => {
            let mat = vfx_material_for_color(vfx, color);
            (*from, vfx.sphere_mesh, mat, Vec3::new(1.0, 1.0, 1.0))
        }
        VfxKind::AreaCircle { center, color, .. } => {
            let mat = vfx_material_for_color(vfx, color);
            // Start at zero scale; tick system will expand it.
            let pos = Vec3::new(center.x, center.y + 0.1, center.z);
            (pos, vfx.disc_mesh, mat, Vec3::new(0.01, 1.0, 0.01))
        }
        VfxKind::FloatingRise { origin, color } => {
            let mat = vfx_material_for_color(vfx, color);
            (*origin, vfx.sphere_mesh, mat, Vec3::new(0.5, 0.5, 0.5))
        }
        VfxKind::MeleeSlash { position } => {
            let pos = Vec3::new(position.x, position.y + 0.5, position.z);
            (
                pos,
                vfx.sphere_mesh,
                vfx.mat_white,
                Vec3::new(0.8, 0.8, 0.8),
            )
        }
    };

    let mut transform = Transform::from_translation(initial_pos);
    transform.scale = initial_scale;

    let entity = world.spawn(LocalTransform(transform));
    world.insert(entity, GlobalTransform(transform));
    world.insert(entity, MeshRenderer { mesh });
    world.insert(entity, MaterialRef { handle: material });
    world.insert(entity, effect);
}

/// Pick the closest pre-uploaded VFX material based on dominant color channel.
fn vfx_material_for_color(vfx: &VfxAssets, color: &[f32; 4]) -> MaterialHandle {
    let [r, g, b, _] = *color;
    // Simple heuristic: pick by dominant channel.
    if g > r && g > b {
        vfx.mat_green
    } else if b > r && b > g {
        vfx.mat_blue
    } else if r > 0.8 && g > 0.6 {
        vfx.mat_yellow
    } else if r > g && r > b {
        vfx.mat_red
    } else if g > 0.5 && b > 0.5 {
        vfx.mat_cyan
    } else {
        vfx.mat_white
    }
}

/// Advance all active `VisualEffect` entities: update positions/scales and
/// despawn completed effects.
fn vfx_tick_system(world: &mut World, dt: f32) {
    // Collect all VFX entities and their current state.
    let vfx_entities: Vec<(Entity, VisualEffect)> = {
        let query = Query::<(Entity, &VisualEffect)>::new(world);
        query.iter().map(|(e, v)| (e, v.clone())).collect()
    };

    let mut to_despawn: Vec<Entity> = Vec::new();

    for (entity, mut effect) in vfx_entities {
        effect.elapsed += dt;
        let t = (effect.elapsed / effect.lifetime).clamp(0.0, 1.0);

        if effect.elapsed >= effect.lifetime {
            to_despawn.push(entity);
            continue;
        }

        match &effect.kind {
            VfxKind::Projectile { from, to, .. } => {
                // Lerp position from start to end.
                let pos = Vec3::new(
                    from.x + (to.x - from.x) * t,
                    from.y + (to.y - from.y) * t,
                    from.z + (to.z - from.z) * t,
                );
                if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
                    lt.0.translation = pos;
                }
                if let Some(gt) = world.get_mut::<GlobalTransform>(entity) {
                    gt.0.translation = pos;
                }
            }
            VfxKind::AreaCircle {
                center, max_radius, ..
            } => {
                // Expand disc from 0 to max_radius.
                let scale = t * max_radius;
                if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
                    lt.0.translation = Vec3::new(center.x, center.y + 0.1, center.z);
                    lt.0.scale = Vec3::new(scale, 1.0, scale);
                }
                if let Some(gt) = world.get_mut::<GlobalTransform>(entity) {
                    gt.0.translation = Vec3::new(center.x, center.y + 0.1, center.z);
                    gt.0.scale = Vec3::new(scale, 1.0, scale);
                }
            }
            VfxKind::FloatingRise { origin, .. } => {
                // Rise 2 units over lifetime.
                let rise = t * 2.0;
                let pos = Vec3::new(origin.x, origin.y + rise, origin.z);
                // Shrink as it rises (fade effect via scale).
                let s = 0.5 * (1.0 - t * 0.7);
                if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
                    lt.0.translation = pos;
                    lt.0.scale = Vec3::new(s, s, s);
                }
                if let Some(gt) = world.get_mut::<GlobalTransform>(entity) {
                    gt.0.translation = pos;
                    gt.0.scale = Vec3::new(s, s, s);
                }
            }
            VfxKind::MeleeSlash { position } => {
                // Flash: scale up quickly then shrink.
                let scale = if t < 0.3 {
                    0.8 + t * 3.0 // expand to ~1.7
                } else {
                    1.7 * (1.0 - (t - 0.3) / 0.7) // shrink back to 0
                };
                let pos = Vec3::new(position.x, position.y + 0.5, position.z);
                if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
                    lt.0.translation = pos;
                    lt.0.scale = Vec3::new(scale, scale, scale);
                }
                if let Some(gt) = world.get_mut::<GlobalTransform>(entity) {
                    gt.0.translation = pos;
                    gt.0.scale = Vec3::new(scale, scale, scale);
                }
            }
        }

        // Write back elapsed time.
        if let Some(ve) = world.get_mut::<VisualEffect>(entity) {
            ve.elapsed = effect.elapsed;
        }
    }

    for entity in to_despawn {
        world.despawn(entity);
    }
}

/// Collect draw commands for all renderable entities, applying death/respawn
/// animation transforms when present.
///
/// - **Death animation** (entity has `Dead` + `DeathAnimation`): tilt forward
///   around X-axis (0 to 90 degrees), scale down (1 to 0), using a smooth
///   ease-out curve. Once the animation completes, the entity is hidden.
/// - **Respawn animation** (entity has `RespawnAnimation`): scale up from 0
///   to 1 with a smooth ease-out curve.
/// - **Dead with completed animation**: not rendered.
/// - **Normal entities**: rendered as usual.
fn collect_draw_commands(world: &World) -> Vec<DrawCommand> {
    let query = Query::<(Entity, &GlobalTransform, &MeshRenderer, &MaterialRef)>::new(world);
    let mut commands = Vec::new();

    for (e, gt, mr, mat) in query.iter() {
        let is_dead = world.get::<euca_gameplay::Dead>(e).is_some();
        let death_anim = world.get::<DeathAnimation>(e);
        let respawn_anim = world.get::<RespawnAnimation>(e);

        // Dead entity whose death animation has finished: skip entirely.
        if is_dead {
            if let Some(anim) = &death_anim {
                if anim.elapsed >= anim.duration {
                    continue;
                }
            } else {
                // Dead but no animation component (shouldn't happen normally,
                // but handle gracefully): skip.
                continue;
            }
        }

        let mut model_matrix = gt.0.to_matrix();

        // Apply visual ground offset so mesh bottoms sit on the ground.
        if let Some(offset) = world.get::<GroundOffset>(e) {
            model_matrix.cols[3][1] += offset.0;
        }

        // Death animation: tilt forward and scale down with ease-out.
        if let Some(anim) = death_anim {
            let t = (anim.elapsed / anim.duration).clamp(0.0, 1.0);
            // Smooth ease-out: fast start, slow finish.
            let smooth = 1.0 - (1.0 - t) * (1.0 - t);

            // Tilt: rotate around X-axis from 0 to 90 degrees (entity falls forward).
            let tilt_angle = smooth * std::f32::consts::FRAC_PI_2;
            let tilt_rot =
                Mat4::from_rotation(Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), tilt_angle));

            // Scale: shrink from 1.0 to 0.0.
            let scale_factor = 1.0 - smooth;
            let scale_mat = Mat4::from_scale(Vec3::new(scale_factor, scale_factor, scale_factor));

            // Extract translation, apply rotation and scale around the entity's
            // world position so it tilts/shrinks in place.
            let translation = Vec3::new(
                model_matrix.cols[3][0],
                model_matrix.cols[3][1],
                model_matrix.cols[3][2],
            );
            let to_origin =
                Mat4::from_translation(Vec3::new(-translation.x, -translation.y, -translation.z));
            let from_origin = Mat4::from_translation(translation);
            model_matrix = from_origin * tilt_rot * scale_mat * to_origin * model_matrix;
        }

        // Respawn animation: scale up from 0 to 1 with ease-out.
        if let Some(anim) = respawn_anim {
            let t = (anim.elapsed / anim.duration).clamp(0.0, 1.0);
            // Ease-out: overshoot slightly for a satisfying "pop" feeling.
            // Deceleration curve: 1 - (1 - t)^2
            let smooth = 1.0 - (1.0 - t) * (1.0 - t);

            let scale_factor = smooth;
            let scale_mat = Mat4::from_scale(Vec3::new(scale_factor, scale_factor, scale_factor));

            let translation = Vec3::new(
                model_matrix.cols[3][0],
                model_matrix.cols[3][1],
                model_matrix.cols[3][2],
            );
            let to_origin =
                Mat4::from_translation(Vec3::new(-translation.x, -translation.y, -translation.z));
            let from_origin = Mat4::from_translation(translation);
            model_matrix = from_origin * scale_mat * to_origin * model_matrix;
        }

        commands.push(DrawCommand {
            mesh: mr.mesh,
            material: mat.handle,
            model_matrix,
            aabb: None,
        });
    }

    commands
}

/// Day/night cycle: modulate lighting based on the DayNightCycle in DotaMobaState.
/// Also applies subtle Radiant vs Dire color grading based on camera position.
fn day_night_system(world: &mut World, _dt: f32) {
    // Use the authoritative DayNightCycle from the MOBA state.
    let day_factor = world
        .resource::<DotaMobaState>()
        .map(|moba| {
            if moba.day_night.is_day() {
                // Smooth transition within the day portion (full bright at midday).
                let progress = moba.day_night.current_time / moba.day_night.day_duration;
                // Bell curve: peak at 0.5, min at edges.
                let t = (progress * std::f32::consts::PI).sin();
                0.5 + 0.5 * t
            } else {
                // Night: dim lighting, smooth transition.
                let night_elapsed = moba.day_night.current_time - moba.day_night.day_duration;
                let progress = night_elapsed / moba.day_night.night_duration;
                let t = (progress * std::f32::consts::PI).sin();
                0.5 - 0.35 * t // 0.15 at deepest night, 0.5 at transitions
            }
        })
        .unwrap_or(1.0);

    // Interpolate directional light — subtle variation, never too dark
    let query_entities: Vec<Entity> = {
        let query = Query::<(Entity, &DirectionalLight)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };
    for entity in query_entities {
        if let Some(light) = world.get_mut::<DirectionalLight>(entity) {
            // Day: neutral white (1.0, 1.0, 1.0), intensity 2.0
            // Night: slightly cool blue (0.85, 0.88, 0.95), intensity 1.2
            light.color = [
                0.85 + 0.15 * day_factor,
                0.88 + 0.12 * day_factor,
                0.95 + 0.05 * day_factor,
            ];
            light.intensity = 1.2 + 0.8 * day_factor;
        }
    }

    // Interpolate ambient light — very subtle
    if let Some(ambient) = world.resource_mut::<AmbientLight>() {
        ambient.intensity = 0.15 + 0.05 * day_factor;
        ambient.color = [1.0, 1.0, 1.0];
    }

    // Keep temperature neutral — the shader's temperature implementation
    // shifts +0.005 red / -0.005 blue per Kelvin, which is too aggressive
    // for subtle color grading. Even ±50K creates a visible orange/blue cast.
    if let Some(pps) = world.resource_mut::<PostProcessSettings>() {
        pps.temperature = 0.0;
    }
}

/// Add glowing point lights on towers and structures for atmosphere.
fn add_structure_lights(world: &mut World) {
    use euca_gameplay::{EntityRole, Team};
    use euca_render::PointLight;

    let structures: Vec<(Entity, Vec3, u8, EntityRole)> = {
        let query = Query::<(Entity, &GlobalTransform, &Team, &EntityRole)>::new(world);
        query
            .iter()
            .filter(|(_, _, _, role)| matches!(role, EntityRole::Tower | EntityRole::Structure))
            .map(|(e, gt, t, r)| (e, gt.0.translation, t.0, *r))
            .collect()
    };

    for &(_entity, pos, team, role) in &structures {
        let (color, intensity, range) = match (team, role) {
            (1, EntityRole::Structure) => ([0.2, 0.8, 0.8], 2.0, 10.0), // Radiant ancient: cyan
            (2, EntityRole::Structure) => ([0.8, 0.2, 0.1], 2.0, 10.0), // Dire ancient: red
            (1, _) => ([0.3, 0.6, 0.7], 1.0, 6.0),                      // Radiant tower: soft cyan
            (2, _) => ([0.7, 0.3, 0.1], 1.0, 6.0),                      // Dire tower: soft orange
            _ => ([0.5, 0.5, 0.5], 0.8, 5.0),
        };

        // Spawn a light entity at the structure's position, elevated
        let light_pos = Vec3::new(pos.x, pos.y + 3.0, pos.z);
        let light_entity = world.spawn(euca_scene::LocalTransform(Transform::from_translation(
            light_pos,
        )));
        world.insert(
            light_entity,
            euca_scene::GlobalTransform(Transform::from_translation(light_pos)),
        );
        world.insert(
            light_entity,
            PointLight {
                color,
                intensity,
                range,
            },
        );
    }

    let count = structures.len();
    if count > 0 {
        log::info!("Added point lights to {count} structures");
    }
}

/// Build health bar UI quads for all alive entities with Health + GlobalTransform.
///
/// Renders Dota 2-style health bars: team-colored, sized by entity role, with
/// dark borders, mana bars for heroes, and level indicators. Only shows bars
/// for damaged entities (full health entities are hidden).
fn build_health_bar_quads(
    world: &World,
    view_proj: &Mat4,
    viewport_w: f32,
    viewport_h: f32,
) -> Vec<UiQuad> {
    use euca_gameplay::player::PlayerHero;
    use euca_gameplay::{Dead, EntityRole, Health, Level, Mana, Team};

    const BORDER: f32 = 1.0;
    const BORDER_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 0.7];
    const BG_COLOR: [f32; 4] = [0.1, 0.1, 0.1, 0.6];

    // Team fill colors.
    const RADIANT_COLOR: [f32; 4] = [0.1, 0.8, 0.2, 0.9];
    const DIRE_COLOR: [f32; 4] = [0.8, 0.15, 0.15, 0.9];
    const NEUTRAL_COLOR: [f32; 4] = [0.8, 0.8, 0.2, 0.9];
    // Player's own hero gets a brighter green.
    const PLAYER_HERO_COLOR: [f32; 4] = [0.15, 0.95, 0.3, 0.95];
    const PLAYER_HERO_OUTLINE: [f32; 4] = [1.0, 1.0, 1.0, 0.6];

    const MANA_COLOR: [f32; 4] = [0.2, 0.3, 0.9, 0.85];
    const MANA_BG_COLOR: [f32; 4] = [0.05, 0.05, 0.15, 0.5];

    let mut quads = Vec::new();
    let query = Query::<(Entity, &GlobalTransform, &Health)>::new(world);

    // Identify the player's hero entity.
    let player_entity = {
        let pq = Query::<(Entity, &PlayerHero)>::new(world);
        pq.iter().next().map(|(e, _)| e)
    };

    for (entity, gt, health) in query.iter() {
        // Skip dead entities.
        if world.get::<Dead>(entity).is_some() {
            continue;
        }
        // Skip entities with no meaningful health.
        if health.max <= 0.0 {
            continue;
        }
        // Skip entities at full health.
        if health.current >= health.max {
            continue;
        }

        let role = world.get::<EntityRole>(entity);
        let is_hero = matches!(role, Some(EntityRole::Hero));
        let is_player_hero = player_entity == Some(entity);

        // Bar dimensions by entity role:
        //   Heroes:           80px wide, 8px tall, world Y offset 2.5
        //   Towers/Structures: 60px wide, 6px tall, world Y offset 4.0
        //   Creeps/Minions:   40px wide, 4px tall, world Y offset 1.0
        let (bar_w, bar_h, world_y_offset) = match role {
            Some(EntityRole::Hero) => (80.0_f32, 8.0_f32, 2.5_f32),
            Some(EntityRole::Tower) | Some(EntityRole::Structure) => (60.0, 6.0, 4.0),
            _ => (40.0, 4.0, 1.0),
        };

        // Project the offset world position (above the entity model) to screen space.
        let world_pos = gt.0.translation;
        let above =
            euca_math::Vec4::new(world_pos.x, world_pos.y + world_y_offset, world_pos.z, 1.0);
        let clip = *view_proj * above;
        if clip.w <= 0.0 {
            continue; // behind camera
        }

        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;

        // NDC to screen pixels.
        let screen_x = (ndc_x + 1.0) * 0.5 * viewport_w;
        let screen_y = (1.0 - ndc_y) * 0.5 * viewport_h; // Y flipped

        let bar_x = screen_x - bar_w * 0.5;
        let bar_y = screen_y - bar_h * 0.5;

        // Skip if off-screen (with border margin).
        if bar_x + bar_w + BORDER < 0.0
            || bar_x - BORDER > viewport_w
            || bar_y + bar_h + BORDER < 0.0
            || bar_y - BORDER > viewport_h
        {
            continue;
        }

        let fill_frac = health.fraction();
        let team = world.get::<Team>(entity).map(|t| t.0).unwrap_or(0);

        // ── Player hero: white outline (rendered as a slightly larger rect behind everything) ──
        if is_player_hero {
            quads.push(UiQuad {
                x: bar_x - BORDER - 1.0,
                y: bar_y - BORDER - 1.0,
                w: bar_w + (BORDER + 1.0) * 2.0,
                h: bar_h + (BORDER + 1.0) * 2.0,
                color: PLAYER_HERO_OUTLINE,
            });
        }

        // ── Dark border ──
        quads.push(UiQuad {
            x: bar_x - BORDER,
            y: bar_y - BORDER,
            w: bar_w + BORDER * 2.0,
            h: bar_h + BORDER * 2.0,
            color: BORDER_COLOR,
        });

        // ── Background ──
        quads.push(UiQuad {
            x: bar_x,
            y: bar_y,
            w: bar_w,
            h: bar_h,
            color: BG_COLOR,
        });

        // ── Health fill ──
        let fill_color = if is_player_hero {
            PLAYER_HERO_COLOR
        } else if team == 0 {
            NEUTRAL_COLOR
        } else if team == 1 {
            RADIANT_COLOR
        } else {
            DIRE_COLOR
        };

        quads.push(UiQuad {
            x: bar_x,
            y: bar_y,
            w: bar_w * fill_frac,
            h: bar_h,
            color: fill_color,
        });

        // ── Mana bar (heroes only) ──
        if is_hero {
            if let Some(mana) = world.get::<Mana>(entity) {
                if mana.max > 0.0 {
                    let mana_h = (bar_h * 0.4).max(2.0); // thin bar below health
                    let mana_y = bar_y + bar_h + BORDER;
                    let mana_frac = (mana.current / mana.max).clamp(0.0, 1.0);

                    // Mana border
                    quads.push(UiQuad {
                        x: bar_x - BORDER,
                        y: mana_y - BORDER,
                        w: bar_w + BORDER * 2.0,
                        h: mana_h + BORDER * 2.0,
                        color: BORDER_COLOR,
                    });

                    // Mana background
                    quads.push(UiQuad {
                        x: bar_x,
                        y: mana_y,
                        w: bar_w,
                        h: mana_h,
                        color: MANA_BG_COLOR,
                    });

                    // Mana fill
                    quads.push(UiQuad {
                        x: bar_x,
                        y: mana_y,
                        w: bar_w * mana_frac,
                        h: mana_h,
                        color: MANA_COLOR,
                    });
                }
            }
        }

        // ── Level indicator (heroes only) ──
        if is_hero {
            if let Some(lvl) = world.get::<Level>(entity) {
                let indicator_size = bar_h + 2.0;
                let indicator_x = bar_x - BORDER - indicator_size - 2.0;
                let indicator_y = bar_y - 1.0;

                // Level brightness scales with level (1-30). Higher = brighter.
                let brightness = 0.3 + 0.7 * (lvl.level as f32 / 30.0).min(1.0);

                // Border
                quads.push(UiQuad {
                    x: indicator_x - BORDER,
                    y: indicator_y - BORDER,
                    w: indicator_size + BORDER * 2.0,
                    h: indicator_size + BORDER * 2.0,
                    color: BORDER_COLOR,
                });

                // Level square — golden tint that brightens with level.
                quads.push(UiQuad {
                    x: indicator_x,
                    y: indicator_y,
                    w: indicator_size,
                    h: indicator_size,
                    color: [brightness * 0.9, brightness * 0.75, brightness * 0.1, 0.9],
                });
            }
        }
    }

    quads
}

/// Build the full Dota 2-style HUD: bottom panel with portrait, HP/mana bars,
/// ability slots (Q/W/E/R) with cooldowns and hotkey labels, item slots (3x2),
/// gold counter, level/XP indicator.
fn build_hud_quads(world: &World, viewport_w: f32, viewport_h: f32) -> Vec<UiQuad> {
    let mut quads = Vec::new();

    // Find the player hero
    let hero = {
        let pq = Query::<(Entity, &euca_gameplay::player::PlayerHero)>::new(world);
        pq.iter().next().map(|(e, _)| e)
    };
    let hero = match hero {
        Some(h) => h,
        None => return quads,
    };

    // ── Layout constants ──
    let panel_h = viewport_h * 0.15; // bottom 15% of screen
    let panel_y = viewport_h - panel_h;
    let panel_w = viewport_w * 0.72; // centered panel width
    let panel_x = (viewport_w - panel_w) * 0.5;

    let portrait_size = panel_h - 16.0; // square portrait area
    let bar_region_w = panel_w * 0.28; // HP/mana bar region width
    let ability_slot_size = (panel_h - 24.0) * 0.85; // ability square size
    let ability_gap = 6.0;
    let item_slot_size = (panel_h - 30.0) * 0.42; // smaller item squares
    let item_gap = 4.0;

    // ── 1. Dark background panel ──
    quads.push(UiQuad {
        x: panel_x,
        y: panel_y,
        w: panel_w,
        h: panel_h,
        color: [0.08, 0.08, 0.1, 0.9],
    });

    // Thin top border for the panel
    quads.push(UiQuad {
        x: panel_x,
        y: panel_y,
        w: panel_w,
        h: 2.0,
        color: [0.25, 0.25, 0.3, 0.8],
    });

    // ── 2. Portrait area (left side of panel) ──
    let portrait_x = panel_x + 8.0;
    let portrait_y = panel_y + 8.0;
    let team = world
        .get::<euca_gameplay::Team>(hero)
        .map(|t| t.0)
        .unwrap_or(1);

    // Portrait border (team-colored)
    let portrait_border_color = if team == 1 {
        [0.1, 0.6, 0.6, 0.8] // Radiant: cyan
    } else {
        [0.6, 0.15, 0.1, 0.8] // Dire: red
    };
    quads.push(UiQuad {
        x: portrait_x - 2.0,
        y: portrait_y - 2.0,
        w: portrait_size + 4.0,
        h: portrait_size + 4.0,
        color: portrait_border_color,
    });

    // Portrait fill (dark with team-tinted color to represent hero)
    let portrait_fill = if team == 1 {
        [0.08, 0.18, 0.22, 0.95] // Radiant: dark teal
    } else {
        [0.22, 0.08, 0.06, 0.95] // Dire: dark crimson
    };
    quads.push(UiQuad {
        x: portrait_x,
        y: portrait_y,
        w: portrait_size,
        h: portrait_size,
        color: portrait_fill,
    });

    // Portrait inner detail — a smaller brighter square to suggest a face
    let inner_margin = portrait_size * 0.2;
    quads.push(UiQuad {
        x: portrait_x + inner_margin,
        y: portrait_y + inner_margin * 0.6,
        w: portrait_size - inner_margin * 2.0,
        h: portrait_size - inner_margin * 1.5,
        color: if team == 1 {
            [0.12, 0.3, 0.35, 0.7]
        } else {
            [0.35, 0.12, 0.1, 0.7]
        },
    });

    // ── 3. HP Bar (thick, 20px) ──
    let bars_x = portrait_x + portrait_size + 12.0;
    let hp_bar_h = 20.0;
    let hp_bar_y = panel_y + 10.0;

    // HP background
    quads.push(UiQuad {
        x: bars_x,
        y: hp_bar_y,
        w: bar_region_w,
        h: hp_bar_h,
        color: [0.12, 0.04, 0.04, 0.85],
    });

    // HP fill
    if let Some(health) = world.get::<euca_gameplay::Health>(hero) {
        let fill = (health.current / health.max).clamp(0.0, 1.0);
        quads.push(UiQuad {
            x: bars_x,
            y: hp_bar_y,
            w: bar_region_w * fill,
            h: hp_bar_h,
            color: [0.15, 0.78, 0.22, 0.92],
        });

        // HP bar tick marks (segmented look, every 250 HP)
        if health.max > 0.0 {
            let segment_hp = 250.0;
            let num_segments = (health.max / segment_hp).floor() as u32;
            for s in 1..num_segments {
                let tick_x = bars_x + (s as f32 * segment_hp / health.max) * bar_region_w;
                quads.push(UiQuad {
                    x: tick_x,
                    y: hp_bar_y,
                    w: 1.0,
                    h: hp_bar_h,
                    color: [0.0, 0.0, 0.0, 0.35],
                });
            }
        }
    }

    // ── 4. Mana Bar (15px, below HP) ──
    let mana_bar_h = 15.0;
    let mana_bar_y = hp_bar_y + hp_bar_h + 4.0;

    // Mana background
    quads.push(UiQuad {
        x: bars_x,
        y: mana_bar_y,
        w: bar_region_w,
        h: mana_bar_h,
        color: [0.04, 0.04, 0.14, 0.85],
    });

    // Mana fill
    if let Some(mana) = world.get::<euca_gameplay::Mana>(hero) {
        let fill = if mana.max > 0.0 {
            (mana.current / mana.max).clamp(0.0, 1.0)
        } else {
            0.0
        };
        quads.push(UiQuad {
            x: bars_x,
            y: mana_bar_y,
            w: bar_region_w * fill,
            h: mana_bar_h,
            color: [0.2, 0.38, 0.95, 0.92],
        });
    }

    // ── 5. Ability Slots (Q/W/E/R) ──
    let abilities_x = bars_x + bar_region_w + 16.0;
    let abilities_y = panel_y + (panel_h - ability_slot_size) * 0.5;

    let abilities = world.get::<euca_gameplay::AbilitySet>(hero);
    // Hotkey colors: Q=blue, W=cyan, E=green, R=yellow
    let hotkey_colors: [[f32; 4]; 4] = [
        [0.3, 0.5, 1.0, 0.9],  // Q: blue
        [0.2, 0.85, 0.9, 0.9], // W: cyan
        [0.3, 0.85, 0.3, 0.9], // E: green
        [0.95, 0.8, 0.2, 0.9], // R: yellow/gold
    ];

    for i in 0..4u32 {
        let slot_x = abilities_x + (ability_slot_size + ability_gap) * i as f32;

        // Slot background
        quads.push(UiQuad {
            x: slot_x,
            y: abilities_y,
            w: ability_slot_size,
            h: ability_slot_size,
            color: [0.15, 0.15, 0.18, 0.9],
        });

        let slot_enum = match i {
            0 => euca_gameplay::AbilitySlot::Q,
            1 => euca_gameplay::AbilitySlot::W,
            2 => euca_gameplay::AbilitySlot::E,
            _ => euca_gameplay::AbilitySlot::R,
        };

        if let Some(ability_set) = abilities {
            if let Some(ability) = ability_set.get(slot_enum) {
                if ability.cooldown_remaining > 0.0 {
                    // On cooldown: dark overlay + clock sweep effect (fill from bottom)
                    let cd_frac = (ability.cooldown_remaining / ability.cooldown).clamp(0.0, 1.0);
                    let cd_h = ability_slot_size * cd_frac;
                    quads.push(UiQuad {
                        x: slot_x,
                        y: abilities_y + (ability_slot_size - cd_h),
                        w: ability_slot_size,
                        h: cd_h,
                        color: [0.0, 0.0, 0.0, 0.65],
                    });
                } else {
                    // Ready: bright border around the entire slot
                    let bw = 2.0;
                    // Top
                    quads.push(UiQuad {
                        x: slot_x,
                        y: abilities_y,
                        w: ability_slot_size,
                        h: bw,
                        color: [0.3, 0.8, 1.0, 0.8],
                    });
                    // Bottom
                    quads.push(UiQuad {
                        x: slot_x,
                        y: abilities_y + ability_slot_size - bw,
                        w: ability_slot_size,
                        h: bw,
                        color: [0.3, 0.8, 1.0, 0.8],
                    });
                    // Left
                    quads.push(UiQuad {
                        x: slot_x,
                        y: abilities_y + bw,
                        w: bw,
                        h: ability_slot_size - bw * 2.0,
                        color: [0.3, 0.8, 1.0, 0.8],
                    });
                    // Right
                    quads.push(UiQuad {
                        x: slot_x + ability_slot_size - bw,
                        y: abilities_y + bw,
                        w: bw,
                        h: ability_slot_size - bw * 2.0,
                        color: [0.3, 0.8, 1.0, 0.8],
                    });
                }
            }
        }

        // Hotkey label: small colored indicator in top-left corner
        let label_size = 10.0;
        quads.push(UiQuad {
            x: slot_x + 2.0,
            y: abilities_y + 2.0,
            w: label_size,
            h: label_size,
            color: hotkey_colors[i as usize],
        });
    }

    // ── 6. Item Slots (3x2 grid) ──
    let items_x = abilities_x + (ability_slot_size + ability_gap) * 4.0 + 16.0;
    let items_y = panel_y + (panel_h - (item_slot_size * 2.0 + item_gap)) * 0.5;

    let inventory = world.get::<euca_gameplay::Inventory>(hero);

    for row in 0..2u32 {
        for col in 0..3u32 {
            let slot_idx = (row * 3 + col) as usize;
            let ix = items_x + (item_slot_size + item_gap) * col as f32;
            let iy = items_y + (item_slot_size + item_gap) * row as f32;

            let has_item = inventory
                .as_ref()
                .and_then(|inv| inv.slots.get(slot_idx).and_then(|s| s.as_ref()))
                .is_some();

            let slot_color = if has_item {
                [0.18, 0.18, 0.22, 0.85] // filled: slightly brighter
            } else {
                [0.1, 0.1, 0.12, 0.7] // empty: dark
            };

            quads.push(UiQuad {
                x: ix,
                y: iy,
                w: item_slot_size,
                h: item_slot_size,
                color: slot_color,
            });

            // Filled item: inner colored indicator
            if has_item {
                let inset = item_slot_size * 0.2;
                quads.push(UiQuad {
                    x: ix + inset,
                    y: iy + inset,
                    w: item_slot_size - inset * 2.0,
                    h: item_slot_size - inset * 2.0,
                    color: [0.35, 0.3, 0.2, 0.6],
                });
            }
        }
    }

    // ── 7. Gold counter (right side of panel) ──
    let gold = world
        .get::<euca_gameplay::HeroEconomy>(hero)
        .map(|e| e.wallet.total() as i32)
        .or_else(|| world.get::<euca_gameplay::Gold>(hero).map(|g| g.0))
        .unwrap_or(0);

    let gold_area_x = items_x + (item_slot_size + item_gap) * 3.0 + 12.0;
    let gold_area_w = 80.0;
    let gold_bar_h = 20.0;
    let gold_y = panel_y + 10.0;

    // Gold background
    quads.push(UiQuad {
        x: gold_area_x,
        y: gold_y,
        w: gold_area_w,
        h: gold_bar_h,
        color: [0.12, 0.12, 0.1, 0.8],
    });

    // Gold fill: width proportional to gold (max display 5000)
    let gold_fill = (gold as f32 / 5000.0).clamp(0.0, 1.0) * gold_area_w;
    quads.push(UiQuad {
        x: gold_area_x,
        y: gold_y,
        w: gold_fill,
        h: gold_bar_h,
        color: [1.0, 0.84, 0.0, 0.9],
    });

    // Gold icon indicator (small bright square)
    quads.push(UiQuad {
        x: gold_area_x + 2.0,
        y: gold_y + 2.0,
        w: 8.0,
        h: 8.0,
        color: [1.0, 0.92, 0.3, 1.0],
    });

    // ── 8. Level / XP indicator (below gold) ──
    let level = world
        .get::<euca_gameplay::Level>(hero)
        .map(|l| (l.level, l.xp, l.xp_to_next))
        .unwrap_or((1, 0, 180));
    let (hero_level, hero_xp, hero_xp_next) = level;

    let level_y = gold_y + gold_bar_h + 8.0;

    // Level square — size encodes level number visually
    let level_box_size = 28.0;
    quads.push(UiQuad {
        x: gold_area_x,
        y: level_y,
        w: level_box_size,
        h: level_box_size,
        color: [0.12, 0.12, 0.15, 0.9],
    });

    // Level fill — brighter interior proportional to level/18
    let level_fill_size = level_box_size * (hero_level as f32 / 18.0).clamp(0.0, 1.0);
    quads.push(UiQuad {
        x: gold_area_x + (level_box_size - level_fill_size) * 0.5,
        y: level_y + (level_box_size - level_fill_size) * 0.5,
        w: level_fill_size,
        h: level_fill_size,
        color: [0.3, 0.7, 1.0, 0.85],
    });

    // XP bar underneath level box
    let xp_bar_w = gold_area_w;
    let xp_bar_h = 6.0;
    let xp_bar_y = level_y + level_box_size + 4.0;
    let xp_fill = if hero_xp_next > 0 {
        (hero_xp as f32 / hero_xp_next as f32).clamp(0.0, 1.0)
    } else {
        1.0
    };

    // XP background
    quads.push(UiQuad {
        x: gold_area_x,
        y: xp_bar_y,
        w: xp_bar_w,
        h: xp_bar_h,
        color: [0.1, 0.1, 0.12, 0.8],
    });

    // XP fill (purple)
    quads.push(UiQuad {
        x: gold_area_x,
        y: xp_bar_y,
        w: xp_bar_w * xp_fill,
        h: xp_bar_h,
        color: [0.6, 0.3, 0.9, 0.85],
    });

    quads
}

/// Build top bar quads: game clock (center), team scores (left/right), day/night indicator.
fn build_top_bar_quads(world: &World, viewport_w: f32, _viewport_h: f32) -> Vec<UiQuad> {
    let mut quads = Vec::new();

    let top_bar_h = 32.0;
    let top_bar_w = viewport_w * 0.4;
    let top_bar_x = (viewport_w - top_bar_w) * 0.5;

    // ── Top bar background ──
    quads.push(UiQuad {
        x: top_bar_x,
        y: 0.0,
        w: top_bar_w,
        h: top_bar_h,
        color: [0.06, 0.06, 0.08, 0.85],
    });

    // Bottom edge highlight
    quads.push(UiQuad {
        x: top_bar_x,
        y: top_bar_h - 1.0,
        w: top_bar_w,
        h: 1.0,
        color: [0.25, 0.25, 0.3, 0.6],
    });

    // ── Game clock (center of top bar) ──
    // Show elapsed time as a proportional fill bar (clock visualization)
    let elapsed = world
        .resource::<GameState>()
        .map(|gs| gs.elapsed)
        .unwrap_or(0.0);
    let total_seconds = elapsed as u32;
    let minutes = total_seconds / 60;
    let seconds = total_seconds % 60;

    // Clock background (center)
    let clock_w = 80.0;
    let clock_h = 22.0;
    let clock_x = (viewport_w - clock_w) * 0.5;
    let clock_y = (top_bar_h - clock_h) * 0.5;

    quads.push(UiQuad {
        x: clock_x,
        y: clock_y,
        w: clock_w,
        h: clock_h,
        color: [0.1, 0.1, 0.12, 0.9],
    });

    // Clock progress bar — fills proportionally within each minute
    let minute_progress = seconds as f32 / 60.0;
    quads.push(UiQuad {
        x: clock_x,
        y: clock_y,
        w: clock_w * minute_progress,
        h: clock_h,
        color: [0.2, 0.2, 0.25, 0.6],
    });

    // Minute counter — series of small pips, one per minute elapsed (max 60)
    let pip_size = 3.0;
    let pip_y = clock_y + clock_h + 2.0;
    let max_pips = minutes.min(20) as usize; // show up to 20 minute pips
    let pip_start_x = (viewport_w - max_pips as f32 * (pip_size + 1.0)) * 0.5;
    for p in 0..max_pips {
        quads.push(UiQuad {
            x: pip_start_x + p as f32 * (pip_size + 1.0),
            y: pip_y,
            w: pip_size,
            h: pip_size,
            color: [0.5, 0.5, 0.55, 0.7],
        });
    }

    // ── Team scores ──
    // Count kills per team by summing individual hero scores
    let (radiant_kills, dire_kills) = {
        let gs = world.resource::<GameState>();
        let q = Query::<(Entity, &euca_gameplay::Team)>::new(world);
        let mut r_kills = 0i32;
        let mut d_kills = 0i32;
        if let Some(state) = gs {
            for (entity, t) in q.iter() {
                let score = state.scores.get(&entity.index()).copied().unwrap_or(0);
                match t.0 {
                    1 => r_kills += score,
                    2 => d_kills += score,
                    _ => {}
                }
            }
        }
        (r_kills, d_kills)
    };

    // Radiant score (left of clock) — green
    let score_w = 50.0;
    let score_h = 22.0;
    let radiant_score_x = clock_x - score_w - 12.0;
    let score_y = (top_bar_h - score_h) * 0.5;

    quads.push(UiQuad {
        x: radiant_score_x,
        y: score_y,
        w: score_w,
        h: score_h,
        color: [0.06, 0.15, 0.08, 0.9],
    });
    // Score fill (proportional indicator, 1 kill = 5px, capped at score_w)
    let r_fill = (radiant_kills as f32 * 5.0).clamp(0.0, score_w);
    quads.push(UiQuad {
        x: radiant_score_x,
        y: score_y,
        w: r_fill,
        h: score_h,
        color: [0.15, 0.7, 0.25, 0.6],
    });

    // Dire score (right of clock) — red
    let dire_score_x = clock_x + clock_w + 12.0;

    quads.push(UiQuad {
        x: dire_score_x,
        y: score_y,
        w: score_w,
        h: score_h,
        color: [0.15, 0.06, 0.06, 0.9],
    });
    let d_fill = (dire_kills as f32 * 5.0).clamp(0.0, score_w);
    quads.push(UiQuad {
        x: dire_score_x + score_w - d_fill,
        y: score_y,
        w: d_fill,
        h: score_h,
        color: [0.7, 0.15, 0.12, 0.6],
    });

    // ── Day/night indicator (right end of top bar) ──
    if let Some(moba) = world.resource::<DotaMobaState>() {
        let indicator_size = 18.0;
        let ix = top_bar_x + top_bar_w - indicator_size - 8.0;
        let iy = (top_bar_h - indicator_size) * 0.5;

        let color = if moba.day_night.is_day() {
            [1.0, 0.9, 0.3, 0.9] // day: sun yellow
        } else {
            [0.15, 0.15, 0.5, 0.9] // night: moon blue
        };
        quads.push(UiQuad {
            x: ix,
            y: iy,
            w: indicator_size,
            h: indicator_size,
            color,
        });
    }

    quads
}

/// Build minimap quads: dark background + colored dots for entities.
fn build_minimap_quads(world: &World, _viewport_w: f32, viewport_h: f32) -> Vec<UiQuad> {
    use euca_gameplay::{Dead, EntityRole, Team};

    let mut quads = Vec::new();

    // Minimap dimensions and position (bottom-left corner)
    let map_size = 160.0f32;
    let map_x = 10.0;
    let map_y = viewport_h - map_size - 10.0;
    let padding = 4.0;

    // Background border
    quads.push(UiQuad {
        x: map_x - padding,
        y: map_y - padding,
        w: map_size + padding * 2.0,
        h: map_size + padding * 2.0,
        color: [0.2, 0.2, 0.2, 0.8],
    });
    // Background fill (darker)
    quads.push(UiQuad {
        x: map_x,
        y: map_y,
        w: map_size,
        h: map_size,
        color: [0.05, 0.1, 0.05, 0.9],
    });

    // Lane lines on minimap — DotA 2 L-shaped layout
    let world_min_x = -30.0f32;
    let world_max_x = 30.0f32;
    let world_min_z = -30.0f32;
    let world_max_z = 30.0f32;

    let to_minimap = |wx: f32, wz: f32| -> (f32, f32) {
        let u = (wx - world_min_x) / (world_max_x - world_min_x);
        let v = (wz - world_min_z) / (world_max_z - world_min_z);
        (map_x + u * map_size, map_y + (1.0 - v) * map_size)
    };

    // Draw L-shaped lane paths on minimap.
    let lane_color = [0.3, 0.25, 0.15, 0.6]; // dirt-colored lane
    let lane_w = 2.0f32;

    // Top lane: left edge (x=-28, from z=-25 to z=25) + top edge (z=25, from x=-28 to x=28)
    {
        let (lx, ly_top) = to_minimap(-28.0, 25.0);
        let (_, ly_bot) = to_minimap(-28.0, -25.0);
        quads.push(UiQuad {
            x: lx - 1.0,
            y: ly_top,
            w: lane_w,
            h: ly_bot - ly_top,
            color: lane_color,
        });
        let (rx, _) = to_minimap(28.0, 25.0);
        quads.push(UiQuad {
            x: lx,
            y: ly_top - 1.0,
            w: rx - lx,
            h: lane_w,
            color: lane_color,
        });
    }
    // Mid lane: diagonal from (-25,-25) to (25,25) — slope = 1.
    // Draw several small quads along the diagonal.
    {
        let steps = 20;
        for i in 0..steps {
            let t0 = i as f32 / steps as f32;
            let t1 = (i + 1) as f32 / steps as f32;
            let x0 = -25.0 + t0 * 50.0;
            let z0 = -25.0 + t0 * 50.0;
            let x1 = -25.0 + t1 * 50.0;
            let z1 = -25.0 + t1 * 50.0;
            let (sx, sy) = to_minimap(x0, z0);
            let (ex, ey) = to_minimap(x1, z1);
            let dx = ex - sx;
            let dy = ey - sy;
            let len = (dx * dx + dy * dy).sqrt().max(1.0);
            quads.push(UiQuad {
                x: sx,
                y: sy.min(ey) - 0.5,
                w: len,
                h: lane_w,
                color: lane_color,
            });
        }
    }
    // Bot lane: bottom edge (z=-28, from x=-28 to x=25) + right edge (x=28, from z=-28 to z=25)
    {
        let (lx, ly) = to_minimap(-28.0, -28.0);
        let (rx, _) = to_minimap(25.0, -28.0);
        quads.push(UiQuad {
            x: lx,
            y: ly - 1.0,
            w: rx - lx,
            h: lane_w,
            color: lane_color,
        });
        let (_, ry_top) = to_minimap(28.0, 25.0);
        let (rx2, ry_bot) = to_minimap(28.0, -28.0);
        quads.push(UiQuad {
            x: rx2 - 1.0,
            y: ry_top,
            w: lane_w,
            h: ry_bot - ry_top,
            color: lane_color,
        });
    }

    // Entity dots
    let query = Query::<(Entity, &GlobalTransform)>::new(world);
    for (entity, gt) in query.iter() {
        if world.get::<Dead>(entity).is_some() {
            continue;
        }

        let pos = gt.0.translation;
        // Skip entities outside map bounds
        if pos.x < world_min_x || pos.x > world_max_x || pos.z < world_min_z || pos.z > world_max_z
        {
            continue;
        }

        let (mx, my) = to_minimap(pos.x, pos.z);
        let team = world.get::<Team>(entity).map(|t| t.0).unwrap_or(0);
        let role = world.get::<EntityRole>(entity);

        let (dot_size, dot_color) = match role {
            Some(EntityRole::Hero) => {
                if team == 1 {
                    (6.0, [0.0, 1.0, 1.0, 1.0]) // Radiant hero: bright cyan
                } else {
                    (6.0, [1.0, 0.2, 0.2, 1.0]) // Dire hero: bright red
                }
            }
            Some(EntityRole::Tower) | Some(EntityRole::Structure) => {
                if team == 1 {
                    (4.0, [0.2, 0.7, 0.7, 0.9]) // Radiant structure: soft cyan
                } else {
                    (4.0, [0.7, 0.2, 0.1, 0.9]) // Dire structure: soft red
                }
            }
            Some(EntityRole::Minion) => {
                if team == 1 {
                    (2.0, [0.3, 0.8, 0.8, 0.7]) // Radiant minion
                } else {
                    (2.0, [0.8, 0.3, 0.2, 0.7]) // Dire minion
                }
            }
            _ => continue, // Skip non-gameplay entities (trees, ground, lights)
        };

        quads.push(UiQuad {
            x: mx - dot_size * 0.5,
            y: my - dot_size * 0.5,
            w: dot_size,
            h: dot_size,
            color: dot_color,
        });
    }

    quads
}

// ── Floating combat text & kill feed systems ────────────────────────────────

/// Damage bar width: proportional to damage amount.
/// 100 damage = 60px wide. Clamped to [20, 120].
fn damage_bar_width(amount: f32) -> f32 {
    (amount * 0.6).clamp(20.0, 120.0)
}

/// Color for a damage type.
fn damage_color(damage_type: DamageType) -> [f32; 4] {
    match damage_type {
        DamageType::Physical => [1.0, 0.2, 0.2, 1.0],
        DamageType::Magical => [0.3, 0.5, 1.0, 1.0],
        DamageType::Pure | DamageType::HpRemoval => [1.0, 1.0, 1.0, 1.0],
    }
}

/// Read DamageEvents and create floating text entries at the target's screen position.
fn harvest_damage_events(world: &mut World) {
    let camera = match world.resource::<Camera>() {
        Some(c) => c.clone(),
        None => return,
    };

    let events: Vec<euca_gameplay::DamageEvent> = world
        .resource::<Events>()
        .map(|e| e.read::<euca_gameplay::DamageEvent>().cloned().collect())
        .unwrap_or_default();

    if events.is_empty() {
        return;
    }

    // Get viewport dimensions from screen size resource.
    let (vw, vh) = world
        .resource::<ScreenSize>()
        .map(|s| (s.width, s.height))
        .unwrap_or((WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32));

    let aspect = if vh > 0.0 { vw / vh } else { 16.0 / 9.0 };
    let vp = camera.view_projection_matrix(aspect);

    let mut new_entries = Vec::new();

    for event in &events {
        // Get target's world position.
        let world_pos = match world.get::<GlobalTransform>(event.target) {
            Some(gt) => gt.0.translation,
            None => continue,
        };

        // Project to screen space.
        let clip = vp * euca_math::Vec4::new(world_pos.x, world_pos.y + 1.5, world_pos.z, 1.0);
        if clip.w <= 0.0 {
            continue;
        }

        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        let screen_x = (ndc_x + 1.0) * 0.5 * vw;
        let screen_y = (1.0 - ndc_y) * 0.5 * vh;

        new_entries.push(FloatingText {
            screen_x,
            screen_y,
            bar_width: damage_bar_width(event.amount),
            color: damage_color(event.damage_type),
            lifetime: 1.5,
            elapsed: 0.0,
        });
    }

    if let Some(texts) = world.resource_mut::<FloatingTexts>() {
        texts.entries.extend(new_entries);
    }
}

/// Read DeathEvents and create kill feed entries.
fn harvest_death_events(world: &mut World) {
    let events: Vec<euca_gameplay::DeathEvent> = world
        .resource::<Events>()
        .map(|e| e.read::<euca_gameplay::DeathEvent>().cloned().collect())
        .unwrap_or_default();

    if events.is_empty() {
        return;
    }

    let player_team = {
        let pq = Query::<(Entity, &euca_gameplay::player::PlayerHero)>::new(world);
        pq.iter()
            .next()
            .and_then(|(e, _)| world.get::<euca_gameplay::Team>(e).map(|t| t.0))
            .unwrap_or(1)
    };

    let mut new_entries = Vec::new();

    for event in &events {
        let victim_team = world
            .get::<euca_gameplay::Team>(event.entity)
            .map(|t| t.0)
            .unwrap_or(0);

        let killer_team = event
            .killer
            .and_then(|k| world.get::<euca_gameplay::Team>(k).map(|t| t.0))
            .unwrap_or(0);

        // Killer color: green if ally kill, red if enemy kill, gray if neutral/unknown.
        let killer_color = if killer_team == player_team {
            [0.1, 0.9, 0.1, 1.0]
        } else if killer_team != 0 {
            [0.9, 0.1, 0.1, 1.0]
        } else {
            [0.5, 0.5, 0.5, 1.0]
        };

        // Victim color: same logic inverted.
        let victim_color = if victim_team == player_team {
            [0.1, 0.9, 0.1, 1.0]
        } else if victim_team != 0 {
            [0.9, 0.1, 0.1, 1.0]
        } else {
            [0.5, 0.5, 0.5, 1.0]
        };

        new_entries.push(KillFeedEntry {
            killer_color,
            victim_color,
            lifetime: 5.0,
            elapsed: 0.0,
        });
    }

    if let Some(feed) = world.resource_mut::<KillFeed>() {
        feed.entries.extend(new_entries);
    }
}

/// Detect gold/XP changes on the player hero and spawn floating popups.
fn harvest_gold_xp_changes(world: &mut World) {
    let hero = {
        let pq = Query::<(Entity, &euca_gameplay::player::PlayerHero)>::new(world);
        match pq.iter().next() {
            Some((e, _)) => e,
            None => return,
        }
    };

    // Current gold (prefer HeroEconomy wallet, fall back to Gold component).
    let current_gold = world
        .get::<euca_gameplay::HeroEconomy>(hero)
        .map(|e| e.wallet.total() as i32)
        .or_else(|| world.get::<euca_gameplay::Gold>(hero).map(|g| g.0))
        .unwrap_or(0);

    let current_xp = world
        .get::<euca_gameplay::Level>(hero)
        .map(|l| l.xp)
        .unwrap_or(0);

    let (prev_gold, prev_xp) = match world.resource::<GoldXpTracker>() {
        Some(t) => (t.prev_gold, t.prev_xp),
        None => return,
    };

    // Get hero screen position for popup placement.
    let camera = match world.resource::<Camera>() {
        Some(c) => c.clone(),
        None => return,
    };
    let (vw, vh) = world
        .resource::<ScreenSize>()
        .map(|s| (s.width, s.height))
        .unwrap_or((WINDOW_WIDTH as f32, WINDOW_HEIGHT as f32));
    let aspect = if vh > 0.0 { vw / vh } else { 16.0 / 9.0 };
    let vp = camera.view_projection_matrix(aspect);

    let hero_pos = world
        .get::<GlobalTransform>(hero)
        .map(|gt| gt.0.translation)
        .unwrap_or(Vec3::ZERO);

    let clip = vp * euca_math::Vec4::new(hero_pos.x, hero_pos.y + 2.0, hero_pos.z, 1.0);
    let (screen_x, screen_y) = if clip.w > 0.0 {
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        ((ndc_x + 1.0) * 0.5 * vw, (1.0 - ndc_y) * 0.5 * vh)
    } else {
        return;
    };

    let mut new_entries = Vec::new();

    // Gold gain popup (yellow).
    let gold_diff = current_gold - prev_gold;
    if gold_diff > 0 {
        new_entries.push(FloatingText {
            screen_x: screen_x + 30.0,
            screen_y,
            bar_width: (gold_diff as f32 * 0.3).clamp(15.0, 80.0),
            color: [1.0, 0.84, 0.0, 1.0],
            lifetime: 1.5,
            elapsed: 0.0,
        });
    }

    // XP gain popup (purple).
    if current_xp > prev_xp {
        let xp_diff = current_xp - prev_xp;
        new_entries.push(FloatingText {
            screen_x: screen_x - 30.0,
            screen_y,
            bar_width: (xp_diff as f32 * 0.3).clamp(15.0, 80.0),
            color: [0.6, 0.2, 0.9, 1.0],
            lifetime: 1.5,
            elapsed: 0.0,
        });
    }

    if let Some(texts) = world.resource_mut::<FloatingTexts>() {
        texts.entries.extend(new_entries);
    }

    // Update tracker for next frame.
    if let Some(tracker) = world.resource_mut::<GoldXpTracker>() {
        tracker.prev_gold = current_gold;
        tracker.prev_xp = current_xp;
    }
}

/// Tick all floating texts: advance elapsed, remove expired.
fn tick_floating_texts(world: &mut World, dt: f32) {
    if let Some(texts) = world.resource_mut::<FloatingTexts>() {
        for entry in &mut texts.entries {
            entry.elapsed += dt;
        }
        texts.entries.retain(|e| e.elapsed < e.lifetime);
    }
}

/// Tick all kill feed entries: advance elapsed, remove expired.
fn tick_kill_feed(world: &mut World, dt: f32) {
    if let Some(feed) = world.resource_mut::<KillFeed>() {
        for entry in &mut feed.entries {
            entry.elapsed += dt;
        }
        feed.entries.retain(|e| e.elapsed < e.lifetime);
    }
}

/// Build UI quads for floating damage/gold/XP indicators.
fn build_floating_text_quads(world: &World) -> Vec<UiQuad> {
    let mut quads = Vec::new();

    let texts = match world.resource::<FloatingTexts>() {
        Some(t) => t,
        None => return quads,
    };

    for entry in &texts.entries {
        let progress = (entry.elapsed / entry.lifetime).clamp(0.0, 1.0);

        // Rise upward: 40px over the lifetime.
        let y = entry.screen_y - progress * 40.0;
        let x = entry.screen_x - entry.bar_width * 0.5;

        // Fade out: full alpha for the first 60%, then linear fade.
        let alpha = if progress < 0.6 {
            entry.color[3]
        } else {
            entry.color[3] * (1.0 - (progress - 0.6) / 0.4)
        };

        let bar_h = 6.0;

        // Damage bar with fading alpha.
        quads.push(UiQuad {
            x,
            y,
            w: entry.bar_width,
            h: bar_h,
            color: [entry.color[0], entry.color[1], entry.color[2], alpha],
        });
    }

    quads
}

/// Build UI quads for the kill feed (top-right corner).
fn build_kill_feed_quads(world: &World, viewport_w: f32) -> Vec<UiQuad> {
    let mut quads = Vec::new();

    let feed = match world.resource::<KillFeed>() {
        Some(f) => f,
        None => return quads,
    };

    let entry_h = 14.0;
    let entry_gap = 4.0;
    let margin_right = 20.0;
    let margin_top = 50.0; // below day/night indicator
    let killer_bar_w = 30.0;
    let skull_w = 10.0;
    let victim_bar_w = 30.0;
    let total_w = killer_bar_w + skull_w + victim_bar_w + 8.0; // 8px internal gaps

    // Show newest entries first (most recent at top).
    for (i, entry) in feed.entries.iter().rev().enumerate() {
        if i >= 8 {
            break; // Show at most 8 entries
        }

        let progress = (entry.elapsed / entry.lifetime).clamp(0.0, 1.0);
        let alpha = if progress < 0.7 {
            0.9
        } else {
            0.9 * (1.0 - (progress - 0.7) / 0.3)
        };

        let y = margin_top + (entry_h + entry_gap) * i as f32;
        let x = viewport_w - margin_right - total_w;

        // Background.
        quads.push(UiQuad {
            x: x - 2.0,
            y: y - 1.0,
            w: total_w + 4.0,
            h: entry_h + 2.0,
            color: [0.0, 0.0, 0.0, 0.4 * alpha],
        });

        // Killer team bar.
        quads.push(UiQuad {
            x,
            y,
            w: killer_bar_w,
            h: entry_h,
            color: [
                entry.killer_color[0],
                entry.killer_color[1],
                entry.killer_color[2],
                alpha,
            ],
        });

        // Skull/separator (white cross).
        let skull_x = x + killer_bar_w + 4.0;
        quads.push(UiQuad {
            x: skull_x,
            y: y + 3.0,
            w: skull_w,
            h: entry_h - 6.0,
            color: [0.9, 0.9, 0.9, alpha],
        });
        quads.push(UiQuad {
            x: skull_x + 2.0,
            y: y + 1.0,
            w: skull_w - 4.0,
            h: entry_h - 2.0,
            color: [0.9, 0.9, 0.9, alpha],
        });

        // Victim team bar.
        quads.push(UiQuad {
            x: skull_x + skull_w + 4.0,
            y,
            w: victim_bar_w,
            h: entry_h,
            color: [
                entry.victim_color[0],
                entry.victim_color[1],
                entry.victim_color[2],
                alpha,
            ],
        });
    }

    quads
}

// ── Window event handling ───────────────────────────────────────────────────

impl ApplicationHandler for DotaClientApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.initialized {
            return;
        }

        let (survey, wgpu_instance) = HardwareSurvey::detect();
        let window = event_loop
            .create_window(self.window_attrs.clone())
            .expect("Failed to create window");
        let gpu = GpuContext::new(window, &survey, &wgpu_instance);
        let renderer = Renderer::new(&gpu);
        let ui_overlay = UiOverlayRenderer::new(&*gpu, gpu.surface_format());
        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
        self.ui_overlay = Some(ui_overlay);
        self.initialized = true;

        // Upload meshes, materials, and create the ground plane + light.
        // Level loading is deferred to the first render_frame (WaitingToLoad
        // phase) so the window is visible before the blocking GLB I/O starts.
        setup_default_assets(
            &mut self.world,
            self.gpu.as_ref().unwrap(),
            self.renderer.as_mut().unwrap(),
        );
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        forward_input(&mut self.world, &event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.logical_key == Key::Named(NamedKey::Escape)
                    && event.state == ElementState::Pressed =>
            {
                event_loop.exit()
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                    if let Some(r) = &mut self.renderer {
                        r.resize(gpu);
                    }
                }
                if let Some(vp) = self.world.resource_mut::<ViewportSize>() {
                    vp.width = size.width as f32;
                    vp.height = size.height as f32;
                }
                if let Some(ss) = self.world.resource_mut::<ScreenSize>() {
                    ss.width = size.width as f32;
                    ss.height = size.height as f32;
                }
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
                if let Some(gpu) = &self.gpu {
                    gpu.window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

/// Translate window events into `InputState` updates.
fn forward_input(world: &mut World, event: &WindowEvent) {
    use euca_input::InputKey;

    let Some(input) = world.resource_mut::<euca_input::InputState>() else {
        return;
    };

    match event {
        WindowEvent::KeyboardInput { event, .. } => {
            let key_name = match &event.logical_key {
                Key::Character(ch) => Some(ch.to_uppercase()),
                Key::Named(named) => match named {
                    NamedKey::Space => Some("Space".to_string()),
                    NamedKey::Escape => Some("Escape".to_string()),
                    _ => None,
                },
                _ => None,
            };
            if let Some(name) = key_name {
                match event.state {
                    ElementState::Pressed => input.press(InputKey::Key(name)),
                    ElementState::Released => input.release(InputKey::Key(name)),
                }
            }
        }
        WindowEvent::MouseInput { state, button, .. } => {
            let key = match button {
                winit::event::MouseButton::Left => Some(InputKey::MouseLeft),
                winit::event::MouseButton::Right => Some(InputKey::MouseRight),
                winit::event::MouseButton::Middle => Some(InputKey::MouseMiddle),
                _ => None,
            };
            if let Some(k) = key {
                match state {
                    ElementState::Pressed => input.press(k),
                    ElementState::Released => input.release(k),
                }
            }
        }
        WindowEvent::CursorMoved { position, .. } => {
            input.set_mouse_position(position.x as f32, position.y as f32);
        }
        WindowEvent::MouseWheel { delta, .. } => {
            let scroll = match delta {
                winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
            };
            input.set_scroll(scroll);
        }
        _ => {}
    }
}

// ── Entry point ─────────────────────────────────────────────────────────────

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("Euca Engine — DotA Client");
    log::info!(
        "Controls: Click to move, Q/W/E/R for abilities, Scroll to zoom, Hold 1 to center camera, Y to toggle lock"
    );

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = DotaClientApp::new();
    event_loop.run_app(&mut app).expect("Event loop failed");
}
