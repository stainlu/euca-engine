//! DotA-style MOBA client — playable single-binary demo.
//!
//! Opens a window, loads the DotA map from `levels/dota.json`, sets up items,
//! heroes, the MOBA camera, and a full gameplay loop with click-to-move, QWER
//! abilities, and shop access.
//!
//! Run: `cargo run -p euca-game --example dota_client`

use std::collections::HashMap;

use euca_core::Time;
use euca_ecs::{Entity, Events, Query, World};
use euca_gameplay::camera::{MobaCamera, ScreenSize};
use euca_gameplay::player_input::ViewportSize;
use euca_gameplay::{
    AbilityDef, AbilityEffect, AbilitySlot, AttrGrowth, BaseAttributes, GameState, HeroDef,
    HeroName, HeroRegistry, HeroTimings, ItemDef, ItemRegistry, PrimaryAttribute,
};
use euca_math::{Mat4, Transform, Vec3};
use euca_physics::PhysicsConfig;
use euca_render::*;
use euca_scene::{GlobalTransform, LocalTransform};

use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{WindowAttributes, WindowId};

// ── Constants ───────────────────────────────────────────────────────────────

const WINDOW_WIDTH: u32 = 1280;
const WINDOW_HEIGHT: u32 = 720;

/// Fixed timestep for gameplay systems (60 Hz).
const DT: f32 = 1.0 / 60.0;

// NOTE: No hardcoded entity indices. All entities are found by their
// ECS components (PlayerHero, Team, EntityRole, etc.), not by creation order.

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
                effect: AbilityEffect::Damage {
                    amount: 150.0,
                    category: "magical".into(),
                },
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
                effect: AbilityEffect::SpawnProjectile {
                    speed: 12.0,
                    range: 8.0,
                    width: 0.5,
                    damage: 100.0,
                    category: "magical".into(),
                },
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

    let grid_tex = renderer.checkerboard_texture(gpu, 512, 32);
    let grid_mat = renderer.upload_material(
        gpu,
        &Material::new([0.25, 0.35, 0.2, 1.0], 0.0, 0.95).with_texture(grid_tex),
    );

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

    // Capture material handles before moving materials into DefaultAssets
    let tree_mat = *materials.get("green").unwrap();

    let mut meshes = HashMap::new();
    meshes.insert("cube".to_string(), cube);
    meshes.insert("sphere".to_string(), sphere);
    meshes.insert("plane".to_string(), plane);

    world.insert_resource(euca_agent::routes::DefaultAssets {
        meshes,
        materials,
        default_material: blue,
    });

    // Ground plane — larger than default to accommodate the DotA map (-30..30 range)
    let g = world.spawn(LocalTransform(Transform::from_translation(Vec3::ZERO)));
    world.insert(g, GlobalTransform::default());
    world.insert(g, MeshRenderer { mesh: plane });
    world.insert(g, MaterialRef { handle: grid_mat });
    world.insert(g, euca_physics::PhysicsBody::fixed());
    world.insert(g, euca_physics::Collider::aabb(40.0, 0.01, 40.0));

    // River — reflective water strip across the center of the map
    let river_mesh = renderer.upload_mesh(gpu, &Mesh::plane(1.0));
    let river_mat = renderer.upload_material(
        gpu,
        &Material::new([0.15, 0.25, 0.45, 0.85], 0.6, 0.15), // blue, metallic, smooth
    );
    let r = world.spawn(LocalTransform({
        let mut t = Transform::from_translation(Vec3::new(0.0, 0.05, 0.0));
        t.scale = Vec3::new(70.0, 1.0, 4.0); // wide and narrow
        t
    }));
    world.insert(r, GlobalTransform::default());
    world.insert(r, MeshRenderer { mesh: river_mesh });
    world.insert(r, MaterialRef { handle: river_mat });

    // Directional light — neutral white sun for the DotA arena
    world.spawn(DirectionalLight {
        direction: [0.4, -0.9, 0.25],
        color: [1.0, 1.0, 1.0],
        intensity: 2.0,
        ..Default::default()
    });

    // Tree lines between lanes — defines MOBA map geography
    spawn_tree_lines(world, cube, tree_mat);
}

/// Spawn tree entities between the 3 lanes to define map geography.
/// Trees are placed in the gaps: between top lane (z=20) and mid (z=0),
/// and between mid (z=0) and bot lane (z=-20), plus jungle flanks.
fn spawn_tree_lines(world: &mut World, mesh: MeshHandle, material: MaterialHandle) {
    // Tree zones: (x_min, x_max, z_min, z_max) — areas between lanes
    let zones: &[(f32, f32, f32, f32)] = &[
        // Between top and mid lanes (z=6..14)
        (-28.0, 28.0, 6.0, 14.0),
        // Between mid and bot lanes (z=-14..-6)
        (-28.0, 28.0, -14.0, -6.0),
        // Top flank (above top lane)
        (-28.0, 28.0, 24.0, 28.0),
        // Bot flank (below bot lane)
        (-28.0, 28.0, -28.0, -24.0),
    ];

    let spacing = 3.0f32;
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
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let scale = 0.8 + ((seed >> 16) as f32 / 65536.0) * 1.2;

                let px = x + ox;
                let pz = z + oz;

                // Skip trees on lane paths (z≈20, z≈0, z≈-20, ±3 wide)
                let on_lane =
                    (pz.abs() < 3.0) || ((pz - 20.0).abs() < 3.0) || ((pz + 20.0).abs() < 3.0);
                if !on_lane {
                    let pos = Vec3::new(px, scale * 0.5, pz);
                    let mut xform = Transform::from_translation(pos);
                    xform.scale = Vec3::new(0.8, scale, 0.8);
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
                    world.insert(t, euca_physics::Collider::aabb(0.4, scale * 0.5, 0.4));
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
    level_loaded: bool,
}

impl DotaClientApp {
    fn new() -> Self {
        let mut world = World::new();
        world.insert_resource(Time::new());
        world.insert_resource(Camera::new(Vec3::new(0.0, 40.0, 30.0), Vec3::ZERO));
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

        Self {
            world,
            initialized: false,
            gpu: None,
            renderer: None,
            ui_overlay: None,
            window_attrs: WindowAttributes::default()
                .with_title("Euca Engine — DotA Client")
                .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT)),
            level_loaded: false,
        }
    }

    fn load_level(&mut self) {
        let path = "levels/dota.json";
        match std::fs::read_to_string(path) {
            Ok(data) => match serde_json::from_str::<serde_json::Value>(&data) {
                Ok(level) => {
                    let count = euca_agent::load_level_into_world(&mut self.world, &level);
                    log::info!("Level loaded: {count} entities from {path}");
                }
                Err(e) => {
                    log::error!("Invalid level JSON in {path}: {e}");
                    return;
                }
            },
            Err(e) => {
                log::error!("Cannot read level file {path}: {e}");
                return;
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

        // Build navmesh for pathfinding
        if self.world.resource::<euca_nav::NavMesh>().is_none() {
            let config = euca_nav::GridConfig {
                min: [-35.0, -25.0],
                max: [35.0, 25.0],
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
    }

    fn render_frame(&mut self) {
        self.world.resource_mut::<Time>().unwrap().update();

        let dt = self.world.resource::<Time>().map(|t| t.delta).unwrap_or(DT);

        // ── Gameplay systems (same order as euca-game main.rs) ──────────

        // Physics & transforms
        euca_physics::physics_step_system(&mut self.world);
        euca_physics::character_controller_system(&mut self.world, dt);

        // Player input -> commands -> execution
        euca_gameplay::player_input_system(&mut self.world);
        euca_gameplay::player::player_command_system(&mut self.world, dt);

        // Stat pipeline
        euca_gameplay::equipment_stat_system(&mut self.world);
        euca_gameplay::zone_system(&mut self.world, dt);
        euca_gameplay::zone_dynamic_system(&mut self.world, dt);
        euca_gameplay::status_effect_tick_system(&mut self.world, dt);
        euca_gameplay::stat_resolution_system(&mut self.world);
        euca_gameplay::attribute_update_system(&mut self.world);

        // Core gameplay
        euca_gameplay::apply_damage_system(&mut self.world);
        euca_gameplay::death_check_system(&mut self.world);
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
        euca_gameplay::xp_on_kill_system(&mut self.world);
        euca_gameplay::ability_tick_system(&mut self.world, dt);
        euca_gameplay::use_ability_system(&mut self.world);

        // Navigation
        euca_nav::pathfinding_system(&mut self.world);
        euca_nav::steering_system(&mut self.world, dt);

        // Visibility
        euca_gameplay::visibility_system(&mut self.world);

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
            ui_quads.extend(build_minimap_quads(&self.world, vw, vh));
            if let Some(ui) = self.ui_overlay.as_mut() {
                ui.render(&*gpu, &mut encoder, &view, &ui_quads, vw, vh);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

/// Collect draw commands for all alive renderable entities.
fn collect_draw_commands(world: &World) -> Vec<DrawCommand> {
    let query = Query::<(Entity, &GlobalTransform, &MeshRenderer, &MaterialRef)>::new(world);
    query
        .iter()
        .filter(|(e, _, _, _)| world.get::<euca_gameplay::Dead>(*e).is_none())
        .map(|(_, gt, mr, mat)| DrawCommand {
            mesh: mr.mesh,
            material: mat.handle,
            model_matrix: gt.0.to_matrix(),
            aabb: None,
        })
        .collect()
}

/// Day/night cycle: slowly oscillate lighting over a 4-minute cycle.
/// Also applies subtle Radiant vs Dire color grading based on camera position.
fn day_night_system(world: &mut World, _dt: f32) {
    let elapsed = world
        .resource::<euca_gameplay::GameState>()
        .map(|gs| gs.elapsed)
        .unwrap_or(0.0);

    // 5-minute day/night cycle (300 seconds), starts at full day
    let cycle = (elapsed / 300.0 * std::f32::consts::TAU + std::f32::consts::FRAC_PI_2).sin();
    let day_factor = cycle * 0.5 + 0.5; // 0 (night) to 1 (day), starts at 1.0

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
fn build_health_bar_quads(
    world: &World,
    view_proj: &Mat4,
    viewport_w: f32,
    viewport_h: f32,
) -> Vec<UiQuad> {
    use euca_gameplay::{Dead, EntityRole, Health, Team};

    let mut quads = Vec::new();
    let query = Query::<(Entity, &GlobalTransform, &Health)>::new(world);

    // Find which team the player is on (for color coding)
    let player_team = {
        let pq = Query::<(Entity, &euca_gameplay::player::PlayerHero)>::new(world);
        pq.iter()
            .next()
            .and_then(|(e, _)| world.get::<Team>(e).map(|t| t.0))
            .unwrap_or(1)
    };

    for (entity, gt, health) in query.iter() {
        if world.get::<Dead>(entity).is_some() {
            continue;
        }
        if health.max <= 0.0 {
            continue;
        }

        let world_pos = gt.0.translation;
        // Project world position to clip space
        let clip = *view_proj * euca_math::Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);
        if clip.w <= 0.0 {
            continue; // behind camera
        }

        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;

        // NDC → screen pixels
        let screen_x = (ndc_x + 1.0) * 0.5 * viewport_w;
        let screen_y = (1.0 - ndc_y) * 0.5 * viewport_h; // Y flipped

        // Bar dimensions based on entity role
        let role = world.get::<EntityRole>(entity);
        let (bar_w, bar_h, y_offset) = match role {
            Some(EntityRole::Hero) => (60.0, 8.0, -45.0),
            Some(EntityRole::Tower) | Some(EntityRole::Structure) => (80.0, 6.0, -60.0),
            _ => (40.0, 5.0, -30.0), // minions, neutrals
        };

        let bar_x = screen_x - bar_w * 0.5;
        let bar_y = screen_y + y_offset;

        // Skip if off-screen
        if bar_x + bar_w < 0.0 || bar_x > viewport_w || bar_y + bar_h < 0.0 || bar_y > viewport_h {
            continue;
        }

        let fill = (health.current / health.max).clamp(0.0, 1.0);
        let team = world.get::<Team>(entity).map(|t| t.0).unwrap_or(0);

        // Background (dark)
        quads.push(UiQuad {
            x: bar_x,
            y: bar_y,
            w: bar_w,
            h: bar_h,
            color: [0.1, 0.1, 0.1, 0.7],
        });

        // Fill bar — green for allies, red for enemies, yellow for neutrals
        let fill_color = if team == 0 {
            [0.8, 0.8, 0.0, 0.9] // neutral: yellow
        } else if team == player_team {
            [0.1, 0.9, 0.1, 0.9] // ally: green
        } else {
            [0.9, 0.1, 0.1, 0.9] // enemy: red
        };

        quads.push(UiQuad {
            x: bar_x,
            y: bar_y,
            w: bar_w * fill,
            h: bar_h,
            color: fill_color,
        });
    }

    quads
}

/// Build HUD quads: ability cooldowns, gold, level display.
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

    // ── Ability bar (bottom-center) ──
    let ability_bar_y = viewport_h - 60.0;
    let slot_size = 50.0;
    let gap = 8.0;
    let total_w = slot_size * 4.0 + gap * 3.0;
    let start_x = (viewport_w - total_w) * 0.5;

    let abilities = world.get::<euca_gameplay::AbilitySet>(hero);
    let _slot_labels = ["Q", "W", "E", "R"];

    for i in 0..4u32 {
        let x = start_x + (slot_size + gap) * i as f32;

        // Slot background
        quads.push(UiQuad {
            x,
            y: ability_bar_y,
            w: slot_size,
            h: slot_size,
            color: [0.15, 0.15, 0.2, 0.85],
        });

        // Cooldown overlay (darken when on cooldown)
        if let Some(ability_set) = abilities {
            let slot = match i {
                0 => euca_gameplay::AbilitySlot::Q,
                1 => euca_gameplay::AbilitySlot::W,
                2 => euca_gameplay::AbilitySlot::E,
                _ => euca_gameplay::AbilitySlot::R,
            };
            if let Some(ability) = ability_set.get(slot) {
                if ability.cooldown_remaining > 0.0 {
                    let cd_frac = (ability.cooldown_remaining / ability.cooldown).clamp(0.0, 1.0);
                    quads.push(UiQuad {
                        x,
                        y: ability_bar_y,
                        w: slot_size,
                        h: slot_size * cd_frac,
                        color: [0.0, 0.0, 0.0, 0.6], // dark overlay for cooldown portion
                    });
                } else {
                    // Ready indicator: bright border at bottom
                    quads.push(UiQuad {
                        x,
                        y: ability_bar_y + slot_size - 3.0,
                        w: slot_size,
                        h: 3.0,
                        color: [0.3, 0.8, 1.0, 0.9], // cyan ready indicator
                    });
                }
            }
        }
    }

    // ── Gold display (bottom-left) ──
    let gold = world
        .get::<euca_gameplay::Gold>(hero)
        .map(|g| g.0)
        .unwrap_or(0);
    // Gold bar: width proportional to gold (max display 5000)
    let gold_bar_w = (gold as f32 / 5000.0).clamp(0.0, 1.0) * 150.0;
    quads.push(UiQuad {
        x: 20.0,
        y: viewport_h - 30.0,
        w: 150.0,
        h: 20.0,
        color: [0.15, 0.15, 0.15, 0.7],
    });
    quads.push(UiQuad {
        x: 20.0,
        y: viewport_h - 30.0,
        w: gold_bar_w,
        h: 20.0,
        color: [1.0, 0.84, 0.0, 0.9], // gold color
    });

    // ── Hero health/mana bars (bottom-center, above abilities) ──
    if let Some(health) = world.get::<euca_gameplay::Health>(hero) {
        let bar_w = total_w;
        let bar_h = 10.0;
        let bar_x = start_x;
        let bar_y = ability_bar_y - bar_h - 4.0;
        let fill = (health.current / health.max).clamp(0.0, 1.0);

        // HP background
        quads.push(UiQuad {
            x: bar_x,
            y: bar_y,
            w: bar_w,
            h: bar_h,
            color: [0.15, 0.05, 0.05, 0.8],
        });
        // HP fill
        quads.push(UiQuad {
            x: bar_x,
            y: bar_y,
            w: bar_w * fill,
            h: bar_h,
            color: [0.1, 0.8, 0.1, 0.9],
        });
    }

    if let Some(mana) = world.get::<euca_gameplay::Mana>(hero) {
        let bar_w = total_w;
        let bar_h = 8.0;
        let bar_x = start_x;
        let bar_y = ability_bar_y - 10.0 - bar_h - 8.0;
        let fill = if mana.max > 0.0 {
            (mana.current / mana.max).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Mana background
        quads.push(UiQuad {
            x: bar_x,
            y: bar_y,
            w: bar_w,
            h: bar_h,
            color: [0.05, 0.05, 0.15, 0.8],
        });
        // Mana fill
        quads.push(UiQuad {
            x: bar_x,
            y: bar_y,
            w: bar_w * fill,
            h: bar_h,
            color: [0.2, 0.4, 1.0, 0.9],
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

    // Lane lines on minimap (3 horizontal lines at z=20, 0, -20)
    let world_min_x = -35.0f32;
    let world_max_x = 35.0f32;
    let world_min_z = -25.0f32;
    let world_max_z = 25.0f32;

    let to_minimap = |wx: f32, wz: f32| -> (f32, f32) {
        let u = (wx - world_min_x) / (world_max_x - world_min_x);
        let v = (wz - world_min_z) / (world_max_z - world_min_z);
        (map_x + u * map_size, map_y + (1.0 - v) * map_size)
    };

    // Draw lane paths as thin horizontal rectangles
    for lane_z in [20.0f32, 0.0, -20.0] {
        let (lx, ly) = to_minimap(world_min_x, lane_z);
        let (rx, _) = to_minimap(world_max_x, lane_z);
        quads.push(UiQuad {
            x: lx,
            y: ly - 1.0,
            w: rx - lx,
            h: 2.0,
            color: [0.3, 0.25, 0.15, 0.6], // dirt-colored lane
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

        // Upload meshes, materials, and create the ground plane + light
        setup_default_assets(
            &mut self.world,
            self.gpu.as_ref().unwrap(),
            self.renderer.as_mut().unwrap(),
        );

        if !self.level_loaded {
            self.load_level();
            self.level_loaded = true;
        }
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
