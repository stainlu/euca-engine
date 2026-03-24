mod commands;

use clap::{Parser, Subcommand};
use serde_json::Value;

use commands::asset::AssetCommands;
use commands::entity::EntityCommands;

#[derive(Parser)]
#[command(
    name = "euca",
    about = "Euca Engine CLI — control the engine from the terminal",
    after_help = "Examples:\n  euca entity create --mesh cube --position 0,2,0 --health 100 --team 1 --color red\n  euca rule create --when death --filter team:2 --do-action \"score source +1\"\n  euca sim play\n  euca screenshot"
)]
// `pub(crate)` so that `commands::discover` can call `Cli::command()` via `CommandFactory`.
pub(crate) struct Cli {
    /// Server URL
    #[arg(short, long, default_value = "http://localhost:3917")]
    server: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Entity operations: create, get, list, update, delete
    Entity {
        #[command(subcommand)]
        command: EntityCommands,
    },

    /// Simulation control: play, pause, step, reset
    Sim {
        #[command(subcommand)]
        command: SimCommands,
    },

    /// Scene management: save, load
    Scene {
        #[command(subcommand)]
        command: SceneCommands,
    },

    /// Camera control: get, set
    Camera {
        #[command(subcommand)]
        command: CameraCommands,
    },

    /// Game match: create, state, scoreboard
    Game {
        #[command(subcommand)]
        command: GameCommands,
    },

    /// Trigger zones: create area-based events
    Trigger {
        #[command(subcommand)]
        command: TriggerCommands,
    },

    /// Projectiles: spawn moving damaging entities
    Projectile {
        #[command(subcommand)]
        command: ProjectileCommands,
    },

    /// AI behavior: set entity AI goals
    Ai {
        #[command(subcommand)]
        command: AiCommands,
    },

    /// Game rules: when X happens, do Y
    Rule {
        #[command(subcommand)]
        command: RuleCommands,
    },

    /// Entity templates: define once, spawn many
    Template {
        #[command(subcommand)]
        command: TemplateCommands,
    },

    /// Abilities: use and list hero abilities
    Ability {
        #[command(subcommand)]
        command: AbilityCommands,
    },

    /// Status effects: apply, list, cleanse modifiers on entities
    Effect {
        #[command(subcommand)]
        command: EffectCommands,
    },

    /// Inventory & items: define, give, equip, list
    Item {
        #[command(subcommand)]
        command: ItemCommands,
    },

    /// Audio: play, stop, list sounds
    Audio {
        #[command(subcommand)]
        command: AudioCommands,
    },

    /// Input: key bindings and input contexts
    Input {
        #[command(subcommand)]
        command: InputCommands,
    },

    /// Navigation: navmesh + pathfinding
    Nav {
        #[command(subcommand)]
        command: NavCommands,
    },

    /// Visual effects: particle emitters
    Vfx {
        #[command(subcommand)]
        command: VfxCommands,
    },

    /// Animation: load glTF models, play/stop skeletal animation
    Animation {
        #[command(subcommand)]
        command: AnimationCommands,
    },

    /// Terrain: create and edit heightmap terrain
    Terrain {
        #[command(subcommand)]
        command: TerrainCommands,
    },
    /// Foliage: scatter instanced vegetation
    Foliage {
        #[command(subcommand)]
        command: FoliageCommands,
    },
    /// Prefab: spawn registered prefabs
    Prefab {
        #[command(subcommand)]
        command: PrefabCommands,
    },
    /// Material: set material properties on entities
    Material {
        #[command(subcommand)]
        command: MaterialCommands,
    },
    /// Post-processing: SSAO, FXAA, bloom, color grading
    Postprocess {
        #[command(subcommand)]
        command: PostprocessCommands,
    },
    /// Volumetric fog: density, scattering, god rays
    Fog {
        #[command(subcommand)]
        command: FogCommands,
    },
    /// Authentication via nit identity
    Auth {
        #[command(subcommand)]
        command: AuthCommands,
    },

    /// Show frame profiler: FPS, frame time, per-section timings
    Profile,

    /// Diagnose engine health — find broken entities
    Diagnose,

    /// Show pending events (damage, death, spawn)
    Events,

    /// Engine status
    Status,

    /// Capture a screenshot of the 3D viewport
    Screenshot {
        /// Output file path (default: temp file)
        #[arg(short, long)]
        output: Option<String>,
    },

    /// HUD elements: text, bars, rectangles on screen
    Ui {
        #[command(subcommand)]
        command: UiCommands,
    },

    /// Inspect component schemas
    Schema {
        /// Component name (e.g. "Collider", "PhysicsBody"). Omit to list all.
        component: Option<String>,
    },

    /// Package game for distribution (build + bundle assets)
    Package {
        /// Path to project directory (containing .eucaproject.json)
        #[arg(short, long, default_value = ".")]
        project: String,
        /// Output directory for the packaged game
        #[arg(short, long, default_value = "dist")]
        output: String,
    },

    /// Asset pipeline: info, optimize, LOD generation (offline, no engine needed)
    Asset {
        #[command(subcommand)]
        command: AssetCommands,
    },

    /// Lua scripting: load scripts, list scripted entities
    Script {
        #[command(subcommand)]
        command: ScriptCommands,
    },

    /// Networking: status, connected peers, tick rate
    Net {
        #[command(subcommand)]
        command: NetCommands,
    },

    /// Discover available commands (for AI agents and humans, works offline)
    Discover {
        /// Output as machine-readable JSON
        #[arg(long)]
        json: bool,
        /// Filter by group name (e.g., "entity", "asset")
        group: Option<String>,
    },

    // ── Hidden aliases for backward compatibility ──
    #[command(hide = true)]
    Observe {
        #[arg(short, long)]
        entity: Option<u32>,
    },
    #[command(hide = true)]
    Spawn {
        #[arg(short, long)]
        position: Option<String>,
        #[arg(long)]
        scale: Option<String>,
        #[arg(long)]
        physics: Option<String>,
        #[arg(long)]
        collider: Option<String>,
    },
    #[command(hide = true)]
    Modify {
        id: u32,
        #[arg(short, long)]
        transform: Option<String>,
        #[arg(long)]
        velocity: Option<String>,
        #[arg(long)]
        physics: Option<String>,
        #[arg(long)]
        collider: Option<String>,
        #[arg(long)]
        json: Option<String>,
    },
    #[command(hide = true)]
    Despawn {
        #[arg(default_value = "0")]
        id: u32,
        #[arg(long)]
        all: bool,
    },
    #[command(hide = true)]
    Play,
    #[command(hide = true)]
    Pause,
    #[command(hide = true)]
    Step {
        #[arg(short, long, default_value = "1")]
        ticks: u64,
    },
    #[command(hide = true)]
    Reset,
}

#[derive(Subcommand)]
enum SimCommands {
    /// Start simulation
    Play,
    /// Pause simulation
    Pause,
    /// Advance simulation by N ticks
    Step {
        /// Number of ticks
        #[arg(short, long, default_value = "1")]
        ticks: u64,
    },
    /// Reset to initial scene
    Reset,
}

#[derive(Subcommand)]
enum SceneCommands {
    /// Save current scene to file
    Save {
        /// Output file path
        path: String,
    },
    /// Load scene from file
    Load {
        /// Input file path
        path: String,
    },
}

#[derive(Subcommand)]
enum CameraCommands {
    /// Get current camera state
    Get,
    /// Set camera position and look-at target
    Set {
        /// Camera position as "x,y,z"
        #[arg(long)]
        eye: Option<String>,
        /// Look-at target as "x,y,z"
        #[arg(long)]
        target: Option<String>,
        /// Field of view in degrees (perspective mode)
        #[arg(long)]
        fov: Option<f32>,
    },
    /// Switch to a preset view: top, front, back, right, left, perspective
    View {
        /// View name
        name: String,
    },
    /// Focus camera on a specific entity
    Focus {
        /// Entity ID to focus on
        id: u32,
    },
}

#[derive(Subcommand)]
enum AbilityCommands {
    /// Use an ability (Q/W/E/R)
    Use {
        /// Entity ID
        entity_id: u32,
        /// Ability slot: Q, W, E, or R
        #[arg(long, default_value = "Q")]
        slot: String,
    },
    /// List an entity's abilities, mana, gold, level
    List {
        /// Entity ID
        entity_id: u32,
    },
}

#[derive(Subcommand)]
enum EffectCommands {
    /// Apply a status effect to an entity
    Apply {
        /// Entity ID
        entity_id: u32,
        /// Effect tag (e.g. "stun", "poison", "buff_speed")
        #[arg(long)]
        tag: String,
        /// Stat modifiers as "stat:op:value" (repeatable). op = set|add|multiply
        #[arg(long)]
        modifier: Vec<String>,
        /// Duration in seconds
        #[arg(long, default_value = "5.0")]
        duration: f32,
        /// Stack policy: "replace" or "stack:N"
        #[arg(long, default_value = "replace")]
        stack: String,
        /// Tick effect: "dps:N", "hps:N", or "custom:name"
        #[arg(long)]
        tick: Option<String>,
        /// Source entity ID (for attribution)
        #[arg(long)]
        source: Option<u32>,
    },
    /// List active status effects on an entity
    List {
        /// Entity ID
        entity_id: u32,
    },
    /// Remove effects matching a tag filter
    Cleanse {
        /// Entity ID
        entity_id: u32,
        /// Tag substring to match (e.g. "debuff" removes all "debuff_*" effects)
        #[arg(long)]
        filter: String,
    },
}

#[derive(Subcommand)]
enum ItemCommands {
    /// Define a new item type
    Define {
        /// Unique item ID
        #[arg(long)]
        id: u32,
        /// Item name
        #[arg(long)]
        name: String,
        /// Properties as "key:value" (repeatable, e.g. --prop damage:50 --prop speed:-5)
        #[arg(long)]
        prop: Vec<String>,
    },
    /// Give items to an entity
    Give {
        /// Entity ID
        entity_id: u32,
        /// Item ID
        item_id: u32,
        /// Count (default 1)
        #[arg(default_value = "1")]
        count: u32,
    },
    /// Equip an item from inventory into a named slot
    Equip {
        /// Entity ID
        entity_id: u32,
        /// Slot name (e.g. "weapon", "head")
        slot: String,
        /// Item ID
        item_id: u32,
    },
    /// List entity's inventory, equipment, and stat modifiers
    List {
        /// Entity ID
        entity_id: u32,
    },
}

#[derive(Subcommand)]
enum AudioCommands {
    /// Play a sound file
    Play {
        /// Path to audio file (WAV, MP3, OGG, FLAC)
        path: String,
        /// Position as "x,y,z" (makes it spatial)
        #[arg(long)]
        position: Option<String>,
        /// Volume (0.0-1.0)
        #[arg(long, default_value = "1.0")]
        volume: f32,
        /// Loop the sound
        #[arg(long, name = "loop")]
        looping: bool,
        /// Max audible distance (spatial only)
        #[arg(long, default_value = "50")]
        max_distance: f32,
    },
    /// Stop an audio source
    Stop {
        /// Entity ID of the audio source
        entity_id: u32,
    },
    /// List active audio sources
    List,
}

#[derive(Subcommand)]
enum InputCommands {
    /// Bind a key to an action
    Bind {
        /// Key name (e.g., "W", "Space", "MouseLeft")
        key: String,
        /// Action name (e.g., "move_forward", "jump")
        action: String,
    },
    /// Remove a key binding
    Unbind {
        /// Key name
        key: String,
    },
    /// List all current bindings
    List,
    /// Push an input context
    ContextPush {
        /// Context name: gameplay, menu, editor
        context: String,
    },
    /// Pop the current input context
    ContextPop,
}

#[derive(Subcommand)]
enum NavCommands {
    /// Generate navmesh from scene colliders
    Generate {
        /// Cell size
        #[arg(long, default_value = "1.0")]
        cell_size: f32,
    },
    /// Compute A* path between two points
    Compute {
        /// Start position as "x,y,z"
        #[arg(long)]
        from: String,
        /// Goal position as "x,y,z"
        #[arg(long)]
        to: String,
    },
    /// Set pathfinding goal on an entity
    Set {
        /// Entity ID
        entity_id: u32,
        /// Target position as "x,y,z"
        #[arg(long)]
        target: String,
        /// Movement speed
        #[arg(long, default_value = "5.0")]
        speed: f32,
    },
}

#[derive(Subcommand)]
enum AnimationCommands {
    /// Load a glTF file with animations and skeleton
    Load {
        /// Path to glTF/glb file
        path: String,
    },
    /// Play an animation clip on an entity
    Play {
        /// Entity ID
        entity_id: u32,
        /// Clip index (from animation list)
        #[arg(long, default_value = "0")]
        clip: usize,
        /// Playback speed
        #[arg(long, default_value = "1.0")]
        speed: f32,
        /// Loop the animation
        #[arg(long, name = "loop")]
        looping: bool,
    },
    /// Stop animation on an entity
    Stop {
        /// Entity ID
        entity_id: u32,
    },
    /// List loaded animation clips
    List,
    /// Create an animation state machine on an entity
    StateMachine {
        /// Entity ID
        entity_id: u32,
        /// Initial state index
        #[arg(long, default_value = "0")]
        initial_state: usize,
        /// States as "name:clip[:speed[:loop]]" (repeatable)
        #[arg(long)]
        state: Vec<String>,
    },
    /// Play an animation montage on an entity
    Montage {
        /// Entity ID
        entity_id: u32,
        /// Clip index
        #[arg(long)]
        clip: usize,
        /// Clip duration in seconds
        #[arg(long)]
        clip_duration: f32,
        /// Playback speed
        #[arg(long, default_value = "1.0")]
        speed: f32,
        /// Blend-in duration (seconds)
        #[arg(long, default_value = "0.1")]
        blend_in: f32,
        /// Blend-out duration (seconds)
        #[arg(long, default_value = "0.1")]
        blend_out: f32,
        /// Bone mask indices as "0,1,2,..."
        #[arg(long)]
        bone_mask: Option<String>,
    },
}

#[derive(Subcommand)]
enum VfxCommands {
    /// Spawn a particle emitter at a position
    Spawn {
        /// Position as "x,y,z"
        #[arg(long)]
        position: Option<String>,
        /// Emission rate (particles/second)
        #[arg(long, default_value = "50")]
        rate: f32,
        /// Particle lifetime (seconds)
        #[arg(long, default_value = "2.0")]
        lifetime: f32,
        /// Max particles alive at once
        #[arg(long, default_value = "1000")]
        max: u32,
    },
    /// Stop a particle emitter
    Stop {
        /// Entity ID of the emitter
        entity_id: u32,
    },
    /// List active particle emitters
    List,
}

#[derive(Subcommand)]
enum AuthCommands {
    /// Login with nit Ed25519 identity
    Login,
    /// Check current authentication status
    Status,
}

#[derive(Subcommand)]
enum GameCommands {
    /// Create a new match
    Create {
        /// Game mode (e.g. "deathmatch")
        #[arg(long, default_value = "deathmatch")]
        mode: String,
        /// Score required to win
        #[arg(long, default_value = "10")]
        score_limit: i32,
    },
    /// Get current match state and scores
    State,
}

#[derive(Subcommand)]
enum TriggerCommands {
    /// Create a trigger zone
    Create {
        /// Position as "x,y,z"
        #[arg(long)]
        position: String,
        /// Zone half-extents as "x,y,z"
        #[arg(long, default_value = "1,1,1")]
        zone: String,
        /// Action: "damage:N" or "heal:N"
        #[arg(long, default_value = "damage:10")]
        action: String,
    },
}

#[derive(Subcommand)]
enum ProjectileCommands {
    /// Spawn a projectile
    Spawn {
        /// Origin position as "x,y,z"
        #[arg(long)]
        from: String,
        /// Direction as "x,y,z"
        #[arg(long)]
        direction: String,
        /// Speed (units/sec)
        #[arg(long, default_value = "20")]
        speed: f32,
        /// Damage on hit
        #[arg(long, default_value = "25")]
        damage: f32,
    },
}

#[derive(Subcommand)]
enum AiCommands {
    /// Set AI behavior on an entity
    Set {
        /// Entity ID
        id: u32,
        /// Behavior: idle, patrol, chase, flee
        #[arg(long)]
        behavior: String,
        /// Target entity ID (for chase/flee)
        #[arg(long)]
        target: Option<u32>,
        /// Movement speed
        #[arg(long, default_value = "3")]
        speed: f32,
    },
}

#[derive(Subcommand)]
enum RuleCommands {
    /// Create a game rule: when condition fires, execute actions
    Create {
        /// Condition: "death", "timer:N", "health-below:N"
        #[arg(long)]
        when: String,
        /// Filter: "any", "entity:N", "team:N"
        #[arg(long, default_value = "any")]
        filter: String,
        /// Actions (can repeat): "spawn cube 0,5,0", "score source +1", "damage this 10"
        #[arg(long)]
        do_action: Vec<String>,
    },
    /// List all rules
    List,
}

#[derive(Subcommand)]
// clippy::large_enum_variant — `Create` carries all template fields for clap
// parsing; this enum is constructed once per CLI invocation and boxing would
// break the `#[derive(Subcommand)]` derive.
#[allow(clippy::large_enum_variant)]
enum TemplateCommands {
    /// Define a named entity template
    Create {
        /// Template name
        name: String,
        /// Mesh
        #[arg(long)]
        mesh: Option<String>,
        /// Color
        #[arg(long)]
        color: Option<String>,
        /// Health
        #[arg(long)]
        health: Option<f32>,
        /// Team
        #[arg(long)]
        team: Option<u8>,
        /// Physics body type
        #[arg(long)]
        physics: Option<String>,
        /// Collider
        #[arg(long)]
        collider: Option<String>,
        /// Enable auto-combat
        #[arg(long)]
        combat: bool,
        /// Combat damage
        #[arg(long)]
        combat_damage: Option<f32>,
        /// Combat range
        #[arg(long)]
        combat_range: Option<f32>,
        /// Combat speed
        #[arg(long)]
        combat_speed: Option<f32>,
        /// Combat cooldown
        #[arg(long)]
        combat_cooldown: Option<f32>,
        /// Combat style: melee or stationary
        #[arg(long)]
        combat_style: Option<String>,
        /// AI patrol waypoints
        #[arg(long)]
        ai_patrol: Option<String>,
        /// Starting gold
        #[arg(long)]
        gold: Option<i32>,
        /// Gold bounty
        #[arg(long)]
        gold_bounty: Option<i32>,
        /// XP bounty
        #[arg(long)]
        xp_bounty: Option<u32>,
        /// Entity role
        #[arg(long)]
        role: Option<String>,
        /// Spawn point for team
        #[arg(long)]
        spawn_point: Option<u8>,
    },
    /// Spawn an entity from a template
    Spawn {
        /// Template name
        name: String,
        /// Position
        #[arg(long)]
        position: Option<String>,
    },
    /// List all templates
    List,
}

#[derive(Subcommand)]
enum UiCommands {
    /// Add text to HUD
    Text {
        /// Text content
        text: String,
        /// X position (0.0-1.0, left to right)
        #[arg(long, default_value = "0.5")]
        x: f32,
        /// Y position (0.0-1.0, top to bottom)
        #[arg(long, default_value = "0.05")]
        y: f32,
        /// Font size in pixels
        #[arg(long, default_value = "20")]
        size: f32,
        /// Color name (red, green, blue, white, yellow, etc.)
        #[arg(long, default_value = "white")]
        color: String,
    },
    /// Add a bar (health bar, progress bar) to HUD
    Bar {
        /// X position (0.0-1.0)
        #[arg(long, default_value = "0.02")]
        x: f32,
        /// Y position (0.0-1.0)
        #[arg(long, default_value = "0.95")]
        y: f32,
        /// Width (0.0-1.0)
        #[arg(long, default_value = "0.2")]
        width: f32,
        /// Height (0.0-1.0)
        #[arg(long, default_value = "0.03")]
        height: f32,
        /// Fill amount (0.0-1.0)
        #[arg(long)]
        fill: f32,
        /// Bar color
        #[arg(long, default_value = "red")]
        color: String,
    },
    /// Remove all HUD elements
    Clear,
    /// List current HUD elements
    List,
}

#[derive(Subcommand)]
enum TerrainCommands {
    Create {
        #[arg(long, default_value = "64")]
        width: u32,
        #[arg(long, default_value = "64")]
        height: u32,
        #[arg(long, default_value = "1.0")]
        cell_size: f32,
    },
    Edit {
        #[arg(long, default_value = "raise")]
        op: String,
        #[arg(long)]
        x: f32,
        #[arg(long)]
        z: f32,
        #[arg(long, default_value = "3")]
        radius: f32,
        #[arg(long, default_value = "0.5")]
        amount: f32,
    },
}
#[derive(Subcommand)]
enum FoliageCommands {
    /// Scatter foliage instances across an area
    Scatter {
        /// Target density (instances per square unit)
        #[arg(long, default_value = "0.5")]
        density: f32,
        /// Scatter area as "min_x,min_z,max_x,max_z"
        #[arg(long, default_value = "-20,-20,20,20")]
        area: String,
        /// Mesh name: cube or sphere
        #[arg(long, default_value = "cube")]
        mesh: String,
        /// Minimum scale factor
        #[arg(long, default_value = "0.8")]
        min_scale: f32,
        /// Maximum scale factor
        #[arg(long, default_value = "1.2")]
        max_scale: f32,
        /// Maximum render distance
        #[arg(long, default_value = "100")]
        max_distance: f32,
    },
    /// List all foliage layers
    List,
}

#[derive(Subcommand)]
enum PrefabCommands {
    Spawn {
        #[arg(long)]
        name: String,
        #[arg(long)]
        position: Option<String>,
    },
    List,
}
#[derive(Subcommand)]
enum MaterialCommands {
    Set {
        #[arg(long)]
        entity: u32,
        #[arg(long)]
        emissive: Option<String>,
        #[arg(long)]
        alpha_mode: Option<String>,
        #[arg(long)]
        metallic: Option<f32>,
        #[arg(long)]
        roughness: Option<f32>,
    },
}
#[derive(Subcommand)]
enum PostprocessCommands {
    Get,
    Set {
        #[arg(long)]
        ssao: Option<bool>,
        #[arg(long)]
        fxaa: Option<bool>,
        #[arg(long)]
        exposure: Option<f32>,
        #[arg(long)]
        bloom: Option<bool>,
        #[arg(long)]
        contrast: Option<f32>,
        #[arg(long)]
        saturation: Option<f32>,
    },
    /// Apply a named quality preset (low, medium, high, ultra)
    Preset {
        /// Quality level: low, medium, high, ultra
        #[arg(value_parser = ["low", "medium", "high", "ultra"])]
        quality: String,
    },
}

#[derive(Subcommand)]
enum FogCommands {
    /// Get current fog settings
    Get,
    /// Set fog parameters
    Set {
        /// Base fog density (higher = thicker fog)
        #[arg(long)]
        density: Option<f32>,
        /// Scattering coefficient (light redirected toward camera)
        #[arg(long)]
        scattering: Option<f32>,
        /// Absorption coefficient (light absorbed by fog)
        #[arg(long)]
        absorption: Option<f32>,
        /// Rate of density decrease with height
        #[arg(long)]
        height_falloff: Option<f32>,
        /// Maximum ray-march distance
        #[arg(long)]
        max_distance: Option<f32>,
        /// God-ray strength (scales light contribution)
        #[arg(long)]
        light_contribution: Option<f32>,
        /// Enable or disable fog
        #[arg(long)]
        enabled: Option<bool>,
    },
}

#[derive(Subcommand)]
enum ScriptCommands {
    /// Attach a Lua script to an entity
    Load {
        /// Entity ID
        entity_id: u32,
        /// Path to .lua script file
        path: String,
    },
    /// List entities with scripts attached
    List,
}

#[derive(Subcommand)]
enum NetCommands {
    /// Show networking state (connected peers, bandwidth, tick rate)
    Status,
}

// ── Helpers ──

pub(crate) fn parse_vec3(s: &str) -> Option<[f32; 3]> {
    let parts: Vec<f32> = s.split(',').filter_map(|p| p.trim().parse().ok()).collect();
    if parts.len() == 3 {
        Some([parts[0], parts[1], parts[2]])
    } else {
        log::warn!("Expected 'x,y,z' format, got '{s}'");
        None
    }
}

fn parse_collider(s: &str) -> Option<Value> {
    if let Some(rest) = s.strip_prefix("sphere:") {
        let radius: f32 = rest.trim().parse().ok()?;
        Some(serde_json::json!({"shape": "Sphere", "radius": radius}))
    } else if let Some(rest) = s.strip_prefix("aabb:") {
        let v = parse_vec3(rest)?;
        Some(serde_json::json!({"shape": "Aabb", "hx": v[0], "hy": v[1], "hz": v[2]}))
    } else if let Some(rest) = s.strip_prefix("capsule:") {
        let parts: Vec<f32> = rest
            .split(',')
            .filter_map(|p| p.trim().parse().ok())
            .collect();
        if parts.len() == 2 {
            Some(
                serde_json::json!({"shape": "Capsule", "radius": parts[0], "half_height": parts[1]}),
            )
        } else {
            log::warn!("Capsule format is 'capsule:radius,half_height'");
            None
        }
    } else {
        log::warn!(
            "Collider format is 'aabb:hx,hy,hz' or 'sphere:radius' or 'capsule:radius,half_height'"
        );
        None
    }
}

/// Build a spawn/create JSON body from friendly flags.
// clippy::too_many_arguments — mirrors the CLI flags 1:1; a wrapper struct
// would duplicate the clap-parsed fields without reducing complexity.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_create_body(
    mesh: &Option<String>,
    color: &Option<String>,
    position: &Option<String>,
    scale: &Option<String>,
    physics: &Option<String>,
    collider: &Option<String>,
    health: Option<f32>,
    team: Option<u8>,
    combat: bool,
    combat_damage: Option<f32>,
    combat_range: Option<f32>,
    combat_speed: Option<f32>,
    combat_cooldown: Option<f32>,
    combat_style: &Option<String>,
    ai_patrol: &Option<String>,
    gold: Option<i32>,
    gold_bounty: Option<i32>,
    xp_bounty: Option<u32>,
    role: &Option<String>,
    spawn_point: Option<u8>,
    player: bool,
) -> Value {
    let mut body = serde_json::json!({});
    if let Some(m) = mesh {
        body["mesh"] = serde_json::json!(m);
    }
    if let Some(c) = color {
        body["color"] = serde_json::json!(c);
    }
    if let Some(s) = position
        && let Some(v) = parse_vec3(s)
    {
        body["position"] = serde_json::json!(v);
    }
    if let Some(s) = scale
        && let Some(v) = parse_vec3(s)
    {
        body["scale"] = serde_json::json!(v);
    }
    if let Some(pb) = physics {
        body["physics_body"] = serde_json::json!(pb);
    }
    if let Some(c) = collider
        && let Some(v) = parse_collider(c)
    {
        body["collider"] = v;
    }
    if let Some(h) = health {
        body["health"] = serde_json::json!(h);
    }
    if let Some(t) = team {
        body["team"] = serde_json::json!(t);
    }
    if combat {
        body["combat"] = serde_json::json!(true);
    }
    if let Some(d) = combat_damage {
        body["combat_damage"] = serde_json::json!(d);
    }
    if let Some(r) = combat_range {
        body["combat_range"] = serde_json::json!(r);
    }
    if let Some(s) = combat_speed {
        body["combat_speed"] = serde_json::json!(s);
    }
    if let Some(c) = combat_cooldown {
        body["combat_cooldown"] = serde_json::json!(c);
    }
    if let Some(s) = combat_style {
        body["combat_style"] = serde_json::json!(s);
    }
    if let Some(patrol_str) = ai_patrol {
        // Parse "x,y,z:x,y,z:x,y,z" into [[x,y,z],[x,y,z]]
        let waypoints: Vec<Vec<f32>> = patrol_str
            .split(':')
            .filter_map(|wp| {
                let parts: Vec<f32> = wp
                    .split(',')
                    .filter_map(|p| p.trim().parse().ok())
                    .collect();
                if parts.len() == 3 { Some(parts) } else { None }
            })
            .collect();
        if !waypoints.is_empty() {
            body["ai_patrol"] = serde_json::json!(waypoints);
        }
    }
    if let Some(g) = gold {
        body["gold"] = serde_json::json!(g);
    }
    if let Some(b) = gold_bounty {
        body["gold_bounty"] = serde_json::json!(b);
    }
    if let Some(xp) = xp_bounty {
        body["xp_bounty"] = serde_json::json!(xp);
    }
    if let Some(r) = role {
        body["role"] = serde_json::json!(r);
    }
    if let Some(sp) = spawn_point {
        body["spawn_point"] = serde_json::json!(sp);
    }
    if player {
        body["player"] = serde_json::json!(true);
    }
    body
}

/// Build an update/patch JSON body from friendly flags.
pub(crate) fn build_update_body(
    color: &Option<String>,
    position: &Option<String>,
    scale: &Option<String>,
    velocity: &Option<String>,
    physics: &Option<String>,
    collider: &Option<String>,
) -> Value {
    let mut body = serde_json::json!({});
    if let Some(c) = color {
        body["color"] = serde_json::json!(c);
    }
    if position.is_some() || scale.is_some() {
        let mut transform = serde_json::json!({});
        if let Some(p) = position
            && let Some(pos) = parse_vec3(p)
        {
            transform["position"] = serde_json::json!(pos);
        }
        if let Some(s) = scale
            && let Some(scl) = parse_vec3(s)
        {
            transform["scale"] = serde_json::json!(scl);
        }
        body["transform"] = transform;
    }
    if let Some(v) = velocity
        && let Some(vel) = parse_vec3(v)
    {
        body["velocity"] = serde_json::json!({"linear": vel, "angular": [0, 0, 0]});
    }
    if let Some(pb) = physics {
        body["physics_body"] = serde_json::json!(pb);
    }
    if let Some(c) = collider
        && let Some(v) = parse_collider(c)
    {
        body["collider"] = v;
    }
    body
}

/// Parse --json flag or exit.
pub(crate) fn parse_json_flag(raw: &str) -> Value {
    match serde_json::from_str::<Value>(raw) {
        Ok(v) => v,
        Err(e) => {
            log::error!("Invalid JSON: {e}");
            std::process::exit(1);
        }
    }
}

// ── Execution ──

fn main() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Warn)
        .parse_default_env()
        .init();

    let cli = Cli::parse();
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .build()
        .expect("Failed to build HTTP client");
    let server = &cli.server;

    let result = match cli.command {
        // ── Entity CRUD ──
        Commands::Entity { command } => commands::entity::run_entity(command, &client, server),

        // ── Simulation ──
        Commands::Sim { command } => match command {
            SimCommands::Play => post_empty(&client, server, "/play"),
            SimCommands::Pause => post_empty(&client, server, "/pause"),
            SimCommands::Step { ticks } => {
                let resp = client
                    .post(format!("{server}/step"))
                    .json(&serde_json::json!({"ticks": ticks}))
                    .send();
                handle_response(resp)
            }
            SimCommands::Reset => post_empty(&client, server, "/reset"),
        },

        // ── Scene ──
        Commands::Scene { command } => match command {
            SceneCommands::Save { path } => {
                let resp = client
                    .post(format!("{server}/scene/save"))
                    .json(&serde_json::json!({"path": path}))
                    .send();
                handle_response(resp)
            }
            SceneCommands::Load { path } => {
                let resp = client
                    .post(format!("{server}/scene/load"))
                    .json(&serde_json::json!({"path": path}))
                    .send();
                handle_response(resp)
            }
        },

        // ── Camera ──
        Commands::Camera { command } => match command {
            CameraCommands::Get => {
                let resp = client.get(format!("{server}/camera")).send();
                handle_response(resp)
            }
            CameraCommands::Set { eye, target, fov } => {
                let mut body = serde_json::json!({});
                if let Some(e) = eye
                    && let Some(v) = parse_vec3(&e)
                {
                    body["eye"] = serde_json::json!(v);
                }
                if let Some(t) = target
                    && let Some(v) = parse_vec3(&t)
                {
                    body["target"] = serde_json::json!(v);
                }
                if let Some(f) = fov {
                    body["fov"] = serde_json::json!(f);
                }
                let resp = client.post(format!("{server}/camera")).json(&body).send();
                handle_response(resp)
            }
            CameraCommands::View { name } => {
                let resp = client
                    .post(format!("{server}/camera/view"))
                    .json(&serde_json::json!({"view": name}))
                    .send();
                handle_response(resp)
            }
            CameraCommands::Focus { id } => {
                let resp = client
                    .post(format!("{server}/camera/focus"))
                    .json(&serde_json::json!({"entity_id": id}))
                    .send();
                handle_response(resp)
            }
        },

        // ── Gameplay ──
        Commands::Game { command } => match command {
            GameCommands::Create { mode, score_limit } => {
                let resp = client
                    .post(format!("{server}/game/create"))
                    .json(&serde_json::json!({"mode": mode, "score_limit": score_limit}))
                    .send();
                handle_response(resp)
            }
            GameCommands::State => {
                let resp = client.get(format!("{server}/game/state")).send();
                handle_response(resp)
            }
        },
        Commands::Trigger { command } => match command {
            TriggerCommands::Create {
                position,
                zone,
                action,
            } => {
                let pos = parse_vec3(&position).unwrap_or([0.0, 0.0, 0.0]);
                let z = parse_vec3(&zone).unwrap_or([1.0, 1.0, 1.0]);
                let resp = client
                    .post(format!("{server}/trigger/create"))
                    .json(&serde_json::json!({
                        "position": pos,
                        "zone": z,
                        "action": action,
                    }))
                    .send();
                handle_response(resp)
            }
        },
        Commands::Projectile { command } => match command {
            ProjectileCommands::Spawn {
                from,
                direction,
                speed,
                damage,
            } => {
                let f = parse_vec3(&from).unwrap_or([0.0, 0.0, 0.0]);
                let d = parse_vec3(&direction).unwrap_or([1.0, 0.0, 0.0]);
                let resp = client
                    .post(format!("{server}/projectile/spawn"))
                    .json(&serde_json::json!({
                        "from": f,
                        "direction": d,
                        "speed": speed,
                        "damage": damage,
                    }))
                    .send();
                handle_response(resp)
            }
        },
        Commands::Ai { command } => match command {
            AiCommands::Set {
                id,
                behavior,
                target,
                speed,
            } => {
                let mut body = serde_json::json!({
                    "entity_id": id,
                    "behavior": behavior,
                    "speed": speed,
                });
                if let Some(t) = target {
                    body["target"] = serde_json::json!(t);
                }
                let resp = client.post(format!("{server}/ai/set")).json(&body).send();
                handle_response(resp)
            }
        },

        // ── Templates ──
        Commands::Template { command } => match command {
            TemplateCommands::Create {
                name,
                mesh,
                color,
                health,
                team,
                physics,
                collider,
                combat,
                combat_damage,
                combat_range,
                combat_speed,
                combat_cooldown,
                combat_style,
                ai_patrol,
                gold,
                gold_bounty,
                xp_bounty,
                role,
                spawn_point,
            } => {
                let mut body = build_create_body(
                    &mesh,
                    &color,
                    &None,
                    &None,
                    &physics,
                    &collider,
                    health,
                    team,
                    combat,
                    combat_damage,
                    combat_range,
                    combat_speed,
                    combat_cooldown,
                    &combat_style,
                    &ai_patrol,
                    gold,
                    gold_bounty,
                    xp_bounty,
                    &role,
                    spawn_point,
                    false,
                );
                body["name"] = serde_json::json!(name);
                let resp = client
                    .post(format!("{server}/template/create"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
            TemplateCommands::Spawn { name, position } => {
                let mut body = serde_json::json!({"name": name});
                if let Some(p) = position
                    && let Some(v) = parse_vec3(&p)
                {
                    body["position"] = serde_json::json!(v);
                }
                let resp = client
                    .post(format!("{server}/template/spawn"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
            TemplateCommands::List => {
                let resp = client.get(format!("{server}/template/list")).send();
                handle_response(resp)
            }
        },

        // ── Rules ──
        Commands::Rule { command } => match command {
            RuleCommands::Create {
                when,
                filter,
                do_action,
            } => {
                let resp = client
                    .post(format!("{server}/rule/create"))
                    .json(&serde_json::json!({
                        "when": when,
                        "filter": filter,
                        "actions": do_action,
                    }))
                    .send();
                handle_response(resp)
            }
            RuleCommands::List => {
                let resp = client.get(format!("{server}/rule/list")).send();
                handle_response(resp)
            }
        },

        // ── Abilities ──
        Commands::Ability { command } => match command {
            AbilityCommands::Use { entity_id, slot } => {
                let resp = client
                    .post(format!("{server}/ability/use"))
                    .json(&serde_json::json!({"entity_id": entity_id, "slot": slot}))
                    .send();
                handle_response(resp)
            }
            AbilityCommands::List { entity_id } => {
                let resp = client
                    .get(format!("{server}/ability/list/{entity_id}"))
                    .send();
                handle_response(resp)
            }
        },

        // ── Status Effects ──
        Commands::Effect { command } => match command {
            EffectCommands::Apply {
                entity_id,
                tag,
                modifier,
                duration,
                stack,
                tick,
                source,
            } => {
                let mut body = serde_json::json!({
                    "entity_id": entity_id,
                    "tag": tag,
                    "duration": duration,
                    "modifiers": modifier,
                    "stack_policy": stack,
                });
                if let Some(t) = tick {
                    body["tick_effect"] = serde_json::json!(t);
                }
                if let Some(s) = source {
                    body["source"] = serde_json::json!(s);
                }
                let resp = client
                    .post(format!("{server}/effect/apply"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
            EffectCommands::List { entity_id } => {
                let resp = client
                    .get(format!("{server}/effect/list/{entity_id}"))
                    .send();
                handle_response(resp)
            }
            EffectCommands::Cleanse { entity_id, filter } => {
                let resp = client
                    .post(format!("{server}/effect/cleanse"))
                    .json(&serde_json::json!({
                        "entity_id": entity_id,
                        "filter": filter,
                    }))
                    .send();
                handle_response(resp)
            }
        },

        // ── Inventory & Items ──
        Commands::Item { command } => match command {
            ItemCommands::Define { id, name, prop } => {
                let properties: serde_json::Map<String, serde_json::Value> = prop
                    .iter()
                    .filter_map(|p| {
                        let (k, v) = p.split_once(':')?;
                        let val: f64 = v.parse().ok()?;
                        Some((k.to_string(), serde_json::json!(val)))
                    })
                    .collect();
                let resp = client
                    .post(format!("{server}/item/define"))
                    .json(&serde_json::json!({
                        "id": id,
                        "name": name,
                        "properties": properties,
                    }))
                    .send();
                handle_response(resp)
            }
            ItemCommands::Give {
                entity_id,
                item_id,
                count,
            } => {
                let resp = client
                    .post(format!("{server}/item/give"))
                    .json(&serde_json::json!({
                        "entity_id": entity_id,
                        "item_id": item_id,
                        "count": count,
                    }))
                    .send();
                handle_response(resp)
            }
            ItemCommands::Equip {
                entity_id,
                slot,
                item_id,
            } => {
                let resp = client
                    .post(format!("{server}/item/equip"))
                    .json(&serde_json::json!({
                        "entity_id": entity_id,
                        "slot": slot,
                        "item_id": item_id,
                    }))
                    .send();
                handle_response(resp)
            }
            ItemCommands::List { entity_id } => {
                let resp = client.get(format!("{server}/item/list/{entity_id}")).send();
                handle_response(resp)
            }
        },

        Commands::Auth { command } => run_auth(command, &client, server),

        // ── Audio ──
        Commands::Audio { command } => match command {
            AudioCommands::Play {
                path,
                position,
                volume,
                looping,
                max_distance,
            } => {
                let mut body = serde_json::json!({
                    "path": path,
                    "volume": volume,
                    "loop": looping,
                    "max_distance": max_distance,
                });
                if let Some(pos_str) = position {
                    let parts: Vec<f32> = pos_str
                        .split(',')
                        .filter_map(|p| p.trim().parse().ok())
                        .collect();
                    if parts.len() == 3 {
                        body["position"] = serde_json::json!([parts[0], parts[1], parts[2]]);
                    }
                }
                let resp = client
                    .post(format!("{server}/audio/play"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
            AudioCommands::Stop { entity_id } => {
                let resp = client
                    .post(format!("{server}/audio/stop"))
                    .json(&serde_json::json!({"entity_id": entity_id}))
                    .send();
                handle_response(resp)
            }
            AudioCommands::List => {
                let resp = client.get(format!("{server}/audio/list")).send();
                handle_response(resp)
            }
        },

        // ── Input ──
        Commands::Input { command } => match command {
            InputCommands::Bind { key, action } => {
                let resp = client
                    .post(format!("{server}/input/bind"))
                    .json(&serde_json::json!({"key": key, "action": action}))
                    .send();
                handle_response(resp)
            }
            InputCommands::Unbind { key } => {
                let resp = client
                    .post(format!("{server}/input/unbind"))
                    .json(&serde_json::json!({"key": key}))
                    .send();
                handle_response(resp)
            }
            InputCommands::List => {
                let resp = client.get(format!("{server}/input/list")).send();
                handle_response(resp)
            }
            InputCommands::ContextPush { context } => {
                let resp = client
                    .post(format!("{server}/input/context/push"))
                    .json(&serde_json::json!({"context": context}))
                    .send();
                handle_response(resp)
            }
            InputCommands::ContextPop => {
                let resp = client
                    .post(format!("{server}/input/context/pop"))
                    .json(&serde_json::json!({}))
                    .send();
                handle_response(resp)
            }
        },

        // ── Navigation ──
        Commands::Nav { command } => match command {
            NavCommands::Generate { cell_size } => {
                let resp = client
                    .post(format!("{server}/navmesh/generate"))
                    .json(&serde_json::json!({"cell_size": cell_size}))
                    .send();
                handle_response(resp)
            }
            NavCommands::Compute { from, to } => {
                let parse_vec3 = |s: &str| -> Vec<f32> {
                    s.split(',').filter_map(|p| p.trim().parse().ok()).collect()
                };
                let from_parts = parse_vec3(&from);
                let to_parts = parse_vec3(&to);
                let body = serde_json::json!({
                    "from": from_parts,
                    "to": to_parts,
                });
                let resp = client
                    .post(format!("{server}/path/compute"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
            NavCommands::Set {
                entity_id,
                target,
                speed,
            } => {
                let parts: Vec<f32> = target
                    .split(',')
                    .filter_map(|p| p.trim().parse().ok())
                    .collect();
                let body = serde_json::json!({
                    "entity_id": entity_id,
                    "target": parts,
                    "speed": speed,
                });
                let resp = client.post(format!("{server}/path/set")).json(&body).send();
                handle_response(resp)
            }
        },

        // ── VFX (Particles) ──
        Commands::Vfx { command } => match command {
            VfxCommands::Spawn {
                position,
                rate,
                lifetime,
                max,
            } => {
                let mut body = serde_json::json!({
                    "rate": rate,
                    "lifetime": lifetime,
                    "max": max,
                });
                if let Some(pos_str) = position {
                    let parts: Vec<f32> = pos_str
                        .split(',')
                        .filter_map(|p| p.trim().parse().ok())
                        .collect();
                    if parts.len() == 3 {
                        body["position"] = serde_json::json!([parts[0], parts[1], parts[2]]);
                    }
                }
                let resp = client
                    .post(format!("{server}/particle/create"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
            VfxCommands::Stop { entity_id } => {
                let resp = client
                    .post(format!("{server}/particle/stop"))
                    .json(&serde_json::json!({"entity_id": entity_id}))
                    .send();
                handle_response(resp)
            }
            VfxCommands::List => {
                let resp = client.get(format!("{server}/particle/list")).send();
                handle_response(resp)
            }
        },

        // ── Animation ──
        Commands::Animation { command } => match command {
            AnimationCommands::Load { path } => {
                let resp = client
                    .post(format!("{server}/animation/load"))
                    .json(&serde_json::json!({"path": path}))
                    .send();
                handle_response(resp)
            }
            AnimationCommands::Play {
                entity_id,
                clip,
                speed,
                looping,
            } => {
                let resp = client
                    .post(format!("{server}/animation/play"))
                    .json(&serde_json::json!({
                        "entity_id": entity_id,
                        "clip": clip,
                        "speed": speed,
                        "loop": looping,
                    }))
                    .send();
                handle_response(resp)
            }
            AnimationCommands::Stop { entity_id } => {
                let resp = client
                    .post(format!("{server}/animation/stop"))
                    .json(&serde_json::json!({"entity_id": entity_id}))
                    .send();
                handle_response(resp)
            }
            AnimationCommands::List => {
                let resp = client.get(format!("{server}/animation/list")).send();
                handle_response(resp)
            }
            AnimationCommands::StateMachine {
                entity_id,
                initial_state,
                state,
            } => {
                let states: Vec<serde_json::Value> = state
                    .iter()
                    .map(|s| {
                        let parts: Vec<&str> = s.split(':').collect();
                        let name = parts.first().copied().unwrap_or("state");
                        let clip: usize = parts.get(1).and_then(|p| p.parse().ok()).unwrap_or(0);
                        let speed: f32 = parts.get(2).and_then(|p| p.parse().ok()).unwrap_or(1.0);
                        let looping: bool = parts.get(3).map(|p| *p != "false").unwrap_or(true);
                        serde_json::json!({
                            "name": name,
                            "clip": clip,
                            "speed": speed,
                            "looping": looping,
                        })
                    })
                    .collect();
                let body = serde_json::json!({
                    "entity_id": entity_id,
                    "initial_state": initial_state,
                    "states": states,
                });
                let resp = client
                    .post(format!("{server}/animation/state-machine"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
            AnimationCommands::Montage {
                entity_id,
                clip,
                clip_duration,
                speed,
                blend_in,
                blend_out,
                bone_mask,
            } => {
                let mut body = serde_json::json!({
                    "entity_id": entity_id,
                    "clip": clip,
                    "clip_duration": clip_duration,
                    "speed": speed,
                    "blend_in": blend_in,
                    "blend_out": blend_out,
                });
                if let Some(mask_str) = bone_mask {
                    let indices: Vec<usize> = mask_str
                        .split(',')
                        .filter_map(|p| p.trim().parse().ok())
                        .collect();
                    if !indices.is_empty() {
                        body["bone_mask"] = serde_json::json!(indices);
                    }
                }
                let resp = client
                    .post(format!("{server}/animation/montage"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
        },

        Commands::Terrain { command } => match command {
            TerrainCommands::Create {
                width,
                height,
                cell_size,
            } => {
                let resp = client.post(format!("{server}/terrain/create")).json(&serde_json::json!({"width": width, "height": height, "cell_size": cell_size})).send();
                handle_response(resp)
            }
            TerrainCommands::Edit {
                op,
                x,
                z,
                radius,
                amount,
            } => {
                let resp = client.post(format!("{server}/terrain/edit")).json(&serde_json::json!({"op": op, "x": x, "z": z, "radius": radius, "amount": amount})).send();
                handle_response(resp)
            }
        },
        Commands::Foliage { command } => match command {
            FoliageCommands::Scatter {
                density,
                area,
                mesh,
                min_scale,
                max_scale,
                max_distance,
            } => {
                let parts: Vec<f32> = area
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                let (area_min, area_max) = if parts.len() == 4 {
                    ([parts[0], 0.0, parts[1]], [parts[2], 0.0, parts[3]])
                } else {
                    ([-20.0, 0.0, -20.0], [20.0, 0.0, 20.0])
                };
                let body = serde_json::json!({
                    "mesh_name": mesh,
                    "density": density,
                    "area_min": area_min,
                    "area_max": area_max,
                    "min_scale": min_scale,
                    "max_scale": max_scale,
                    "max_distance": max_distance,
                });
                let resp = client
                    .post(format!("{server}/foliage/scatter"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
            FoliageCommands::List => {
                let resp = client.get(format!("{server}/foliage/list")).send();
                handle_response(resp)
            }
        },
        Commands::Prefab { command } => match command {
            PrefabCommands::Spawn { name, position } => {
                let mut body = serde_json::json!({"name": name});
                if let Some(p) = position
                    && let Some(v) = parse_vec3(&p)
                {
                    body["position"] = serde_json::json!(v);
                }
                let resp = client
                    .post(format!("{server}/prefab/spawn"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
            PrefabCommands::List => {
                let resp = client.get(format!("{server}/prefab/list")).send();
                handle_response(resp)
            }
        },
        Commands::Material { command } => match command {
            MaterialCommands::Set {
                entity,
                emissive,
                alpha_mode,
                metallic,
                roughness,
            } => {
                let mut body = serde_json::json!({"entity_id": entity});
                if let Some(e) = emissive
                    && let Some(v) = parse_vec3(&e)
                {
                    body["emissive"] = serde_json::json!(v);
                }
                if let Some(a) = alpha_mode {
                    body["alpha_mode"] = serde_json::json!(a);
                }
                if let Some(m) = metallic {
                    body["metallic"] = serde_json::json!(m);
                }
                if let Some(r) = roughness {
                    body["roughness"] = serde_json::json!(r);
                }
                let resp = client
                    .post(format!("{server}/material/set"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
        },
        Commands::Postprocess { command } => match command {
            PostprocessCommands::Get => {
                let resp = client.get(format!("{server}/postprocess/settings")).send();
                handle_response(resp)
            }
            PostprocessCommands::Set {
                ssao,
                fxaa,
                exposure,
                bloom,
                contrast,
                saturation,
            } => {
                let mut body = serde_json::json!({});
                if let Some(v) = ssao {
                    body["ssao_enabled"] = serde_json::json!(v);
                }
                if let Some(v) = fxaa {
                    body["fxaa_enabled"] = serde_json::json!(v);
                }
                if let Some(v) = exposure {
                    body["exposure"] = serde_json::json!(v);
                }
                if let Some(v) = bloom {
                    body["bloom_enabled"] = serde_json::json!(v);
                }
                if let Some(v) = contrast {
                    body["contrast"] = serde_json::json!(v);
                }
                if let Some(v) = saturation {
                    body["saturation"] = serde_json::json!(v);
                }
                let resp = client
                    .post(format!("{server}/postprocess/settings"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
            PostprocessCommands::Preset { quality } => {
                let body = serde_json::json!({ "quality": quality });
                let resp = client
                    .post(format!("{server}/postprocess/preset"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
        },

        // ── Volumetric Fog ──
        Commands::Fog { command } => match command {
            FogCommands::Get => {
                let resp = client.get(format!("{server}/fog/settings")).send();
                handle_response(resp)
            }
            FogCommands::Set {
                density,
                scattering,
                absorption,
                height_falloff,
                max_distance,
                light_contribution,
                enabled,
            } => {
                let mut body = serde_json::json!({});
                if let Some(v) = density {
                    body["density"] = serde_json::json!(v);
                }
                if let Some(v) = scattering {
                    body["scattering"] = serde_json::json!(v);
                }
                if let Some(v) = absorption {
                    body["absorption"] = serde_json::json!(v);
                }
                if let Some(v) = height_falloff {
                    body["height_falloff"] = serde_json::json!(v);
                }
                if let Some(v) = max_distance {
                    body["max_distance"] = serde_json::json!(v);
                }
                if let Some(v) = light_contribution {
                    body["light_contribution"] = serde_json::json!(v);
                }
                if let Some(v) = enabled {
                    body["enabled"] = serde_json::json!(v);
                }
                let resp = client
                    .post(format!("{server}/fog/settings"))
                    .json(&body)
                    .send();
                handle_response(resp)
            }
        },

        // ── HUD ──
        Commands::Ui { command } => match command {
            UiCommands::Text {
                text,
                x,
                y,
                size,
                color,
            } => {
                let resp = client
                    .post(format!("{server}/ui/text"))
                    .json(&serde_json::json!({"type": "text", "text": text, "x": x, "y": y, "size": size, "color": color}))
                    .send();
                handle_response(resp)
            }
            UiCommands::Bar {
                x,
                y,
                width,
                height,
                fill,
                color,
            } => {
                let resp = client
                    .post(format!("{server}/ui/bar"))
                    .json(&serde_json::json!({"type": "bar", "x": x, "y": y, "width": width, "height": height, "fill": fill, "color": color}))
                    .send();
                handle_response(resp)
            }
            UiCommands::Clear => post_empty(&client, server, "/ui/clear"),
            UiCommands::List => {
                let resp = client.get(format!("{server}/ui/list")).send();
                handle_response(resp)
            }
        },

        // ── Scripting ──
        Commands::Script { command } => match command {
            ScriptCommands::Load { entity_id, path } => {
                let resp = client
                    .post(format!("{server}/script/load"))
                    .json(&serde_json::json!({"entity_id": entity_id, "path": path}))
                    .send();
                handle_response(resp)
            }
            ScriptCommands::List => {
                let resp = client.get(format!("{server}/script/list")).send();
                handle_response(resp)
            }
        },

        // ── Networking ──
        Commands::Net { command } => match command {
            NetCommands::Status => {
                let resp = client.get(format!("{server}/net/status")).send();
                handle_response(resp)
            }
        },

        // ── Standalone ──
        Commands::Profile => run_profile(&client, server),
        Commands::Diagnose => {
            let resp = client.get(format!("{server}/diagnose")).send();
            handle_response(resp)
        }
        Commands::Events => {
            let resp = client.get(format!("{server}/events")).send();
            handle_response(resp)
        }
        Commands::Status => {
            let resp = client.get(format!("{server}/")).send();
            handle_response(resp)
        }
        Commands::Screenshot { output } => run_screenshot(&client, server, output),
        Commands::Schema {
            component: _component,
        } => {
            let resp = client.get(format!("{server}/schema")).send();
            handle_response(resp)
        }

        Commands::Package { project, output } => {
            commands::package::package_game(&project, &output);
            Ok(())
        }

        Commands::Discover { json, group } => {
            commands::discover::run_discover(json, group.as_deref());
            Ok(())
        }

        Commands::Asset { command } => {
            commands::asset::run_asset(command);
            Ok(())
        }

        // ── Hidden backward-compat aliases ──
        Commands::Observe { entity } => {
            if let Some(id) = entity {
                let resp = client.get(format!("{server}/entities/{id}")).send();
                handle_response(resp)
            } else {
                let resp = client
                    .post(format!("{server}/observe"))
                    .header("Content-Type", "application/json")
                    .body("{}")
                    .send();
                handle_response(resp)
            }
        }
        Commands::Spawn {
            position,
            scale,
            physics,
            collider,
        } => {
            let body = build_create_body(
                &None, &None, &position, &scale, &physics, &collider, None, None, false, None,
                None, None, None, &None, &None, None, None, None, &None, None, false,
            );
            let resp = client.post(format!("{server}/spawn")).json(&body).send();
            handle_response(resp)
        }
        Commands::Modify {
            id,
            transform,
            velocity,
            physics,
            collider,
            json,
        } => {
            let body = if let Some(ref raw) = json {
                parse_json_flag(raw)
            } else {
                // Map old --transform to --position
                build_update_body(&None, &transform, &None, &velocity, &physics, &collider)
            };
            let resp = client
                .post(format!("{server}/entities/{id}/components"))
                .json(&body)
                .send();
            handle_response(resp)
        }
        Commands::Despawn { id, all } => {
            if all {
                post_empty(&client, server, "/reset")
            } else {
                let body = serde_json::json!({"entity_id": id, "entity_generation": 0});
                let resp = client.post(format!("{server}/despawn")).json(&body).send();
                handle_response(resp)
            }
        }
        Commands::Play => post_empty(&client, server, "/play"),
        Commands::Pause => post_empty(&client, server, "/pause"),
        Commands::Step { ticks } => {
            let resp = client
                .post(format!("{server}/step"))
                .json(&serde_json::json!({"ticks": ticks}))
                .send();
            handle_response(resp)
        }
        Commands::Reset => post_empty(&client, server, "/reset"),
    };

    if let Err(e) = result {
        log::error!("{}", e);
        log::error!("Is the engine running? Start with: cargo run -p euca-editor --example editor");
        std::process::exit(1);
    }
}

// ── Shared request helpers ──

pub(crate) fn post_empty(
    client: &reqwest::blocking::Client,
    server: &str,
    path: &str,
) -> Result<(), String> {
    let resp = client
        .post(format!("{server}{path}"))
        .header("Content-Type", "application/json")
        .body("{}")
        .send();
    handle_response(resp)
}

pub(crate) fn handle_response(
    resp: Result<reqwest::blocking::Response, reqwest::Error>,
) -> Result<(), String> {
    let resp = resp.map_err(|e| e.to_string())?;
    let status = resp.status();
    let text = resp.text().map_err(|e| e.to_string())?;

    match serde_json::from_str::<Value>(&text) {
        Ok(json) => {
            let pretty = serde_json::to_string_pretty(&json).unwrap_or_default();
            println!("{pretty}");
        }
        Err(_) => {
            println!("{text}");
        }
    }

    if !status.is_success() {
        return Err(format!("HTTP {status}"));
    }
    Ok(())
}

fn run_profile(client: &reqwest::blocking::Client, server: &str) -> Result<(), String> {
    let resp = client
        .get(format!("{server}/profile"))
        .send()
        .map_err(|e| e.to_string())?;
    let json: Value = resp.json().map_err(|e| e.to_string())?;
    let fps = json["fps"].as_f64().unwrap_or(0.0);
    let frame_ms = json["frame_ms"].as_f64().unwrap_or(0.0);
    println!("FPS: {fps:.1}  frame: {frame_ms:.1} ms");
    if let Some(sections) = json["sections"].as_array() {
        for s in sections {
            let name = s["name"].as_str().unwrap_or("?");
            let us = s["us"].as_f64().unwrap_or(0.0);
            println!("  {name:<20} {us:>8.1} us");
        }
    }
    Ok(())
}

fn run_screenshot(
    client: &reqwest::blocking::Client,
    server: &str,
    output: Option<String>,
) -> Result<(), String> {
    let resp = client
        .post(format!("{server}/screenshot"))
        .header("Content-Type", "application/json")
        .body("{}")
        .send();
    match resp {
        Ok(r) if r.status().is_success() => {
            let text = r.text().unwrap_or_default();
            if let Ok(json) = serde_json::from_str::<Value>(&text) {
                let server_path = json["path"].as_str().unwrap_or("");
                if let Some(ref out) = output {
                    if let Err(e) = std::fs::copy(server_path, out) {
                        log::error!("Failed to copy screenshot: {e}");
                        std::process::exit(1);
                    }
                    println!("{out}");
                } else {
                    println!("{server_path}");
                }
            }
            Ok(())
        }
        Ok(r) => {
            let status = r.status();
            let text = r.text().unwrap_or_default();
            log::error!("{}", text);
            Err(format!("HTTP {status}"))
        }
        Err(e) => Err(e.to_string()),
    }
}

fn run_auth(
    command: AuthCommands,
    client: &reqwest::blocking::Client,
    server: &str,
) -> Result<(), String> {
    match command {
        AuthCommands::Login => {
            let nit_output = std::process::Command::new("nit")
                .args(["sign", "--login", "eucaengine.local"])
                .output();

            let nit_output = match nit_output {
                Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
                Ok(o) => {
                    let err = String::from_utf8_lossy(&o.stderr);
                    log::error!("nit sign --login failed: {err}");
                    std::process::exit(1);
                }
                Err(_) => {
                    log::error!("nit not found. Install nit: npm install -g @newtype-ai/nit");
                    std::process::exit(1);
                }
            };

            let nit_data: Value = match serde_json::from_str(nit_output.trim()) {
                Ok(v) => v,
                Err(e) => {
                    log::error!("Failed to parse nit output: {e}");
                    std::process::exit(1);
                }
            };

            let home = std::env::var("HOME").unwrap_or_default();
            let pub_key_path = format!("{home}/.nit/identity/agent.pub");
            let public_key = match std::fs::read_to_string(&pub_key_path) {
                Ok(k) => k,
                Err(_) => {
                    log::error!("Cannot read {pub_key_path}. Run 'nit init' first.");
                    std::process::exit(1);
                }
            };

            let payload = serde_json::json!({
                "agent_id": nit_data["agent_id"],
                "domain": nit_data["domain"],
                "timestamp": nit_data["timestamp"],
                "signature": nit_data["signature"],
                "public_key": public_key.trim(),
            });

            let resp = client
                .post(format!("{server}/auth/login"))
                .json(&payload)
                .send();
            handle_response(resp)
        }
        AuthCommands::Status => {
            let resp = client.get(format!("{server}/auth/status")).send();
            handle_response(resp)
        }
    }
}
