use euca_agent::AgentServer;
use euca_ecs::{Events, Schedule, World};

const DT: f32 = 1.0 / 60.0;

/// Register all gameplay systems into the schedule so that `sim step`
/// processes damage, death, respawn, scoring, rules, and combat.
fn build_gameplay_schedule() -> Schedule {
    let mut schedule = Schedule::new();

    // Input
    schedule.add_system(euca_gameplay::player_input_system);

    // Physics & transforms
    schedule.add_system(|w: &mut World| euca_physics::physics_step_system(w));
    schedule.add_system(|w: &mut World| euca_physics::character_controller_system(w, DT));
    schedule.add_system(euca_scene::transform_propagation_system);

    // Stat pipeline
    schedule.add_system(euca_gameplay::equipment_stat_system);
    schedule.add_system(|w: &mut World| euca_gameplay::zone_system(w, DT));
    schedule.add_system(|w: &mut World| euca_gameplay::zone_dynamic_system(w, DT));
    schedule.add_system(|w: &mut World| euca_gameplay::status_effect_tick_system(w, DT));
    schedule.add_system(euca_gameplay::stat_resolution_system);

    // Core gameplay
    schedule.add_system(euca_gameplay::apply_damage_system);
    schedule.add_system(euca_gameplay::death_check_system);
    schedule.add_system(|w: &mut World| euca_gameplay::projectile_system(w, DT));
    schedule.add_system(euca_gameplay::trigger_system);
    schedule.add_system(|w: &mut World| euca_gameplay::ai_system(w, DT));
    schedule.add_system(|w: &mut World| euca_gameplay::auto_combat_system(w, DT));

    // Game state & scoring
    schedule.add_system(|w: &mut World| euca_gameplay::game_state_system(w, DT));
    schedule.add_system(euca_gameplay::on_death_rule_system);
    schedule.add_system(|w: &mut World| euca_gameplay::timer_rule_system(w, DT));
    schedule.add_system(euca_gameplay::health_below_rule_system);
    schedule.add_system(euca_gameplay::on_score_rule_system);
    schedule.add_system(euca_gameplay::on_phase_rule_system);

    // Respawn & cleanup
    schedule.add_system(|w: &mut World| {
        let delay = w
            .resource::<euca_gameplay::GameState>()
            .map(|gs| gs.config.respawn_delay)
            .unwrap_or(5.0);
        euca_gameplay::start_respawn_on_death(w, delay);
    });
    schedule.add_system(|w: &mut World| euca_gameplay::respawn_system(w, DT));
    schedule.add_system(|w: &mut World| euca_gameplay::corpse_cleanup_system(w, DT));

    // Economy & abilities
    schedule.add_system(euca_gameplay::gold_on_kill_system);
    schedule.add_system(euca_gameplay::xp_on_kill_system);
    schedule.add_system(|w: &mut World| euca_gameplay::ability_tick_system(w, DT));
    schedule.add_system(euca_gameplay::use_ability_system);

    // Perception
    schedule.add_system(euca_gameplay::visibility_system);

    // Turn-based
    schedule.add_system(euca_gameplay::turn_system);

    // SLG economy
    schedule.add_system(|w: &mut World| euca_gameplay::tile_income_system(w, DT));

    schedule
}

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut world = World::new();
    world.insert_resource(Events::default());

    log::info!("Euca Engine headless server — full gameplay schedule");

    let schedule = build_gameplay_schedule();

    let port = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);

    let server = AgentServer::new(world, schedule, port);

    log::info!("Listening on http://127.0.0.1:{port}");
    log::info!("Run: ./scripts/moba.sh to set up a MOBA game");
    log::info!("Then: euca sim step --ticks 60 to advance the simulation");

    server.run().await;
}
