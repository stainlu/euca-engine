use euca_agent::AgentServer;
use euca_ecs::{Entity, Query, Schedule, World};
use euca_math::{Transform, Vec3};
use euca_scene::{GlobalTransform, LocalTransform};

/// Simple movement system: entities with Velocity move each tick.
#[derive(Clone, Copy, Debug)]
struct Velocity {
    dx: f32,
    dy: f32,
    dz: f32,
}

fn movement_system(world: &mut World) {
    let updates: Vec<(Entity, f32, f32, f32)> = {
        let query = Query::<(Entity, &Velocity)>::new(world);
        query.iter().map(|(e, v)| (e, v.dx, v.dy, v.dz)).collect()
    };
    for (entity, dx, dy, dz) in updates {
        if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
            lt.0.translation = lt.0.translation + Vec3::new(dx, dy, dz);
        }
    }
    // Propagate transforms
    euca_scene::transform_propagation_system(world);
}

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let mut world = World::new();

    // Spawn some entities
    let e1 = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
        0.0, 0.0, 0.0,
    ))));
    world.insert(e1, GlobalTransform::default());
    world.insert(
        e1,
        Velocity {
            dx: 1.0,
            dy: 0.0,
            dz: 0.0,
        },
    );

    let e2 = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
        10.0, 0.0, 0.0,
    ))));
    world.insert(e2, GlobalTransform::default());

    let e3 = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
        0.0, 5.0, 0.0,
    ))));
    world.insert(e3, GlobalTransform::default());
    world.insert(
        e3,
        Velocity {
            dx: 0.0,
            dy: -0.5,
            dz: 0.0,
        },
    );

    log::info!("Spawned {} entities", world.entity_count());

    // Create schedule with movement system
    let mut schedule = Schedule::new();
    schedule.add_system(movement_system);

    // Start the agent server
    let port = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);

    let server = AgentServer::new(world, schedule, port);

    log::info!("Euca Engine running in headless mode");
    log::info!("Try: curl http://localhost:{port}/");
    log::info!("Try: curl -X POST http://localhost:{port}/observe");
    log::info!(
        "Try: curl -X POST http://localhost:{port}/step -H 'Content-Type: application/json' -d '{{\"ticks\": 10}}'"
    );

    server.run().await;
}
