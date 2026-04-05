//! Spinning cubes — WASM proof of concept for Euca Engine.
//!
//! Implements [`WebApp`] to render three PBR cubes with an orbiting camera.
//! Works both natively (for development) and compiled to WASM (for browsers).
//!
//! Native:  `cargo run -p euca-web-demo`
//! WASM:    `cd games/web-demo && wasm-pack build --target web --out-dir pkg`

use euca_web::euca_ecs::{Entity, Query, World};
use euca_web::euca_math::{Quat, Transform, Vec3};
use euca_web::euca_render::{GpuContext, Material, MaterialRef, Mesh, MeshRenderer, Renderer};
use euca_web::euca_scene::{GlobalTransform, LocalTransform};
use euca_web::{WebApp, euca_core::Time, euca_render::Camera};

#[derive(Clone, Copy)]
struct Spin {
    speed: f32,
}

#[derive(Default)]
pub struct SpinningCubes;

impl WebApp for SpinningCubes {
    fn init(&mut self, world: &mut World, renderer: &mut Renderer, gpu: &GpuContext) {
        let cube = renderer.upload_mesh(gpu, &Mesh::cube());
        let red = renderer.upload_material(gpu, &Material::red_plastic());
        let green = renderer.upload_material(gpu, &Material::green());
        let blue = renderer.upload_material(gpu, &Material::blue_plastic());

        let spawn = |world: &mut World,
                     pos: Vec3,
                     mat: euca_web::euca_render::MaterialHandle,
                     speed: f32| {
            let e = world.spawn(LocalTransform(Transform::from_translation(pos)));
            world.insert(e, GlobalTransform::default());
            world.insert(e, MeshRenderer { mesh: cube });
            world.insert(e, MaterialRef { handle: mat });
            world.insert(e, Spin { speed });
        };

        spawn(world, Vec3::new(-2.5, 0.0, 0.0), red, 1.0);
        spawn(world, Vec3::new(0.0, 0.0, 0.0), green, 1.5);
        spawn(world, Vec3::new(2.5, 0.0, 0.0), blue, 2.0);

        log::info!("SpinningCubes initialized: 3 cubes spawned");
    }

    fn update(&mut self, world: &mut World, _dt: f32) {
        let elapsed = world.resource::<Time>().unwrap().elapsed as f32;

        // Spin cubes
        let updates: Vec<(Entity, f32)> = {
            let query = Query::<(Entity, &Spin)>::new(world);
            query.iter().map(|(e, s)| (e, s.speed)).collect()
        };
        for (entity, speed) in updates {
            if let Some(lt) = world.get_mut::<LocalTransform>(entity) {
                lt.0.rotation = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), elapsed * speed);
            }
        }

        // Orbit camera
        let angle = elapsed * 0.3;
        let cam = world.resource_mut::<Camera>().unwrap();
        cam.eye = Vec3::new(angle.cos() * 7.0, 4.0, angle.sin() * 7.0);
    }
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn main() {
    euca_web::run_web_app::<SpinningCubes>();
}

#[cfg(not(target_arch = "wasm32"))]
pub fn main() {
    euca_web::run_web_app::<SpinningCubes>();
}
