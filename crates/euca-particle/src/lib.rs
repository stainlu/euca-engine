//! CPU particle system for EucaEngine.
//!
//! Particles are managed per-emitter (not individual ECS entities) for performance.
//! Each `ParticleEmitter` component owns a `Vec<Particle>` pool.
//!
//! # Systems
//! - `emit_particles_system`: spawns new particles based on emission rate
//! - `particle_update_system`: advances particles (velocity, gravity, aging), removes dead ones
//!
//! # Usage
//! ```ignore
//! let e = world.spawn(ParticleEmitter::new(EmitterConfig {
//!     rate: 50.0,
//!     particle_lifetime: 2.0,
//!     speed_range: [2.0, 5.0],
//!     ..Default::default()
//! }));
//! world.insert(e, LocalTransform(Transform::from_translation(pos)));
//! ```

use euca_ecs::{Entity, Query, World};
use euca_math::Vec3;
use euca_scene::GlobalTransform;
use serde::{Deserialize, Serialize};

// ── Particle (internal, not an ECS component) ──

/// A single particle — lives inside a ParticleEmitter's pool.
#[derive(Clone, Debug)]
pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub age: f32,
    pub lifetime: f32,
    pub size: f32,
}

impl Particle {
    /// Returns 0.0 (newborn) to 1.0 (dead).
    pub fn age_fraction(&self) -> f32 {
        (self.age / self.lifetime).clamp(0.0, 1.0)
    }

    pub fn is_dead(&self) -> bool {
        self.age >= self.lifetime
    }
}

// ── Emitter shape ──

/// Shape from which particles are spawned.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EmitterShape {
    /// All particles spawn at the emitter origin.
    Point,
    /// Particles spawn randomly within a sphere.
    Sphere { radius: f32 },
    /// Particles spawn in a cone (upward by default).
    Cone { angle: f32 },
}

impl Default for EmitterShape {
    fn default() -> Self {
        Self::Point
    }
}

// ── Emitter config ──

/// Configuration for a particle emitter.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmitterConfig {
    /// Particles spawned per second.
    pub rate: f32,
    /// Lifetime of each particle (seconds).
    pub particle_lifetime: f32,
    /// Min and max initial speed.
    pub speed_range: [f32; 2],
    /// Min and max particle size.
    pub size_range: [f32; 2],
    /// Start color (RGBA).
    pub color_start: [f32; 4],
    /// End color (RGBA) — fades toward this as particle ages.
    pub color_end: [f32; 4],
    /// Shape from which particles are emitted.
    pub shape: EmitterShape,
    /// Maximum particles alive at once (pool cap).
    pub max_particles: u32,
    /// Gravity applied to particles (default: [0, -9.81, 0]).
    pub gravity: [f32; 3],
}

impl Default for EmitterConfig {
    fn default() -> Self {
        Self {
            rate: 50.0,
            particle_lifetime: 2.0,
            speed_range: [2.0, 5.0],
            size_range: [0.1, 0.3],
            color_start: [1.0, 0.8, 0.2, 1.0],
            color_end: [1.0, 0.2, 0.0, 0.0],
            shape: EmitterShape::Point,
            max_particles: 1000,
            gravity: [0.0, -9.81, 0.0],
        }
    }
}

// ── ParticleEmitter component ──

/// ECS component: emits and manages a pool of particles.
#[derive(Clone, Debug)]
pub struct ParticleEmitter {
    pub config: EmitterConfig,
    pub active: bool,
    /// Internal: particle pool.
    pub particles: Vec<Particle>,
    /// Internal: fractional particles not yet emitted.
    accumulator: f32,
}

impl ParticleEmitter {
    pub fn new(config: EmitterConfig) -> Self {
        Self {
            config,
            active: true,
            particles: Vec::new(),
            accumulator: 0.0,
        }
    }

    /// Get current particle color for a given age fraction (0..1).
    pub fn color_at(&self, age_frac: f32) -> [f32; 4] {
        let t = age_frac.clamp(0.0, 1.0);
        [
            self.config.color_start[0]
                + (self.config.color_end[0] - self.config.color_start[0]) * t,
            self.config.color_start[1]
                + (self.config.color_end[1] - self.config.color_start[1]) * t,
            self.config.color_start[2]
                + (self.config.color_end[2] - self.config.color_start[2]) * t,
            self.config.color_start[3]
                + (self.config.color_end[3] - self.config.color_start[3]) * t,
        ]
    }
}

// ── Simple RNG (avoids pulling in rand for particles) ──

fn simple_hash(seed: u32) -> u32 {
    let mut x = seed;
    x = x.wrapping_mul(0x9E3779B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x21F0AAAD);
    x ^= x >> 15;
    x
}

fn random_f32(seed: &mut u32) -> f32 {
    *seed = simple_hash(*seed);
    (*seed as f32) / (u32::MAX as f32)
}

fn random_range(seed: &mut u32, min: f32, max: f32) -> f32 {
    min + random_f32(seed) * (max - min)
}

fn random_direction(seed: &mut u32) -> Vec3 {
    let theta = random_range(seed, 0.0, std::f32::consts::TAU);
    let z = random_range(seed, -1.0, 1.0);
    let r = (1.0 - z * z).sqrt();
    Vec3::new(r * theta.cos(), r * theta.sin(), z)
}

// ── Systems ──

/// Spawn new particles based on emission rate.
pub fn emit_particles_system(world: &mut World, dt: f32) {
    // We use tick as a seed base for deterministic-ish randomness
    let tick = world.current_tick() as u32;

    let entities: Vec<Entity> = {
        let query = Query::<(Entity, &ParticleEmitter)>::new(world);
        query
            .iter()
            .filter(|(_, em)| em.active)
            .map(|(e, _)| e)
            .collect()
    };

    for (idx, entity) in entities.iter().enumerate() {
        let emitter_pos = world
            .get::<GlobalTransform>(*entity)
            .map(|gt| gt.0.translation)
            .unwrap_or(Vec3::ZERO);

        let emitter = match world.get_mut::<ParticleEmitter>(*entity) {
            Some(e) => e,
            None => continue,
        };

        emitter.accumulator += emitter.config.rate * dt;
        let to_spawn = emitter.accumulator as u32;
        emitter.accumulator -= to_spawn as f32;

        let max = emitter.config.max_particles;
        let speed_min = emitter.config.speed_range[0];
        let speed_max = emitter.config.speed_range[1];
        let size_min = emitter.config.size_range[0];
        let size_max = emitter.config.size_range[1];
        let lifetime = emitter.config.particle_lifetime;
        let shape = emitter.config.shape.clone();

        let mut seed = tick.wrapping_add(idx as u32 * 7919);

        for _ in 0..to_spawn {
            if emitter.particles.len() >= max as usize {
                break;
            }

            let spawn_offset = match &shape {
                EmitterShape::Point => Vec3::ZERO,
                EmitterShape::Sphere { radius } => {
                    random_direction(&mut seed) * random_range(&mut seed, 0.0, *radius)
                }
                EmitterShape::Cone { angle } => {
                    let half_angle = angle.to_radians() * 0.5;
                    let theta = random_range(&mut seed, 0.0, std::f32::consts::TAU);
                    let phi = random_range(&mut seed, 0.0, half_angle);
                    Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin())
                }
            };

            let dir = random_direction(&mut seed);
            let speed = random_range(&mut seed, speed_min, speed_max);

            emitter.particles.push(Particle {
                position: emitter_pos + spawn_offset,
                velocity: dir * speed,
                age: 0.0,
                lifetime,
                size: random_range(&mut seed, size_min, size_max),
            });
        }
    }
}

/// Update particles: apply velocity + gravity, age, remove dead.
pub fn particle_update_system(world: &mut World, dt: f32) {
    let entities: Vec<Entity> = {
        let query = Query::<(Entity, &ParticleEmitter)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };

    for entity in entities {
        let emitter = match world.get_mut::<ParticleEmitter>(entity) {
            Some(e) => e,
            None => continue,
        };

        let gravity = Vec3::new(
            emitter.config.gravity[0],
            emitter.config.gravity[1],
            emitter.config.gravity[2],
        );

        for particle in emitter.particles.iter_mut() {
            particle.velocity = particle.velocity + gravity * dt;
            particle.position = particle.position + particle.velocity * dt;
            particle.age += dt;
        }

        // Remove dead particles
        emitter.particles.retain(|p| !p.is_dead());
    }
}

/// Collect all live particles for rendering. Returns (position, size, color_rgba).
pub fn collect_particle_data(world: &World) -> Vec<(Vec3, f32, [f32; 4])> {
    let mut data = Vec::new();
    let query = Query::<&ParticleEmitter>::new(world);
    for emitter in query.iter() {
        for particle in &emitter.particles {
            let color = emitter.color_at(particle.age_fraction());
            data.push((particle.position, particle.size, color));
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Transform;
    use euca_scene::LocalTransform;

    #[test]
    fn emitter_spawns_particles() {
        let mut world = World::new();
        let e = world.spawn(ParticleEmitter::new(EmitterConfig {
            rate: 100.0,
            particle_lifetime: 1.0,
            ..Default::default()
        }));
        world.insert(e, LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(e, GlobalTransform::default());

        emit_particles_system(&mut world, 0.1); // should spawn ~10

        let emitter = world.get::<ParticleEmitter>(e).unwrap();
        assert!(emitter.particles.len() >= 5); // at least some spawned
    }

    #[test]
    fn particles_age_and_die() {
        let mut world = World::new();
        let e = world.spawn(ParticleEmitter::new(EmitterConfig {
            rate: 10.0,
            particle_lifetime: 0.5,
            ..Default::default()
        }));
        world.insert(e, LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(e, GlobalTransform::default());

        emit_particles_system(&mut world, 1.0); // spawn ~10
        let count_before = world.get::<ParticleEmitter>(e).unwrap().particles.len();
        assert!(count_before > 0);

        particle_update_system(&mut world, 1.0); // age by 1s, lifetime is 0.5s

        let count_after = world.get::<ParticleEmitter>(e).unwrap().particles.len();
        assert_eq!(count_after, 0); // all dead
    }

    #[test]
    fn color_interpolation() {
        let emitter = ParticleEmitter::new(EmitterConfig {
            color_start: [1.0, 1.0, 1.0, 1.0],
            color_end: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        });
        let mid = emitter.color_at(0.5);
        assert!((mid[0] - 0.5).abs() < 0.01);
        assert!((mid[3] - 0.5).abs() < 0.01);
    }

    #[test]
    fn max_particles_capped() {
        let mut world = World::new();
        let e = world.spawn(ParticleEmitter::new(EmitterConfig {
            rate: 10000.0,
            max_particles: 50,
            particle_lifetime: 10.0,
            ..Default::default()
        }));
        world.insert(e, LocalTransform(Transform::from_translation(Vec3::ZERO)));
        world.insert(e, GlobalTransform::default());

        emit_particles_system(&mut world, 1.0);

        let count = world.get::<ParticleEmitter>(e).unwrap().particles.len();
        assert!(count <= 50);
    }
}
