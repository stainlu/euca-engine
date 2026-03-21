//! Raycast-based vehicle physics.
//!
//! Provides a `Vehicle` component (wheel configs + runtime state), a
//! `VehicleInput` component for player input, an `EngineCurve` for
//! torque-vs-RPM modelling with automatic gear shifting, and a
//! per-frame `vehicle_physics_system` that resolves suspension, tire
//! forces, and drivetrain each tick.

use euca_ecs::{Entity, Query, World};
use euca_math::{Quat, Vec3};
use euca_scene::LocalTransform;

use crate::components::{Mass, Velocity};
use crate::raycast::{Ray, raycast_world};

// ── Configuration ────────────────────────────────────────────────────────────

/// Per-wheel configuration (immutable at runtime).
#[derive(Clone, Debug)]
pub struct WheelConfig {
    /// Offset from the vehicle body center (in local space).
    pub offset: Vec3,
    /// Wheel radius in meters.
    pub radius: f32,
    /// Spring constant (N/m) for the suspension.
    pub spring_constant: f32,
    /// Damping coefficient (Ns/m) for the suspension.
    pub damping: f32,
    /// Natural length of the suspension spring (meters).
    pub rest_length: f32,
    /// Maximum steering angle (radians). 0.0 for non-steering wheels.
    pub max_steer_angle: f32,
}

impl WheelConfig {
    /// Create a wheel config with sensible defaults.
    pub fn new(offset: Vec3) -> Self {
        Self {
            offset,
            radius: 0.35,
            spring_constant: 35_000.0,
            damping: 4_500.0,
            rest_length: 0.5,
            max_steer_angle: 0.0,
        }
    }

    pub fn with_radius(mut self, radius: f32) -> Self {
        self.radius = radius;
        self
    }

    pub fn with_spring(mut self, spring_constant: f32, damping: f32) -> Self {
        self.spring_constant = spring_constant;
        self.damping = damping;
        self
    }

    pub fn with_rest_length(mut self, rest_length: f32) -> Self {
        self.rest_length = rest_length;
        self
    }

    pub fn with_max_steer_angle(mut self, angle: f32) -> Self {
        self.max_steer_angle = angle;
        self
    }
}

/// Per-wheel runtime state (updated each frame by the system).
#[derive(Clone, Debug, Default)]
pub struct WheelState {
    /// Current suspension compression (0 = fully extended, rest_length = fully compressed).
    pub compression: f32,
    /// Whether this wheel is touching the ground.
    pub is_grounded: bool,
    /// Longitudinal slip ratio (dimensionless). Positive = wheelspin.
    pub slip_ratio: f32,
    /// Lateral slip angle in radians.
    pub slip_angle: f32,
    /// World-space contact point (valid only when `is_grounded`).
    pub contact_point: Vec3,
    /// World-space contact normal (valid only when `is_grounded`).
    pub contact_normal: Vec3,
    /// Previous compression for damping calculation.
    prev_compression: f32,
    /// Angular velocity of the wheel (rad/s) for drivetrain integration.
    pub angular_velocity: f32,
}

// ── Engine ───────────────────────────────────────────────────────────────────

/// A single point on the engine torque curve.
#[derive(Clone, Copy, Debug)]
pub struct TorquePoint {
    /// RPM at this sample.
    pub rpm: f32,
    /// Torque (Nm) at this RPM.
    pub torque: f32,
}

/// Engine torque curve with gear ratios and automatic shifting.
#[derive(Clone, Debug)]
pub struct EngineCurve {
    /// Torque samples sorted by RPM. The system linearly interpolates
    /// between adjacent points.
    pub samples: Vec<TorquePoint>,
    /// Gear ratios (index 0 = first gear, etc.). Each ratio multiplies
    /// engine torque and divides wheel speed to compute RPM.
    pub gear_ratios: Vec<f32>,
    /// Final drive ratio (differential).
    pub final_drive_ratio: f32,
    /// RPM at which the auto-shifter upshifts.
    pub upshift_rpm: f32,
    /// RPM at which the auto-shifter downshifts.
    pub downshift_rpm: f32,
    /// Current gear index (0-based). Managed by the auto-shifter.
    pub current_gear: usize,
    /// Current engine RPM (computed from wheel speed + gear ratio).
    pub current_rpm: f32,
    /// Idle RPM floor.
    pub idle_rpm: f32,
    /// Maximum RPM (rev limiter).
    pub max_rpm: f32,
}

impl EngineCurve {
    /// Create an engine curve with a basic torque profile.
    pub fn new(samples: Vec<TorquePoint>, gear_ratios: Vec<f32>) -> Self {
        assert!(
            samples.len() >= 2,
            "EngineCurve requires at least 2 torque samples"
        );
        assert!(
            !gear_ratios.is_empty(),
            "EngineCurve requires at least 1 gear ratio"
        );
        let max_rpm = samples.last().map(|s| s.rpm).unwrap_or(6000.0);
        Self {
            samples,
            gear_ratios,
            final_drive_ratio: 3.5,
            upshift_rpm: max_rpm * 0.85,
            downshift_rpm: max_rpm * 0.35,
            current_gear: 0,
            current_rpm: 800.0,
            idle_rpm: 800.0,
            max_rpm,
        }
    }

    /// Evaluate engine torque at a given RPM by linearly interpolating
    /// the sample curve.
    pub fn torque_at_rpm(&self, rpm: f32) -> f32 {
        let clamped = rpm.clamp(
            self.samples[0].rpm,
            self.samples.last().expect("non-empty torque samples").rpm,
        );
        // Find the two bracketing samples.
        for window in self.samples.windows(2) {
            let lo = &window[0];
            let hi = &window[1];
            if clamped >= lo.rpm && clamped <= hi.rpm {
                let t = (clamped - lo.rpm) / (hi.rpm - lo.rpm);
                return lo.torque + (hi.torque - lo.torque) * t;
            }
        }
        self.samples
            .last()
            .expect("non-empty torque samples")
            .torque
    }

    /// Current combined gear ratio (gear * final drive).
    pub fn combined_ratio(&self) -> f32 {
        self.gear_ratios[self.current_gear] * self.final_drive_ratio
    }

    /// Automatic gear shifting logic. Call once per frame.
    fn auto_shift(&mut self) {
        if self.current_rpm >= self.upshift_rpm && self.current_gear + 1 < self.gear_ratios.len() {
            self.current_gear += 1;
        } else if self.current_rpm <= self.downshift_rpm && self.current_gear > 0 {
            self.current_gear -= 1;
        }
    }
}

/// Default engine curve modelling a typical sedan.
impl Default for EngineCurve {
    fn default() -> Self {
        Self::new(
            vec![
                TorquePoint {
                    rpm: 800.0,
                    torque: 150.0,
                },
                TorquePoint {
                    rpm: 2500.0,
                    torque: 280.0,
                },
                TorquePoint {
                    rpm: 4500.0,
                    torque: 320.0,
                },
                TorquePoint {
                    rpm: 6000.0,
                    torque: 250.0,
                },
            ],
            vec![3.5, 2.1, 1.4, 1.0, 0.8],
        )
    }
}

// ── Components ───────────────────────────────────────────────────────────────

/// ECS component describing a vehicle's wheel layout and drivetrain.
///
/// Attach alongside `LocalTransform`, `Velocity`, and `Mass` on an entity.
/// The entity also needs a collision layer mask set up so the wheel raycasts
/// do not hit the vehicle's own collider (use `query_mask`).
#[derive(Clone, Debug)]
pub struct Vehicle {
    /// Per-wheel configuration.
    pub wheels: Vec<WheelConfig>,
    /// Per-wheel runtime state (same length as `wheels`).
    pub wheel_states: Vec<WheelState>,
    /// Engine / drivetrain.
    pub engine: EngineCurve,
    /// Collision mask for wheel raycasts (should exclude the vehicle's
    /// own layer). Default: `u32::MAX`.
    pub query_mask: u32,
    /// Maximum braking torque (Nm) per wheel.
    pub max_brake_torque: f32,
}

impl Vehicle {
    /// Create a new vehicle from a set of wheel configs.
    pub fn new(wheels: Vec<WheelConfig>) -> Self {
        let count = wheels.len();
        Self {
            wheels,
            wheel_states: vec![WheelState::default(); count],
            engine: EngineCurve::default(),
            query_mask: u32::MAX,
            max_brake_torque: 3000.0,
        }
    }

    pub fn with_engine(mut self, engine: EngineCurve) -> Self {
        self.engine = engine;
        self
    }

    pub fn with_query_mask(mut self, mask: u32) -> Self {
        self.query_mask = mask;
        self
    }

    pub fn with_max_brake_torque(mut self, torque: f32) -> Self {
        self.max_brake_torque = torque;
        self
    }
}

/// ECS component for vehicle driver input.
#[derive(Clone, Copy, Debug, Default)]
pub struct VehicleInput {
    /// Throttle amount (0.0 = none, 1.0 = full).
    pub throttle: f32,
    /// Brake amount (0.0 = none, 1.0 = full).
    pub brake: f32,
    /// Steering amount (-1.0 = full left, 1.0 = full right).
    pub steer: f32,
}

// ── Tire model constants ─────────────────────────────────────────────────────

/// Peak coefficient of friction for the simplified tire model.
const TIRE_MU_PEAK: f32 = 1.5;

/// Stiffness factor for the linear region of the longitudinal slip curve.
const LONGITUDINAL_STIFFNESS: f32 = 12.0;

/// Stiffness factor for the linear region of the lateral slip curve.
const LATERAL_STIFFNESS: f32 = 10.0;

/// Simplified longitudinal force from slip ratio.
/// Uses a linear-then-plateau model: force rises linearly up to peak
/// friction, then clamps. This approximates a Pacejka curve without
/// the complexity.
fn longitudinal_force(normal_force: f32, slip_ratio: f32) -> f32 {
    let raw = LONGITUDINAL_STIFFNESS * slip_ratio;
    let clamped = raw.clamp(-TIRE_MU_PEAK, TIRE_MU_PEAK);
    normal_force * clamped
}

/// Simplified lateral force from slip angle.
fn lateral_force(normal_force: f32, slip_angle: f32) -> f32 {
    let raw = LATERAL_STIFFNESS * slip_angle;
    let clamped = raw.clamp(-TIRE_MU_PEAK, TIRE_MU_PEAK);
    normal_force * clamped
}

// ── System ───────────────────────────────────────────────────────────────────

/// Per-frame vehicle physics system.
///
/// For each entity with `Vehicle`, `VehicleInput`, `Velocity`, `Mass`, and
/// `LocalTransform`, this system:
/// 1. Casts a ray per wheel to detect ground contact.
/// 2. Computes spring + damper suspension forces.
/// 3. Computes longitudinal and lateral tire forces.
/// 4. Integrates engine torque through the drivetrain.
/// 5. Applies the net force and torque to the vehicle body.
pub fn vehicle_physics_system(world: &mut World, dt: f32) {
    if dt <= 0.0 {
        return;
    }

    // Collect entities to process.
    let entities: Vec<Entity> = {
        let q = Query::<(
            Entity,
            &Vehicle,
            &VehicleInput,
            &Velocity,
            &Mass,
            &LocalTransform,
        )>::new(world);
        q.iter().map(|(e, _, _, _, _, _)| e).collect()
    };

    for entity in entities {
        // Copy out all the data we need to avoid borrow conflicts.
        let (mut vehicle, input, velocity, mass, transform) = {
            let v = match world.get::<Vehicle>(entity) {
                Some(v) => v.clone(),
                None => continue,
            };
            let i = match world.get::<VehicleInput>(entity) {
                Some(i) => *i,
                None => continue,
            };
            let vel = match world.get::<Velocity>(entity) {
                Some(v) => *v,
                None => continue,
            };
            let m = match world.get::<Mass>(entity) {
                Some(m) => *m,
                None => continue,
            };
            let lt = match world.get::<LocalTransform>(entity) {
                Some(lt) => lt.0,
                None => continue,
            };
            (v, i, vel, m, lt)
        };

        let rotation = transform.rotation;
        let body_pos = transform.translation;

        // Compute vehicle-space basis vectors.
        let forward = rotation * Vec3::Z; // local Z = forward
        let right = rotation * Vec3::X; // local X = right
        let up = rotation * Vec3::Y; // local Y = up

        let body_velocity = velocity.linear;

        let mut total_force = Vec3::ZERO;
        let mut total_torque = Vec3::ZERO;

        let wheel_count = vehicle.wheels.len();
        let mut grounded_count = 0u32;
        let mut avg_wheel_speed = 0.0_f32;

        for i in 0..wheel_count {
            let config = &vehicle.wheels[i];
            let state = &mut vehicle.wheel_states[i];

            // ── 1. Suspension raycast ──
            let wheel_world_pos = body_pos + rotation * config.offset;
            let ray_length = config.rest_length + config.radius;
            let ray = Ray::new(wheel_world_pos, -up);

            let hits = raycast_world(world, &ray, ray_length, vehicle.query_mask);
            // Skip hits against our own entity.
            let hit = hits.iter().find(|h| h.entity != entity);

            if let Some(hit) = hit {
                state.is_grounded = true;
                state.contact_point = hit.point;
                state.contact_normal = hit.normal;

                // Compression = how much the spring is compressed.
                let spring_length = hit.t - config.radius;
                state.compression = (config.rest_length - spring_length).max(0.0);

                // ── 2. Spring + damper force ──
                let compression_velocity = (state.compression - state.prev_compression) / dt;
                let spring_force = config.spring_constant * state.compression
                    + config.damping * compression_velocity;
                // Force is applied along the contact normal (upward).
                let suspension_force = hit.normal * spring_force.max(0.0);
                let normal_force = spring_force.max(0.0);

                total_force = total_force + suspension_force;
                // Torque from off-center force application.
                let arm = wheel_world_pos - body_pos;
                total_torque = total_torque + arm.cross(suspension_force);

                // ── 3. Tire forces ──
                // Velocity at the contact patch.
                let contact_vel =
                    body_velocity + velocity.angular.cross(wheel_world_pos - body_pos);

                // Compute steered forward direction for this wheel.
                let steer_angle = input.steer * config.max_steer_angle;
                let steer_rot = Quat::from_axis_angle(up, steer_angle);
                let wheel_forward = steer_rot * forward;
                let wheel_right = steer_rot * right;

                // Longitudinal velocity (along wheel forward).
                let v_long = contact_vel.dot(wheel_forward);
                // Lateral velocity (perpendicular to wheel forward, in ground plane).
                let v_lat = contact_vel.dot(wheel_right);

                // Slip ratio: (wheel_speed - ground_speed) / max(|ground_speed|, epsilon).
                let wheel_linear_speed = state.angular_velocity * config.radius;
                let ground_speed = v_long.abs().max(0.5); // avoid division by zero
                state.slip_ratio = (wheel_linear_speed - v_long) / ground_speed;

                // Slip angle: atan2(lateral_velocity, |longitudinal_velocity|).
                state.slip_angle = if v_long.abs() > 0.5 {
                    (v_lat / v_long.abs()).atan()
                } else {
                    v_lat.signum() * std::f32::consts::FRAC_PI_4 * (v_lat.abs() / 0.5).min(1.0)
                };

                // Compute tire forces.
                let fx = longitudinal_force(normal_force, state.slip_ratio);
                let fy = lateral_force(normal_force, state.slip_angle);

                let tire_force = wheel_forward * fx - wheel_right * fy;
                total_force = total_force + tire_force;
                total_torque = total_torque + arm.cross(tire_force);

                grounded_count += 1;
                avg_wheel_speed += state.angular_velocity.abs();
            } else {
                state.is_grounded = false;
                state.compression = 0.0;
                state.slip_ratio = 0.0;
                state.slip_angle = 0.0;
            }

            state.prev_compression = state.compression;
        }

        // ── 4. Engine / drivetrain ──
        if grounded_count > 0 {
            avg_wheel_speed /= grounded_count as f32;

            // Compute engine RPM from average driven wheel angular velocity.
            let combined_ratio = vehicle.engine.combined_ratio();
            let engine_rpm = (avg_wheel_speed * combined_ratio * 60.0
                / (2.0 * std::f32::consts::PI))
                .abs()
                .max(vehicle.engine.idle_rpm);
            vehicle.engine.current_rpm = engine_rpm.min(vehicle.engine.max_rpm);

            // Auto-shift.
            vehicle.engine.auto_shift();

            // Engine torque at the wheels.
            let engine_torque = vehicle.engine.torque_at_rpm(vehicle.engine.current_rpm)
                * input.throttle
                * vehicle.engine.combined_ratio();
            let brake_torque = vehicle.max_brake_torque * input.brake;

            // Distribute torque across grounded wheels (simple equal split).
            let torque_per_wheel = engine_torque / grounded_count as f32;
            let brake_per_wheel = brake_torque / grounded_count as f32;

            for i in 0..wheel_count {
                let state = &mut vehicle.wheel_states[i];
                if !state.is_grounded {
                    continue;
                }
                let config = &vehicle.wheels[i];

                // Apply engine torque as angular acceleration (tau = I * alpha).
                // Approximate wheel inertia as a solid cylinder: I = 0.5 * m * r^2.
                let wheel_inertia = 0.5 * 20.0 * config.radius * config.radius;
                let net_torque =
                    torque_per_wheel - brake_per_wheel * state.angular_velocity.signum();
                state.angular_velocity += (net_torque / wheel_inertia) * dt;

                // Clamp wheel speed on braking to prevent sign reversal.
                if input.brake > 0.0 && state.angular_velocity.abs() < 1.0 {
                    state.angular_velocity *= 0.9;
                }
            }
        } else {
            // Airborne: RPM decays toward idle.
            vehicle.engine.current_rpm = vehicle
                .engine
                .current_rpm
                .lerp(vehicle.engine.idle_rpm, dt * 2.0);
        }

        // ── 5. Apply forces to body ──
        let inv_mass = mass.inverse_mass;
        let inv_inertia = mass.inverse_inertia;

        let new_linear = velocity.linear + total_force * (inv_mass * dt);
        let new_angular = velocity.angular + total_torque * (inv_inertia * dt);

        // Write back.
        if let Some(vel) = world.get_mut::<Velocity>(entity) {
            vel.linear = new_linear;
            vel.angular = new_angular;
        }
        if let Some(v) = world.get_mut::<Vehicle>(entity) {
            *v = vehicle;
        }
    }
}

/// f32 lerp helper (f32::lerp is not stable in all Rust editions).
trait Lerp {
    fn lerp(self, target: Self, t: Self) -> Self;
}

impl Lerp for f32 {
    #[inline]
    fn lerp(self, target: f32, t: f32) -> f32 {
        self + (target - self) * t
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use euca_math::Transform;
    use euca_scene::GlobalTransform;

    use crate::components::{Collider, PhysicsBody};
    use crate::world::PhysicsConfig;

    /// Helper: create a world with physics config (no gravity to isolate vehicle forces).
    fn world_no_gravity() -> World {
        let mut world = World::new();
        world.insert_resource(PhysicsConfig {
            gravity: Vec3::ZERO,
            fixed_dt: 1.0 / 60.0,
            max_substeps: 1,
        });
        world
    }

    /// Helper: spawn a static ground plane at y=0.
    fn spawn_ground(world: &mut World) -> Entity {
        let e = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, -0.5, 0.0,
        ))));
        world.insert(e, GlobalTransform::default());
        world.insert(e, PhysicsBody::fixed());
        world.insert(e, Collider::aabb(100.0, 0.5, 100.0));
        e
    }

    /// Helper: build a simple 4-wheel vehicle entity above the ground.
    fn spawn_vehicle(world: &mut World, height: f32) -> Entity {
        let e = world.spawn(LocalTransform(Transform::from_translation(Vec3::new(
            0.0, height, 0.0,
        ))));
        world.insert(e, GlobalTransform::default());
        world.insert(e, Velocity::default());
        world.insert(e, Mass::new(1500.0, 2500.0));

        let fl = WheelConfig::new(Vec3::new(-0.8, -0.3, 1.2)).with_max_steer_angle(0.5);
        let fr = WheelConfig::new(Vec3::new(0.8, -0.3, 1.2)).with_max_steer_angle(0.5);
        let rl = WheelConfig::new(Vec3::new(-0.8, -0.3, -1.2));
        let rr = WheelConfig::new(Vec3::new(0.8, -0.3, -1.2));

        world.insert(e, Vehicle::new(vec![fl, fr, rl, rr]));
        world.insert(e, VehicleInput::default());
        e
    }

    // ── Test 1: Suspension supports the vehicle ──

    #[test]
    fn suspension_generates_upward_force() {
        let mut world = world_no_gravity();
        spawn_ground(&mut world);
        let car = spawn_vehicle(&mut world, 1.0);

        let dt = 1.0 / 60.0;
        vehicle_physics_system(&mut world, dt);

        let vel = world.get::<Velocity>(car).unwrap();
        // Suspension should push the car upward (positive Y force).
        assert!(
            vel.linear.y > 0.0,
            "Suspension should generate upward force, vy={}",
            vel.linear.y
        );
    }

    // ── Test 2: Throttle accelerates the vehicle ──

    #[test]
    fn throttle_accelerates_forward() {
        let mut world = world_no_gravity();
        spawn_ground(&mut world);
        let car = spawn_vehicle(&mut world, 1.0);

        // Set throttle.
        if let Some(input) = world.get_mut::<VehicleInput>(car) {
            input.throttle = 1.0;
        }

        let dt = 1.0 / 60.0;
        for _ in 0..30 {
            vehicle_physics_system(&mut world, dt);
        }

        let vel = world.get::<Velocity>(car).unwrap();
        // Forward is +Z in local space; with identity rotation the vehicle
        // should gain forward velocity.
        assert!(
            vel.linear.z.abs() > 0.01 || vel.linear.x.abs() > 0.01,
            "Throttle should produce acceleration, vel={:?}",
            vel.linear
        );
    }

    // ── Test 3: Braking decelerates wheel spin ──

    #[test]
    fn brake_reduces_wheel_speed() {
        let mut world = world_no_gravity();
        spawn_ground(&mut world);
        let car = spawn_vehicle(&mut world, 1.0);

        let dt = 1.0 / 60.0;

        // First, spin up the wheels with throttle.
        if let Some(input) = world.get_mut::<VehicleInput>(car) {
            input.throttle = 1.0;
        }
        for _ in 0..60 {
            vehicle_physics_system(&mut world, dt);
        }

        let speed_before: f32 = {
            let v = world.get::<Vehicle>(car).unwrap();
            v.wheel_states
                .iter()
                .map(|s| s.angular_velocity.abs())
                .sum::<f32>()
        };

        // Now brake hard.
        if let Some(input) = world.get_mut::<VehicleInput>(car) {
            input.throttle = 0.0;
            input.brake = 1.0;
        }
        for _ in 0..60 {
            vehicle_physics_system(&mut world, dt);
        }

        let speed_after: f32 = {
            let v = world.get::<Vehicle>(car).unwrap();
            v.wheel_states
                .iter()
                .map(|s| s.angular_velocity.abs())
                .sum::<f32>()
        };

        assert!(
            speed_after < speed_before,
            "Braking should reduce wheel speed: before={speed_before}, after={speed_after}"
        );
    }

    // ── Test 4: Wheels detect ground contact ──

    #[test]
    fn wheels_detect_ground() {
        let mut world = world_no_gravity();
        spawn_ground(&mut world);
        let car = spawn_vehicle(&mut world, 1.0);

        let dt = 1.0 / 60.0;
        vehicle_physics_system(&mut world, dt);

        let vehicle = world.get::<Vehicle>(car).unwrap();
        let grounded = vehicle
            .wheel_states
            .iter()
            .filter(|s| s.is_grounded)
            .count();
        assert!(
            grounded > 0,
            "At least one wheel should detect ground contact"
        );
    }

    // ── Test 5: Engine curve interpolation ──

    #[test]
    fn engine_curve_interpolation() {
        let curve = EngineCurve::default();

        // At 800 RPM (first sample) should return the first torque value.
        let t0 = curve.torque_at_rpm(800.0);
        assert!(
            (t0 - 150.0).abs() < 0.01,
            "Torque at 800 RPM should be 150, got {t0}"
        );

        // At 6000 RPM (last sample) should return the last torque value.
        let t_max = curve.torque_at_rpm(6000.0);
        assert!(
            (t_max - 250.0).abs() < 0.01,
            "Torque at 6000 RPM should be 250, got {t_max}"
        );

        // Midpoint (2500 RPM) should give the second sample value.
        let t_mid = curve.torque_at_rpm(2500.0);
        assert!(
            (t_mid - 280.0).abs() < 0.01,
            "Torque at 2500 RPM should be 280, got {t_mid}"
        );

        // Between samples should interpolate.
        let t_interp = curve.torque_at_rpm(1650.0);
        // 1650 is halfway between 800 and 2500. t = (1650-800)/(2500-800) = 0.5
        let expected = 150.0 + (280.0 - 150.0) * 0.5;
        assert!(
            (t_interp - expected).abs() < 1.0,
            "Torque at 1650 RPM should be ~{expected}, got {t_interp}"
        );
    }

    // ── Test 6: Auto-shift logic ──

    #[test]
    fn auto_shift_upshift_and_downshift() {
        let mut engine = EngineCurve::default();

        // Force RPM above upshift threshold.
        engine.current_rpm = engine.upshift_rpm + 100.0;
        engine.current_gear = 0;
        engine.auto_shift();
        assert_eq!(engine.current_gear, 1, "Should upshift from gear 0 to 1");

        // Force RPM below downshift threshold.
        engine.current_rpm = engine.downshift_rpm - 100.0;
        engine.auto_shift();
        assert_eq!(engine.current_gear, 0, "Should downshift from gear 1 to 0");

        // Already in lowest gear, should not underflow.
        engine.current_gear = 0;
        engine.current_rpm = engine.downshift_rpm - 100.0;
        engine.auto_shift();
        assert_eq!(engine.current_gear, 0, "Should not downshift below gear 0");

        // Already in highest gear, should not overflow.
        engine.current_gear = engine.gear_ratios.len() - 1;
        engine.current_rpm = engine.upshift_rpm + 100.0;
        engine.auto_shift();
        assert_eq!(
            engine.current_gear,
            engine.gear_ratios.len() - 1,
            "Should not upshift beyond max gear"
        );
    }

    // ── Test 7: Zero dt is a no-op ──

    #[test]
    fn zero_dt_is_noop() {
        let mut world = world_no_gravity();
        spawn_ground(&mut world);
        let car = spawn_vehicle(&mut world, 1.0);

        if let Some(input) = world.get_mut::<VehicleInput>(car) {
            input.throttle = 1.0;
        }

        vehicle_physics_system(&mut world, 0.0);

        let vel = world.get::<Velocity>(car).unwrap();
        assert!(
            vel.linear.length_squared() < 1e-6,
            "Zero dt should not change velocity"
        );
    }

    // ── Test 8: Airborne vehicle has no suspension force ──

    #[test]
    fn airborne_vehicle_no_suspension() {
        let mut world = world_no_gravity();
        // No ground spawned -- vehicle is in the air.
        let car = spawn_vehicle(&mut world, 100.0);

        let dt = 1.0 / 60.0;
        vehicle_physics_system(&mut world, dt);

        let vel = world.get::<Velocity>(car).unwrap();
        assert!(
            vel.linear.length_squared() < 1e-6,
            "Airborne vehicle should have no forces applied, vel={:?}",
            vel.linear
        );

        let vehicle = world.get::<Vehicle>(car).unwrap();
        let grounded = vehicle
            .wheel_states
            .iter()
            .filter(|s| s.is_grounded)
            .count();
        assert_eq!(grounded, 0, "No wheels should be grounded when airborne");
    }
}
