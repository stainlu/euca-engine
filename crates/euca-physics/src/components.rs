use euca_ecs::Entity;
use euca_math::Vec3;
use euca_reflect::Reflect;

/// Rigid body type.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub enum RigidBodyType {
    /// Fully simulated, responds to forces and gravity.
    Dynamic,
    /// Immovable.
    Static,
    /// Moved programmatically, pushes dynamic bodies.
    Kinematic,
}

/// ECS component marking an entity as a physics body.
#[derive(Clone, Copy, Debug, Reflect)]
pub struct PhysicsBody {
    pub body_type: RigidBodyType,
}

impl PhysicsBody {
    /// Create a fully simulated dynamic body.
    pub fn dynamic() -> Self {
        Self {
            body_type: RigidBodyType::Dynamic,
        }
    }
    /// Create a static (immovable) body.
    pub fn fixed() -> Self {
        Self {
            body_type: RigidBodyType::Static,
        }
    }
    /// Create a kinematic body (moved by gameplay code, pushes dynamic bodies).
    pub fn kinematic() -> Self {
        Self {
            body_type: RigidBodyType::Kinematic,
        }
    }
}

/// Mass and inertia properties for physics bodies.
///
/// Static bodies should use `Mass::infinite()` (zero inverse mass).
/// Dynamic bodies use `Mass::new(mass, inertia)` for spherically symmetric
/// objects, or the shape-specific factories (`from_sphere`, `from_aabb`,
/// `from_capsule`) for accurate diagonal inertia tensors.
#[derive(Clone, Copy, Debug, Reflect)]
pub struct Mass {
    /// Total mass in kilograms.
    pub mass: f32,
    /// Precomputed `1 / mass` (0 for static bodies).
    pub inverse_mass: f32,
    /// Scalar moment of inertia (kg*m^2). Retained for backward compatibility.
    pub inertia: f32,
    /// Precomputed `1 / inertia` (0 for static bodies). Retained for backward compatibility.
    pub inverse_inertia: f32,
    /// Diagonal of the inverse inertia tensor in body-local frame.
    /// For spherically symmetric bodies this equals `Vec3::new(1/I, 1/I, 1/I)`.
    /// For static bodies this is `Vec3::ZERO`.
    pub inverse_inertia_tensor: Vec3,
}

impl Mass {
    /// Build a `Mass` from a mass value and the three principal moments of
    /// inertia. The scalar `inertia` / `inverse_inertia` fields are set to
    /// the mean of the three moments for backward compatibility.
    fn from_principal_moments(mass: f32, ix: f32, iy: f32, iz: f32) -> Self {
        let scalar_i = (ix + iy + iz) / 3.0;
        Self {
            mass,
            inverse_mass: 1.0 / mass,
            inertia: scalar_i,
            inverse_inertia: 1.0 / scalar_i,
            inverse_inertia_tensor: Vec3::new(1.0 / ix, 1.0 / iy, 1.0 / iz),
        }
    }

    /// Create a mass component with the given mass and scalar inertia.
    ///
    /// The inertia tensor is set to `Vec3::new(1/I, 1/I, 1/I)` (spherically
    /// symmetric), which is backward-compatible with the previous scalar model.
    /// For a solid sphere of uniform density: `inertia = 0.4 * mass * radius^2`.
    /// For a solid box: `inertia = mass * (w^2 + h^2) / 12`.
    pub fn new(mass: f32, inertia: f32) -> Self {
        assert!(
            mass > 0.0,
            "Mass must be positive; use Mass::infinite() for static bodies"
        );
        assert!(inertia > 0.0, "Inertia must be positive");
        Self::from_principal_moments(mass, inertia, inertia, inertia)
    }

    /// Infinite mass (for static/kinematic bodies). Zero inverse mass and
    /// zero inverse inertia tensor mean the body is immovable in all
    /// mass-weighted calculations.
    pub fn infinite() -> Self {
        Self {
            mass: f32::INFINITY,
            inverse_mass: 0.0,
            inertia: f32::INFINITY,
            inverse_inertia: 0.0,
            inverse_inertia_tensor: Vec3::ZERO,
        }
    }

    /// Default mass for a 1-unit cube of density 1.
    pub fn default_dynamic() -> Self {
        Self::from_aabb(1.0, 0.5, 0.5, 0.5)
    }

    /// Create mass properties for a solid sphere of uniform density.
    ///
    /// Inertia: `I = 2/5 * mass * radius^2` (all axes equal).
    pub fn from_sphere(mass: f32, radius: f32) -> Self {
        assert!(mass > 0.0, "Mass must be positive");
        assert!(radius > 0.0, "Radius must be positive");
        let i = 0.4 * mass * radius * radius;
        Self::from_principal_moments(mass, i, i, i)
    }

    /// Create mass properties for a solid axis-aligned box (AABB) of uniform density.
    ///
    /// `hx`, `hy`, `hz` are half-extents along each axis.
    /// - `Ix = m * (hy^2 + hz^2) / 3`
    /// - `Iy = m * (hx^2 + hz^2) / 3`
    /// - `Iz = m * (hx^2 + hy^2) / 3`
    ///
    /// (Using half-extents: `(2h)^2 / 12 = h^2 / 3`.)
    pub fn from_aabb(mass: f32, hx: f32, hy: f32, hz: f32) -> Self {
        assert!(mass > 0.0, "Mass must be positive");
        let ix = mass * (hy * hy + hz * hz) / 3.0;
        let iy = mass * (hx * hx + hz * hz) / 3.0;
        let iz = mass * (hx * hx + hy * hy) / 3.0;
        Self::from_principal_moments(mass, ix, iy, iz)
    }

    /// Create mass properties for a Y-axis capsule of uniform density.
    ///
    /// The capsule is modeled as a cylinder of half-height `half_height` and
    /// radius `radius`, capped by two hemispheres. The inertia is computed as
    /// the sum of the cylinder and hemisphere contributions.
    pub fn from_capsule(mass: f32, radius: f32, half_height: f32) -> Self {
        assert!(mass > 0.0, "Mass must be positive");
        assert!(radius > 0.0, "Radius must be positive");
        assert!(half_height >= 0.0, "Half-height must be non-negative");

        let r2 = radius * radius;
        let h = 2.0 * half_height; // full cylinder height

        // Volume fractions for mass distribution
        let vol_cyl = std::f32::consts::PI * r2 * h;
        let vol_sphere = (4.0 / 3.0) * std::f32::consts::PI * r2 * radius;
        let vol_total = vol_cyl + vol_sphere;

        let m_cyl = mass * vol_cyl / vol_total;
        let m_hemi = mass * vol_sphere / vol_total; // both hemispheres combined

        // Cylinder inertia (Y-axis aligned):
        //   Iy_cyl = m_cyl * r^2 / 2
        //   Ix_cyl = Iz_cyl = m_cyl * (3*r^2 + h^2) / 12
        let iy_cyl = m_cyl * r2 / 2.0;
        let ix_cyl = m_cyl * (3.0 * r2 + h * h) / 12.0;

        // Hemisphere inertia (two hemispheres = one sphere, shifted along Y):
        //   Iy_sphere = 2/5 * m_hemi * r^2 (spin axis, no parallel axis shift)
        //   Transverse: each hemisphere has mass m_hemi/2 and COM at distance
        //   d = half_height + 3r/8 from the capsule center. By the parallel
        //   axis theorem: Ix_hemi = I_sphere_center + m_hemi * d^2
        //   where I_sphere_center = 2/5 * m_hemi * r^2 (full sphere about center).
        let iy_hemi = 0.4 * m_hemi * r2;
        let d = half_height + 3.0 * radius / 8.0;
        let ix_hemi = 0.4 * m_hemi * r2 + m_hemi * d * d;

        let ix = ix_cyl + ix_hemi;
        let iy = iy_cyl + iy_hemi;
        Self::from_principal_moments(mass, ix, iy, ix) // Iz == Ix (Y-axis symmetric)
    }
}

/// Velocity component for dynamic bodies.
#[derive(Clone, Copy, Debug, Default, Reflect)]
pub struct Velocity {
    /// Linear velocity in world units per second.
    pub linear: Vec3,
    /// Angular velocity in radians per second (axis-angle representation).
    pub angular: Vec3,
}

/// Accumulated external force and torque applied to a physics body.
///
/// Gameplay code, wind, thrusters, and other systems write force/torque here;
/// the physics step reads it, applies `F = m*a` integration, then optionally
/// clears it (one-shot mode) so the caller does not need to reset it each frame.
#[derive(Clone, Copy, Debug, Default, Reflect)]
pub struct ExternalForce {
    /// World-space force in Newtons.
    pub force: Vec3,
    /// World-space torque in Newton-meters.
    pub torque: Vec3,
    /// If `true`, force and torque persist across physics steps (caller clears
    /// manually). If `false`, they are zeroed after each step (one-shot impulse).
    pub persistent: bool,
}

/// Gravity component (overrides global gravity for this entity).
#[derive(Clone, Copy, Debug)]
pub struct Gravity(pub Vec3);

/// Marker component for sleeping (deactivated) physics bodies.
/// Sleeping bodies skip gravity and integration until woken by a collision.
#[derive(Clone, Copy, Debug)]
pub struct Sleeping;

/// Velocity threshold below which a body is put to sleep.
pub const SLEEP_THRESHOLD: f32 = 0.05;

/// Collider shape.
#[derive(Clone, Debug, Reflect)]
pub enum ColliderShape {
    /// Axis-aligned bounding box (half-extents).
    Aabb { hx: f32, hy: f32, hz: f32 },
    /// Sphere.
    Sphere { radius: f32 },
    /// Capsule (two hemispheres connected by a cylinder, aligned to Y axis).
    /// `radius` is the hemisphere/cylinder radius, `half_height` is the
    /// half-length of the cylinder segment (total height = 2*(half_height + radius)).
    Capsule { radius: f32, half_height: f32 },
}

/// ECS component defining collision shape and filtering.
///
/// `layer` is a bitmask indicating which layer(s) this collider belongs to.
/// `mask` is a bitmask indicating which layers this collider can collide with.
/// Two colliders A and B collide only if `(a.layer & b.mask) != 0 && (b.layer & a.mask) != 0`.
#[derive(Clone, Debug, Reflect)]
pub struct Collider {
    pub shape: ColliderShape,
    pub restitution: f32,
    pub friction: f32,
    /// Collision layer bits this collider belongs to. Default: 1 (layer 0 only).
    pub layer: u32,
    /// Collision mask bits indicating which layers this collider interacts with.
    /// Default: `u32::MAX` (all layers).
    pub mask: u32,
}

impl Collider {
    /// Create an axis-aligned bounding box collider with the given half-extents.
    pub fn aabb(hx: f32, hy: f32, hz: f32) -> Self {
        Self {
            shape: ColliderShape::Aabb { hx, hy, hz },
            restitution: 0.3,
            friction: 0.5,
            layer: 1,
            mask: u32::MAX,
        }
    }

    /// Create a sphere collider with the given radius.
    pub fn sphere(radius: f32) -> Self {
        Self {
            shape: ColliderShape::Sphere { radius },
            restitution: 0.3,
            friction: 0.5,
            layer: 1,
            mask: u32::MAX,
        }
    }

    /// Y-axis capsule (good for character controllers).
    pub fn capsule(radius: f32, half_height: f32) -> Self {
        Self {
            shape: ColliderShape::Capsule {
                radius,
                half_height,
            },
            restitution: 0.3,
            friction: 0.5,
            layer: 1,
            mask: u32::MAX,
        }
    }

    /// Set the bounciness (0.0 = no bounce, 1.0 = perfectly elastic).
    pub fn with_restitution(mut self, r: f32) -> Self {
        self.restitution = r;
        self
    }

    /// Set the friction coefficient (0.0 = frictionless, 1.0 = high friction).
    pub fn with_friction(mut self, f: f32) -> Self {
        self.friction = f;
        self
    }

    /// Set the collision layer bits.
    pub fn with_layer(mut self, layer: u32) -> Self {
        self.layer = layer;
        self
    }

    /// Set the collision mask bits.
    pub fn with_mask(mut self, mask: u32) -> Self {
        self.mask = mask;
        self
    }
}

/// Pre-cached collider shape, updated only when `Collider` changes.
///
/// This component eliminates per-frame shape cloning in the physics solver.
/// The `sync_cached_shapes` system detects `Changed<Collider>` and writes
/// the shape here once; the solver then reads this cached copy every tick.
#[derive(Clone, Debug)]
pub struct CachedColliderShape(pub ColliderShape);

/// Returns true if two colliders should interact based on their layer/mask.
pub fn layers_interact(layer_a: u32, mask_a: u32, layer_b: u32, mask_b: u32) -> bool {
    (layer_a & mask_b) != 0 && (layer_b & mask_a) != 0
}

/// Event emitted when two colliders overlap during collision resolution.
#[derive(Clone, Debug)]
pub struct CollisionEvent {
    /// First colliding entity.
    pub entity_a: Entity,
    /// Second colliding entity.
    pub entity_b: Entity,
    /// Contact normal (from A toward B).
    pub normal: Vec3,
    /// Penetration depth (positive when overlapping).
    pub penetration: f32,
    /// World-space contact point.
    pub contact_point: Vec3,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mass_new_produces_splat_tensor() {
        let m = Mass::new(2.0, 0.5);
        let expected_inv = 1.0 / 0.5;
        assert!((m.inverse_inertia_tensor.x - expected_inv).abs() < 1e-6);
        assert!((m.inverse_inertia_tensor.y - expected_inv).abs() < 1e-6);
        assert!((m.inverse_inertia_tensor.z - expected_inv).abs() < 1e-6);
    }

    #[test]
    fn mass_infinite_produces_zero_tensor() {
        let m = Mass::infinite();
        assert_eq!(m.inverse_inertia_tensor.x, 0.0);
        assert_eq!(m.inverse_inertia_tensor.y, 0.0);
        assert_eq!(m.inverse_inertia_tensor.z, 0.0);
    }

    #[test]
    fn mass_from_sphere() {
        let m = Mass::from_sphere(5.0, 2.0);
        let expected_i = 0.4 * 5.0 * 4.0; // 2/5 * m * r^2
        assert!((m.inertia - expected_i).abs() < 1e-6);
        let expected_inv = 1.0 / expected_i;
        assert!((m.inverse_inertia_tensor.x - expected_inv).abs() < 1e-6);
        assert!((m.inverse_inertia_tensor.y - expected_inv).abs() < 1e-6);
        assert!((m.inverse_inertia_tensor.z - expected_inv).abs() < 1e-6);
    }

    #[test]
    fn mass_from_aabb() {
        let m = Mass::from_aabb(12.0, 1.0, 2.0, 3.0);
        // Ix = m*(hy^2+hz^2)/3 = 12*(4+9)/3 = 52
        let ix = 12.0 * (4.0 + 9.0) / 3.0;
        // Iy = m*(hx^2+hz^2)/3 = 12*(1+9)/3 = 40
        let iy = 12.0 * (1.0 + 9.0) / 3.0;
        // Iz = m*(hx^2+hy^2)/3 = 12*(1+4)/3 = 20
        let iz = 12.0 * (1.0 + 4.0) / 3.0;
        assert!((m.inverse_inertia_tensor.x - 1.0 / ix).abs() < 1e-6);
        assert!((m.inverse_inertia_tensor.y - 1.0 / iy).abs() < 1e-6);
        assert!((m.inverse_inertia_tensor.z - 1.0 / iz).abs() < 1e-6);
    }

    #[test]
    fn mass_from_capsule() {
        let m = Mass::from_capsule(10.0, 0.5, 1.0);
        // Basic sanity: mass is correct, all tensor components are finite and positive.
        assert!((m.mass - 10.0).abs() < 1e-6);
        assert!(m.inverse_inertia_tensor.x > 0.0);
        assert!(m.inverse_inertia_tensor.y > 0.0);
        assert!(m.inverse_inertia_tensor.z > 0.0);
        // Capsule is Y-axis symmetric: Ix == Iz.
        assert!(
            (m.inverse_inertia_tensor.x - m.inverse_inertia_tensor.z).abs() < 1e-6,
            "Capsule Ix should equal Iz"
        );
        // Iy (spin around Y) should be larger inverse (smaller moment) than Ix
        // because mass is concentrated closer to the Y axis.
        assert!(
            m.inverse_inertia_tensor.y > m.inverse_inertia_tensor.x,
            "Capsule should have smaller Iy moment than Ix"
        );
    }
}
