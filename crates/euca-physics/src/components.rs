use euca_ecs::Entity;
use euca_math::Vec3;
use euca_reflect::Reflect;

/// Rigid body type.
#[derive(Clone, Copy, Debug, PartialEq)]
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
    pub fn dynamic() -> Self {
        Self {
            body_type: RigidBodyType::Dynamic,
        }
    }
    pub fn fixed() -> Self {
        Self {
            body_type: RigidBodyType::Static,
        }
    }
    pub fn kinematic() -> Self {
        Self {
            body_type: RigidBodyType::Kinematic,
        }
    }
}

/// Mass and inertia properties for physics bodies.
///
/// Static bodies should use `Mass::infinite()` (zero inverse mass).
/// Dynamic bodies use `Mass::new(mass, inertia)`.
#[derive(Clone, Copy, Debug, Reflect)]
pub struct Mass {
    pub mass: f32,
    pub inverse_mass: f32,
    pub inertia: f32,
    pub inverse_inertia: f32,
}

impl Mass {
    /// Create a mass component with the given mass and scalar inertia.
    /// For a solid sphere of uniform density: `inertia = 0.4 * mass * radius^2`.
    /// For a solid box: `inertia = mass * (w^2 + h^2) / 12`.
    pub fn new(mass: f32, inertia: f32) -> Self {
        assert!(
            mass > 0.0,
            "Mass must be positive; use Mass::infinite() for static bodies"
        );
        assert!(inertia > 0.0, "Inertia must be positive");
        Self {
            mass,
            inverse_mass: 1.0 / mass,
            inertia,
            inverse_inertia: 1.0 / inertia,
        }
    }

    /// Infinite mass (for static/kinematic bodies). Zero inverse mass means
    /// the body is immovable in mass-weighted calculations.
    pub fn infinite() -> Self {
        Self {
            mass: f32::INFINITY,
            inverse_mass: 0.0,
            inertia: f32::INFINITY,
            inverse_inertia: 0.0,
        }
    }

    /// Default mass for a 1-unit cube of density 1.
    pub fn default_dynamic() -> Self {
        Self::new(1.0, 1.0 / 6.0)
    }
}

/// Velocity component for dynamic bodies.
#[derive(Clone, Copy, Debug, Default, Reflect)]
pub struct Velocity {
    pub linear: Vec3,
    pub angular: Vec3,
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
#[derive(Clone, Debug)]
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
    pub fn aabb(hx: f32, hy: f32, hz: f32) -> Self {
        Self {
            shape: ColliderShape::Aabb { hx, hy, hz },
            restitution: 0.3,
            friction: 0.5,
            layer: 1,
            mask: u32::MAX,
        }
    }

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

    pub fn with_restitution(mut self, r: f32) -> Self {
        self.restitution = r;
        self
    }

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

/// Returns true if two colliders should interact based on their layer/mask.
pub fn layers_interact(layer_a: u32, mask_a: u32, layer_b: u32, mask_b: u32) -> bool {
    (layer_a & mask_b) != 0 && (layer_b & mask_a) != 0
}

/// Event emitted when two colliders overlap during collision resolution.
#[derive(Clone, Debug)]
pub struct CollisionEvent {
    pub entity_a: Entity,
    pub entity_b: Entity,
    /// Contact normal (from A toward B).
    pub normal: Vec3,
    /// Penetration depth (positive when overlapping).
    pub penetration: f32,
}
