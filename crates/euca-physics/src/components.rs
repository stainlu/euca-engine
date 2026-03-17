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

/// ECS component defining collision shape.
#[derive(Clone, Debug, Reflect)]
pub struct Collider {
    pub shape: ColliderShape,
    pub restitution: f32,
    pub friction: f32,
}

impl Collider {
    pub fn aabb(hx: f32, hy: f32, hz: f32) -> Self {
        Self {
            shape: ColliderShape::Aabb { hx, hy, hz },
            restitution: 0.3,
            friction: 0.5,
        }
    }

    pub fn sphere(radius: f32) -> Self {
        Self {
            shape: ColliderShape::Sphere { radius },
            restitution: 0.3,
            friction: 0.5,
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
}
