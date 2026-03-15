use euca_math::Vec3;

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
#[derive(Clone, Copy, Debug)]
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
#[derive(Clone, Copy, Debug, Default)]
pub struct Velocity {
    pub linear: Vec3,
    pub angular: Vec3,
}

/// Gravity component (overrides global gravity for this entity).
#[derive(Clone, Copy, Debug)]
pub struct Gravity(pub Vec3);

/// Collider shape.
#[derive(Clone, Debug)]
pub enum ColliderShape {
    /// Axis-aligned bounding box (half-extents).
    Aabb { hx: f32, hy: f32, hz: f32 },
    /// Sphere.
    Sphere { radius: f32 },
}

/// ECS component defining collision shape.
#[derive(Clone, Debug)]
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

    pub fn with_restitution(mut self, r: f32) -> Self {
        self.restitution = r;
        self
    }

    pub fn with_friction(mut self, f: f32) -> Self {
        self.friction = f;
        self
    }
}
