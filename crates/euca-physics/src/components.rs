/// The type of rigid body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RigidBodyType {
    /// Fully simulated, responsive to forces and gravity.
    Dynamic,
    /// Immovable, infinite mass.
    Static,
    /// Moved programmatically, not by physics forces.
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

/// Collider shape for physics simulation.
#[derive(Clone, Debug)]
pub enum ColliderShape {
    Cuboid { hx: f32, hy: f32, hz: f32 },
    Sphere { radius: f32 },
    Capsule { half_height: f32, radius: f32 },
}

/// ECS component defining the collision shape of a physics body.
#[derive(Clone, Debug)]
pub struct PhysicsCollider {
    pub shape: ColliderShape,
    pub restitution: f32,
    pub friction: f32,
}

impl PhysicsCollider {
    pub fn cuboid(hx: f32, hy: f32, hz: f32) -> Self {
        Self {
            shape: ColliderShape::Cuboid { hx, hy, hz },
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

/// Marker component: set by the physics system once a body is registered in Rapier.
/// Absence of this marker means the body needs initial registration.
#[derive(Clone, Copy, Debug)]
pub struct PhysicsRegistered;
