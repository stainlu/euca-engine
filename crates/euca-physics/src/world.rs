use euca_math::Vec3;

/// Physics configuration resource.
#[derive(Clone, Debug)]
pub struct PhysicsConfig {
    pub gravity: Vec3,
    pub fixed_dt: f32,
}

impl PhysicsConfig {
    pub fn new() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            fixed_dt: 1.0 / 60.0,
        }
    }

    pub fn with_gravity(mut self, gravity: Vec3) -> Self {
        self.gravity = gravity;
        self
    }
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self::new()
    }
}
