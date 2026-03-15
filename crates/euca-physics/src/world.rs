use std::collections::HashMap;
use euca_ecs::Entity;
use rapier3d::prelude::*;

/// Holds all Rapier3D simulation state as an ECS resource.
pub struct PhysicsWorld {
    pub gravity: rapier3d::math::Vec3,
    pub integration_parameters: IntegrationParameters,
    pub(crate) pipeline: PhysicsPipeline,
    pub(crate) island_manager: IslandManager,
    pub(crate) broad_phase: DefaultBroadPhase,
    pub(crate) narrow_phase: NarrowPhase,
    pub(crate) bodies: RigidBodySet,
    pub(crate) colliders: ColliderSet,
    pub(crate) impulse_joints: ImpulseJointSet,
    pub(crate) multibody_joints: MultibodyJointSet,
    pub(crate) ccd_solver: CCDSolver,
    pub(crate) entity_to_body: HashMap<Entity, RigidBodyHandle>,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            gravity: rapier3d::math::Vec3::new(0.0, -9.81, 0.0),
            integration_parameters: IntegrationParameters::default(),
            pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            entity_to_body: HashMap::new(),
        }
    }

    pub fn with_gravity(mut self, x: f32, y: f32, z: f32) -> Self {
        self.gravity = rapier3d::math::Vec3::new(x, y, z);
        self
    }

    pub(crate) fn step(&mut self) {
        self.pipeline.step(
            self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            &(),
            &(),
        );
    }
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}
