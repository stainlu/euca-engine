//! Reciprocal Velocity Obstacles (RVO) for local agent avoidance.
//!
//! Each agent computes a velocity that avoids collisions with nearby agents
//! while staying as close as possible to its preferred velocity (toward waypoint).

use euca_math::Vec3;

/// An agent for RVO computation.
#[derive(Clone, Debug)]
pub struct RvoAgent {
    pub position: Vec3,
    pub velocity: Vec3,
    pub preferred_velocity: Vec3,
    pub radius: f32,
    pub max_speed: f32,
}

/// Compute collision-avoiding velocities for all agents.
///
/// Each agent adjusts its velocity to avoid other agents while staying
/// close to its preferred velocity. The adjustment is reciprocal: both
/// agents in a pair share responsibility for avoidance.
pub fn compute_rvo_velocities(agents: &mut [RvoAgent], dt: f32) {
    let n = agents.len();
    if n < 2 {
        for agent in agents.iter_mut() {
            agent.velocity = agent.preferred_velocity;
            // Clamp to max speed even for single agents
            let speed = agent.velocity.length();
            if speed > agent.max_speed {
                agent.velocity = agent.velocity * (agent.max_speed / speed);
            }
            agent.velocity.y = 0.0;
        }
        return;
    }

    // Compute adjusted velocities for each agent
    let mut new_velocities = Vec::with_capacity(n);

    for i in 0..n {
        let mut adjustment = Vec3::ZERO;
        let mut num_adjustments = 0u32;

        for j in 0..n {
            if i == j {
                continue;
            }

            let rel_pos = agents[j].position - agents[i].position;
            let dist = rel_pos.length();
            let combined_radius = agents[i].radius + agents[j].radius;

            // Only avoid agents within interaction range (3x combined radius)
            if dist > combined_radius * 3.0 || dist < 1e-6 {
                continue;
            }

            // Time to collision if both maintain current preferred velocities
            let rel_vel = agents[i].preferred_velocity - agents[j].preferred_velocity;
            let time_horizon = 2.0 * dt.max(0.1); // look ahead ~2 frames minimum

            // Direction to push away from collision
            let push_dir = if dist < combined_radius {
                // Already overlapping — push directly apart
                let d = (agents[i].position - agents[j].position).normalize();
                d * 2.0 // strong push
            } else {
                // Approaching — deflect perpendicular to relative velocity
                let approach_speed = rel_vel.dot(rel_pos.normalize());
                if approach_speed <= 0.0 {
                    continue; // moving apart, no avoidance needed
                }

                let penetration_time = (dist - combined_radius) / approach_speed.max(0.01);
                if penetration_time > time_horizon {
                    continue; // collision is far in the future
                }

                // Perpendicular avoidance direction (avoid the other agent)
                let forward = rel_pos.normalize();
                let perp = Vec3::new(-forward.z, 0.0, forward.x); // 90° rotation on XZ

                // Choose the perpendicular that's more aligned with our velocity
                let side = if perp.dot(agents[i].preferred_velocity) > 0.0 {
                    perp
                } else {
                    -perp
                };

                // Strength: stronger when closer to collision
                let urgency = 1.0 - (penetration_time / time_horizon).clamp(0.0, 1.0);
                side * urgency * agents[i].max_speed * 0.5
            };

            adjustment = adjustment + push_dir;
            num_adjustments += 1;
        }

        // Blend preferred velocity with avoidance adjustment
        let mut new_vel = agents[i].preferred_velocity;
        if num_adjustments > 0 {
            // Reciprocal: each agent takes half the avoidance responsibility
            new_vel = new_vel + adjustment * (0.5 / num_adjustments as f32);
        }

        // Clamp to max speed
        let speed = new_vel.length();
        if speed > agents[i].max_speed {
            new_vel = new_vel * (agents[i].max_speed / speed);
        }

        // Keep movement on XZ plane
        new_vel.y = 0.0;

        new_velocities.push(new_vel);
    }

    // Apply computed velocities
    for (agent, new_vel) in agents.iter_mut().zip(new_velocities) {
        agent.velocity = new_vel;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_agent_keeps_preferred() {
        let mut agents = vec![RvoAgent {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            preferred_velocity: Vec3::new(1.0, 0.0, 0.0),
            radius: 0.5,
            max_speed: 5.0,
        }];
        compute_rvo_velocities(&mut agents, 0.016);
        assert!((agents[0].velocity.x - 1.0).abs() < 0.01);
    }

    #[test]
    fn head_on_collision_deflects() {
        let mut agents = vec![
            RvoAgent {
                position: Vec3::new(-1.0, 0.0, 0.0),
                velocity: Vec3::ZERO,
                preferred_velocity: Vec3::new(3.0, 0.0, 0.0),
                radius: 0.5,
                max_speed: 5.0,
            },
            RvoAgent {
                position: Vec3::new(1.0, 0.0, 0.0),
                velocity: Vec3::ZERO,
                preferred_velocity: Vec3::new(-3.0, 0.0, 0.0),
                radius: 0.5,
                max_speed: 5.0,
            },
        ];
        compute_rvo_velocities(&mut agents, 0.016);
        // Both agents should deflect — velocity should have a Z component
        assert!(
            agents[0].velocity.z.abs() > 0.01 || agents[1].velocity.z.abs() > 0.01,
            "Agents should deflect: a0={:?} a1={:?}",
            agents[0].velocity,
            agents[1].velocity
        );
    }

    #[test]
    fn distant_agents_no_avoidance() {
        let mut agents = vec![
            RvoAgent {
                position: Vec3::new(-100.0, 0.0, 0.0),
                velocity: Vec3::ZERO,
                preferred_velocity: Vec3::new(1.0, 0.0, 0.0),
                radius: 0.5,
                max_speed: 5.0,
            },
            RvoAgent {
                position: Vec3::new(100.0, 0.0, 0.0),
                velocity: Vec3::ZERO,
                preferred_velocity: Vec3::new(-1.0, 0.0, 0.0),
                radius: 0.5,
                max_speed: 5.0,
            },
        ];
        compute_rvo_velocities(&mut agents, 0.016);
        // Far apart — should keep preferred velocity
        assert!((agents[0].velocity.x - 1.0).abs() < 0.01);
        assert!((agents[1].velocity.x - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn speed_clamped_to_max() {
        let mut agents = vec![RvoAgent {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            preferred_velocity: Vec3::new(100.0, 0.0, 0.0),
            radius: 0.5,
            max_speed: 5.0,
        }];
        compute_rvo_velocities(&mut agents, 0.016);
        assert!(agents[0].velocity.length() <= 5.01);
    }
}
