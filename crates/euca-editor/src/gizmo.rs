use euca_math::Vec3;
use euca_physics::{Ray, raycast_aabb};
use euca_render::*;

/// Which axis the user is interacting with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
}

impl GizmoAxis {
    /// Unit direction vector for this axis.
    pub fn direction(self) -> Vec3 {
        match self {
            GizmoAxis::X => Vec3::X,
            GizmoAxis::Y => Vec3::Y,
            GizmoAxis::Z => Vec3::Z,
        }
    }
}

/// Active gizmo drag state.
#[derive(Clone, Debug)]
pub struct GizmoDrag {
    pub axis: GizmoAxis,
    pub entity_index: u32,
    /// World position of entity when drag started.
    pub start_position: Vec3,
    /// Point on the axis line where the mouse first grabbed.
    pub grab_point: Vec3,
}

/// Gizmo rendering and interaction state.
pub struct GizmoState {
    pub active_drag: Option<GizmoDrag>,
    /// Cube mesh handle (reused for shafts and tips).
    pub mesh: Option<MeshHandle>,
    /// Material handles: [X=red, Y=green, Z=blue].
    pub materials: [Option<MaterialHandle>; 3],
}

impl GizmoState {
    pub fn new() -> Self {
        Self {
            active_drag: None,
            mesh: None,
            materials: [None; 3],
        }
    }
}

impl Default for GizmoState {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize gizmo GPU resources (materials). Call once during setup.
pub fn init_gizmo(renderer: &mut Renderer, gpu: &GpuContext, cube_mesh: MeshHandle) -> GizmoState {
    let red = renderer.upload_material(gpu, &Material::new([1.0, 0.15, 0.15, 1.0], 0.0, 1.0));
    let green = renderer.upload_material(gpu, &Material::new([0.15, 1.0, 0.15, 1.0], 0.0, 1.0));
    let blue = renderer.upload_material(gpu, &Material::new([0.15, 0.15, 1.0, 1.0], 0.0, 1.0));

    GizmoState {
        active_drag: None,
        mesh: Some(cube_mesh),
        materials: [Some(red), Some(green), Some(blue)],
    }
}

/// Generate DrawCommands for the gizmo at the given entity position.
/// Returns 6 commands: 3 shafts + 3 tips.
pub fn gizmo_draw_commands(
    entity_pos: Vec3,
    camera_eye: Vec3,
    state: &GizmoState,
) -> Vec<DrawCommand> {
    let mesh = match state.mesh {
        Some(m) => m,
        None => return Vec::new(),
    };

    let cam_dist = (camera_eye - entity_pos).length();
    let scale = cam_dist * 0.08; // gizmo length
    let shaft_thick = cam_dist * 0.004; // shaft thickness
    let tip_size = cam_dist * 0.015; // tip cube size

    let axes = [(GizmoAxis::X, 0), (GizmoAxis::Y, 1), (GizmoAxis::Z, 2)];

    let mut cmds = Vec::with_capacity(6);

    for (axis, mat_idx) in &axes {
        let mat = match state.materials[*mat_idx] {
            Some(m) => m,
            None => continue,
        };
        let dir = axis.direction();

        // Shaft: thin cube along the axis
        let shaft_center = entity_pos + dir * (scale * 0.5);
        let shaft_scale = match axis {
            GizmoAxis::X => Vec3::new(scale, shaft_thick, shaft_thick),
            GizmoAxis::Y => Vec3::new(shaft_thick, scale, shaft_thick),
            GizmoAxis::Z => Vec3::new(shaft_thick, shaft_thick, scale),
        };
        let shaft_mat = euca_math::Mat4::from_scale_rotation_translation(
            shaft_scale,
            euca_math::Quat::IDENTITY,
            shaft_center,
        );
        cmds.push(DrawCommand {
            mesh,
            material: mat,
            model_matrix: shaft_mat,
        });

        // Tip: small cube at the end of the axis
        let tip_center = entity_pos + dir * scale;
        let tip_scale = Vec3::new(tip_size, tip_size, tip_size);
        let tip_mat = euca_math::Mat4::from_scale_rotation_translation(
            tip_scale,
            euca_math::Quat::IDENTITY,
            tip_center,
        );
        cmds.push(DrawCommand {
            mesh,
            material: mat,
            model_matrix: tip_mat,
        });
    }

    cmds
}

/// Test if a ray hits any gizmo axis. Returns the closest axis + hit distance.
/// Gizmo picking uses fattened AABBs for comfortable clicking.
pub fn pick_gizmo_axis(ray: &Ray, entity_pos: Vec3, camera_eye: Vec3) -> Option<(GizmoAxis, f32)> {
    let cam_dist = (camera_eye - entity_pos).length();
    let scale = cam_dist * 0.08;
    let pick_radius = cam_dist * 0.012; // slightly wider than visual for easier clicking

    let axes = [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z];
    let mut closest: Option<(GizmoAxis, f32)> = None;

    for axis in &axes {
        let dir = axis.direction();
        let center = entity_pos + dir * (scale * 0.5);

        // Half-extents for the pickable AABB
        let (hx, hy, hz) = match axis {
            GizmoAxis::X => (scale * 0.5, pick_radius, pick_radius),
            GizmoAxis::Y => (pick_radius, scale * 0.5, pick_radius),
            GizmoAxis::Z => (pick_radius, pick_radius, scale * 0.5),
        };

        if let Some(hit) = raycast_aabb(ray, center, hx, hy, hz)
            && hit.t >= 0.0
            && (closest.is_none() || hit.t < closest.unwrap().1)
        {
            closest = Some((*axis, hit.t));
        }
    }

    closest
}

/// Compute the new entity position during a gizmo drag.
/// Projects the mouse ray onto the drag axis and returns the updated position.
pub fn update_gizmo_drag(drag: &GizmoDrag, ray_origin: Vec3, ray_dir: Vec3) -> Vec3 {
    let axis_dir = drag.axis.direction();

    // Find current closest point on the axis line to the mouse ray
    let t = Vec3::closest_line_param(drag.start_position, axis_dir, ray_origin, ray_dir);

    // Find the grab offset (how far along the axis the initial grab was)
    let grab_axis_t = axis_dir.dot(drag.grab_point - drag.start_position);

    // New position = start + axis movement delta
    drag.start_position + axis_dir * (t - grab_axis_t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_directions() {
        assert_eq!(GizmoAxis::X.direction(), Vec3::X);
        assert_eq!(GizmoAxis::Y.direction(), Vec3::Y);
        assert_eq!(GizmoAxis::Z.direction(), Vec3::Z);
    }

    #[test]
    fn gizmo_state_default() {
        let state = GizmoState::new();
        assert!(state.active_drag.is_none());
        assert!(state.mesh.is_none());
    }
}
