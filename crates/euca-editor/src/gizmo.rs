use euca_math::{Quat, Vec3};
use euca_physics::{Ray, raycast_aabb};
use euca_render::*;

/// Active gizmo manipulation mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GizmoMode {
    #[default]
    Translate,
    Rotate,
    Scale,
}

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
    pub mode: GizmoMode,
    pub axis: GizmoAxis,
    pub entity_index: u32,
    /// World position of entity when drag started.
    pub start_position: Vec3,
    /// Point on the axis line where the mouse first grabbed.
    pub grab_point: Vec3,
    /// Entity rotation when drag started (for rotate mode).
    pub start_rotation: Quat,
    /// Entity scale when drag started (for scale mode).
    pub start_scale: Vec3,
    /// Accumulated angle during rotation drag (radians).
    pub accumulated_angle: f32,
}

/// Gizmo rendering and interaction state.
pub struct GizmoState {
    pub active_drag: Option<GizmoDrag>,
    /// Current manipulation mode.
    pub mode: GizmoMode,
    /// Cube mesh handle (reused for shafts, tips, and ring segments).
    pub mesh: Option<MeshHandle>,
    /// Material handles: [X=red, Y=green, Z=blue].
    pub materials: [Option<MaterialHandle>; 3],
}

impl GizmoState {
    pub fn new() -> Self {
        Self {
            active_drag: None,
            mode: GizmoMode::Translate,
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
        mode: GizmoMode::Translate,
        mesh: Some(cube_mesh),
        materials: [Some(red), Some(green), Some(blue)],
    }
}

/// Number of segments per rotation ring.
const RING_SEGMENTS: usize = 24;

/// Generate DrawCommands for the gizmo at the given entity position.
/// Output depends on `state.mode`: translate (shafts+tips), rotate (rings),
/// or scale (shafts+cubes).
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
    let scale = cam_dist * 0.08;
    let shaft_thick = cam_dist * 0.004;
    let tip_size = cam_dist * 0.015;

    let axes = [(GizmoAxis::X, 0), (GizmoAxis::Y, 1), (GizmoAxis::Z, 2)];

    match state.mode {
        GizmoMode::Translate => {
            let mut cmds = Vec::with_capacity(6);
            for (axis, mat_idx) in &axes {
                let mat = match state.materials[*mat_idx] {
                    Some(m) => m,
                    None => continue,
                };
                let dir = axis.direction();

                // Shaft
                let shaft_center = entity_pos + dir * (scale * 0.5);
                let shaft_scale = match axis {
                    GizmoAxis::X => Vec3::new(scale, shaft_thick, shaft_thick),
                    GizmoAxis::Y => Vec3::new(shaft_thick, scale, shaft_thick),
                    GizmoAxis::Z => Vec3::new(shaft_thick, shaft_thick, scale),
                };
                cmds.push(DrawCommand {
                    mesh,
                    material: mat,
                    model_matrix: euca_math::Mat4::from_scale_rotation_translation(
                        shaft_scale,
                        Quat::IDENTITY,
                        shaft_center,
                    ),
                    aabb: None,
                    is_water: false,
                });

                // Arrow tip
                let tip_center = entity_pos + dir * scale;
                cmds.push(DrawCommand {
                    mesh,
                    material: mat,
                    model_matrix: euca_math::Mat4::from_scale_rotation_translation(
                        Vec3::new(tip_size, tip_size, tip_size),
                        Quat::IDENTITY,
                        tip_center,
                    ),
                    aabb: None,
                    is_water: false,
                });
            }
            cmds
        }

        GizmoMode::Rotate => {
            let seg_thick = cam_dist * 0.003;
            let seg_len = scale * (std::f32::consts::TAU / RING_SEGMENTS as f32);
            let radius = scale * 0.8;
            let mut cmds = Vec::with_capacity(RING_SEGMENTS * 3);

            for (axis, mat_idx) in &axes {
                let mat = match state.materials[*mat_idx] {
                    Some(m) => m,
                    None => continue,
                };

                for i in 0..RING_SEGMENTS {
                    let angle = (i as f32 + 0.5) * std::f32::consts::TAU / RING_SEGMENTS as f32;
                    let (sin_a, cos_a) = angle.sin_cos();

                    // Position on the ring circle
                    let pos = match axis {
                        GizmoAxis::X => entity_pos + Vec3::new(0.0, cos_a * radius, sin_a * radius),
                        GizmoAxis::Y => entity_pos + Vec3::new(cos_a * radius, 0.0, sin_a * radius),
                        GizmoAxis::Z => entity_pos + Vec3::new(cos_a * radius, sin_a * radius, 0.0),
                    };

                    // Tangent direction (for scaling the segment along the ring)
                    let seg_scale = match axis {
                        GizmoAxis::X => Vec3::new(seg_thick, seg_len, seg_thick),
                        GizmoAxis::Y => Vec3::new(seg_len, seg_thick, seg_thick),
                        GizmoAxis::Z => Vec3::new(seg_thick, seg_len, seg_thick),
                    };

                    cmds.push(DrawCommand {
                        mesh,
                        material: mat,
                        model_matrix: euca_math::Mat4::from_scale_rotation_translation(
                            seg_scale,
                            Quat::IDENTITY,
                            pos,
                        ),
                        aabb: None,
                        is_water: false,
                    });
                }
            }
            cmds
        }

        GizmoMode::Scale => {
            let mut cmds = Vec::with_capacity(6);
            let cube_size = cam_dist * 0.012;

            for (axis, mat_idx) in &axes {
                let mat = match state.materials[*mat_idx] {
                    Some(m) => m,
                    None => continue,
                };
                let dir = axis.direction();

                // Shaft (same as translate)
                let shaft_center = entity_pos + dir * (scale * 0.5);
                let shaft_scale = match axis {
                    GizmoAxis::X => Vec3::new(scale, shaft_thick, shaft_thick),
                    GizmoAxis::Y => Vec3::new(shaft_thick, scale, shaft_thick),
                    GizmoAxis::Z => Vec3::new(shaft_thick, shaft_thick, scale),
                };
                cmds.push(DrawCommand {
                    mesh,
                    material: mat,
                    model_matrix: euca_math::Mat4::from_scale_rotation_translation(
                        shaft_scale,
                        Quat::IDENTITY,
                        shaft_center,
                    ),
                    aabb: None,
                    is_water: false,
                });

                // Cube endpoint (instead of arrow tip)
                let cube_center = entity_pos + dir * scale;
                cmds.push(DrawCommand {
                    mesh,
                    material: mat,
                    model_matrix: euca_math::Mat4::from_scale_rotation_translation(
                        Vec3::new(cube_size, cube_size, cube_size),
                        Quat::IDENTITY,
                        cube_center,
                    ),
                    aabb: None,
                    is_water: false,
                });
            }
            cmds
        }
    }
}

/// Test if a ray hits any gizmo axis. Returns the closest axis + hit distance.
/// Works for translate, rotate, and scale modes.
pub fn pick_gizmo_axis(
    ray: &Ray,
    entity_pos: Vec3,
    camera_eye: Vec3,
    mode: GizmoMode,
) -> Option<(GizmoAxis, f32)> {
    let cam_dist = (camera_eye - entity_pos).length();
    let scale = cam_dist * 0.08;
    let pick_radius = cam_dist * 0.012;

    let axes = [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z];
    let mut closest: Option<(GizmoAxis, f32)> = None;

    match mode {
        GizmoMode::Translate | GizmoMode::Scale => {
            // Pick against shaft AABBs (same geometry for both modes)
            for axis in &axes {
                let dir = axis.direction();
                let center = entity_pos + dir * (scale * 0.5);
                let (hx, hy, hz) = match axis {
                    GizmoAxis::X => (scale * 0.5, pick_radius, pick_radius),
                    GizmoAxis::Y => (pick_radius, scale * 0.5, pick_radius),
                    GizmoAxis::Z => (pick_radius, pick_radius, scale * 0.5),
                };
                if let Some(hit) = raycast_aabb(ray, center, hx, hy, hz)
                    && hit.t >= 0.0
                    && closest.as_ref().is_none_or(|(_, t)| hit.t < *t)
                {
                    closest = Some((*axis, hit.t));
                }
            }
        }
        GizmoMode::Rotate => {
            // Pick against ring segments (fat AABBs along the circle)
            let radius = scale * 0.8;
            let seg_pick = cam_dist * 0.015;
            for axis in &axes {
                for i in 0..RING_SEGMENTS {
                    let angle = (i as f32 + 0.5) * std::f32::consts::TAU / RING_SEGMENTS as f32;
                    let (sin_a, cos_a) = angle.sin_cos();
                    let pos = match axis {
                        GizmoAxis::X => entity_pos + Vec3::new(0.0, cos_a * radius, sin_a * radius),
                        GizmoAxis::Y => entity_pos + Vec3::new(cos_a * radius, 0.0, sin_a * radius),
                        GizmoAxis::Z => entity_pos + Vec3::new(cos_a * radius, sin_a * radius, 0.0),
                    };
                    if let Some(hit) = raycast_aabb(ray, pos, seg_pick, seg_pick, seg_pick)
                        && hit.t >= 0.0
                        && closest.as_ref().is_none_or(|(_, t)| hit.t < *t)
                    {
                        closest = Some((*axis, hit.t));
                    }
                }
            }
        }
    }

    closest
}

/// Compute the new entity position during a translate gizmo drag.
/// Projects the mouse ray onto the drag axis and returns the updated position.
pub fn update_translate_drag(drag: &GizmoDrag, ray_origin: Vec3, ray_dir: Vec3) -> Vec3 {
    let axis_dir = drag.axis.direction();
    let t = Vec3::closest_line_param(drag.start_position, axis_dir, ray_origin, ray_dir);
    let grab_axis_t = axis_dir.dot(drag.grab_point - drag.start_position);
    drag.start_position + axis_dir * (t - grab_axis_t)
}

/// Compute the new entity rotation during a rotate gizmo drag.
/// Projects the mouse ray onto the rotation plane and computes an angle delta.
pub fn update_rotate_drag(drag: &GizmoDrag, ray_origin: Vec3, ray_dir: Vec3) -> Quat {
    let axis_dir = drag.axis.direction();

    // Project mouse ray onto the plane perpendicular to the axis through the entity
    let denom = axis_dir.dot(ray_dir);
    if denom.abs() < 1e-6 {
        return drag.start_rotation;
    }
    let t = axis_dir.dot(drag.start_position - ray_origin) / denom;
    if t < 0.0 {
        return drag.start_rotation;
    }
    let hit = ray_origin + ray_dir * t;

    // Compute angle from entity center to hit point in the rotation plane
    let offset = hit - drag.start_position;
    let grab_offset = drag.grab_point - drag.start_position;

    // Get two basis vectors in the rotation plane
    let (basis_a, basis_b) = match drag.axis {
        GizmoAxis::X => (Vec3::Y, Vec3::Z),
        GizmoAxis::Y => (Vec3::X, Vec3::Z),
        GizmoAxis::Z => (Vec3::X, Vec3::Y),
    };

    let current_angle = offset.dot(basis_b).atan2(offset.dot(basis_a));
    let grab_angle = grab_offset.dot(basis_b).atan2(grab_offset.dot(basis_a));
    let delta_angle = current_angle - grab_angle;

    let rotation_delta = Quat::from_axis_angle(axis_dir, delta_angle);
    rotation_delta * drag.start_rotation
}

/// Compute the new entity scale during a scale gizmo drag.
/// Projects the mouse ray onto the drag axis and computes a scale factor.
pub fn update_scale_drag(drag: &GizmoDrag, ray_origin: Vec3, ray_dir: Vec3) -> Vec3 {
    let axis_dir = drag.axis.direction();
    let t = Vec3::closest_line_param(drag.start_position, axis_dir, ray_origin, ray_dir);
    let grab_axis_t = axis_dir.dot(drag.grab_point - drag.start_position);

    // Scale factor = ratio of current distance to grab distance along axis
    let scale_factor = if grab_axis_t.abs() > 1e-6 {
        (t / grab_axis_t).clamp(0.01, 100.0)
    } else {
        1.0
    };

    // Apply scale factor to the dragged axis only
    let mut new_scale = drag.start_scale;
    match drag.axis {
        GizmoAxis::X => new_scale.x *= scale_factor,
        GizmoAxis::Y => new_scale.y *= scale_factor,
        GizmoAxis::Z => new_scale.z *= scale_factor,
    }
    new_scale
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
