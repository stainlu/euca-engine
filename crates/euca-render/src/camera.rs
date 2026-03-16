use euca_math::{Mat4, Vec3, Vec4};

/// Camera component with view + projection.
#[derive(Clone, Debug)]
pub struct Camera {
    /// Eye position in world space.
    pub eye: Vec3,
    /// Look-at target in world space.
    pub target: Vec3,
    /// Up direction.
    pub up: Vec3,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near clip plane.
    pub near: f32,
    /// Far clip plane.
    pub far: f32,
}

impl Camera {
    pub fn new(eye: Vec3, target: Vec3) -> Self {
        Self {
            eye,
            target,
            up: Vec3::new(0.0, 1.0, 0.0),
            fov_y: std::f32::consts::FRAC_PI_4, // 45 degrees
            near: 0.1,
            far: 1000.0,
        }
    }

    /// Build the view matrix.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_lh(self.eye, self.target, self.up)
    }

    /// Build the projection matrix for the given aspect ratio.
    pub fn projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        Mat4::perspective_lh(self.fov_y, aspect_ratio, self.near, self.far)
    }

    /// Combined view-projection matrix.
    pub fn view_projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        self.projection_matrix(aspect_ratio) * self.view_matrix()
    }

    /// Convert a screen pixel position to a world-space ray (origin, direction).
    /// Used for viewport click-to-select (raycasting from camera through mouse cursor).
    pub fn screen_to_ray(
        &self,
        pixel_x: f32,
        pixel_y: f32,
        screen_w: f32,
        screen_h: f32,
    ) -> (Vec3, Vec3) {
        let aspect = screen_w / screen_h;
        let inv_vp = self.view_projection_matrix(aspect).inverse();

        // Convert pixel → NDC [-1, 1] (Y is flipped: top=1, bottom=-1)
        let ndc_x = (pixel_x / screen_w) * 2.0 - 1.0;
        let ndc_y = 1.0 - (pixel_y / screen_h) * 2.0;

        // Unproject near and far points
        let near_ndc = Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far_ndc = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

        let near_world = inv_vp * near_ndc;
        let far_world = inv_vp * far_ndc;

        // Perspective divide
        let near_pos = Vec3::new(
            near_world.x / near_world.w,
            near_world.y / near_world.w,
            near_world.z / near_world.w,
        );
        let far_pos = Vec3::new(
            far_world.x / far_world.w,
            far_world.y / far_world.w,
            far_world.z / far_world.w,
        );

        let direction = (far_pos - near_pos).normalize();
        (near_pos, direction)
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(Vec3::new(0.0, 2.0, -5.0), Vec3::ZERO)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_projection_not_identity() {
        let cam = Camera::default();
        let vp = cam.view_projection_matrix(16.0 / 9.0);
        assert_ne!(vp, Mat4::IDENTITY);
    }

    #[test]
    fn aspect_ratio_affects_projection() {
        let cam = Camera::default();
        let wide = cam.projection_matrix(2.0);
        let narrow = cam.projection_matrix(0.5);
        assert_ne!(wide, narrow);
    }
}
