use euca_math::{Mat4, Vec3, Vec4};

/// Halton low-discrepancy sequence value for the given index and base.
/// Used to generate sub-pixel jitter patterns for TAA.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    let inv_base = 1.0 / base as f32;
    while index > 0 {
        f *= inv_base;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

/// Camera component defining the viewer's position and projection.
///
/// Uses a left-handed coordinate system (+X right, +Y up, +Z forward).
/// Supports both perspective and orthographic projection modes, TAA jitter,
/// and view-preset helpers for common viewpoints.
#[derive(Clone, Debug)]
pub struct Camera {
    /// Eye position in world space.
    pub eye: Vec3,
    /// Look-at target in world space.
    pub target: Vec3,
    /// Up direction.
    pub up: Vec3,
    /// Vertical field of view in radians (perspective mode).
    pub fov_y: f32,
    /// Near clip plane.
    pub near: f32,
    /// Far clip plane.
    pub far: f32,
    /// Use orthographic projection instead of perspective.
    pub orthographic: bool,
    /// Half-extent of the orthographic view in world units.
    pub ortho_size: f32,
    /// Sub-pixel jitter offset in clip space for TAA (set each frame).
    pub jitter: [f32; 2],
    /// Previous frame's view-projection matrix for TAA reprojection.
    pub prev_view_proj: Option<Mat4>,
}

impl Camera {
    /// Create a perspective camera at `eye` looking toward `target`.
    ///
    /// Defaults: 45-degree vertical FOV, near = 0.1, far = 1000, Y-up.
    pub fn new(eye: Vec3, target: Vec3) -> Self {
        Self {
            eye,
            target,
            up: Vec3::new(0.0, 1.0, 0.0),
            fov_y: std::f32::consts::FRAC_PI_4, // 45 degrees
            near: 0.1,
            far: 1000.0,
            orthographic: false,
            ortho_size: 10.0,
            jitter: [0.0, 0.0],
            prev_view_proj: None,
        }
    }

    /// Build the view matrix.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_lh(self.eye, self.target, self.up)
    }

    /// Build the projection matrix for the given aspect ratio.
    pub fn projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        if self.orthographic {
            let half_w = self.ortho_size * aspect_ratio;
            let half_h = self.ortho_size;
            Mat4::orthographic_lh(-half_w, half_w, -half_h, half_h, self.near, self.far)
        } else {
            Mat4::perspective_lh(self.fov_y, aspect_ratio, self.near, self.far)
        }
    }

    /// Combined view-projection matrix.
    pub fn view_projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        self.projection_matrix(aspect_ratio) * self.view_matrix()
    }

    /// Build a jittered view-projection matrix for TAA.
    ///
    /// Applies a sub-pixel offset (Halton 2,3 sequence) to the projection matrix
    /// and stores the current VP as `prev_view_proj` for next frame's reprojection.
    pub fn jittered_view_projection_matrix(
        &mut self,
        aspect_ratio: f32,
        frame_index: u32,
        screen_w: f32,
        screen_h: f32,
    ) -> Mat4 {
        // Halton(2,3) sequence for sub-pixel jitter
        let idx = (frame_index % 16) + 1;
        let jx = halton(idx, 2) - 0.5; // center around 0 → [-0.5, 0.5]
        let jy = halton(idx, 3) - 0.5;

        // Convert pixel-space jitter to clip-space offset
        self.jitter = [jx / screen_w * 2.0, jy / screen_h * 2.0];

        // Apply jitter to projection (offset in clip space, column-major: cols[col][row])
        let mut proj = self.projection_matrix(aspect_ratio);
        proj.cols[2][0] += self.jitter[0];
        proj.cols[2][1] += self.jitter[1];

        let jittered_vp = proj * self.view_matrix();

        // Store the jittered VP for next frame's TAA reprojection
        self.prev_view_proj = Some(jittered_vp);

        jittered_vp
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

/// A frustum defined by 6 planes (left, right, bottom, top, near, far).
/// Each plane is `(nx, ny, nz, d)` where `nx*x + ny*y + nz*z + d >= 0` is inside.
#[derive(Clone, Debug)]
pub struct Frustum {
    pub planes: [[f32; 4]; 6],
}

impl Frustum {
    /// Extract frustum planes from a view-projection matrix.
    /// Uses the Gribb/Hartmann method.
    pub fn from_view_projection(vp: &Mat4) -> Self {
        let m = vp.to_cols_array_2d();
        // Row extraction from column-major matrix
        let row = |r: usize| -> [f32; 4] { [m[0][r], m[1][r], m[2][r], m[3][r]] };
        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);

        let mut planes = [[0.0f32; 4]; 6];
        // Left:   row3 + row0
        // Right:  row3 - row0
        // Bottom: row3 + row1
        // Top:    row3 - row1
        // Near:   row3 + row2
        // Far:    row3 - row2
        for i in 0..4 {
            planes[0][i] = r3[i] + r0[i]; // left
            planes[1][i] = r3[i] - r0[i]; // right
            planes[2][i] = r3[i] + r1[i]; // bottom
            planes[3][i] = r3[i] - r1[i]; // top
            planes[4][i] = r3[i] + r2[i]; // near
            planes[5][i] = r3[i] - r2[i]; // far
        }

        // Normalize each plane
        for plane in &mut planes {
            let len = (plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]).sqrt();
            if len > 1e-8 {
                plane[0] /= len;
                plane[1] /= len;
                plane[2] /= len;
                plane[3] /= len;
            }
        }

        Self { planes }
    }

    /// Test if an AABB (center + half-extents) is at least partially inside the frustum.
    pub fn intersects_aabb(&self, center: Vec3, half_extents: Vec3) -> bool {
        for plane in &self.planes {
            let nx = plane[0];
            let ny = plane[1];
            let nz = plane[2];
            let d = plane[3];

            // Compute the effective radius of the AABB projected onto the plane normal
            let r =
                half_extents.x * nx.abs() + half_extents.y * ny.abs() + half_extents.z * nz.abs();

            // Distance from center to plane
            let dist = nx * center.x + ny * center.y + nz * center.z + d;

            // If the AABB is entirely outside this plane, it's outside the frustum
            if dist < -r {
                return false;
            }
        }
        true
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(Vec3::new(0.0, 2.0, -5.0), Vec3::ZERO)
    }
}

/// Standard view presets for agent use.
impl Camera {
    /// Apply a named view preset. Returns true if the name was recognized.
    pub fn apply_preset(&mut self, name: &str) -> bool {
        match name {
            "top" => {
                self.eye = Vec3::new(0.0, 20.0, 0.001); // slight offset to avoid degenerate up
                self.target = Vec3::ZERO;
                self.up = Vec3::new(0.0, 0.0, -1.0); // Z-forward when looking down
                self.orthographic = true;
                self.ortho_size = 12.0;
                true
            }
            "front" => {
                self.eye = Vec3::new(0.0, 2.0, 20.0);
                self.target = Vec3::new(0.0, 2.0, 0.0);
                self.up = Vec3::Y;
                self.orthographic = true;
                self.ortho_size = 8.0;
                true
            }
            "back" => {
                self.eye = Vec3::new(0.0, 2.0, -20.0);
                self.target = Vec3::new(0.0, 2.0, 0.0);
                self.up = Vec3::Y;
                self.orthographic = true;
                self.ortho_size = 8.0;
                true
            }
            "right" => {
                self.eye = Vec3::new(20.0, 2.0, 0.0);
                self.target = Vec3::new(0.0, 2.0, 0.0);
                self.up = Vec3::Y;
                self.orthographic = true;
                self.ortho_size = 8.0;
                true
            }
            "left" => {
                self.eye = Vec3::new(-20.0, 2.0, 0.0);
                self.target = Vec3::new(0.0, 2.0, 0.0);
                self.up = Vec3::Y;
                self.orthographic = true;
                self.ortho_size = 8.0;
                true
            }
            "perspective" => {
                self.eye = Vec3::new(8.0, 6.0, 8.0);
                self.target = Vec3::new(0.0, 1.0, 0.0);
                self.up = Vec3::Y;
                self.orthographic = false;
                true
            }
            _ => false,
        }
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

    #[test]
    fn prev_view_proj_stores_jittered() {
        let mut cam = Camera::default();
        let aspect = 16.0 / 9.0;

        let jittered_vp = cam.jittered_view_projection_matrix(aspect, 1, 1920.0, 1080.0);
        let stored = cam.prev_view_proj.expect("prev_view_proj should be set");

        // The stored matrix must equal the jittered VP, not the unjittered one
        assert_eq!(stored, jittered_vp);

        let unjittered_vp = cam.view_projection_matrix(aspect);
        assert_ne!(
            stored, unjittered_vp,
            "prev_view_proj must differ from the unjittered VP"
        );
    }
}
