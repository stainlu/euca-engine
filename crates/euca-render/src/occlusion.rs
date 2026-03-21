//! Occlusion culling via a Hierarchical Z-Buffer (HZB).
//!
//! The HZB is a mip chain of the depth buffer where each texel stores the
//! **maximum** depth of the corresponding 2x2 region in the level below. This
//! allows conservative occlusion queries: if the *closest* depth of a projected
//! AABB is farther than the HZB sample at the appropriate mip level, the AABB
//! is guaranteed to be fully occluded.
//!
//! Two execution paths are provided:
//!
//! * **CPU path** (`HzbPyramid::build_from_depth_buffer`) -- operates on a plain
//!   `&[f32]` depth image.  Useful for unit tests and software rasterisation.
//!
//! * **GPU path** (`setup_hzb_pipeline` + `dispatch_hzb_downsample`) -- dispatches
//!   the [`HZB_DOWNSAMPLE_SHADER`] compute shader through the `ComputeManager`.

use euca_math::{Mat4, Vec3, Vec4};

use crate::compute::{ComputeManager, ComputePipeline, ComputePipelineDesc, GpuBuffer};

// ---------------------------------------------------------------------------
// WGSL compute shader -- HZB downsample (MAX of 2x2)
// ---------------------------------------------------------------------------

/// WGSL compute shader that downsamples a depth mip level by taking the MAX of
/// each 2x2 texel block.
///
/// Bindings (group 0):
///   @binding(0) `src` -- storage (read) array of `f32` (source mip)
///   @binding(1) `dst` -- storage (read_write) array of `f32` (destination mip)
///   @binding(2) `params` -- uniform `HzbParams { src_width, src_height, dst_width, dst_height }`
pub const HZB_DOWNSAMPLE_SHADER: &str = r#"
struct HzbParams {
    src_width:  u32,
    src_height: u32,
    dst_width:  u32,
    dst_height: u32,
}

@group(0) @binding(0) var<storage, read>       src:    array<f32>;
@group(0) @binding(1) var<storage, read_write> dst:    array<f32>;
@group(0) @binding(2) var<uniform>             params: HzbParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_x = gid.x;
    let dst_y = gid.y;
    if dst_x >= params.dst_width || dst_y >= params.dst_height {
        return;
    }

    let src_x = dst_x * 2u;
    let src_y = dst_y * 2u;

    // Sample up to 4 texels from the source mip, clamping at edges.
    let x0 = src_x;
    let y0 = src_y;
    let x1 = min(src_x + 1u, params.src_width - 1u);
    let y1 = min(src_y + 1u, params.src_height - 1u);

    let s00 = src[y0 * params.src_width + x0];
    let s10 = src[y0 * params.src_width + x1];
    let s01 = src[y1 * params.src_width + x0];
    let s11 = src[y1 * params.src_width + x1];

    let max_depth = max(max(s00, s10), max(s01, s11));
    dst[dst_y * params.dst_width + dst_x] = max_depth;
}
"#;

/// GPU-side parameters for the HZB downsample shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HzbParams {
    pub src_width: u32,
    pub src_height: u32,
    pub dst_width: u32,
    pub dst_height: u32,
}

// ---------------------------------------------------------------------------
// HzbPyramid
// ---------------------------------------------------------------------------

/// A hierarchical Z-buffer: a chain of depth mip levels.
///
/// Level 0 is the full-resolution depth buffer.  Each subsequent level is half
/// the resolution (rounded up) and stores the MAX of each 2x2 block.
pub struct HzbPyramid {
    /// Mip levels, index 0 = full resolution.
    levels: Vec<HzbMipLevel>,
}

/// A single mip level in the HZB pyramid.
struct HzbMipLevel {
    width: u32,
    height: u32,
    /// Row-major depth data.  `data[y * width + x]`.
    data: Vec<f32>,
}

impl HzbPyramid {
    /// Build the HZB pyramid on the CPU from a row-major depth buffer.
    ///
    /// `depth` must have exactly `width * height` elements.  Depth values are
    /// in clip-space (0.0 = near, 1.0 = far) as output by a left-handed depth
    /// buffer with depth range [0, 1].
    pub fn build_from_depth_buffer(depth: &[f32], width: u32, height: u32) -> Self {
        assert_eq!(
            depth.len(),
            (width * height) as usize,
            "depth buffer size mismatch"
        );

        let mut levels = Vec::new();

        // Level 0 = copy of the original depth buffer.
        levels.push(HzbMipLevel {
            width,
            height,
            data: depth.to_vec(),
        });

        let mut src_w = width;
        let mut src_h = height;

        while src_w > 1 || src_h > 1 {
            let dst_w = src_w.div_ceil(2);
            let dst_h = src_h.div_ceil(2);

            let src = &levels.last().unwrap().data;
            let mut dst = vec![0.0f32; (dst_w * dst_h) as usize];

            for dy in 0..dst_h {
                for dx in 0..dst_w {
                    let sx = dx * 2;
                    let sy = dy * 2;

                    let x0 = sx;
                    let y0 = sy;
                    let x1 = (sx + 1).min(src_w - 1);
                    let y1 = (sy + 1).min(src_h - 1);

                    let s00 = src[(y0 * src_w + x0) as usize];
                    let s10 = src[(y0 * src_w + x1) as usize];
                    let s01 = src[(y1 * src_w + x0) as usize];
                    let s11 = src[(y1 * src_w + x1) as usize];

                    dst[(dy * dst_w + dx) as usize] = s00.max(s10).max(s01).max(s11);
                }
            }

            levels.push(HzbMipLevel {
                width: dst_w,
                height: dst_h,
                data: dst,
            });

            src_w = dst_w;
            src_h = dst_h;
        }

        Self { levels }
    }

    /// Number of mip levels in the pyramid.
    pub fn mip_count(&self) -> usize {
        self.levels.len()
    }

    /// Width of a given mip level.
    pub fn mip_width(&self, level: usize) -> u32 {
        self.levels[level].width
    }

    /// Height of a given mip level.
    pub fn mip_height(&self, level: usize) -> u32 {
        self.levels[level].height
    }

    /// Sample a depth value from a specific mip level at integer coordinates.
    ///
    /// Coordinates are clamped to the level dimensions.
    pub fn sample(&self, level: usize, x: u32, y: u32) -> f32 {
        let mip = &self.levels[level];
        let cx = x.min(mip.width.saturating_sub(1));
        let cy = y.min(mip.height.saturating_sub(1));
        mip.data[(cy * mip.width + cx) as usize]
    }

    /// Sample the maximum depth in a screen-space rectangle at an appropriate
    /// mip level.
    ///
    /// `min_x`, `min_y`, `max_x`, `max_y` are in pixel coordinates of level 0.
    /// The mip level is chosen such that the rectangle maps to roughly one
    /// texel, giving a conservative (maximum) depth.
    pub fn sample_rect_max(&self, min_x: f32, min_y: f32, max_x: f32, max_y: f32) -> f32 {
        let rect_w = (max_x - min_x).max(1.0);
        let rect_h = (max_y - min_y).max(1.0);

        // Choose the mip level where the rectangle covers ~1-2 texels.
        let longest_side = rect_w.max(rect_h);
        let level = (longest_side.log2().ceil() as usize).min(self.levels.len() - 1);

        let mip = &self.levels[level];
        let scale = 1.0 / (1u32 << level) as f32;

        // Map the rectangle corners into this mip level's coordinate space.
        let mx0 = (min_x * scale).floor() as u32;
        let my0 = (min_y * scale).floor() as u32;
        let mx1 = (max_x * scale).ceil() as u32;
        let my1 = (max_y * scale).ceil() as u32;

        let mut max_depth = 0.0f32;
        for y in my0..=my1.min(mip.height.saturating_sub(1)) {
            for x in mx0..=mx1.min(mip.width.saturating_sub(1)) {
                max_depth = max_depth.max(self.sample(level, x, y));
            }
        }

        max_depth
    }
}

// ---------------------------------------------------------------------------
// OcclusionResult
// ---------------------------------------------------------------------------

/// Per-entity visibility result from an occlusion query.
pub struct OcclusionResult {
    /// `visible[i]` is `true` if entity `i` passed the occlusion test.
    pub visible: Vec<bool>,
}

impl OcclusionResult {
    /// Returns the number of entities that are visible.
    pub fn visible_count(&self) -> usize {
        self.visible.iter().filter(|&&v| v).count()
    }

    /// Returns the number of entities that are occluded.
    pub fn occluded_count(&self) -> usize {
        self.visible.iter().filter(|&&v| !v).count()
    }
}

// ---------------------------------------------------------------------------
// test_occlusion -- CPU occlusion test
// ---------------------------------------------------------------------------

/// Project each AABB to screen space and test against the HZB.
///
/// * `hzb`        -- the hierarchical Z-buffer pyramid.
/// * `aabbs`      -- slice of `(min_corner, max_corner)` world-space AABBs.
/// * `view_proj`  -- the combined view-projection matrix.
/// * `screen_w`   -- width of the depth buffer in pixels (= hzb level 0 width).
/// * `screen_h`   -- height of the depth buffer in pixels (= hzb level 0 height).
///
/// An AABB is **occluded** when the maximum depth stored in the HZB at the
/// appropriate mip level is *less than* the AABB's nearest projected depth.
pub fn test_occlusion(
    hzb: &HzbPyramid,
    aabbs: &[(Vec3, Vec3)],
    view_proj: Mat4,
    screen_w: u32,
    screen_h: u32,
) -> OcclusionResult {
    let sw = screen_w as f32;
    let sh = screen_h as f32;

    let visible: Vec<bool> = aabbs
        .iter()
        .map(|(aabb_min, aabb_max)| {
            // Generate all 8 corners of the AABB.
            let corners = aabb_corners(*aabb_min, *aabb_max);

            // Project each corner to clip space, then to NDC, then to screen.
            let mut min_sx = f32::MAX;
            let mut min_sy = f32::MAX;
            let mut max_sx = f32::MIN;
            let mut max_sy = f32::MIN;
            let mut near_depth = f32::MAX; // Smallest (closest) depth

            for corner in &corners {
                let clip = view_proj * Vec4::new(corner.x, corner.y, corner.z, 1.0);

                // If w <= 0, the point is behind the camera.  We treat the
                // AABB as visible (conservative).
                if clip.w <= 0.0 {
                    return true;
                }

                let inv_w = 1.0 / clip.w;
                let ndc_x = clip.x * inv_w;
                let ndc_y = clip.y * inv_w;
                let ndc_z = clip.z * inv_w; // Depth in [0, 1] for LH depth [0,1]

                // NDC [-1, 1] -> screen [0, width/height]
                // Note: Y is flipped (NDC +Y = screen top)
                let sx = (ndc_x * 0.5 + 0.5) * sw;
                let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * sh;

                min_sx = min_sx.min(sx);
                min_sy = min_sy.min(sy);
                max_sx = max_sx.max(sx);
                max_sy = max_sy.max(sy);
                near_depth = near_depth.min(ndc_z);
            }

            // Clamp the screen rect to valid pixel bounds.
            min_sx = min_sx.max(0.0);
            min_sy = min_sy.max(0.0);
            max_sx = max_sx.min(sw - 1.0);
            max_sy = max_sy.min(sh - 1.0);

            // If the projected rect is degenerate (entirely off-screen), cull.
            if min_sx > max_sx || min_sy > max_sy {
                return false;
            }

            // Depth behind the far plane: not visible.
            if near_depth > 1.0 {
                return false;
            }

            // Depth in front of the near plane: conservatively visible.
            if near_depth < 0.0 {
                return true;
            }

            // Sample the HZB.  The returned value is the maximum depth in the
            // rectangle -- if even that maximum is less than the AABB's nearest
            // depth, the AABB is fully behind existing geometry.
            let hzb_depth = hzb.sample_rect_max(min_sx, min_sy, max_sx, max_sy);

            // Visible when: the farthest depth already rendered in that region
            // is at least as far as the nearest depth of this AABB.
            hzb_depth >= near_depth
        })
        .collect();

    OcclusionResult { visible }
}

/// Compute the 8 corner points of an axis-aligned bounding box.
fn aabb_corners(min: Vec3, max: Vec3) -> [Vec3; 8] {
    [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, max.y, max.z),
    ]
}

// ---------------------------------------------------------------------------
// OcclusionCuller -- high-level resource
// ---------------------------------------------------------------------------

/// High-level occlusion culling resource.
///
/// Owns an HZB pyramid and provides a simple interface for the rendering
/// pipeline to update the pyramid each frame and query visibility.
pub struct OcclusionCuller {
    /// The current HZB pyramid (rebuilt each frame from the depth buffer).
    pyramid: Option<HzbPyramid>,
    /// Screen dimensions used to build the current pyramid.
    screen_width: u32,
    screen_height: u32,
}

impl OcclusionCuller {
    /// Create a new, empty culler.
    pub fn new() -> Self {
        Self {
            pyramid: None,
            screen_width: 0,
            screen_height: 0,
        }
    }

    /// (Re)build the HZB pyramid from a new depth buffer (CPU path).
    pub fn update_from_depth_buffer(&mut self, depth: &[f32], width: u32, height: u32) {
        self.pyramid = Some(HzbPyramid::build_from_depth_buffer(depth, width, height));
        self.screen_width = width;
        self.screen_height = height;
    }

    /// Test a batch of AABBs against the current HZB.
    ///
    /// Returns `None` if the pyramid has not been built yet.
    pub fn test(&self, aabbs: &[(Vec3, Vec3)], view_proj: Mat4) -> Option<OcclusionResult> {
        let pyramid = self.pyramid.as_ref()?;
        Some(test_occlusion(
            pyramid,
            aabbs,
            view_proj,
            self.screen_width,
            self.screen_height,
        ))
    }

    /// Access the underlying HZB pyramid, if built.
    pub fn pyramid(&self) -> Option<&HzbPyramid> {
        self.pyramid.as_ref()
    }

    /// Current screen width.
    pub fn screen_width(&self) -> u32 {
        self.screen_width
    }

    /// Current screen height.
    pub fn screen_height(&self) -> u32 {
        self.screen_height
    }
}

impl Default for OcclusionCuller {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GPU path helpers
// ---------------------------------------------------------------------------

/// Pipeline name constant for the GPU HZB path.
const HZB_PIPELINE_NAME: &str = "hzb_downsample";

/// Register the HZB downsample compute pipeline with a `ComputeManager`.
///
/// Call once at startup.  Subsequent frames call `dispatch_hzb_downsample`
/// for each mip transition.
pub fn setup_hzb_pipeline(device: &wgpu::Device, manager: &mut ComputeManager) {
    let pipeline = ComputePipeline::new(
        device,
        &ComputePipelineDesc {
            label: "hzb_downsample_pipeline",
            shader_source: HZB_DOWNSAMPLE_SHADER,
            entry_point: "main",
        },
    );
    manager.add_pipeline(HZB_PIPELINE_NAME, pipeline);
}

/// Create GPU buffers for a single mip-transition dispatch.
///
/// This is a helper that allocates `src` and `dst` storage buffers sized for
/// the given dimensions.  In a full implementation these would be backed by
/// texture views; for now we use flat `f32` storage buffers to keep the
/// compute infrastructure consistent with the frustum-culling path.
pub fn create_hzb_buffers(
    device: &wgpu::Device,
    manager: &mut ComputeManager,
    src_width: u32,
    src_height: u32,
) {
    let src_size = (src_width as u64) * (src_height as u64) * 4; // f32
    let dst_w = src_width.div_ceil(2);
    let dst_h = src_height.div_ceil(2);
    let dst_size = (dst_w as u64) * (dst_h as u64) * 4;

    let src_buf = GpuBuffer::new_storage(device, src_size, "hzb_src");
    let dst_buf = GpuBuffer::new_storage(device, dst_size, "hzb_dst");
    let params_buf = GpuBuffer::new_uniform_with_data(
        device,
        &HzbParams {
            src_width,
            src_height,
            dst_width: dst_w,
            dst_height: dst_h,
        },
        "hzb_params",
    );

    manager.add_buffer("hzb_src", src_buf);
    manager.add_buffer("hzb_dst", dst_buf);
    manager.add_buffer("hzb_params", params_buf);
}

/// Create a bind group for one HZB downsample dispatch.
pub fn create_hzb_bind_group(device: &wgpu::Device, manager: &ComputeManager) -> wgpu::BindGroup {
    let pipeline = manager
        .pipeline(HZB_PIPELINE_NAME)
        .expect("hzb_downsample pipeline not set up");
    let src_buf = manager.buffer("hzb_src").expect("hzb_src buffer missing");
    let dst_buf = manager.buffer("hzb_dst").expect("hzb_dst buffer missing");
    let params_buf = manager
        .buffer("hzb_params")
        .expect("hzb_params buffer missing");

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hzb_downsample_bind_group"),
        layout: pipeline.bind_group_layout(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buf.raw().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dst_buf.raw().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.raw().as_entire_binding(),
            },
        ],
    })
}

/// Dispatch one mip-transition of the HZB downsample shader.
pub fn dispatch_hzb_downsample(
    encoder: &mut wgpu::CommandEncoder,
    manager: &ComputeManager,
    bind_group: &wgpu::BindGroup,
    dst_width: u32,
    dst_height: u32,
) {
    let pipeline = manager
        .pipeline(HZB_PIPELINE_NAME)
        .expect("hzb_downsample pipeline not set up");

    let wg_x = dst_width.div_ceil(8);
    let wg_y = dst_height.div_ceil(8);
    crate::compute::dispatch_compute(encoder, pipeline, &[bind_group], [wg_x, wg_y, 1], None);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- 1. Pyramid mip levels --

    #[test]
    fn pyramid_mip_levels_power_of_two() {
        // 8x8 -> 4x4 -> 2x2 -> 1x1 = 4 levels
        let depth = vec![0.5f32; 64];
        let hzb = HzbPyramid::build_from_depth_buffer(&depth, 8, 8);
        assert_eq!(hzb.mip_count(), 4);
        assert_eq!((hzb.mip_width(0), hzb.mip_height(0)), (8, 8));
        assert_eq!((hzb.mip_width(1), hzb.mip_height(1)), (4, 4));
        assert_eq!((hzb.mip_width(2), hzb.mip_height(2)), (2, 2));
        assert_eq!((hzb.mip_width(3), hzb.mip_height(3)), (1, 1));
    }

    #[test]
    fn pyramid_mip_levels_non_power_of_two() {
        // 5x3 -> 3x2 -> 2x1 -> 1x1 = 4 levels
        let depth = vec![0.5f32; 15];
        let hzb = HzbPyramid::build_from_depth_buffer(&depth, 5, 3);
        assert_eq!(hzb.mip_count(), 4);
        assert_eq!((hzb.mip_width(0), hzb.mip_height(0)), (5, 3));
        assert_eq!((hzb.mip_width(1), hzb.mip_height(1)), (3, 2));
        assert_eq!((hzb.mip_width(2), hzb.mip_height(2)), (2, 1));
        assert_eq!((hzb.mip_width(3), hzb.mip_height(3)), (1, 1));
    }

    #[test]
    fn pyramid_max_reduction() {
        // 4x4 depth buffer with known values.
        #[rustfmt::skip]
        let depth = vec![
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.2, 0.3, 0.9, 0.1,
            0.4, 0.5, 0.6, 0.7,
        ];
        let hzb = HzbPyramid::build_from_depth_buffer(&depth, 4, 4);

        // Level 1 (2x2): MAX of each 2x2 block.
        // Block (0,0): max(0.1, 0.2, 0.5, 0.6) = 0.6
        // Block (1,0): max(0.3, 0.4, 0.7, 0.8) = 0.8
        // Block (0,1): max(0.2, 0.3, 0.4, 0.5) = 0.5
        // Block (1,1): max(0.9, 0.1, 0.6, 0.7) = 0.9
        assert!((hzb.sample(1, 0, 0) - 0.6).abs() < 1e-6);
        assert!((hzb.sample(1, 1, 0) - 0.8).abs() < 1e-6);
        assert!((hzb.sample(1, 0, 1) - 0.5).abs() < 1e-6);
        assert!((hzb.sample(1, 1, 1) - 0.9).abs() < 1e-6);

        // Level 2 (1x1): MAX of the 2x2 at level 1.
        assert!((hzb.sample(2, 0, 0) - 0.9).abs() < 1e-6);
    }

    // -- 2. Occluded behind wall --

    #[test]
    fn occluded_behind_wall() {
        // A "wall" fills the depth buffer at depth 0.3 (close to camera).
        // An AABB sits behind it at depth ~0.7.  It should be occluded.
        let width = 16u32;
        let height = 16u32;

        let depth = vec![0.3f32; (width * height) as usize];
        let hzb = HzbPyramid::build_from_depth_buffer(&depth, width, height);

        let view_proj = Mat4::orthographic_lh(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0);

        // AABB centred at z=0.7 (depth 0.7), well behind the wall at 0.3.
        let aabbs = vec![(Vec3::new(-0.5, -0.5, 0.6), Vec3::new(0.5, 0.5, 0.8))];

        let result = test_occlusion(&hzb, &aabbs, view_proj, width, height);
        assert_eq!(result.visible.len(), 1);
        // HZB max depth = 0.3.  AABB near depth ~0.6.  0.3 < 0.6 => occluded.
        assert!(!result.visible[0], "AABB behind wall should be occluded");
    }

    // -- 3. Visible in front --

    #[test]
    fn visible_in_front_of_wall() {
        let width = 16u32;
        let height = 16u32;

        // Wall at depth 0.8 (far from camera).
        let depth = vec![0.8f32; (width * height) as usize];
        let hzb = HzbPyramid::build_from_depth_buffer(&depth, width, height);

        let view_proj = Mat4::orthographic_lh(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0);

        // AABB in front of the wall at depth ~0.2.
        let aabbs = vec![(Vec3::new(-0.5, -0.5, 0.1), Vec3::new(0.5, 0.5, 0.3))];

        let result = test_occlusion(&hzb, &aabbs, view_proj, width, height);
        assert_eq!(result.visible.len(), 1);
        // HZB max depth = 0.8.  AABB near depth ~0.1.  0.8 >= 0.1 => visible.
        assert!(result.visible[0], "AABB in front of wall should be visible");
    }

    // -- 4. Partial overlap -- some depth closer, some farther --

    #[test]
    fn partial_overlap_visibility() {
        let width = 16u32;
        let height = 16u32;

        // Left half has a close wall (depth 0.2), right half is clear (depth 1.0).
        let mut depth = vec![0.0f32; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                depth[idx] = if x < width / 2 { 0.2 } else { 1.0 };
            }
        }
        let hzb = HzbPyramid::build_from_depth_buffer(&depth, width, height);

        let view_proj = Mat4::orthographic_lh(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0);

        // AABB that spans the full screen at depth 0.5.
        let aabbs = vec![(Vec3::new(-0.9, -0.9, 0.4), Vec3::new(0.9, 0.9, 0.6))];

        let result = test_occlusion(&hzb, &aabbs, view_proj, width, height);
        assert_eq!(result.visible.len(), 1);
        // The HZB max depth at the coarsest level covering the full rect
        // includes 1.0 (from the right half).  Since 1.0 >= 0.4, visible.
        assert!(
            result.visible[0],
            "AABB spanning partially occluded region should be visible (conservative)"
        );
    }

    // -- 5. Empty scene --

    #[test]
    fn empty_scene_no_entities() {
        let width = 8u32;
        let height = 8u32;
        let depth = vec![1.0f32; (width * height) as usize];
        let hzb = HzbPyramid::build_from_depth_buffer(&depth, width, height);

        let view_proj = Mat4::orthographic_lh(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0);

        let result = test_occlusion(&hzb, &[], view_proj, width, height);
        assert!(result.visible.is_empty());
        assert_eq!(result.visible_count(), 0);
        assert_eq!(result.occluded_count(), 0);
    }

    // -- 6. OcclusionCuller resource lifecycle --

    #[test]
    fn occlusion_culler_lifecycle() {
        let mut culler = OcclusionCuller::new();
        assert!(culler.pyramid().is_none());
        assert!(
            culler.test(&[], Mat4::IDENTITY).is_none(),
            "Test before update should return None"
        );

        let depth = vec![0.5f32; 64];
        culler.update_from_depth_buffer(&depth, 8, 8);
        assert!(culler.pyramid().is_some());
        assert_eq!(culler.screen_width(), 8);
        assert_eq!(culler.screen_height(), 8);

        let result = culler.test(&[], Mat4::IDENTITY).unwrap();
        assert!(result.visible.is_empty());
    }

    // -- 7. HZB shader source sanity --

    #[test]
    fn hzb_shader_is_valid_wgsl_source() {
        assert!(!HZB_DOWNSAMPLE_SHADER.is_empty());
        assert!(HZB_DOWNSAMPLE_SHADER.contains("@compute"));
        assert!(HZB_DOWNSAMPLE_SHADER.contains("@workgroup_size(8, 8)"));
        assert!(HZB_DOWNSAMPLE_SHADER.contains("fn main"));
    }

    // -- 8. HzbParams GPU struct layout --

    #[test]
    fn hzb_params_layout() {
        assert_eq!(std::mem::size_of::<HzbParams>(), 16); // 4 * u32
        let params = HzbParams {
            src_width: 1024,
            src_height: 768,
            dst_width: 512,
            dst_height: 384,
        };
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 16);
        assert_eq!(
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            1024
        );
    }

    // -- 9. 1x1 depth buffer edge case --

    #[test]
    fn single_pixel_depth_buffer() {
        let depth = vec![0.42f32];
        let hzb = HzbPyramid::build_from_depth_buffer(&depth, 1, 1);
        assert_eq!(hzb.mip_count(), 1);
        assert!((hzb.sample(0, 0, 0) - 0.42).abs() < 1e-6);
    }

    // -- 10. Multiple entities mixed visibility --

    #[test]
    fn multiple_entities_mixed_visibility() {
        let width = 16u32;
        let height = 16u32;
        let depth = vec![0.5f32; (width * height) as usize];
        let hzb = HzbPyramid::build_from_depth_buffer(&depth, width, height);

        let view_proj = Mat4::orthographic_lh(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0);

        let aabbs = vec![
            // In front of depth buffer (visible): near depth ~0.1, HZB = 0.5
            (Vec3::new(-0.5, -0.5, 0.1), Vec3::new(0.5, 0.5, 0.3)),
            // Behind depth buffer (occluded): near depth ~0.7, HZB = 0.5
            (Vec3::new(-0.5, -0.5, 0.7), Vec3::new(0.5, 0.5, 0.9)),
        ];

        let result = test_occlusion(&hzb, &aabbs, view_proj, width, height);
        assert_eq!(result.visible.len(), 2);
        assert!(result.visible[0], "First AABB should be visible");
        assert!(!result.visible[1], "Second AABB should be occluded");
        assert_eq!(result.visible_count(), 1);
        assert_eq!(result.occluded_count(), 1);
    }
}
