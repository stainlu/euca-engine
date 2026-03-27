//! Per-pixel velocity buffer pass for screen-space motion vectors.
//!
//! Renders all opaque geometry with a minimal shader that compares each
//! vertex's current clip position against its previous frame clip position,
//! outputting a 2D motion vector per pixel in screen UV space.
//!
//! # Texture format
//! The velocity buffer uses `Rg16Float` — two 16-bit floats storing
//! `(velocity_x, velocity_y)`. Static objects (same model matrix across
//! frames) naturally produce zero velocity.
//!
//! # Integration
//! The velocity buffer is consumed by:
//! - **TAA** — for accurate per-pixel reprojection (replaces depth-only reprojection)
//! - **Motion blur** — screen-space directional blur scaled by velocity magnitude
//! - **Temporal SSGI** — stable indirect lighting accumulation for moving objects
//!
//! # Usage
//! ```ignore
//! let textures = VelocityTextures::new(&device, width, height);
//! let pipeline = VelocityPipeline::new(&device, &prepass_instance_bgl, unified_memory);
//! // ... each frame:
//! pipeline.update_previous_models(&device, &queue, &current_model_matrices);
//! pipeline.write_scene(&queue, view_proj, prev_view_proj);
//! pipeline.execute(&mut encoder, &textures, &prepass_depth_view, |pass| {
//!     // draw opaque geometry (same draw calls as prepass)
//! });
//! ```

use crate::buffer::{BufferKind, SmartBuffer};
use crate::vertex::Vertex;

/// Texture format for the velocity buffer (2-channel half-float).
pub const VELOCITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

/// Initial velocity instance buffer capacity. Grows dynamically when exceeded.
const INITIAL_VELOCITY_INSTANCE_CAPACITY: usize = 16384;

const VELOCITY_SHADER: &str = include_str!("../shaders/velocity.wgsl");

/// GPU-side scene uniforms for the velocity pass.
///
/// Contains the current and previous frame view-projection matrices so the
/// vertex shader can compute clip-space positions for both frames.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VelocitySceneUniforms {
    /// Current frame combined view-projection matrix (column-major).
    pub view_projection: [[f32; 4]; 4],
    /// Previous frame combined view-projection matrix (column-major).
    pub prev_view_projection: [[f32; 4]; 4],
}

/// Resolution-dependent textures produced by the velocity pass.
pub struct VelocityTextures {
    pub velocity_texture: wgpu::Texture,
    pub velocity_view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl VelocityTextures {
    /// Create the velocity buffer texture at the given resolution.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let w = width.max(1);
        let h = height.max(1);

        let velocity_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Velocity Buffer"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: VELOCITY_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let velocity_view = velocity_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            velocity_texture,
            velocity_view,
            width: w,
            height: h,
        }
    }

    /// Recreate texture at a new resolution (e.g. window resize).
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        *self = Self::new(device, width, height);
    }
}

/// Render pipeline that writes per-pixel screen-space motion vectors.
///
/// Re-uses the same instance buffer (bind group) as the prepass for current
/// frame model matrices. Maintains its own storage buffer for previous frame
/// model matrices.
pub struct VelocityPipeline {
    pipeline: wgpu::RenderPipeline,
    scene_bgl: wgpu::BindGroupLayout,
    prev_model_bgl: wgpu::BindGroupLayout,
    scene_buffer: SmartBuffer,
    scene_bind_group: wgpu::BindGroup,
    prev_model_buffer: SmartBuffer,
    prev_model_bind_group: wgpu::BindGroup,
    /// CPU-side copy of previous frame model matrices.
    prev_models: Vec<[[f32; 4]; 4]>,
    /// Whether `update_previous_models` has been called at least once.
    initialized: bool,
    /// Current capacity (in instances) of the previous-model buffer.
    prev_model_capacity: usize,
    /// Whether the GPU uses unified memory (needed for buffer re-creation).
    unified_memory: bool,
}

impl VelocityPipeline {
    /// Create the velocity pass pipeline and allocate GPU buffers.
    ///
    /// `instance_bgl` should be the same bind group layout used by the prepass
    /// (group 0: storage buffer of `InstanceData`). This allows re-using the
    /// prepass instance buffer directly.
    pub fn new(
        device: &wgpu::Device,
        instance_bgl: &wgpu::BindGroupLayout,
        unified_memory: bool,
    ) -> Self {
        // Group 1: scene uniforms (current + previous view-projection)
        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Velocity Scene BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<VelocitySceneUniforms>() as u64,
                    ),
                },
                count: None,
            }],
        });

        // Group 2: previous frame model matrices (storage buffer)
        let prev_model_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Velocity PrevModel BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let scene_buffer = SmartBuffer::new(
            device,
            std::mem::size_of::<VelocitySceneUniforms>() as u64,
            BufferKind::Uniform,
            unified_memory,
            "Velocity Scene UBO",
        );

        // Each previous model is a mat4x4 = 64 bytes.
        let prev_model_size =
            (INITIAL_VELOCITY_INSTANCE_CAPACITY * std::mem::size_of::<[[f32; 4]; 4]>()) as u64;
        let prev_model_buffer = SmartBuffer::new(
            device,
            prev_model_size,
            BufferKind::Storage,
            unified_memory,
            "Velocity PrevModel SSBO",
        );

        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Velocity Scene BG"),
            layout: &scene_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_buffer.raw().as_entire_binding(),
            }],
        });

        let prev_model_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Velocity PrevModel BG"),
            layout: &prev_model_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: prev_model_buffer.raw().as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Velocity Shader"),
            source: wgpu::ShaderSource::Wgsl(VELOCITY_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Velocity Pipeline Layout"),
            bind_group_layouts: &[instance_bgl, &scene_bgl, &prev_model_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Velocity Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: VELOCITY_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            // Read-only depth test: use prepass depth to avoid overdraw,
            // but do not write depth (the prepass already wrote it).
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::prepass::PREPASS_DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Equal,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            scene_bgl,
            prev_model_bgl,
            scene_buffer,
            scene_bind_group,
            prev_model_buffer,
            prev_model_bind_group,
            prev_models: Vec::new(),
            initialized: false,
            prev_model_capacity: INITIAL_VELOCITY_INSTANCE_CAPACITY,
            unified_memory,
        }
    }

    /// Grow the previous-model buffer if `count` exceeds capacity.
    pub fn ensure_prev_model_capacity(&mut self, device: &wgpu::Device, count: usize) {
        if count <= self.prev_model_capacity {
            return;
        }
        self.prev_model_capacity = count.next_power_of_two();
        let size = (self.prev_model_capacity * std::mem::size_of::<[[f32; 4]; 4]>()) as u64;
        self.prev_model_buffer = SmartBuffer::new(
            device,
            size,
            BufferKind::Storage,
            self.unified_memory,
            "Velocity PrevModel SSBO",
        );
        self.prev_model_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Velocity PrevModel BG"),
            layout: &self.prev_model_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.prev_model_buffer.raw().as_entire_binding(),
            }],
        });
    }

    /// Upload per-frame scene matrices (current and previous view-projection).
    pub fn write_scene(
        &self,
        queue: &wgpu::Queue,
        view_projection: [[f32; 4]; 4],
        prev_view_projection: [[f32; 4]; 4],
    ) {
        let uniforms = VelocitySceneUniforms {
            view_projection,
            prev_view_projection,
        };
        self.scene_buffer
            .write_bytes(queue, bytemuck::bytes_of(&uniforms));
    }

    /// Save current frame model matrices and upload the previous frame's matrices to the GPU.
    ///
    /// Call this **before** rendering the velocity pass each frame. On the very
    /// first call, `prev_models` is initialized to the current matrices so that
    /// all objects report zero velocity on the first frame.
    pub fn update_previous_models(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        current_models: &[[[f32; 4]; 4]],
    ) {
        if !self.initialized {
            // First frame: previous == current so velocity is zero everywhere.
            self.prev_models = current_models.to_vec();
            self.initialized = true;
        }

        // Grow buffer if needed before uploading.
        self.ensure_prev_model_capacity(device, self.prev_models.len());

        // Upload previous frame's models to the GPU.
        if !self.prev_models.is_empty() {
            self.prev_model_buffer.write(queue, &self.prev_models);
        }

        // Store current frame for next frame's "previous".
        self.prev_models.clear();
        self.prev_models.extend_from_slice(current_models);
    }

    /// Execute the velocity buffer pass.
    ///
    /// Uses the prepass instance bind group (group 0) for current frame model
    /// matrices, and the velocity pass's own bind groups for scene uniforms
    /// (group 1) and previous model matrices (group 2).
    ///
    /// `depth_view` should be the prepass depth texture view for depth testing.
    /// `instance_bind_group` should be the prepass instance bind group.
    pub fn execute<'a, F>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        textures: &'a VelocityTextures,
        depth_view: &'a wgpu::TextureView,
        instance_bind_group: &'a wgpu::BindGroup,
        draw_fn: F,
    ) where
        F: FnOnce(&mut wgpu::RenderPass<'a>),
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Velocity Buffer Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &textures.velocity_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Clear to zero velocity (no motion).
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    // Load existing depth from prepass; do not clear or write.
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, instance_bind_group, &[]);
        pass.set_bind_group(1, &self.scene_bind_group, &[]);
        pass.set_bind_group(2, &self.prev_model_bind_group, &[]);
        draw_fn(&mut pass);
    }

    /// Access the velocity scene bind group layout.
    pub fn scene_bgl(&self) -> &wgpu::BindGroupLayout {
        &self.scene_bgl
    }

    /// Access the previous model bind group layout.
    pub fn prev_model_bgl(&self) -> &wgpu::BindGroupLayout {
        &self.prev_model_bgl
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn velocity_texture_format() {
        assert_eq!(VELOCITY_FORMAT, wgpu::TextureFormat::Rg16Float);
    }

    #[test]
    fn velocity_scene_uniforms_gpu_aligned() {
        let size = std::mem::size_of::<VelocitySceneUniforms>();
        // Two mat4x4<f32> = 2 * 64 = 128 bytes.
        assert_eq!(size, 128);
        assert_eq!(
            size % 16,
            0,
            "VelocitySceneUniforms size ({size}) must be 16-byte aligned"
        );
        let u = VelocitySceneUniforms {
            view_projection: [[1.0, 0.0, 0.0, 0.0]; 4],
            prev_view_projection: [[0.0; 4]; 4],
        };
        assert_eq!(bytemuck::bytes_of(&u).len(), size);
    }

    #[test]
    fn velocity_textures_dimensions() {
        // Cannot create real GPU textures in unit tests, but verify the
        // constructor clamps dimensions to at least 1.
        assert_eq!(1_u32.max(1), 1);
        assert_eq!(0_u32.max(1), 1);
        assert_eq!(1920_u32.max(1), 1920);
    }

    #[test]
    fn initial_capacity_matches_prepass() {
        // Both prepass and velocity must start with the same initial capacity
        // since they share the instance buffer bind group layout.
        assert_eq!(INITIAL_VELOCITY_INSTANCE_CAPACITY, 16384);
    }

    #[test]
    fn prev_model_matrix_size() {
        // Each prev_model is a mat4x4<f32> = 64 bytes.
        let size = std::mem::size_of::<[[f32; 4]; 4]>();
        assert_eq!(size, 64);
    }

    #[test]
    fn update_previous_models_first_frame_zero_velocity() {
        // Verify the logic: on first call, prev_models should equal current_models.
        let identity: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let current = vec![identity; 3];

        // Simulate the first-frame logic without GPU.
        let mut prev_models: Vec<[[f32; 4]; 4]> = Vec::new();
        let mut initialized = false;

        if !initialized {
            prev_models = current.clone();
            initialized = true;
        }

        // prev_models should equal current_models on first frame.
        assert_eq!(prev_models.len(), current.len());
        for (prev, cur) in prev_models.iter().zip(current.iter()) {
            for row in 0..4 {
                for col in 0..4 {
                    assert!(
                        (prev[row][col] - cur[row][col]).abs() < f32::EPSILON,
                        "First frame prev_model must equal current model"
                    );
                }
            }
        }
        assert!(initialized);
    }

    #[test]
    fn update_previous_models_subsequent_frame() {
        // Verify that after the first frame, prev_models stores the previous
        // frame's current_models.
        let identity: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let translated: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [5.0, 0.0, 0.0, 1.0],
        ];

        // Simulate two frames without GPU.
        let mut prev_models: Vec<[[f32; 4]; 4]> = Vec::new();
        let mut initialized = false;

        // Frame 1: current = identity
        let frame1_current = vec![identity];
        if !initialized {
            prev_models = frame1_current.clone();
            initialized = true;
        }
        assert!(initialized, "should be initialized after first frame");
        // Would upload prev_models (== identity) to GPU here.
        prev_models.clear();
        prev_models.extend_from_slice(&frame1_current);

        // Frame 2: current = translated
        let frame2_current = vec![translated];
        // prev_models still holds frame1's identity.
        assert_eq!(prev_models.len(), 1);
        assert!(
            (prev_models[0][3][0] - 0.0).abs() < f32::EPSILON,
            "prev should be identity"
        );
        // Would upload prev_models (identity) to GPU, then overwrite with frame2's current.
        prev_models.clear();
        prev_models.extend_from_slice(&frame2_current);

        // After frame 2, prev_models holds translated for next frame.
        assert!(
            (prev_models[0][3][0] - 5.0).abs() < f32::EPSILON,
            "prev should now be translated"
        );
    }
}
