//! Lightweight UI overlay renderer — draws screen-space colored quads.
//!
//! This provides a minimal 2D rendering pipeline for health bars, HUD panels,
//! and other UI elements. It renders AFTER the 3D scene and post-processing,
//! blended on top of the final framebuffer.
//!
//! # Usage
//! ```ignore
//! let mut ui = UiOverlayRenderer::new(&device, surface_format);
//! // Each frame:
//! let quads = vec![UiQuad { x: 100.0, y: 50.0, w: 200.0, h: 20.0, color: [0.0, 1.0, 0.0, 0.8] }];
//! ui.render(&device, &queue, encoder, color_view, &quads, viewport_width, viewport_height);
//! ```

use wgpu;

/// A single screen-space colored rectangle.
#[derive(Clone, Debug)]
pub struct UiQuad {
    /// X position in screen pixels (left edge).
    pub x: f32,
    /// Y position in screen pixels (top edge).
    pub y: f32,
    /// Width in screen pixels.
    pub w: f32,
    /// Height in screen pixels.
    pub h: f32,
    /// RGBA color [0..1].
    pub color: [f32; 4],
}

/// GPU instance data for one quad (matches shader layout).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuUiQuad {
    /// xy = top-left in NDC, zw = size in NDC.
    pos_size: [f32; 4],
    /// RGBA color.
    color: [f32; 4],
}

/// Renders 2D colored quads as screen-space overlays.
pub struct UiOverlayRenderer {
    pipeline: wgpu::RenderPipeline,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
}

const INITIAL_CAPACITY: usize = 256;

impl UiOverlayRenderer {
    /// Create a new UI overlay renderer.
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ui_quad.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ui_quad.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui_quad_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ui_quad_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuUiQuad>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        // pos_size: vec4<f32> at location 0
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 0,
                        },
                        // color: vec4<f32> at location 1
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 1,
                        },
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui_quad_instances"),
            size: (INITIAL_CAPACITY * std::mem::size_of::<GpuUiQuad>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            instance_buffer,
            instance_capacity: INITIAL_CAPACITY,
        }
    }

    /// Render UI quads onto the given color view.
    ///
    /// Call this AFTER the main 3D render and post-processing, on the same
    /// encoder before `encoder.finish()`.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        quads: &[UiQuad],
        viewport_width: f32,
        viewport_height: f32,
    ) {
        if quads.is_empty() {
            return;
        }

        // Convert screen-space quads to NDC instances.
        let instances: Vec<GpuUiQuad> = quads
            .iter()
            .map(|q| {
                // Screen pixels → NDC: x: [0,w] → [-1,1], y: [0,h] → [1,-1] (Y flipped)
                let ndc_x = (q.x / viewport_width) * 2.0 - 1.0;
                let ndc_y = 1.0 - (q.y / viewport_height) * 2.0;
                let ndc_w = (q.w / viewport_width) * 2.0;
                let ndc_h = -(q.h / viewport_height) * 2.0; // negative because Y is flipped
                GpuUiQuad {
                    pos_size: [ndc_x, ndc_y, ndc_w, ndc_h],
                    color: q.color,
                }
            })
            .collect();

        // Grow buffer if needed.
        if instances.len() > self.instance_capacity {
            self.instance_capacity = instances.len().next_power_of_two();
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ui_quad_instances"),
                size: (self.instance_capacity * std::mem::size_of::<GpuUiQuad>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&instances),
        );

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ui_overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // preserve 3D scene
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None, // no depth testing for UI
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
            // 6 vertices per quad (2 triangles), N instances
            pass.draw(0..6, 0..instances.len() as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ui_quad_construction() {
        let q = UiQuad {
            x: 100.0,
            y: 50.0,
            w: 200.0,
            h: 20.0,
            color: [0.0, 1.0, 0.0, 0.8],
        };
        assert_eq!(q.w, 200.0);
        assert_eq!(q.color[1], 1.0);
    }

    #[test]
    fn gpu_quad_is_pod() {
        // Verify the struct is correctly aligned for GPU upload.
        assert_eq!(std::mem::size_of::<GpuUiQuad>(), 32);
    }
}
