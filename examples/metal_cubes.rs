//! Native Metal rendering demo — colored cubes rendered directly through
//! MetalDevice, proving the Metal backend works end-to-end on Apple Silicon.
//!
//! Run: `cargo run --example metal_cubes --features euca-rhi/metal-backend`

fn main() {
    use euca_rhi::metal_backend::MetalDevice;
    use euca_rhi::*;
    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::window::{Window, WindowAttributes, WindowId};

    struct App {
        window: Option<Window>,
        device: Option<MetalDevice>,
        pipeline: Option<<MetalDevice as RenderDevice>::RenderPipeline>,
        vertex_buffer: Option<<MetalDevice as RenderDevice>::Buffer>,
        uniform_buffer: Option<<MetalDevice as RenderDevice>::Buffer>,
        frame: u32,
    }

    // MSL shader: reads position+color from buffer, applies MVP transform
    const SHADER_SRC: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        struct VertexData {
            packed_float3 position;
            packed_float3 color;
        };

        struct VertexOut {
            float4 position [[position]];
            float3 color;
        };

        vertex VertexOut vertex_main(
            uint vid [[vertex_id]],
            const device VertexData* vertices [[buffer(0)]],
            constant float4x4& mvp [[buffer(1)]]
        ) {
            VertexOut out;
            float3 pos = float3(vertices[vid].position);
            out.position = mvp * float4(pos, 1.0);
            out.color = float3(vertices[vid].color);
            return out;
        }

        fragment float4 fragment_main(VertexOut in [[stage_in]]) {
            return float4(in.color, 1.0);
        }
    "#;

    // Cube vertex data: 36 vertices (6 faces × 2 triangles × 3 vertices)
    // Each vertex: [px, py, pz, r, g, b]
    #[rustfmt::skip]
    const CUBE_VERTICES: &[[f32; 6]] = &[
        // Front (red)
        [-0.5, -0.5,  0.5,  0.9, 0.2, 0.2], [ 0.5, -0.5,  0.5,  0.9, 0.2, 0.2], [ 0.5,  0.5,  0.5,  0.9, 0.2, 0.2],
        [-0.5, -0.5,  0.5,  0.9, 0.2, 0.2], [ 0.5,  0.5,  0.5,  0.9, 0.2, 0.2], [-0.5,  0.5,  0.5,  0.9, 0.2, 0.2],
        // Back (green)
        [ 0.5, -0.5, -0.5,  0.2, 0.9, 0.2], [-0.5, -0.5, -0.5,  0.2, 0.9, 0.2], [-0.5,  0.5, -0.5,  0.2, 0.9, 0.2],
        [ 0.5, -0.5, -0.5,  0.2, 0.9, 0.2], [-0.5,  0.5, -0.5,  0.2, 0.9, 0.2], [ 0.5,  0.5, -0.5,  0.2, 0.9, 0.2],
        // Top (blue)
        [-0.5,  0.5,  0.5,  0.2, 0.2, 0.9], [ 0.5,  0.5,  0.5,  0.2, 0.2, 0.9], [ 0.5,  0.5, -0.5,  0.2, 0.2, 0.9],
        [-0.5,  0.5,  0.5,  0.2, 0.2, 0.9], [ 0.5,  0.5, -0.5,  0.2, 0.2, 0.9], [-0.5,  0.5, -0.5,  0.2, 0.2, 0.9],
        // Bottom (yellow)
        [-0.5, -0.5, -0.5,  0.9, 0.9, 0.2], [ 0.5, -0.5, -0.5,  0.9, 0.9, 0.2], [ 0.5, -0.5,  0.5,  0.9, 0.9, 0.2],
        [-0.5, -0.5, -0.5,  0.9, 0.9, 0.2], [ 0.5, -0.5,  0.5,  0.9, 0.9, 0.2], [-0.5, -0.5,  0.5,  0.9, 0.9, 0.2],
        // Right (cyan)
        [ 0.5, -0.5,  0.5,  0.2, 0.9, 0.9], [ 0.5, -0.5, -0.5,  0.2, 0.9, 0.9], [ 0.5,  0.5, -0.5,  0.2, 0.9, 0.9],
        [ 0.5, -0.5,  0.5,  0.2, 0.9, 0.9], [ 0.5,  0.5, -0.5,  0.2, 0.9, 0.9], [ 0.5,  0.5,  0.5,  0.2, 0.9, 0.9],
        // Left (magenta)
        [-0.5, -0.5, -0.5,  0.9, 0.2, 0.9], [-0.5, -0.5,  0.5,  0.9, 0.2, 0.9], [-0.5,  0.5,  0.5,  0.9, 0.2, 0.9],
        [-0.5, -0.5, -0.5,  0.9, 0.2, 0.9], [-0.5,  0.5,  0.5,  0.9, 0.2, 0.9], [-0.5,  0.5, -0.5,  0.9, 0.2, 0.9],
    ];

    // All matrices are ROW-MAJOR: mat[row][col]. Transposed before Metal upload.
    fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
        let f = 1.0 / (fov_y / 2.0).tan();
        let nf = 1.0 / (near - far);
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) * nf, 2.0 * far * near * nf],
            [0.0, 0.0, -1.0, 0.0],
        ]
    }

    fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
        let f = normalize(sub(target, eye));
        let s = normalize(cross(f, up));
        let u = cross(s, f);
        [
            [s[0], s[1], s[2], -dot(s, eye)],
            [u[0], u[1], u[2], -dot(u, eye)],
            [-f[0], -f[1], -f[2], dot(f, eye)],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
        let mut out = [[0.0f32; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    out[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        out
    }

    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let len = dot(v, v).sqrt();
        [v[0] / len, v[1] / len, v[2] / len]
    }

    fn transpose(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
        [
            [m[0][0], m[1][0], m[2][0], m[3][0]],
            [m[0][1], m[1][1], m[2][1], m[3][1]],
            [m[0][2], m[1][2], m[2][2], m[3][2]],
            [m[0][3], m[1][3], m[2][3], m[3][3]],
        ]
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() {
                return;
            }
            let attrs = WindowAttributes::default()
                .with_title("Euca Engine — Native Metal Demo")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
            let window = event_loop.create_window(attrs).unwrap();

            let device = MetalDevice::from_window(&window);
            println!(
                "Metal device: {} (Apple Silicon: {})",
                device.capabilities().device_name,
                device.capabilities().apple_silicon,
            );

            // Compile MSL shader
            let shader = device.create_shader(&ShaderDesc {
                label: Some("Cube Shader"),
                source: ShaderSource::Msl(SHADER_SRC.into()),
            });

            // Create render pipeline
            let pipeline = device.create_render_pipeline(&RenderPipelineDesc {
                label: Some("Cube Pipeline"),
                layout: &[],
                vertex: VertexState {
                    module: &shader,
                    entry_point: "vertex_main",
                    buffers: &[],
                },
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: "fragment_main",
                    targets: &[Some(ColorTargetState {
                        format: device.surface_format(),
                        blend: Some(BlendState::REPLACE),
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    cull_mode: Some(Face::Back),
                    front_face: FrontFace::Ccw,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: MultisampleState::default(),
            });

            // Upload cube vertex data
            let vertex_data: Vec<u8> = CUBE_VERTICES
                .iter()
                .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())
                .collect();
            let vertex_buffer = device.create_buffer(&BufferDesc {
                label: Some("Cube Vertices"),
                size: vertex_data.len() as u64,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            device.write_buffer(&vertex_buffer, 0, &vertex_data);

            // Uniform buffer for MVP matrix
            let uniform_buffer = device.create_buffer(&BufferDesc {
                label: Some("Uniforms"),
                size: 64, // 4x4 f32 matrix
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            self.pipeline = Some(pipeline);
            self.vertex_buffer = Some(vertex_buffer);
            self.uniform_buffer = Some(uniform_buffer);
            self.device = Some(device);
            window.request_redraw(); // Kick off the render loop
            self.window = Some(window);
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _id: WindowId,
            event: WindowEvent,
        ) {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => {
                    if let Some(ref mut device) = self.device {
                        device.resize_surface(size.width, size.height);
                    }
                }
                WindowEvent::RedrawRequested => {
                    let device = self.device.as_ref().unwrap();
                    let pipeline = self.pipeline.as_ref().unwrap();
                    let vb = self.vertex_buffer.as_ref().unwrap();
                    let ub = self.uniform_buffer.as_ref().unwrap();

                    self.frame += 1;
                    let t = self.frame as f32 * 0.01;

                    // Spinning camera
                    let eye = [t.cos() * 3.0, 1.5, t.sin() * 3.0];
                    let view = look_at(eye, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
                    let proj = perspective(
                        std::f32::consts::FRAC_PI_4,
                        device.aspect_ratio(),
                        0.1,
                        100.0,
                    );
                    let mvp = mat4_mul(proj, view);
                    // Transpose for Metal: our mat[i] = row i, Metal reads mat[i] as column i
                    let mvp = transpose(mvp);

                    // Upload MVP
                    let mvp_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            mvp.as_ptr() as *const u8,
                            std::mem::size_of_val(&mvp),
                        )
                    };
                    device.write_buffer(ub, 0, mvp_bytes);

                    // Render
                    let output = match device.get_current_texture() {
                        Ok(t) => t,
                        Err(_) => return,
                    };
                    let view = device.surface_texture_view(&output);
                    let mut encoder = device.create_command_encoder(Some("Frame"));
                    {
                        let mut pass = device.begin_render_pass(
                            &mut encoder,
                            &RenderPassDesc {
                                label: Some("Main"),
                                color_attachments: &[Some(RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: Operations {
                                        load: LoadOp::Clear(Color {
                                            r: 0.05,
                                            g: 0.05,
                                            b: 0.2,
                                            a: 1.0,
                                        }),
                                        store: StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                            },
                        );
                        pass.set_pipeline(pipeline);
                        // Bind vertex buffer at slot 0, uniform buffer at slot 1
                        pass.set_vertex_buffer(
                            0,
                            vb,
                            0,
                            std::mem::size_of_val(CUBE_VERTICES) as u64,
                        );
                        // For Metal: uniform buffer bound via set_vertex_buffer at slot 1
                        pass.set_vertex_buffer(1, ub, 0, 64);
                        pass.draw(0..36, 0..1);
                    }
                    encoder.schedule_present(&output);
                    device.submit(encoder);
                    device.present(output);

                    self.window.as_ref().unwrap().request_redraw();
                }
                _ => {}
            }
        }
    }

    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        window: None,
        device: None,
        pipeline: None,
        vertex_buffer: None,
        uniform_buffer: None,
        frame: 0,
    };
    event_loop.run_app(&mut app).unwrap();
}
