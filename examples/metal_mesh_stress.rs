//! Metal mesh shader stress test — renders N cubes using mesh shaders to
//! bypass the Apple TBDR vertex processing bottleneck.
//!
//! Mesh shaders eliminate the traditional vertex pipeline (vertex fetch →
//! vertex shader → primitive assembly → binning). Instead, a cooperative
//! threadgroup outputs vertices and triangles directly to the rasterizer.
//!
//! Run: `EUCA_ENTITIES=50000 cargo run --release --example metal_mesh_stress --features euca-rhi/metal-backend`

fn main() {
    use euca_rhi::metal_backend::MetalDevice;
    use euca_rhi::*;
    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::window::{Window, WindowAttributes, WindowId};

    let entity_count: u32 = std::env::var("EUCA_ENTITIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    // MSL mesh shader: each threadgroup outputs one cube (24 vertices, 12 triangles).
    // No vertex shader — geometry is produced cooperatively by the mesh threadgroup.
    //
    // Mesh shaders bypass the traditional vertex pipeline (fetch → shade → assemble → bin)
    // and output directly to the rasterizer. On Apple Silicon, this eliminates TBDR
    // binning-phase overhead for vertex processing.
    const MESH_SHADER_SRC: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        struct InstanceData {
            float4x4 model;
        };

        struct MeshVertexOut {
            float4 position [[position]];
            float3 color;
        };

        // 24 unique vertices: 4 per face, each with face color.
        constant float3 face_verts[24] = {
            // Front (z = +0.5)
            float3(-0.5, -0.5,  0.5), float3( 0.5, -0.5,  0.5),
            float3( 0.5,  0.5,  0.5), float3(-0.5,  0.5,  0.5),
            // Back (z = -0.5)
            float3( 0.5, -0.5, -0.5), float3(-0.5, -0.5, -0.5),
            float3(-0.5,  0.5, -0.5), float3( 0.5,  0.5, -0.5),
            // Top (y = +0.5)
            float3(-0.5,  0.5,  0.5), float3( 0.5,  0.5,  0.5),
            float3( 0.5,  0.5, -0.5), float3(-0.5,  0.5, -0.5),
            // Bottom (y = -0.5)
            float3(-0.5, -0.5, -0.5), float3( 0.5, -0.5, -0.5),
            float3( 0.5, -0.5,  0.5), float3(-0.5, -0.5,  0.5),
            // Right (x = +0.5)
            float3( 0.5, -0.5,  0.5), float3( 0.5, -0.5, -0.5),
            float3( 0.5,  0.5, -0.5), float3( 0.5,  0.5,  0.5),
            // Left (x = -0.5)
            float3(-0.5, -0.5, -0.5), float3(-0.5, -0.5,  0.5),
            float3(-0.5,  0.5,  0.5), float3(-0.5,  0.5, -0.5),
        };

        constant float3 face_colors[6] = {
            float3(0.9, 0.2, 0.2),  // Front: red
            float3(0.2, 0.9, 0.2),  // Back: green
            float3(0.2, 0.2, 0.9),  // Top: blue
            float3(0.9, 0.9, 0.2),  // Bottom: yellow
            float3(0.2, 0.9, 0.9),  // Right: cyan
            float3(0.9, 0.2, 0.9),  // Left: magenta
        };

        constant ushort quad_indices[6] = { 0, 1, 2, 0, 2, 3 };

        using CubeMesh = metal::mesh<MeshVertexOut, void, 24, 12, topology::triangle>;

        [[mesh, max_total_threads_per_threadgroup(32)]]
        void mesh_cube(
            CubeMesh output,
            uint tid [[thread_position_in_threadgroup]],
            uint gid [[threadgroup_position_in_grid]],
            constant float4x4& vp [[buffer(0)]],
            const device InstanceData* instances [[buffer(1)]]
        ) {
            if (tid == 0) {
                output.set_primitive_count(12);
            }

            if (tid < 24) {
                uint face = tid / 4;
                MeshVertexOut v;
                v.position = vp * (instances[gid].model * float4(face_verts[tid], 1.0));
                v.color = face_colors[face];
                output.set_vertex(tid, v);
            }

            if (tid < 12) {
                uint face = tid / 2;
                uint tri = tid % 2;
                uint base_vertex = face * 4;
                uint base_idx = tri * 3;
                output.set_index(tid * 3 + 0, base_vertex + quad_indices[base_idx + 0]);
                output.set_index(tid * 3 + 1, base_vertex + quad_indices[base_idx + 1]);
                output.set_index(tid * 3 + 2, base_vertex + quad_indices[base_idx + 2]);
            }
        }

        // ---------------------------------------------------------------
        // Fragment shader
        // ---------------------------------------------------------------

        [[fragment]]
        float4 fragment_mesh(MeshVertexOut in [[stage_in]]) {
            return float4(in.color, 1.0);
        }
    "#;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct InstanceData {
        model: [[f32; 4]; 4],
    }

    fn identity() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn translation(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
        let mut m = identity();
        m[3] = [x, y, z, 1.0];
        m
    }

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
        let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let cross = |a: [f32; 3], b: [f32; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let normalize = |v: [f32; 3]| {
            let l = dot(v, v).sqrt();
            [v[0] / l, v[1] / l, v[2] / l]
        };
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

    fn transpose(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
        [
            [m[0][0], m[1][0], m[2][0], m[3][0]],
            [m[0][1], m[1][1], m[2][1], m[3][1]],
            [m[0][2], m[1][2], m[2][2], m[3][2]],
            [m[0][3], m[1][3], m[2][3], m[3][3]],
        ]
    }

    struct App {
        entity_count: u32,
        window: Option<Window>,
        device: Option<MetalDevice>,
        pipeline: Option<<MetalDevice as RenderDevice>::RenderPipeline>,
        vp_buffer: Option<<MetalDevice as RenderDevice>::Buffer>,
        instance_buffer: Option<<MetalDevice as RenderDevice>::Buffer>,
        depth_view: Option<<MetalDevice as RenderDevice>::TextureView>,
        instances: Vec<InstanceData>,
        frame: u64,
        last_fps_time: std::time::Instant,
        fps_frame_count: u32,
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() {
                return;
            }
            let attrs = WindowAttributes::default()
                .with_title(format!(
                    "Euca Engine -- Metal Mesh Shader Stress ({} entities)",
                    self.entity_count
                ))
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
            let window = event_loop.create_window(attrs).unwrap();
            let device = MetalDevice::from_window(&window);

            // Disable vsync for true FPS measurement
            device.set_display_sync_enabled(false);

            let (sw, sh) = device.surface_size();
            println!(
                "Metal device: {} | {} entities | drawable: {}x{} | MESH SHADERS",
                device.capabilities().device_name,
                self.entity_count,
                sw,
                sh,
            );

            // Compile mesh shader
            let shader = device.create_shader(&ShaderDesc {
                label: Some("Mesh Cube Shader"),
                source: ShaderSource::Msl(MESH_SHADER_SRC.into()),
            });

            // Create mesh render pipeline (no vertex shader!)
            let pipeline = device.create_mesh_render_pipeline(
                &shader,
                "mesh_cube",
                "fragment_mesh",
                None,
                &[device.surface_format()],
                &[Some(BlendState::REPLACE)],
                Some(TextureFormat::Depth32Float),
                Some("Mesh Cube Pipeline"),
            );

            // VP uniform buffer
            let vp_buffer = device.create_buffer(&BufferDesc {
                label: Some("VP"),
                size: 64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Instance data: grid layout
            let side = (self.entity_count as f32).sqrt().ceil() as u32;
            let spacing = 2.5_f32;
            let mut instances = Vec::with_capacity(self.entity_count as usize);
            for i in 0..self.entity_count {
                let row = i / side;
                let col = i % side;
                let x = (col as f32 - side as f32 / 2.0) * spacing;
                let z = (row as f32 - side as f32 / 2.0) * spacing;
                // Transpose for Metal column-major
                instances.push(InstanceData {
                    model: transpose(translation(x, 0.5, z)),
                });
            }
            let inst_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    instances.as_ptr() as *const u8,
                    instances.len() * std::mem::size_of::<InstanceData>(),
                )
            };
            let instance_buffer = device.create_buffer(&BufferDesc {
                label: Some("Instances"),
                size: inst_bytes.len() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            device.write_buffer(&instance_buffer, 0, inst_bytes);

            // Depth buffer
            let depth_tex = device.create_texture(&TextureDesc {
                label: Some("Depth"),
                size: Extent3d {
                    width: sw.max(1),
                    height: sh.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let depth_view = device.create_texture_view(&depth_tex, &TextureViewDesc::default());

            self.pipeline = Some(pipeline);
            self.vp_buffer = Some(vp_buffer);
            self.instance_buffer = Some(instance_buffer);
            self.depth_view = Some(depth_view);
            self.instances = instances;
            self.device = Some(device);
            window.request_redraw();
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
                    let vp_buf = self.vp_buffer.as_ref().unwrap();
                    let inst_buf = self.instance_buffer.as_ref().unwrap();
                    let depth = self.depth_view.as_ref().unwrap();

                    self.frame += 1;
                    let t = self.frame as f32 * 0.005;

                    // FPS counter
                    self.fps_frame_count += 1;
                    let now = std::time::Instant::now();
                    let elapsed = now.duration_since(self.last_fps_time).as_secs_f32();
                    if elapsed >= 1.0 {
                        let fps = self.fps_frame_count as f32 / elapsed;
                        println!(
                            "[frame {}] {} entities | FPS: {:.1} (mesh shaders)",
                            self.frame, self.entity_count, fps
                        );
                        if let Some(ref w) = self.window {
                            w.set_title(&format!(
                                "Euca Engine -- Mesh Shader Stress ({} entities) | FPS: {:.0}",
                                self.entity_count, fps
                            ));
                        }
                        self.fps_frame_count = 0;
                        self.last_fps_time = now;
                    }

                    // Camera: scales with grid size. At each entity count, cubes
                    // are ~1-2 pixels on screen (consistent vertex/tiling load).
                    let area = (self.entity_count as f32).sqrt() * 2.5;
                    let eye = [t.cos() * area * 1.2, area * 0.6, t.sin() * area * 1.2];
                    let view = look_at(eye, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
                    let proj = perspective(
                        std::f32::consts::FRAC_PI_4,
                        device.aspect_ratio(),
                        0.1,
                        area * 3.0,
                    );
                    let vp = transpose(mat4_mul(proj, view));
                    let vp_bytes: &[u8] =
                        unsafe { std::slice::from_raw_parts(vp.as_ptr() as *const u8, 64) };
                    device.write_buffer(vp_buf, 0, vp_bytes);

                    // Render
                    let output = match device.get_current_texture() {
                        Ok(t) => t,
                        Err(_) => return,
                    };
                    let color_view = device.surface_texture_view(&output);
                    let mut encoder = device.create_command_encoder(Some("Frame"));
                    {
                        let mut pass = device.begin_render_pass(
                            &mut encoder,
                            &RenderPassDesc {
                                label: Some("Mesh"),
                                color_attachments: &[Some(RenderPassColorAttachment {
                                    view: &color_view,
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
                                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                                    view: depth,
                                    depth_ops: Some(Operations {
                                        load: LoadOp::Clear(1.0),
                                        store: StoreOp::Discard,
                                    }),
                                    stencil_ops: None,
                                }),
                                timestamp_writes: None,
                            },
                        );
                        pass.set_pipeline(pipeline);
                        // Mesh shader buffers (not vertex buffers!)
                        pass.set_mesh_buffer(0, vp_buf, 0);
                        pass.set_mesh_buffer(1, inst_buf, 0);

                        // Dispatch one mesh threadgroup per cube instance.
                        pass.draw_mesh_threadgroups(
                            [self.entity_count, 1, 1],
                            [1, 1, 1], // no object shader
                            [32, 1, 1],
                        );
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
        entity_count,
        window: None,
        device: None,
        pipeline: None,
        vp_buffer: None,
        instance_buffer: None,
        depth_view: None,
        instances: Vec::new(),
        frame: 0,
        last_fps_time: std::time::Instant::now(),
        fps_frame_count: 0,
    };
    event_loop.run_app(&mut app).unwrap();
}
