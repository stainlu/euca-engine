//! MetalFX temporal upscaling demo — renders at 50% resolution and
//! reconstructs full resolution via Apple's MetalFX temporal scaler.
//!
//! This reduces fragment load by 4x (half width × half height) while
//! maintaining visual quality through temporal accumulation.
//!
//! Run: `EUCA_ENTITIES=50000 cargo run --release --example metal_fx_upscale --features euca-rhi/metal-backend`

fn main() {
    use euca_rhi::metal_backend::{MetalDevice, MetalFXUpscaler};
    use euca_rhi::*;
    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::window::{Window, WindowAttributes, WindowId};

    let entity_count: u32 = std::env::var("EUCA_ENTITIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10000);

    // Simple vertex+fragment shader (same as metal_stress)
    const SHADER_SRC: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        struct VertexData { packed_float3 position; packed_float3 color; };
        struct InstanceData { float4x4 model; };
        struct VertexOut { float4 position [[position]]; float3 color; };

        vertex VertexOut vertex_main(
            uint vid [[vertex_id]], uint iid [[instance_id]],
            const device VertexData* vertices [[buffer(0)]],
            constant float4x4& vp [[buffer(1)]],
            const device InstanceData* instances [[buffer(2)]]
        ) {
            VertexOut out;
            float3 pos = float3(vertices[vid].position);
            out.position = vp * (instances[iid].model * float4(pos, 1.0));
            out.color = float3(vertices[vid].color);
            return out;
        }

        fragment float4 fragment_main(VertexOut in [[stage_in]]) {
            return float4(in.color, 1.0);
        }
    "#;

    #[rustfmt::skip]
    const CUBE_VERTICES: &[[f32; 6]] = &[
        [-0.5,-0.5, 0.5, 0.9,0.2,0.2],[ 0.5,-0.5, 0.5, 0.9,0.2,0.2],[ 0.5, 0.5, 0.5, 0.9,0.2,0.2],
        [-0.5,-0.5, 0.5, 0.9,0.2,0.2],[ 0.5, 0.5, 0.5, 0.9,0.2,0.2],[-0.5, 0.5, 0.5, 0.9,0.2,0.2],
        [ 0.5,-0.5,-0.5, 0.2,0.9,0.2],[-0.5,-0.5,-0.5, 0.2,0.9,0.2],[-0.5, 0.5,-0.5, 0.2,0.9,0.2],
        [ 0.5,-0.5,-0.5, 0.2,0.9,0.2],[-0.5, 0.5,-0.5, 0.2,0.9,0.2],[ 0.5, 0.5,-0.5, 0.2,0.9,0.2],
        [-0.5, 0.5, 0.5, 0.2,0.2,0.9],[ 0.5, 0.5, 0.5, 0.2,0.2,0.9],[ 0.5, 0.5,-0.5, 0.2,0.2,0.9],
        [-0.5, 0.5, 0.5, 0.2,0.2,0.9],[ 0.5, 0.5,-0.5, 0.2,0.2,0.9],[-0.5, 0.5,-0.5, 0.2,0.2,0.9],
        [-0.5,-0.5,-0.5, 0.9,0.9,0.2],[ 0.5,-0.5,-0.5, 0.9,0.9,0.2],[ 0.5,-0.5, 0.5, 0.9,0.9,0.2],
        [-0.5,-0.5,-0.5, 0.9,0.9,0.2],[ 0.5,-0.5, 0.5, 0.9,0.9,0.2],[-0.5,-0.5, 0.5, 0.9,0.9,0.2],
        [ 0.5,-0.5, 0.5, 0.2,0.9,0.9],[ 0.5,-0.5,-0.5, 0.2,0.9,0.9],[ 0.5, 0.5,-0.5, 0.2,0.9,0.9],
        [ 0.5,-0.5, 0.5, 0.2,0.9,0.9],[ 0.5, 0.5,-0.5, 0.2,0.9,0.9],[ 0.5, 0.5, 0.5, 0.2,0.9,0.9],
        [-0.5,-0.5,-0.5, 0.9,0.2,0.9],[-0.5,-0.5, 0.5, 0.9,0.2,0.9],[-0.5, 0.5, 0.5, 0.9,0.2,0.9],
        [-0.5,-0.5,-0.5, 0.9,0.2,0.9],[-0.5, 0.5, 0.5, 0.9,0.2,0.9],[-0.5, 0.5,-0.5, 0.9,0.2,0.9],
    ];

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct InstanceData {
        model: [[f32; 4]; 4],
    }

    fn identity() -> [[f32; 4]; 4] {
        [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]
    }
    fn translation(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
        let mut m = identity(); m[3] = [x, y, z, 1.0]; m
    }
    fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
        let f = 1.0 / (fov_y / 2.0).tan(); let nf = 1.0 / (near - far);
        [[f/aspect,0.0,0.0,0.0],[0.0,f,0.0,0.0],[0.0,0.0,(far+near)*nf,2.0*far*near*nf],[0.0,0.0,-1.0,0.0]]
    }
    fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
        let sub = |a: [f32;3], b: [f32;3]| [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
        let dot = |a: [f32;3], b: [f32;3]| a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
        let cross = |a: [f32;3], b: [f32;3]| [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
        let normalize = |v: [f32;3]| { let l = dot(v,v).sqrt(); [v[0]/l, v[1]/l, v[2]/l] };
        let f = normalize(sub(target, eye)); let s = normalize(cross(f, up)); let u = cross(s, f);
        [[s[0],s[1],s[2],-dot(s,eye)],[u[0],u[1],u[2],-dot(u,eye)],[-f[0],-f[1],-f[2],dot(f,eye)],[0.0,0.0,0.0,1.0]]
    }
    fn mat4_mul(a: [[f32;4];4], b: [[f32;4];4]) -> [[f32;4];4] {
        let mut o = [[0.0f32;4];4];
        for i in 0..4 { for j in 0..4 { for k in 0..4 { o[i][j] += a[i][k] * b[k][j]; }}} o
    }
    fn transpose(m: [[f32;4];4]) -> [[f32;4];4] {
        [[m[0][0],m[1][0],m[2][0],m[3][0]],[m[0][1],m[1][1],m[2][1],m[3][1]],
         [m[0][2],m[1][2],m[2][2],m[3][2]],[m[0][3],m[1][3],m[2][3],m[3][3]]]
    }

    struct App {
        entity_count: u32,
        window: Option<Window>,
        device: Option<MetalDevice>,
        pipeline: Option<<MetalDevice as RenderDevice>::RenderPipeline>,
        vertex_buffer: Option<<MetalDevice as RenderDevice>::Buffer>,
        vp_buffer: Option<<MetalDevice as RenderDevice>::Buffer>,
        instance_buffer: Option<<MetalDevice as RenderDevice>::Buffer>,
        // Low-res render targets
        color_tex: Option<<MetalDevice as RenderDevice>::Texture>,
        color_view: Option<<MetalDevice as RenderDevice>::TextureView>,
        depth_tex: Option<<MetalDevice as RenderDevice>::Texture>,
        depth_view: Option<<MetalDevice as RenderDevice>::TextureView>,
        motion_tex: Option<<MetalDevice as RenderDevice>::Texture>,
        // Full-res output
        output_tex: Option<<MetalDevice as RenderDevice>::Texture>,
        upscaler: Option<MetalFXUpscaler>,
        instances: Vec<InstanceData>,
        frame: u64,
        last_fps_time: std::time::Instant,
        fps_frame_count: u32,
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() { return; }
            let attrs = WindowAttributes::default()
                .with_title(format!("Euca Engine -- MetalFX Upscale ({} entities)", self.entity_count))
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
            let window = event_loop.create_window(attrs).unwrap();
            let device = MetalDevice::from_window(&window);
            device.set_display_sync_enabled(false);

            let (sw, sh) = device.surface_size();
            let (rw, rh) = (sw / 2, sh / 2); // Render at 50% resolution
            println!("MetalFX upscale: {} → {} | {} entities", format!("{}x{}", rw, rh), format!("{}x{}", sw, sh), self.entity_count);

            let shader = device.create_shader(&ShaderDesc {
                label: Some("Shader"), source: ShaderSource::Msl(SHADER_SRC.into()),
            });
            let pipeline = device.create_render_pipeline(&RenderPipelineDesc {
                label: Some("Pipeline"), layout: &[],
                vertex: VertexState { module: &shader, entry_point: "vertex_main", buffers: &[] },
                fragment: Some(FragmentState {
                    module: &shader, entry_point: "fragment_main",
                    targets: &[Some(ColorTargetState {
                        format: TextureFormat::Rgba16Float,
                        blend: Some(BlendState::REPLACE), write_mask: ColorWrites::ALL,
                    })],
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    cull_mode: Some(Face::Back), front_face: FrontFace::Ccw, ..Default::default()
                },
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float, depth_write_enabled: true,
                    depth_compare: CompareFunction::Less,
                    stencil: StencilState::default(), bias: DepthBiasState::default(),
                }),
                multisample: MultisampleState::default(),
            });

            // Vertex + instance buffers
            let vdata: Vec<u8> = CUBE_VERTICES.iter().flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes())).collect();
            let vertex_buffer = device.create_buffer(&BufferDesc {
                label: Some("Vertices"), size: vdata.len() as u64,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST, mapped_at_creation: false,
            });
            device.write_buffer(&vertex_buffer, 0, &vdata);
            let vp_buffer = device.create_buffer(&BufferDesc {
                label: Some("VP"), size: 64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST, mapped_at_creation: false,
            });

            let side = (self.entity_count as f32).sqrt().ceil() as u32;
            let spacing = 2.5_f32;
            let mut instances = Vec::with_capacity(self.entity_count as usize);
            for i in 0..self.entity_count {
                let (row, col) = (i / side, i % side);
                let x = (col as f32 - side as f32 / 2.0) * spacing;
                let z = (row as f32 - side as f32 / 2.0) * spacing;
                instances.push(InstanceData { model: transpose(translation(x, 0.5, z)) });
            }
            let inst_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(instances.as_ptr() as *const u8, instances.len() * std::mem::size_of::<InstanceData>())
            };
            let instance_buffer = device.create_buffer(&BufferDesc {
                label: Some("Instances"), size: inst_bytes.len() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST, mapped_at_creation: false,
            });
            device.write_buffer(&instance_buffer, 0, inst_bytes);

            // Low-res render targets (half resolution)
            let color_tex = device.create_texture(&TextureDesc {
                label: Some("LR Color"), size: Extent3d { width: rw.max(1), height: rh.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let color_view = device.create_texture_view(&color_tex, &TextureViewDesc::default());
            let depth_tex = device.create_texture(&TextureDesc {
                label: Some("LR Depth"), size: Extent3d { width: rw.max(1), height: rh.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let depth_view = device.create_texture_view(&depth_tex, &TextureViewDesc::default());
            // Motion vectors (zero for now — static scene)
            let motion_tex = device.create_texture(&TextureDesc {
                label: Some("Motion"), size: Extent3d { width: rw.max(1), height: rh.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: TextureDimension::D2,
                format: TextureFormat::Rg16Float,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            // Full-res output texture (MetalFX writes here)
            let output_tex = device.create_texture(&TextureDesc {
                label: Some("FX Output"), size: Extent3d { width: sw.max(1), height: sh.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });

            // MetalFX temporal upscaler
            let upscaler = device.create_temporal_upscaler(
                rw, rh, sw, sh,
                TextureFormat::Rgba16Float,
                TextureFormat::Depth32Float,
                TextureFormat::Rg16Float,
            );

            self.pipeline = Some(pipeline);
            self.vertex_buffer = Some(vertex_buffer);
            self.vp_buffer = Some(vp_buffer);
            self.instance_buffer = Some(instance_buffer);
            self.color_tex = Some(color_tex);
            self.color_view = Some(color_view);
            self.depth_tex = Some(depth_tex);
            self.depth_view = Some(depth_view);
            self.motion_tex = Some(motion_tex);
            self.output_tex = Some(output_tex);
            self.upscaler = Some(upscaler);
            self.instances = instances;
            self.device = Some(device);
            window.request_redraw();
            self.window = Some(window);
        }

        fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => {
                    if let Some(ref mut device) = self.device { device.resize_surface(size.width, size.height); }
                }
                WindowEvent::RedrawRequested => {
                    let device = self.device.as_ref().unwrap();
                    let pipeline = self.pipeline.as_ref().unwrap();
                    let vb = self.vertex_buffer.as_ref().unwrap();
                    let vp_buf = self.vp_buffer.as_ref().unwrap();
                    let inst_buf = self.instance_buffer.as_ref().unwrap();
                    let color_view = self.color_view.as_ref().unwrap();
                    let depth_view = self.depth_view.as_ref().unwrap();
                    let color_tex = self.color_tex.as_ref().unwrap();
                    let depth_tex = self.depth_tex.as_ref().unwrap();
                    let motion_tex = self.motion_tex.as_ref().unwrap();
                    let output_tex = self.output_tex.as_ref().unwrap();
                    let upscaler = self.upscaler.as_ref().unwrap();

                    self.frame += 1;
                    let t = self.frame as f32 * 0.005;

                    // FPS counter
                    self.fps_frame_count += 1;
                    let now = std::time::Instant::now();
                    let elapsed = now.duration_since(self.last_fps_time).as_secs_f32();
                    if elapsed >= 1.0 {
                        let fps = self.fps_frame_count as f32 / elapsed;
                        println!("[frame {}] {} entities | FPS: {:.1} (MetalFX 50% upscale)", self.frame, self.entity_count, fps);
                        if let Some(ref w) = self.window {
                            w.set_title(&format!("Euca Engine -- MetalFX Upscale ({} ent) | FPS: {:.0}", self.entity_count, fps));
                        }
                        self.fps_frame_count = 0;
                        self.last_fps_time = now;
                    }

                    // Camera
                    let area = (self.entity_count as f32).sqrt() * 2.5;
                    let eye = [t.cos() * area * 1.2, area * 0.6, t.sin() * area * 1.2];
                    let view = look_at(eye, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
                    let proj = perspective(std::f32::consts::FRAC_PI_4, device.aspect_ratio(), 0.1, area * 3.0);
                    let vp = transpose(mat4_mul(proj, view));
                    let vp_bytes: &[u8] = unsafe { std::slice::from_raw_parts(vp.as_ptr() as *const u8, 64) };
                    device.write_buffer(vp_buf, 0, vp_bytes);

                    // === Pass 1: Render at half resolution ===
                    let mut encoder = device.create_command_encoder(Some("Frame"));
                    {
                        let mut pass = device.begin_render_pass(&mut encoder, &RenderPassDesc {
                            label: Some("LowRes"),
                            color_attachments: &[Some(RenderPassColorAttachment {
                                view: color_view, resolve_target: None,
                                ops: Operations { load: LoadOp::Clear(Color { r: 0.05, g: 0.05, b: 0.2, a: 1.0 }), store: StoreOp::Store },
                            })],
                            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                                view: depth_view,
                                depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                                stencil_ops: None,
                            }),
                        });
                        pass.set_pipeline(pipeline);
                        pass.set_vertex_buffer(0, vb, 0, std::mem::size_of_val(CUBE_VERTICES) as u64);
                        pass.set_vertex_buffer(1, vp_buf, 0, 64);
                        pass.set_vertex_buffer(2, inst_buf, 0, (self.instances.len() * 64) as u64);
                        pass.draw(0..36, 0..self.entity_count);
                    }

                    // === Pass 2: MetalFX upscale (half → full res) ===
                    let reset = self.frame == 1;
                    upscaler.encode(&encoder, color_tex, depth_tex, motion_tex, output_tex, 0.0, 0.0, reset);

                    // === Pass 3: Blit upscaled result to swapchain ===
                    let surface = match device.get_current_texture() {
                        Ok(t) => t,
                        Err(_) => return,
                    };
                    device.blit_to_surface(&mut encoder, output_tex, &surface);

                    encoder.schedule_present(&surface);
                    device.submit(encoder);
                    device.present(surface);
                    self.window.as_ref().unwrap().request_redraw();
                }
                _ => {}
            }
        }
    }

    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        entity_count, window: None, device: None, pipeline: None,
        vertex_buffer: None, vp_buffer: None, instance_buffer: None,
        color_tex: None, color_view: None, depth_tex: None, depth_view: None,
        motion_tex: None, output_tex: None, upscaler: None,
        instances: Vec::new(), frame: 0,
        last_fps_time: std::time::Instant::now(), fps_frame_count: 0,
    };
    event_loop.run_app(&mut app).unwrap();
}
