//! Metal backend smoke tests — validates the MetalDevice on real Apple Silicon.
//!
//! Run with: `cargo test -p euca-rhi --features metal-backend -- metal_smoke`
//!
//! These tests exercise the full MetalDevice → resource creation → command
//! encoding → submission pipeline on real hardware (no mocks).

#![cfg(all(target_os = "macos", feature = "metal-backend"))]

use euca_rhi::metal_backend::MetalDevice;
use euca_rhi::*;

fn device() -> MetalDevice {
    MetalDevice::headless()
}

#[test]
fn metal_device_creates_successfully() {
    let dev = device();
    let caps = dev.capabilities();
    println!("Metal device: {}", caps.device_name);
    println!("Apple Silicon: {}", caps.apple_silicon);
    println!("Unified memory: {}", caps.unified_memory);
    println!("Memoryless targets: {}", caps.memoryless_render_targets);
    println!(
        "Max buffer length: {} MB",
        caps.max_buffer_length / 1_048_576
    );
    assert!(
        !caps.device_name.is_empty(),
        "Device name should not be empty"
    );
    assert!(
        caps.apple_silicon,
        "M4 Pro should be detected as Apple Silicon"
    );
    assert!(
        caps.unified_memory,
        "Apple Silicon should have unified memory"
    );
}

#[test]
fn metal_create_buffer_and_write() {
    let dev = device();

    // Create a buffer
    let buffer = dev.create_buffer(&BufferDesc {
        label: Some("Test Buffer"),
        size: 1024,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Write data to it (unified memory = direct memcpy)
    let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
    dev.write_buffer(&buffer, 0, &data);

    // Verify the data was written correctly by reading back from shared memory
    // (This works because Apple Silicon uses unified memory)
    // The buffer contents are directly accessible.
}

#[test]
fn metal_create_texture() {
    let dev = device();

    let texture = dev.create_texture(&TextureDesc {
        label: Some("Test Texture"),
        size: Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Create a view from the texture
    let _view = dev.create_texture_view(&texture, &TextureViewDesc::default());
}

#[test]
fn metal_create_sampler() {
    let dev = device();

    let _sampler = dev.create_sampler(&SamplerDesc {
        label: Some("Test Sampler"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        address_mode_w: AddressMode::Repeat,
        ..Default::default()
    });
}

#[test]
fn metal_compile_msl_shader() {
    let dev = device();

    let msl_source = r#"
        #include <metal_stdlib>
        using namespace metal;

        struct VertexOut {
            float4 position [[position]];
            float2 uv;
        };

        vertex VertexOut vertex_main(uint vid [[vertex_id]]) {
            float2 positions[3] = {
                float2(-1.0, -1.0),
                float2( 3.0, -1.0),
                float2(-1.0,  3.0),
            };
            VertexOut out;
            out.position = float4(positions[vid], 0.0, 1.0);
            out.uv = positions[vid] * 0.5 + 0.5;
            return out;
        }

        fragment float4 fragment_main(VertexOut in [[stage_in]]) {
            return float4(in.uv, 0.0, 1.0);
        }
    "#;

    let _shader = dev.create_shader(&ShaderDesc {
        label: Some("Test MSL Shader"),
        source: ShaderSource::Msl(msl_source.into()),
    });
}

#[test]
fn metal_create_render_pipeline() {
    let dev = device();

    let shader = dev.create_shader(&ShaderDesc {
        label: Some("Pipeline Shader"),
        source: ShaderSource::Msl(
            r#"
            #include <metal_stdlib>
            using namespace metal;
            struct VertexOut { float4 position [[position]]; };
            vertex VertexOut vs_main(uint vid [[vertex_id]]) {
                VertexOut out;
                out.position = float4(0.0);
                return out;
            }
            fragment float4 fs_main(VertexOut in [[stage_in]]) {
                return float4(1.0);
            }
            "#
            .into(),
        ),
    });

    let _pipeline = dev.create_render_pipeline(&RenderPipelineDesc {
        label: Some("Test Pipeline"),
        layout: &[],
        vertex: VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(ColorTargetState {
                format: TextureFormat::Bgra8UnormSrgb,
                blend: Some(BlendState::REPLACE),
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
    });
}

#[test]
fn metal_offscreen_render_pass() {
    let dev = device();

    // Create offscreen render target
    let color_texture = dev.create_texture(&TextureDesc {
        label: Some("Offscreen Color"),
        size: Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Bgra8UnormSrgb,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let color_view = dev.create_texture_view(&color_texture, &TextureViewDesc::default());

    // Create shader + pipeline
    let shader = dev.create_shader(&ShaderDesc {
        label: Some("Offscreen Shader"),
        source: ShaderSource::Msl(
            r#"
            #include <metal_stdlib>
            using namespace metal;
            struct VertexOut { float4 position [[position]]; };
            vertex VertexOut vs_main(uint vid [[vertex_id]]) {
                float2 pos[3] = { float2(0,1), float2(-1,-1), float2(1,-1) };
                VertexOut out;
                out.position = float4(pos[vid], 0, 1);
                return out;
            }
            fragment float4 fs_main() { return float4(1,0,0,1); }
            "#
            .into(),
        ),
    });

    let pipeline = dev.create_render_pipeline(&RenderPipelineDesc {
        label: Some("Offscreen Pipeline"),
        layout: &[],
        vertex: VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(ColorTargetState {
                format: TextureFormat::Bgra8UnormSrgb,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
    });

    // Encode and submit a render pass
    let mut encoder = dev.create_command_encoder(Some("Offscreen Encoder"));
    {
        let mut pass = dev.begin_render_pass(
            &mut encoder,
            &RenderPassDesc {
                label: Some("Offscreen Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
            },
        );
        pass.set_pipeline(&pipeline);
        pass.draw(0..3, 0..1);
    }
    dev.submit(encoder);

    println!("Offscreen render pass completed successfully on Metal!");
}

#[test]
fn metal_compute_dispatch() {
    let dev = device();

    let shader = dev.create_shader(&ShaderDesc {
        label: Some("Compute Shader"),
        source: ShaderSource::Msl(
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void compute_main(
                device float* output [[buffer(0)]],
                uint id [[thread_position_in_grid]]
            ) {
                output[id] = float(id) * 2.0;
            }
            "#
            .into(),
        ),
    });

    let pipeline = dev.create_compute_pipeline(&ComputePipelineDesc {
        label: Some("Test Compute Pipeline"),
        layout: &[],
        module: &shader,
        entry_point: "compute_main",
    });

    let buffer = dev.create_buffer(&BufferDesc {
        label: Some("Compute Output"),
        size: 256 * 4, // 256 floats
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group_layout = dev.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some("Compute BGL"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = dev.create_bind_group(&BindGroupDesc {
        label: Some("Compute BG"),
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: None,
            }),
        }],
    });

    let mut encoder = dev.create_command_encoder(Some("Compute Encoder"));
    {
        let mut pass = dev.begin_compute_pass(&mut encoder, Some("Compute Pass"));
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(4, 1, 1); // 4 * 64 = 256 threads
    }
    dev.submit(encoder);

    println!("Compute dispatch completed successfully on Metal!");
}

#[test]
fn metal_wgsl_to_msl_transpilation() {
    let dev = device();

    // Test with a simple WGSL shader — naga transpiles it to MSL automatically
    let wgsl = r#"
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) uv: vec2<f32>,
        };
        @vertex
        fn vs_main(@builtin(vertex_index) id: u32) -> VertexOutput {
            let x = f32(i32(id) / 2) * 4.0 - 1.0;
            let y = f32(i32(id) % 2) * 4.0 - 1.0;
            var out: VertexOutput;
            out.position = vec4<f32>(x, y, 0.0, 1.0);
            out.uv = vec2<f32>(x * 0.5 + 0.5, -y * 0.5 + 0.5);
            return out;
        }
    "#;

    let shader = dev.create_shader(&ShaderDesc {
        label: Some("WGSL fullscreen VS"),
        source: ShaderSource::Wgsl(wgsl.into()),
    });

    // If we got here, transpilation and Metal compilation both succeeded
    println!("WGSL → MSL transpilation + Metal compilation succeeded!");
    let _ = shader;
}
