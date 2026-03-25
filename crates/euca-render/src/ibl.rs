//! Image-Based Lighting (IBL) resource generation pipeline.
//!
//! Generates the three GPU textures required for split-sum PBR environment
//! lighting:
//!
//! 1. **BRDF LUT** -- A 512x512 Rg16Float lookup table encoding the GGX
//!    split-sum scale and bias factors as a function of `(NdotV, roughness)`.
//!    Generated once at init and never changes.
//!
//! 2. **Irradiance cubemap** -- A 64x64-per-face Rgba16Float cubemap storing
//!    cosine-weighted hemisphere integrals of the environment map. Used for
//!    diffuse indirect lighting.
//!
//! 3. **Pre-filtered specular cubemap** -- A 256x256-per-face Rgba16Float
//!    cubemap with 5 mip levels. Each mip corresponds to increasing roughness.
//!    Used with the BRDF LUT for specular indirect lighting.
//!
//! # Usage
//!
//! ```ignore
//! // With an environment cubemap:
//! let ibl = IblResources::generate(&device, &queue, &env_cubemap_view);
//!
//! // Without an environment cubemap (uniform color fallback):
//! let ibl = IblResources::from_uniform_color(&device, &queue, [0.2, 0.2, 0.3]);
//! ```

/// BRDF LUT resolution (square).
pub const BRDF_LUT_SIZE: u32 = 512;
/// Irradiance cubemap face resolution.
pub const IRRADIANCE_SIZE: u32 = 64;
/// Specular cubemap base mip resolution.
pub const SPECULAR_SIZE: u32 = 256;
/// Number of mip levels in the specular cubemap.
pub const SPECULAR_MIP_COUNT: u32 = 5;
/// Number of cubemap faces.
const CUBEMAP_FACES: u32 = 6;

const BRDF_LUT_SHADER: &str = include_str!("../shaders/brdf_lut.wgsl");
const IRRADIANCE_SHADER: &str = include_str!("../shaders/ibl_irradiance.wgsl");
const SPECULAR_SHADER: &str = include_str!("../shaders/ibl_specular.wgsl");

// ---------------------------------------------------------------------------
// GPU uniform structs (must match WGSL layouts)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct IrradianceParams {
    face: u32,
    size: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SpecularParams {
    face: u32,
    mip_level: u32,
    mip_count: u32,
    size: u32,
}

// ---------------------------------------------------------------------------
// IblResources
// ---------------------------------------------------------------------------

/// Pre-computed IBL textures ready for binding in the PBR shader.
pub struct IblResources {
    /// BRDF integration LUT (Rg16Float, 512x512).
    pub brdf_lut: wgpu::Texture,
    /// View into the BRDF LUT for shader binding.
    pub brdf_lut_view: wgpu::TextureView,
    /// Diffuse irradiance cubemap (Rgba16Float, 64x64 per face).
    pub irradiance_cubemap: wgpu::Texture,
    /// Cube view into the irradiance cubemap.
    pub irradiance_view: wgpu::TextureView,
    /// Pre-filtered specular cubemap (Rgba16Float, 256x256 base, 5 mip levels).
    pub specular_cubemap: wgpu::Texture,
    /// Cube view into the specular cubemap (all mip levels).
    pub specular_view: wgpu::TextureView,
    /// Trilinear-filtering sampler for cubemap lookups.
    pub cubemap_sampler: wgpu::Sampler,
}

impl IblResources {
    /// Generate IBL resources from a source environment cubemap.
    ///
    /// Dispatches three compute shaders to produce the BRDF LUT, irradiance
    /// cubemap, and pre-filtered specular cubemap. All work is recorded into
    /// a single command buffer and submitted synchronously.
    pub fn generate(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source_cubemap_view: &wgpu::TextureView,
    ) -> Self {
        let brdf_lut = Self::create_brdf_lut_texture(device);
        let brdf_lut_view = brdf_lut.create_view(&wgpu::TextureViewDescriptor::default());

        let (irradiance_cubemap, irradiance_view) =
            Self::create_cubemap_texture(device, "ibl_irradiance", IRRADIANCE_SIZE, 1);
        let (specular_cubemap, specular_view) =
            Self::create_cubemap_texture(device, "ibl_specular", SPECULAR_SIZE, SPECULAR_MIP_COUNT);

        let cubemap_sampler = Self::create_cubemap_sampler(device);
        let source_sampler = Self::create_source_sampler(device);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ibl_generate"),
        });

        // --- 1. BRDF LUT ---
        Self::dispatch_brdf_lut(device, &mut encoder, &brdf_lut);

        // --- 2. Irradiance convolution ---
        Self::dispatch_irradiance(
            device,
            queue,
            &mut encoder,
            source_cubemap_view,
            &source_sampler,
            &irradiance_cubemap,
        );

        // --- 3. Specular pre-filter ---
        Self::dispatch_specular(
            device,
            queue,
            &mut encoder,
            source_cubemap_view,
            &source_sampler,
            &specular_cubemap,
        );

        queue.submit(std::iter::once(encoder.finish()));

        Self {
            brdf_lut,
            brdf_lut_view,
            irradiance_cubemap,
            irradiance_view,
            specular_cubemap,
            specular_view,
            cubemap_sampler,
        }
    }

    /// Generate IBL resources from a uniform solid color.
    ///
    /// Creates a tiny solid-color cubemap as the source environment, then runs
    /// the full generation pipeline. Use this when no HDR environment map is
    /// available (e.g. during initial engine bring-up).
    pub fn from_uniform_color(device: &wgpu::Device, queue: &wgpu::Queue, color: [f32; 3]) -> Self {
        let (source_cubemap, source_view) = Self::create_solid_color_cubemap(device, queue, color);
        let _ = source_cubemap; // keep alive until generate() submits
        Self::generate(device, queue, &source_view)
    }

    // -----------------------------------------------------------------------
    // Texture creation helpers
    // -----------------------------------------------------------------------

    fn create_brdf_lut_texture(device: &wgpu::Device) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl_brdf_lut"),
            size: wgpu::Extent3d {
                width: BRDF_LUT_SIZE,
                height: BRDF_LUT_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // The compute shader writes rg32float for precision; we store in
            // Rg16Float for the final binding. We use rg32float as the storage
            // format since compute writes require exact format match, then the
            // view will interpret as Rg16Float for sampling.
            //
            // Actually, wgpu requires storage texture format to match the
            // texture format exactly. We use Rgba16Float because Rg32Float is
            // NOT filterable on Apple Silicon (Metal) — binding it with a
            // filtering sampler panics. Rgba16Float is filterable and a valid
            // compute storage format. We store (scale, bias) in .rg channels.
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn create_cubemap_texture(
        device: &wgpu::Device,
        label: &str,
        face_size: u32,
        mip_levels: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: face_size,
                height: face_size,
                depth_or_array_layers: CUBEMAP_FACES,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(label),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        (texture, view)
    }

    fn create_cubemap_sampler(device: &wgpu::Device) -> wgpu::Sampler {
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ibl_cubemap_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        })
    }

    fn create_source_sampler(device: &wgpu::Device) -> wgpu::Sampler {
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ibl_source_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        })
    }

    /// Create a tiny solid-color cubemap to use as the source environment.
    fn create_solid_color_cubemap(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color: [f32; 3],
    ) -> (wgpu::Texture, wgpu::TextureView) {
        // 1x1 per face is sufficient for a uniform environment.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ibl_solid_color_source"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: CUBEMAP_FACES,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Encode color as f16 (using half crate would be cleaner, but we can
        // use f32-to-f16 bit conversion inline to avoid an extra dependency).
        let pixel = [
            f32_to_f16(color[0]),
            f32_to_f16(color[1]),
            f32_to_f16(color[2]),
            f32_to_f16(1.0),
        ];
        let pixel_bytes = bytemuck::cast_slice(&pixel);

        for face in 0..CUBEMAP_FACES {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: face,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                pixel_bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(8), // 4 x f16 = 8 bytes
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ibl_solid_color_source"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        (texture, view)
    }

    // -----------------------------------------------------------------------
    // Compute dispatch helpers
    // -----------------------------------------------------------------------

    fn dispatch_brdf_lut(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        brdf_lut: &wgpu::Texture,
    ) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brdf_lut_shader"),
            source: wgpu::ShaderSource::Wgsl(BRDF_LUT_SHADER.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brdf_lut_pipeline"),
            layout: None, // auto layout
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bgl = pipeline.get_bind_group_layout(0);
        let output_view = brdf_lut.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brdf_lut_bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&output_view),
            }],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("brdf_lut_compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(BRDF_LUT_SIZE.div_ceil(8), BRDF_LUT_SIZE.div_ceil(8), 1);
    }

    fn dispatch_irradiance(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        source_cubemap_view: &wgpu::TextureView,
        source_sampler: &wgpu::Sampler,
        irradiance_cubemap: &wgpu::Texture,
    ) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ibl_irradiance_shader"),
            source: wgpu::ShaderSource::Wgsl(IRRADIANCE_SHADER.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ibl_irradiance_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bgl = pipeline.get_bind_group_layout(0);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ibl_irradiance_params"),
            size: std::mem::size_of::<IrradianceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        for face in 0..CUBEMAP_FACES {
            let params = IrradianceParams {
                face,
                size: IRRADIANCE_SIZE,
                _pad0: 0,
                _pad1: 0,
            };
            queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&params));

            let output_view = irradiance_cubemap.create_view(&wgpu::TextureViewDescriptor {
                label: Some("ibl_irradiance_face_view"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                base_array_layer: 0,
                array_layer_count: Some(CUBEMAP_FACES),
                base_mip_level: 0,
                mip_level_count: Some(1),
                ..Default::default()
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ibl_irradiance_bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(source_cubemap_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(source_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&output_view),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ibl_irradiance_compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(IRRADIANCE_SIZE.div_ceil(8), IRRADIANCE_SIZE.div_ceil(8), 1);
        }
    }

    fn dispatch_specular(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        source_cubemap_view: &wgpu::TextureView,
        source_sampler: &wgpu::Sampler,
        specular_cubemap: &wgpu::Texture,
    ) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ibl_specular_shader"),
            source: wgpu::ShaderSource::Wgsl(SPECULAR_SHADER.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ibl_specular_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bgl = pipeline.get_bind_group_layout(0);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ibl_specular_params"),
            size: std::mem::size_of::<SpecularParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        for mip in 0..SPECULAR_MIP_COUNT {
            let mip_size = SPECULAR_SIZE >> mip;

            for face in 0..CUBEMAP_FACES {
                let params = SpecularParams {
                    face,
                    mip_level: mip,
                    mip_count: SPECULAR_MIP_COUNT,
                    size: mip_size,
                };
                queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&params));

                let output_view = specular_cubemap.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("ibl_specular_mip_view"),
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    base_array_layer: 0,
                    array_layer_count: Some(CUBEMAP_FACES),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ibl_specular_bg"),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(source_cubemap_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(source_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&output_view),
                        },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ibl_specular_compute"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, Some(&bind_group), &[]);
                pass.dispatch_workgroups(mip_size.div_ceil(8), mip_size.div_ceil(8), 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// f32 -> f16 conversion (IEEE 754 half-precision)
// ---------------------------------------------------------------------------

/// Convert an f32 to an f16 stored as u16 (IEEE 754 half-precision).
///
/// Handles normal numbers, denorms, infinities, and NaN. Used to write
/// f16 texture data without pulling in a dedicated half-float crate.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 0xFF {
        // Inf or NaN
        let half_mantissa = if mantissa != 0 { 0x0200 } else { 0 };
        return (sign | 0x7C00 | half_mantissa) as u16;
    }

    // Rebias exponent from f32 bias (127) to f16 bias (15).
    let new_exp = exponent - 127 + 15;

    if new_exp >= 0x1F {
        // Overflow -> infinity.
        return (sign | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        // Denormalized or zero.
        if new_exp < -10 {
            return sign as u16; // too small, flush to zero
        }
        let m = (mantissa | 0x0080_0000) >> (1 - new_exp + 13);
        return (sign | m) as u16;
    }

    (sign | ((new_exp as u32) << 10) | (mantissa >> 13)) as u16
}

// ---------------------------------------------------------------------------
// Pure helper functions (testable without GPU)
// ---------------------------------------------------------------------------

/// Compute the expected number of workgroups for a BRDF LUT dispatch.
pub fn brdf_lut_workgroups() -> [u32; 3] {
    [BRDF_LUT_SIZE.div_ceil(8), BRDF_LUT_SIZE.div_ceil(8), 1]
}

/// Compute the expected number of workgroups for one face of the irradiance
/// convolution dispatch.
pub fn irradiance_workgroups() -> [u32; 3] {
    [IRRADIANCE_SIZE.div_ceil(8), IRRADIANCE_SIZE.div_ceil(8), 1]
}

/// Compute the expected number of workgroups for one face of the specular
/// pre-filter at the given mip level.
pub fn specular_workgroups(mip: u32) -> [u32; 3] {
    let mip_size = SPECULAR_SIZE >> mip;
    [mip_size.div_ceil(8), mip_size.div_ceil(8), 1]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brdf_lut_dimensions() {
        assert_eq!(BRDF_LUT_SIZE, 512);
    }

    #[test]
    fn irradiance_cubemap_dimensions() {
        assert_eq!(IRRADIANCE_SIZE, 64);
    }

    #[test]
    fn specular_cubemap_has_five_mip_levels() {
        assert_eq!(SPECULAR_MIP_COUNT, 5);
        // Verify mip chain: 256, 128, 64, 32, 16
        for mip in 0..SPECULAR_MIP_COUNT {
            let expected = SPECULAR_SIZE >> mip;
            assert!(expected > 0, "Mip level {mip} should have non-zero size");
        }
        assert_eq!(SPECULAR_SIZE >> 0, 256);
        assert_eq!(SPECULAR_SIZE >> 1, 128);
        assert_eq!(SPECULAR_SIZE >> 2, 64);
        assert_eq!(SPECULAR_SIZE >> 3, 32);
        assert_eq!(SPECULAR_SIZE >> 4, 16);
    }

    #[test]
    fn brdf_lut_workgroups_correct() {
        let wg = brdf_lut_workgroups();
        assert_eq!(wg, [64, 64, 1]);
    }

    #[test]
    fn irradiance_workgroups_correct() {
        let wg = irradiance_workgroups();
        assert_eq!(wg, [8, 8, 1]);
    }

    #[test]
    fn specular_workgroups_decrease_with_mip() {
        let wg0 = specular_workgroups(0);
        let wg1 = specular_workgroups(1);
        let wg4 = specular_workgroups(4);
        assert_eq!(wg0, [32, 32, 1]);
        assert_eq!(wg1, [16, 16, 1]);
        assert_eq!(wg4, [2, 2, 1]);
    }

    #[test]
    fn uniform_params_layout() {
        assert_eq!(std::mem::size_of::<IrradianceParams>(), 16);
        assert_eq!(std::mem::size_of::<SpecularParams>(), 16);
    }

    #[test]
    fn uniform_params_are_pod() {
        let irr = IrradianceParams {
            face: 3,
            size: 64,
            _pad0: 0,
            _pad1: 0,
        };
        let bytes = bytemuck::bytes_of(&irr);
        assert_eq!(bytes.len(), 16);

        let spec = SpecularParams {
            face: 5,
            mip_level: 2,
            mip_count: 5,
            size: 64,
        };
        let bytes = bytemuck::bytes_of(&spec);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn f32_to_f16_basic_values() {
        // 0.0
        assert_eq!(f32_to_f16(0.0), 0x0000);
        // 1.0 = sign=0, exp=15, mantissa=0 => 0_01111_0000000000 = 0x3C00
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        // -1.0
        assert_eq!(f32_to_f16(-1.0), 0xBC00);
        // Infinity
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
        // -Infinity
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xFC00);
        // NaN should have exponent=0x1F and non-zero mantissa
        let nan_h = f32_to_f16(f32::NAN);
        assert_eq!(nan_h & 0x7C00, 0x7C00);
        assert_ne!(nan_h & 0x03FF, 0);
    }

    #[test]
    fn f32_to_f16_small_values() {
        // 0.5 = 0_01110_0000000000 = 0x3800
        assert_eq!(f32_to_f16(0.5), 0x3800);
        // 2.0 = 0_10000_0000000000 = 0x4000
        assert_eq!(f32_to_f16(2.0), 0x4000);
    }

    #[test]
    fn shader_sources_are_valid() {
        assert!(!BRDF_LUT_SHADER.is_empty());
        assert!(BRDF_LUT_SHADER.contains("@compute"));
        assert!(BRDF_LUT_SHADER.contains("@workgroup_size(8, 8)"));

        assert!(!IRRADIANCE_SHADER.is_empty());
        assert!(IRRADIANCE_SHADER.contains("@compute"));
        assert!(IRRADIANCE_SHADER.contains("@workgroup_size(8, 8)"));

        assert!(!SPECULAR_SHADER.is_empty());
        assert!(SPECULAR_SHADER.contains("@compute"));
        assert!(SPECULAR_SHADER.contains("@workgroup_size(8, 8)"));
    }

    #[test]
    fn total_specular_dispatches() {
        // 6 faces * 5 mip levels = 30 total dispatches.
        assert_eq!(CUBEMAP_FACES * SPECULAR_MIP_COUNT, 30);
    }

    #[test]
    fn total_irradiance_dispatches() {
        // 6 faces * 1 = 6 total dispatches.
        assert_eq!(CUBEMAP_FACES, 6);
    }
}
