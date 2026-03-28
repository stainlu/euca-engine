//! Native Metal backend — implements [`RenderDevice`] using Apple's Metal API
//! via `objc2-metal` for direct GPU access on Apple Silicon.
//!
//! This backend bypasses wgpu to access Metal 3/4 features that the WebGPU
//! spec cannot express: mesh shaders, tile shading, indirect command buffers,
//! MetalFX upscaling, memoryless render targets, and MPS compute.

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

use crate::RenderDevice;
use crate::pass::{ComputePassOps, RenderPassOps};
use crate::types::*;

// ===========================================================================
// Metal wrapper types (Send + Sync)
// ===========================================================================

// Metal protocol objects from objc2 are not Send/Sync by default.
// Metal guarantees thread safety for resource creation and command buffer
// creation from different threads. Encoding must happen on one thread per
// encoder, which our single-encoder-per-frame model satisfies.

struct SendSync<T>(T);
unsafe impl<T> Send for SendSync<T> {}
unsafe impl<T> Sync for SendSync<T> {}

impl<T> std::ops::Deref for SendSync<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

// ===========================================================================
// Associated types for MetalDevice
// ===========================================================================

pub struct MetalBuffer(SendSync<Retained<ProtocolObject<dyn MTLBuffer>>>);
pub struct MetalTexture(SendSync<Retained<ProtocolObject<dyn MTLTexture>>>);
pub struct MetalTextureView(SendSync<Retained<ProtocolObject<dyn MTLTexture>>>);
pub struct MetalSampler(SendSync<Retained<ProtocolObject<dyn MTLSamplerState>>>);
pub struct MetalShaderModule(SendSync<Retained<ProtocolObject<dyn MTLLibrary>>>);
pub struct MetalRenderPipeline(SendSync<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>);
pub struct MetalComputePipeline(SendSync<Retained<ProtocolObject<dyn MTLComputePipelineState>>>);
pub struct MetalBindGroupLayout {
    entries: Vec<BindGroupLayoutEntry>,
}
pub struct MetalBindGroup {
    buffers: Vec<(u32, MetalBuffer)>,
    textures: Vec<(u32, MetalTextureView)>,
    samplers: Vec<(u32, MetalSampler)>,
}

pub struct MetalCommandEncoder {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

pub struct MetalRenderPass<'a> {
    encoder: Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>,
    _marker: std::marker::PhantomData<&'a ()>,
}

pub struct MetalComputePass<'a> {
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    _marker: std::marker::PhantomData<&'a ()>,
}

pub struct MetalSurfaceTexture {
    drawable: Retained<ProtocolObject<dyn CAMetalDrawable>>,
}

// ===========================================================================
// MetalDevice
// ===========================================================================

/// Native Metal GPU backend for Apple Silicon.
///
/// Provides direct access to Metal 3/4 features unavailable through wgpu:
/// mesh shaders, tile shading, indirect command buffers, MetalFX upscaling,
/// memoryless render targets, and MPS compute.
pub struct MetalDevice {
    device: SendSync<Retained<ProtocolObject<dyn MTLDevice>>>,
    queue: SendSync<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
    layer: SendSync<Retained<CAMetalLayer>>,
    surface_width: u32,
    surface_height: u32,
    surface_format: TextureFormat,
    capabilities: Capabilities,
}

impl MetalDevice {
    /// Create a new MetalDevice from a CAMetalLayer (obtained from the window).
    ///
    /// # Safety
    /// The `layer` must be a valid CAMetalLayer attached to a visible view.
    pub unsafe fn new(layer: Retained<CAMetalLayer>, width: u32, height: u32) -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal-capable GPU found");

        let queue = device
            .newCommandQueue()
            .expect("Failed to create Metal command queue");

        // Configure the layer
        layer.setDevice(Some(&device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm_sRGB);
        layer.setDrawableSize(objc2_foundation::NSSize {
            width: width as f64,
            height: height as f64,
        });

        let capabilities = Capabilities {
            unified_memory: true, // All Apple Silicon has unified memory
            multi_draw_indirect: true,
            multi_draw_indirect_count: true,
            texture_binding_array: true,
            non_uniform_indexing: true,
            max_texture_dimension_2d: 16384,
            max_bind_groups: 31, // Metal argument buffer slots
            max_bindings_per_bind_group: 1024,
            max_binding_array_elements: 500_000, // Metal has very high limits
        };

        Self {
            device: SendSync(device),
            queue: SendSync(queue),
            layer: SendSync(layer),
            surface_width: width,
            surface_height: height,
            surface_format: TextureFormat::Bgra8UnormSrgb,
            capabilities,
        }
    }
}

// ===========================================================================
// Type conversion helpers
// ===========================================================================

fn to_mtl_pixel_format(format: TextureFormat) -> MTLPixelFormat {
    match format {
        TextureFormat::R8Unorm => MTLPixelFormat::R8Unorm,
        TextureFormat::R8Snorm => MTLPixelFormat::R8Snorm,
        TextureFormat::R8Uint => MTLPixelFormat::R8Uint,
        TextureFormat::R16Float => MTLPixelFormat::R16Float,
        TextureFormat::Rg8Unorm => MTLPixelFormat::RG8Unorm,
        TextureFormat::R32Float => MTLPixelFormat::R32Float,
        TextureFormat::R32Uint => MTLPixelFormat::R32Uint,
        TextureFormat::Rg16Float => MTLPixelFormat::RG16Float,
        TextureFormat::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        TextureFormat::Rgba8UnormSrgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        TextureFormat::Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        TextureFormat::Bgra8UnormSrgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        TextureFormat::Rg32Float => MTLPixelFormat::RG32Float,
        TextureFormat::Rgba16Float => MTLPixelFormat::RGBA16Float,
        TextureFormat::Rgba32Float => MTLPixelFormat::RGBA32Float,
        TextureFormat::Depth16Unorm => MTLPixelFormat::Depth16Unorm,
        TextureFormat::Depth32Float => MTLPixelFormat::Depth32Float,
        TextureFormat::Depth24Plus => MTLPixelFormat::Depth32Float, // no exact match
        TextureFormat::Depth24PlusStencil8 => MTLPixelFormat::Depth32Float_Stencil8,
        TextureFormat::Depth32FloatStencil8 => MTLPixelFormat::Depth32Float_Stencil8,
    }
}

fn to_mtl_storage_mode_for_usage(usage: TextureUsages) -> MTLStorageMode {
    // Apple Silicon: always use shared memory for best performance
    if usage.contains(TextureUsages::RENDER_ATTACHMENT) {
        MTLStorageMode::Private // render targets in tile memory
    } else {
        MTLStorageMode::Shared // everything else in unified memory
    }
}

fn to_mtl_texture_usage(usage: TextureUsages) -> MTLTextureUsage {
    let mut mtl = MTLTextureUsage::empty();
    if usage.contains(TextureUsages::TEXTURE_BINDING) {
        mtl |= MTLTextureUsage::ShaderRead;
    }
    if usage.contains(TextureUsages::STORAGE_BINDING) {
        mtl |= MTLTextureUsage::ShaderWrite;
    }
    if usage.contains(TextureUsages::RENDER_ATTACHMENT) {
        mtl |= MTLTextureUsage::RenderTarget;
    }
    mtl
}

fn to_mtl_sampler_address_mode(mode: AddressMode) -> MTLSamplerAddressMode {
    match mode {
        AddressMode::ClampToEdge => MTLSamplerAddressMode::ClampToEdge,
        AddressMode::Repeat => MTLSamplerAddressMode::Repeat,
        AddressMode::MirrorRepeat => MTLSamplerAddressMode::MirrorRepeat,
    }
}

fn to_mtl_sampler_filter(mode: FilterMode) -> MTLSamplerMinMagFilter {
    match mode {
        FilterMode::Nearest => MTLSamplerMinMagFilter::Nearest,
        FilterMode::Linear => MTLSamplerMinMagFilter::Linear,
    }
}

fn to_mtl_sampler_mip_filter(mode: FilterMode) -> MTLSamplerMipFilter {
    match mode {
        FilterMode::Nearest => MTLSamplerMipFilter::Nearest,
        FilterMode::Linear => MTLSamplerMipFilter::Linear,
    }
}

fn to_mtl_compare_function(cf: CompareFunction) -> MTLCompareFunction {
    match cf {
        CompareFunction::Never => MTLCompareFunction::Never,
        CompareFunction::Less => MTLCompareFunction::Less,
        CompareFunction::Equal => MTLCompareFunction::Equal,
        CompareFunction::LessEqual => MTLCompareFunction::LessEqual,
        CompareFunction::Greater => MTLCompareFunction::Greater,
        CompareFunction::NotEqual => MTLCompareFunction::NotEqual,
        CompareFunction::GreaterEqual => MTLCompareFunction::GreaterEqual,
        CompareFunction::Always => MTLCompareFunction::Always,
    }
}

fn to_mtl_primitive_type(topology: PrimitiveTopology) -> MTLPrimitiveType {
    match topology {
        PrimitiveTopology::PointList => MTLPrimitiveType::Point,
        PrimitiveTopology::LineList => MTLPrimitiveType::Line,
        PrimitiveTopology::LineStrip => MTLPrimitiveType::LineStrip,
        PrimitiveTopology::TriangleList => MTLPrimitiveType::Triangle,
        PrimitiveTopology::TriangleStrip => MTLPrimitiveType::TriangleStrip,
    }
}

fn to_mtl_index_type(format: IndexFormat) -> MTLIndexType {
    match format {
        IndexFormat::Uint16 => MTLIndexType::UInt16,
        IndexFormat::Uint32 => MTLIndexType::UInt32,
    }
}

fn to_mtl_load_action_color(op: &LoadOp<Color>) -> (MTLLoadAction, MTLClearColor) {
    match op {
        LoadOp::Clear(c) => (
            MTLLoadAction::Clear,
            MTLClearColor {
                red: c.r,
                green: c.g,
                blue: c.b,
                alpha: c.a,
            },
        ),
        LoadOp::Load => (
            MTLLoadAction::Load,
            MTLClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            },
        ),
    }
}

fn to_mtl_store_action(op: StoreOp) -> MTLStoreAction {
    match op {
        StoreOp::Store => MTLStoreAction::Store,
        StoreOp::Discard => MTLStoreAction::DontCare,
    }
}

// ===========================================================================
// RenderDevice implementation
// ===========================================================================

impl RenderDevice for MetalDevice {
    type Buffer = MetalBuffer;
    type Texture = MetalTexture;
    type TextureView = MetalTextureView;
    type Sampler = MetalSampler;
    type BindGroupLayout = MetalBindGroupLayout;
    type BindGroup = MetalBindGroup;
    type ShaderModule = MetalShaderModule;
    type RenderPipeline = MetalRenderPipeline;
    type ComputePipeline = MetalComputePipeline;
    type CommandEncoder = MetalCommandEncoder;
    type RenderPass<'a> = MetalRenderPass<'a>;
    type ComputePass<'a> = MetalComputePass<'a>;
    type SurfaceTexture = MetalSurfaceTexture;

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn create_buffer(&self, desc: &BufferDesc) -> MetalBuffer {
        // Apple Silicon: always use StorageModeShared (unified memory)
        let options = MTLResourceOptions::StorageModeShared;
        let buffer = self
            .device
            .newBufferWithLength_options(desc.size as usize, options)
            .expect("Failed to create Metal buffer");
        if let Some(label) = desc.label {
            buffer.setLabel(Some(&NSString::from_str(label)));
        }
        MetalBuffer(SendSync(buffer))
    }

    fn create_texture(&self, desc: &TextureDesc) -> MetalTexture {
        let mtl_desc = unsafe {
            let d = MTLTextureDescriptor::new();
            d.setTextureType(match desc.dimension {
                TextureDimension::D1 => MTLTextureType::Type1D,
                TextureDimension::D2 => {
                    if desc.size.depth_or_array_layers > 1 {
                        MTLTextureType::Type2DArray
                    } else if desc.sample_count > 1 {
                        MTLTextureType::Type2DMultisample
                    } else {
                        MTLTextureType::Type2D
                    }
                }
                TextureDimension::D3 => MTLTextureType::Type3D,
            });
            d.setPixelFormat(to_mtl_pixel_format(desc.format));
            d.setWidth(desc.size.width as usize);
            d.setHeight(desc.size.height as usize);
            d.setDepth(if desc.dimension == TextureDimension::D3 {
                desc.size.depth_or_array_layers as usize
            } else {
                1
            });
            if desc.dimension != TextureDimension::D3 && desc.size.depth_or_array_layers > 1 {
                d.setArrayLength(desc.size.depth_or_array_layers as usize);
            }
            d.setMipmapLevelCount(desc.mip_level_count as usize);
            d.setSampleCount(desc.sample_count as usize);
            d.setStorageMode(to_mtl_storage_mode_for_usage(desc.usage));
            d.setUsage(to_mtl_texture_usage(desc.usage));
            d
        };
        let texture = self
            .device
            .newTextureWithDescriptor(&mtl_desc)
            .expect("Failed to create Metal texture");
        if let Some(label) = desc.label {
            texture.setLabel(Some(&NSString::from_str(label)));
        }
        MetalTexture(SendSync(texture))
    }

    fn create_texture_view(
        &self,
        texture: &MetalTexture,
        desc: &TextureViewDesc,
    ) -> MetalTextureView {
        // Metal doesn't have separate "views" like wgpu. A texture IS its own view.
        // For different view dimensions (e.g., cube from 2D array), use newTextureViewWithPixelFormat.
        let view = if let Some(dim) = desc.dimension {
            let mtl_type = match dim {
                TextureViewDimension::D1 => MTLTextureType::Type1D,
                TextureViewDimension::D2 => MTLTextureType::Type2D,
                TextureViewDimension::D2Array => MTLTextureType::Type2DArray,
                TextureViewDimension::Cube => MTLTextureType::TypeCube,
                TextureViewDimension::CubeArray => MTLTextureType::TypeCubeArray,
                TextureViewDimension::D3 => MTLTextureType::Type3D,
            };
            let pixel_format = desc
                .format
                .map(to_mtl_pixel_format)
                .unwrap_or_else(|| texture.0.pixelFormat());
            let mip_count = desc
                .mip_level_count
                .unwrap_or(texture.0.mipmapLevelCount() as u32 - desc.base_mip_level);
            let levels = objc2_foundation::NSRange {
                location: desc.base_mip_level as usize,
                length: mip_count as usize,
            };
            let array_count = desc
                .array_layer_count
                .unwrap_or(texture.0.arrayLength().max(1) as u32 - desc.base_array_layer);
            let slices = objc2_foundation::NSRange {
                location: desc.base_array_layer as usize,
                length: array_count as usize,
            };
            unsafe {
                texture
                    .0
                    .newTextureViewWithPixelFormat_textureType_levels_slices(
                        pixel_format,
                        mtl_type,
                        levels,
                        slices,
                    )
                    .expect("Failed to create Metal texture view")
            }
        } else {
            // Return the texture itself as a "view" (Metal textures are their own views)
            Retained::clone(&texture.0.0)
        };
        MetalTextureView(SendSync(view))
    }

    fn create_sampler(&self, desc: &SamplerDesc) -> MetalSampler {
        let mtl_desc = {
            let d = MTLSamplerDescriptor::new();
            d.setMinFilter(to_mtl_sampler_filter(desc.min_filter));
            d.setMagFilter(to_mtl_sampler_filter(desc.mag_filter));
            d.setMipFilter(to_mtl_sampler_mip_filter(desc.mipmap_filter));
            d.setSAddressMode(to_mtl_sampler_address_mode(desc.address_mode_u));
            d.setTAddressMode(to_mtl_sampler_address_mode(desc.address_mode_v));
            d.setRAddressMode(to_mtl_sampler_address_mode(desc.address_mode_w));
            d.setLodMinClamp(desc.lod_min_clamp);
            d.setLodMaxClamp(desc.lod_max_clamp);
            d.setMaxAnisotropy(desc.anisotropy_clamp as usize);
            if let Some(cf) = desc.compare {
                d.setCompareFunction(to_mtl_compare_function(cf));
            }
            if let Some(label) = desc.label {
                d.setLabel(Some(&NSString::from_str(label)));
            }
            d
        };
        let sampler = self
            .device
            .newSamplerStateWithDescriptor(&mtl_desc)
            .expect("Failed to create Metal sampler");
        MetalSampler(SendSync(sampler))
    }

    fn create_shader(&self, desc: &ShaderDesc) -> MetalShaderModule {
        let source = match &desc.source {
            ShaderSource::Msl(src) => src.as_ref(),
            ShaderSource::Wgsl(_) => {
                panic!("WGSL shaders not supported by Metal backend — use MSL or SPIR-V")
            }
            ShaderSource::SpirV(_) => {
                panic!("SPIR-V shaders not directly supported by Metal backend — use MSL")
            }
        };
        let ns_source = NSString::from_str(source);
        let options = MTLCompileOptions::new();
        let library = self
            .device
            .newLibraryWithSource_options_error(&ns_source, Some(&options))
            .expect("Failed to compile Metal shader");
        MetalShaderModule(SendSync(library))
    }

    fn create_bind_group_layout(&self, desc: &BindGroupLayoutDesc) -> MetalBindGroupLayout {
        // Metal doesn't have explicit bind group layouts — argument buffers are
        // created from the function's reflection data. We store the layout
        // description for validation and bind group creation.
        MetalBindGroupLayout {
            entries: desc.entries.to_vec(),
        }
    }

    fn create_bind_group(&self, desc: &BindGroupDesc<Self>) -> MetalBindGroup {
        // Metal binds resources directly to encoder slots, not via bind groups.
        // We store the bindings and apply them when encoding render/compute passes.
        let mut buffers = Vec::new();
        let mut textures = Vec::new();
        let mut samplers = Vec::new();

        for entry in desc.entries {
            match &entry.resource {
                BindingResource::Buffer(b) => {
                    let buf = MetalBuffer(SendSync(Retained::clone(&b.buffer.0.0)));
                    buffers.push((entry.binding, buf));
                }
                BindingResource::TextureView(v) => {
                    let view = MetalTextureView(SendSync(Retained::clone(&v.0.0)));
                    textures.push((entry.binding, view));
                }
                BindingResource::Sampler(s) => {
                    let sam = MetalSampler(SendSync(Retained::clone(&s.0.0)));
                    samplers.push((entry.binding, sam));
                }
                BindingResource::TextureViewArray(views) => {
                    for (i, v) in views.iter().enumerate() {
                        let view = MetalTextureView(SendSync(Retained::clone(&v.0.0)));
                        textures.push((entry.binding + i as u32, view));
                    }
                }
            }
        }

        MetalBindGroup {
            buffers,
            textures,
            samplers,
        }
    }

    fn create_render_pipeline(&self, desc: &RenderPipelineDesc<Self>) -> MetalRenderPipeline {
        unsafe {
            let pipeline_desc = MTLRenderPipelineDescriptor::new();

            // Vertex function
            let vertex_fn = desc
                .vertex
                .module
                .0
                .newFunctionWithName(&NSString::from_str(desc.vertex.entry_point))
                .expect("Vertex function not found in Metal library");
            pipeline_desc.setVertexFunction(Some(&vertex_fn));

            // Fragment function
            if let Some(ref frag) = desc.fragment {
                let fragment_fn = frag
                    .module
                    .0
                    .newFunctionWithName(&NSString::from_str(frag.entry_point))
                    .expect("Fragment function not found in Metal library");
                pipeline_desc.setFragmentFunction(Some(&fragment_fn));

                // Color attachments
                let color_attachments = pipeline_desc.colorAttachments();
                for (i, target) in frag.targets.iter().enumerate() {
                    if let Some(target) = target {
                        let attachment = color_attachments.objectAtIndexedSubscript(i);
                        attachment.setPixelFormat(to_mtl_pixel_format(target.format));
                        if let Some(blend) = &target.blend {
                            attachment.setBlendingEnabled(true);
                            attachment.setSourceRGBBlendFactor(to_mtl_blend_factor(
                                blend.color.src_factor,
                            ));
                            attachment.setDestinationRGBBlendFactor(to_mtl_blend_factor(
                                blend.color.dst_factor,
                            ));
                            attachment.setRgbBlendOperation(to_mtl_blend_op(blend.color.operation));
                            attachment.setSourceAlphaBlendFactor(to_mtl_blend_factor(
                                blend.alpha.src_factor,
                            ));
                            attachment.setDestinationAlphaBlendFactor(to_mtl_blend_factor(
                                blend.alpha.dst_factor,
                            ));
                            attachment
                                .setAlphaBlendOperation(to_mtl_blend_op(blend.alpha.operation));
                        }
                    }
                }
            }

            // Depth attachment
            if let Some(ref ds) = desc.depth_stencil {
                pipeline_desc.setDepthAttachmentPixelFormat(to_mtl_pixel_format(ds.format));
            }

            // Multisample
            pipeline_desc.setRasterSampleCount(desc.multisample.count as usize);

            if let Some(label) = desc.label {
                pipeline_desc.setLabel(Some(&NSString::from_str(label)));
            }

            let pipeline = self
                .device
                .newRenderPipelineStateWithDescriptor_error(&pipeline_desc)
                .expect("Failed to create Metal render pipeline");

            MetalRenderPipeline(SendSync(pipeline))
        }
    }

    fn create_compute_pipeline(&self, desc: &ComputePipelineDesc<Self>) -> MetalComputePipeline {
        let function = desc
            .module
            .0
            .newFunctionWithName(&NSString::from_str(desc.entry_point))
            .expect("Compute function not found in Metal library");

        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .expect("Failed to create Metal compute pipeline");

        MetalComputePipeline(SendSync(pipeline))
    }

    fn write_buffer(&self, buffer: &MetalBuffer, offset: u64, data: &[u8]) {
        // Apple Silicon unified memory: direct memcpy into shared buffer
        unsafe {
            let contents = buffer.0.contents().as_ptr() as *mut u8;
            let dst = contents.add(offset as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
    }

    fn write_texture(
        &self,
        dst: &TexelCopyTextureInfo<Self>,
        data: &[u8],
        layout: &TextureDataLayout,
        size: Extent3d,
    ) {
        let bytes_per_row = layout.bytes_per_row.unwrap_or(0) as usize;
        let region = MTLRegion {
            origin: MTLOrigin {
                x: dst.origin.x as usize,
                y: dst.origin.y as usize,
                z: dst.origin.z as usize,
            },
            size: MTLSize {
                width: size.width as usize,
                height: size.height as usize,
                depth: size.depth_or_array_layers as usize,
            },
        };
        unsafe {
            let ptr = data[layout.offset as usize..].as_ptr() as *mut std::ffi::c_void;
            let nn = NonNull::new_unchecked(ptr);
            dst.texture
                .0
                .replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    region,
                    dst.mip_level as usize,
                    dst.origin.z as usize,
                    nn,
                    bytes_per_row,
                    0, // bytesPerImage (0 for 2D textures)
                );
        }
    }

    fn create_command_encoder(&self, _label: Option<&str>) -> MetalCommandEncoder {
        let command_buffer = self
            .queue
            .commandBuffer()
            .expect("Failed to create Metal command buffer");
        MetalCommandEncoder { command_buffer }
    }

    fn begin_render_pass<'a>(
        &self,
        encoder: &'a mut MetalCommandEncoder,
        desc: &RenderPassDesc<'_, Self>,
    ) -> MetalRenderPass<'a> {
        unsafe {
            let pass_desc = MTLRenderPassDescriptor::new();
            let color_attachments = pass_desc.colorAttachments();

            for (i, attachment) in desc.color_attachments.iter().enumerate() {
                if let Some(a) = attachment {
                    let ca = color_attachments.objectAtIndexedSubscript(i);
                    ca.setTexture(Some(&a.view.0));
                    if let Some(resolve) = &a.resolve_target {
                        ca.setResolveTexture(Some(&resolve.0));
                        ca.setStoreAction(MTLStoreAction::MultisampleResolve);
                    } else {
                        ca.setStoreAction(to_mtl_store_action(a.ops.store));
                    }
                    let (load_action, clear_color) = to_mtl_load_action_color(&a.ops.load);
                    ca.setLoadAction(load_action);
                    ca.setClearColor(clear_color);
                }
            }

            if let Some(ref ds) = desc.depth_stencil_attachment {
                let da = pass_desc.depthAttachment();
                da.setTexture(Some(&ds.view.0));
                if let Some(ref ops) = ds.depth_ops {
                    match &ops.load {
                        LoadOp::Clear(v) => {
                            da.setLoadAction(MTLLoadAction::Clear);
                            da.setClearDepth(*v as f64);
                        }
                        LoadOp::Load => {
                            da.setLoadAction(MTLLoadAction::Load);
                        }
                    }
                    da.setStoreAction(to_mtl_store_action(ops.store));
                }
            }

            let render_encoder = encoder
                .command_buffer
                .renderCommandEncoderWithDescriptor(&pass_desc)
                .expect("Failed to create Metal render encoder");

            MetalRenderPass {
                encoder: render_encoder,
                _marker: std::marker::PhantomData,
            }
        }
    }

    fn begin_compute_pass<'a>(
        &self,
        encoder: &'a mut MetalCommandEncoder,
        _label: Option<&str>,
    ) -> MetalComputePass<'a> {
        let compute_encoder = encoder
            .command_buffer
            .computeCommandEncoder()
            .expect("Failed to create Metal compute encoder");
        MetalComputePass {
            encoder: compute_encoder,
            _marker: std::marker::PhantomData,
        }
    }

    fn submit(&self, encoder: MetalCommandEncoder) {
        encoder.command_buffer.commit();
    }

    fn get_current_texture(&self) -> Result<MetalSurfaceTexture, SurfaceError> {
        let drawable = self.layer.nextDrawable().ok_or(SurfaceError::Timeout)?;
        Ok(MetalSurfaceTexture { drawable })
    }

    fn surface_texture_view(&self, surface_texture: &MetalSurfaceTexture) -> MetalTextureView {
        let texture = surface_texture.drawable.texture();
        MetalTextureView(SendSync(texture))
    }

    fn present(&self, texture: MetalSurfaceTexture) {
        // Present via the command queue — create a tiny command buffer just for present
        if let Some(cmd_buf) = self.queue.commandBuffer() {
            // Upcast CAMetalDrawable → MTLDrawable for presentDrawable()
            let drawable: &ProtocolObject<dyn MTLDrawable> =
                ProtocolObject::from_ref(&*texture.drawable);
            cmd_buf.presentDrawable(drawable);
            cmd_buf.commit();
        }
    }

    fn resize_surface(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.surface_width = width;
            self.surface_height = height;
            self.layer.setDrawableSize(objc2_foundation::NSSize {
                width: width as f64,
                height: height as f64,
            });
        }
    }

    fn surface_format(&self) -> TextureFormat {
        self.surface_format
    }

    fn surface_size(&self) -> (u32, u32) {
        (self.surface_width, self.surface_height)
    }
}

// ===========================================================================
// Blend factor / operation conversion
// ===========================================================================

fn to_mtl_blend_factor(f: BlendFactor) -> MTLBlendFactor {
    match f {
        BlendFactor::Zero => MTLBlendFactor::Zero,
        BlendFactor::One => MTLBlendFactor::One,
        BlendFactor::Src => MTLBlendFactor::SourceColor,
        BlendFactor::OneMinusSrc => MTLBlendFactor::OneMinusSourceColor,
        BlendFactor::SrcAlpha => MTLBlendFactor::SourceAlpha,
        BlendFactor::OneMinusSrcAlpha => MTLBlendFactor::OneMinusSourceAlpha,
        BlendFactor::Dst => MTLBlendFactor::DestinationColor,
        BlendFactor::OneMinusDst => MTLBlendFactor::OneMinusDestinationColor,
        BlendFactor::DstAlpha => MTLBlendFactor::DestinationAlpha,
        BlendFactor::OneMinusDstAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
        BlendFactor::SrcAlphaSaturated => MTLBlendFactor::SourceAlphaSaturated,
        BlendFactor::Constant => MTLBlendFactor::BlendColor,
        BlendFactor::OneMinusConstant => MTLBlendFactor::OneMinusBlendColor,
    }
}

fn to_mtl_blend_op(op: BlendOperation) -> MTLBlendOperation {
    match op {
        BlendOperation::Add => MTLBlendOperation::Add,
        BlendOperation::Subtract => MTLBlendOperation::Subtract,
        BlendOperation::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
        BlendOperation::Min => MTLBlendOperation::Min,
        BlendOperation::Max => MTLBlendOperation::Max,
    }
}

// ===========================================================================
// RenderPassOps for MetalRenderPass
// ===========================================================================

impl<'a> RenderPassOps<MetalDevice> for MetalRenderPass<'a> {
    fn set_pipeline(&mut self, pipeline: &MetalRenderPipeline) {
        self.encoder.setRenderPipelineState(&pipeline.0);
    }

    fn set_bind_group(&mut self, _index: u32, bind_group: &MetalBindGroup, _offsets: &[u32]) {
        // Metal: bind resources directly to encoder slots
        unsafe {
            for (binding, buf) in &bind_group.buffers {
                self.encoder
                    .setVertexBuffer_offset_atIndex(Some(&buf.0), 0, *binding as usize);
                self.encoder
                    .setFragmentBuffer_offset_atIndex(Some(&buf.0), 0, *binding as usize);
            }
            for (binding, tex) in &bind_group.textures {
                self.encoder
                    .setVertexTexture_atIndex(Some(&tex.0), *binding as usize);
                self.encoder
                    .setFragmentTexture_atIndex(Some(&tex.0), *binding as usize);
            }
            for (binding, sam) in &bind_group.samplers {
                self.encoder
                    .setVertexSamplerState_atIndex(Some(&sam.0), *binding as usize);
                self.encoder
                    .setFragmentSamplerState_atIndex(Some(&sam.0), *binding as usize);
            }
        }
    }

    fn set_vertex_buffer(&mut self, slot: u32, buffer: &MetalBuffer, offset: u64, _size: u64) {
        unsafe {
            self.encoder.setVertexBuffer_offset_atIndex(
                Some(&buffer.0),
                offset as usize,
                slot as usize,
            );
        }
    }

    fn set_index_buffer(
        &mut self,
        _buffer: &MetalBuffer,
        _format: IndexFormat,
        _offset: u64,
        _size: u64,
    ) {
        // Metal: index buffer is passed directly to drawIndexedPrimitives
        // Store it for the next draw call (handled in draw_indexed)
    }

    fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        let vertex_count = vertices.end - vertices.start;
        let instance_count = instances.end - instances.start;
        unsafe {
            self.encoder
                .drawPrimitives_vertexStart_vertexCount_instanceCount(
                    MTLPrimitiveType::Triangle,
                    vertices.start as usize,
                    vertex_count as usize,
                    instance_count as usize,
                );
        }
    }

    fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        _base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        let index_count = indices.end - indices.start;
        let instance_count = instances.end - instances.start;
        // Note: index buffer must have been set via set_index_buffer
        // In a full implementation, we'd store the index buffer reference
        // For now, this is a placeholder that will be completed in Phase C
        let _ = (index_count, instance_count);
        // TODO: implement with stored index buffer reference
    }

    fn draw_indexed_indirect(&mut self, _indirect_buffer: &MetalBuffer, _indirect_offset: u64) {
        // TODO: Phase C — indirect command buffers
    }

    fn multi_draw_indexed_indirect(
        &mut self,
        _indirect_buffer: &MetalBuffer,
        _indirect_offset: u64,
        _count: u32,
    ) {
        // TODO: Phase C — indirect command buffers
    }

    fn multi_draw_indexed_indirect_count(
        &mut self,
        _indirect_buffer: &MetalBuffer,
        _indirect_offset: u64,
        _count_buffer: &MetalBuffer,
        _count_offset: u64,
        _max_count: u32,
    ) {
        // TODO: Phase C — indirect command buffers
    }

    fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        self.encoder.setViewport(MTLViewport {
            originX: x as f64,
            originY: y as f64,
            width: w as f64,
            height: h as f64,
            znear: min_depth as f64,
            zfar: max_depth as f64,
        });
    }

    fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        self.encoder.setScissorRect(MTLScissorRect {
            x: x as usize,
            y: y as usize,
            width: w as usize,
            height: h as usize,
        });
    }
}

// ===========================================================================
// ComputePassOps for MetalComputePass
// ===========================================================================

impl<'a> ComputePassOps<MetalDevice> for MetalComputePass<'a> {
    fn set_pipeline(&mut self, pipeline: &MetalComputePipeline) {
        self.encoder.setComputePipelineState(&pipeline.0);
    }

    fn set_bind_group(&mut self, _index: u32, bind_group: &MetalBindGroup, _offsets: &[u32]) {
        unsafe {
            for (binding, buf) in &bind_group.buffers {
                self.encoder
                    .setBuffer_offset_atIndex(Some(&buf.0), 0, *binding as usize);
            }
            for (binding, tex) in &bind_group.textures {
                self.encoder
                    .setTexture_atIndex(Some(&tex.0), *binding as usize);
            }
            for (binding, sam) in &bind_group.samplers {
                self.encoder
                    .setSamplerState_atIndex(Some(&sam.0), *binding as usize);
            }
        }
    }

    fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        let threadgroups = MTLSize {
            width: x as usize,
            height: y as usize,
            depth: z as usize,
        };
        // Use a reasonable threadgroup size — ideally from pipeline reflection
        let threads_per_group = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        self.encoder
            .dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_group);
    }

    fn dispatch_workgroups_indirect(
        &mut self,
        indirect_buffer: &MetalBuffer,
        indirect_offset: u64,
    ) {
        let threads_per_group = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        unsafe {
            self.encoder
                .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    &indirect_buffer.0,
                    indirect_offset as usize,
                    threads_per_group,
                );
        }
    }
}

// ===========================================================================
// Drop impls — end encoding when passes are dropped
// ===========================================================================

impl Drop for MetalRenderPass<'_> {
    fn drop(&mut self) {
        self.encoder.endEncoding();
    }
}

impl Drop for MetalComputePass<'_> {
    fn drop(&mut self) {
        self.encoder.endEncoding();
    }
}
