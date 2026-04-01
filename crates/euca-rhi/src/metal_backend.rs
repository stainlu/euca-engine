//! Native Metal backend — implements [`RenderDevice`] using Apple's Metal API
//! via `objc2-metal` for direct GPU access on Apple Silicon.
//!
//! This backend bypasses wgpu to access Metal 3/4 features that the WebGPU
//! spec cannot express: mesh shaders, tile shading, indirect command buffers,
//! MetalFX upscaling, memoryless render targets, and MPS compute.

use std::ptr::NonNull;

use objc2::rc::Retained;

// Grand Central Dispatch semaphore for frame pipelining
unsafe extern "C" {
    fn dispatch_semaphore_create(value: isize) -> *mut std::ffi::c_void;
    fn dispatch_semaphore_wait(dsema: *mut std::ffi::c_void, timeout: u64) -> isize;
    fn dispatch_semaphore_signal(dsema: *mut std::ffi::c_void) -> isize;
    fn dispatch_release(object: *mut std::ffi::c_void);
}
const DISPATCH_TIME_FOREVER: u64 = !0;
const MAX_FRAMES_IN_FLIGHT: usize = 3;
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
pub struct MetalRenderPipeline {
    state: SendSync<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    primitive_type: MTLPrimitiveType,
}
pub struct MetalComputePipeline(SendSync<Retained<ProtocolObject<dyn MTLComputePipelineState>>>);
pub struct MetalBindGroupLayout {
    // Stored for validation during bind group creation; not read by the GPU.
    #[allow(dead_code)]
    entries: Vec<BindGroupLayoutEntry>,
}
pub struct MetalBindGroup {
    buffers: Vec<(u32, MetalBuffer)>,
    textures: Vec<(u32, MetalTextureView)>,
    samplers: Vec<(u32, MetalSampler)>,
}

pub struct MetalCommandEncoder {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    /// Drawable to present when this encoder is submitted.
    pending_drawable: Option<Retained<ProtocolObject<dyn CAMetalDrawable>>>,
}

/// Retained index buffer state: (buffer, index type, byte offset).
type IndexBufferBinding = (Retained<ProtocolObject<dyn MTLBuffer>>, MTLIndexType, u64);

pub struct MetalRenderPass<'a> {
    encoder: Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>,
    /// Current primitive topology, set when a pipeline is bound.
    primitive_type: MTLPrimitiveType,
    /// Stored index buffer for `draw_indexed` calls. Metal requires the index
    /// buffer to be passed directly to the draw call, unlike wgpu which has
    /// separate `set_index_buffer` / `draw_indexed` steps.
    index_buffer: Option<IndexBufferBinding>,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl MetalRenderPass<'_> {
    /// Returns a reference to the currently bound index buffer, panicking if none is set.
    fn require_index_buffer(&self, caller: &str) -> &IndexBufferBinding {
        self.index_buffer
            .as_ref()
            .unwrap_or_else(|| panic!("{caller} called without a prior set_index_buffer"))
    }

    /// Bind a buffer to the object shader at the given slot index.
    pub fn set_object_buffer(&mut self, slot: u32, buffer: &MetalBuffer, offset: u64) {
        unsafe {
            self.encoder.setObjectBuffer_offset_atIndex(
                Some(&buffer.0),
                offset as usize,
                slot as usize,
            );
        }
    }

    /// Bind a buffer to the mesh shader at the given slot index.
    ///
    /// Mesh shaders use separate buffer slots from vertex shaders — you must
    /// use this method (not `set_vertex_buffer`) when using a mesh pipeline.
    pub fn set_mesh_buffer(&mut self, slot: u32, buffer: &MetalBuffer, offset: u64) {
        unsafe {
            self.encoder.setMeshBuffer_offset_atIndex(
                Some(&buffer.0),
                offset as usize,
                slot as usize,
            );
        }
    }

    /// Draw using a mesh shader pipeline.
    ///
    /// Dispatches `threadgroups_per_grid` mesh shader threadgroups. Each
    /// threadgroup cooperatively outputs vertices and primitives directly to
    /// the rasterizer, bypassing the traditional vertex processing pipeline.
    ///
    /// When no object shader is present, `threads_per_object_threadgroup` is
    /// ignored (pass `[1, 1, 1]`).
    pub fn draw_mesh_threadgroups(
        &self,
        threadgroups_per_grid: [u32; 3],
        threads_per_object_threadgroup: [u32; 3],
        threads_per_mesh_threadgroup: [u32; 3],
    ) {
        self.encoder
            .drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                MTLSize {
                    width: threadgroups_per_grid[0] as usize,
                    height: threadgroups_per_grid[1] as usize,
                    depth: threadgroups_per_grid[2] as usize,
                },
                MTLSize {
                    width: threads_per_object_threadgroup[0] as usize,
                    height: threads_per_object_threadgroup[1] as usize,
                    depth: threads_per_object_threadgroup[2] as usize,
                },
                MTLSize {
                    width: threads_per_mesh_threadgroup[0] as usize,
                    height: threads_per_mesh_threadgroup[1] as usize,
                    depth: threads_per_mesh_threadgroup[2] as usize,
                },
            );
    }
}

pub struct MetalComputePass<'a> {
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    _marker: std::marker::PhantomData<&'a ()>,
}

pub struct MetalSurfaceTexture {
    drawable: Retained<ProtocolObject<dyn CAMetalDrawable>>,
}

impl MetalCommandEncoder {
    /// Schedule a drawable for presentation when this encoder is submitted.
    /// Must be called before `device.submit(encoder)`.
    pub fn schedule_present(&mut self, surface_texture: &MetalSurfaceTexture) {
        self.pending_drawable = Some(Retained::clone(&surface_texture.drawable));
    }
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
    /// GCD semaphore for triple-buffered frame pipelining.
    /// Limits in-flight frames to MAX_FRAMES_IN_FLIGHT so CPU and GPU
    /// work overlaps without exhausting the drawable pool.
    frame_semaphore: SendSync<*mut std::ffi::c_void>,
}

impl MetalDevice {
    /// Query device capabilities from a Metal device.
    fn query_capabilities(device: &ProtocolObject<dyn MTLDevice>) -> Capabilities {
        let device_name = device.name().to_string();
        let is_apple_silicon = device.supportsFamily(MTLGPUFamily::Apple7);
        let supports_memoryless = is_apple_silicon;
        let max_buffer_len = device.maxBufferLength() as u64;

        Capabilities {
            unified_memory: is_apple_silicon,
            multi_draw_indirect: true,
            multi_draw_indirect_count: true,
            texture_binding_array: true,
            non_uniform_indexing: true,
            max_texture_dimension_2d: 16384,
            max_bind_groups: 31,
            max_bindings_per_bind_group: 1024,
            max_binding_array_elements: 500_000,
            device_name,
            apple_silicon: is_apple_silicon,
            max_buffer_length: max_buffer_len,
            memoryless_render_targets: supports_memoryless,
        }
    }

    /// Create a new MetalDevice from a CAMetalLayer (obtained from the window).
    ///
    /// # Safety
    /// The `layer` must be a valid CAMetalLayer attached to a visible view.
    pub unsafe fn new(layer: Retained<CAMetalLayer>, width: u32, height: u32) -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal-capable GPU found");

        let queue = device
            .newCommandQueue()
            .expect("Failed to create Metal command queue");

        layer.setDevice(Some(&device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm_sRGB);
        // Use framebufferOnly for best performance (no readback needed)
        layer.setFramebufferOnly(true);
        layer.setDrawableSize(objc2_foundation::NSSize {
            width: width as f64,
            height: height as f64,
        });

        let capabilities = Self::query_capabilities(&device);
        let frame_semaphore =
            SendSync(unsafe { dispatch_semaphore_create(MAX_FRAMES_IN_FLIGHT as isize) });

        Self {
            device: SendSync(device),
            queue: SendSync(queue),
            layer: SendSync(layer),
            surface_width: width,
            surface_height: height,
            surface_format: TextureFormat::Bgra8UnormSrgb,
            capabilities,
            frame_semaphore,
        }
    }

    /// Create a headless MetalDevice for testing (no surface/window required).
    ///
    /// Surface operations (`get_current_texture`, `present`, `resize_surface`)
    /// will panic. Use this only for resource creation, shader compilation,
    /// and offscreen rendering tests.
    pub fn headless() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal-capable GPU found");
        let queue = device
            .newCommandQueue()
            .expect("Failed to create Metal command queue");
        let capabilities = Self::query_capabilities(&device);
        let frame_semaphore =
            SendSync(unsafe { dispatch_semaphore_create(MAX_FRAMES_IN_FLIGHT as isize) });

        let layer = CAMetalLayer::new();

        Self {
            device: SendSync(device),
            queue: SendSync(queue),
            layer: SendSync(layer),
            surface_width: 0,
            surface_height: 0,
            surface_format: TextureFormat::Bgra8UnormSrgb,
            capabilities,
            frame_semaphore,
        }
    }

    /// Create a MetalDevice from a winit window (macOS only).
    ///
    /// Extracts the NSView, creates and attaches a CAMetalLayer, and
    /// initializes the Metal device with the system default GPU.
    #[cfg(feature = "metal-backend")]
    pub fn from_window(window: &winit::window::Window) -> Self {
        use raw_window_handle::HasWindowHandle;

        let size = window.inner_size();
        let scale_factor = window.scale_factor();
        let handle = window.window_handle().expect("Failed to get window handle");
        let raw = handle.as_raw();

        let layer = unsafe {
            let raw_window_handle::RawWindowHandle::AppKit(appkit) = raw else {
                panic!("Expected AppKit window handle on macOS");
            };
            let ns_view = appkit.ns_view.as_ptr() as *const objc2::runtime::AnyObject;

            // Create and configure CAMetalLayer
            let metal_layer = CAMetalLayer::new();

            // Configure for Retina display
            metal_layer.setContentsScale(scale_factor);

            // Make the view layer-backed and set our Metal layer
            let _: () = objc2::msg_send![ns_view, setWantsLayer: true];
            let _: () = objc2::msg_send![ns_view, setLayer: &*metal_layer];

            // Set the layer's frame to match the view's bounds
            let bounds: objc2_foundation::NSRect = objc2::msg_send![ns_view, bounds];
            let _: () = objc2::msg_send![&*metal_layer, setFrame: bounds];

            metal_layer
        };

        unsafe { Self::new(layer, size.width, size.height) }
    }

    /// Access the raw MTL device (for advanced Metal-specific operations).
    pub fn mtl_device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Blit a texture to the current surface drawable via a blit encoder.
    ///
    /// Use this to copy an offscreen render result (e.g., MetalFX output) to
    /// the swapchain for presentation.
    pub fn blit_to_surface(
        &self,
        encoder: &mut MetalCommandEncoder,
        src: &MetalTexture,
        surface: &MetalSurfaceTexture,
    ) {
        unsafe {
            let blit = encoder
                .command_buffer
                .blitCommandEncoder()
                .expect("Failed to create blit encoder");
            let dst_texture = surface.drawable.texture();
            let width = dst_texture.width().min(src.0.width());
            let height = dst_texture.height().min(src.0.height());
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &src.0,
                0, 0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize { width, height, depth: 1 },
                &dst_texture,
                0, 0,
                MTLOrigin { x: 0, y: 0, z: 0 },
            );
            blit.endEncoding();
        }
    }

    /// Enable or disable vsync (display sync).
    /// When disabled, frames render as fast as possible (uncapped FPS).
    pub fn set_display_sync_enabled(&self, enabled: bool) {
        unsafe {
            // setDisplaySyncEnabled controls vsync
            let _: () = objc2::msg_send![&*self.layer, setDisplaySyncEnabled: enabled];
        }
    }

    /// Create a mesh render pipeline (mesh + fragment, optionally object shader).
    ///
    /// Metal mesh shaders bypass the traditional vertex pipeline entirely:
    /// Object shader → Mesh shader → Rasterizer → Fragment shader.
    /// This eliminates the TBDR binning-phase bottleneck that limits vertex
    /// throughput on Apple Silicon.
    ///
    /// This is Metal-specific and not part of the `RenderDevice` trait.
    pub fn create_mesh_render_pipeline(
        &self,
        shader: &MetalShaderModule,
        mesh_entry: &str,
        fragment_entry: &str,
        object_entry: Option<&str>,
        color_formats: &[TextureFormat],
        color_blends: &[Option<BlendState>],
        depth_format: Option<TextureFormat>,
        label: Option<&str>,
    ) -> MetalRenderPipeline {
        unsafe {
            let desc = MTLMeshRenderPipelineDescriptor::new();

            if let Some(lbl) = label {
                desc.setLabel(Some(&NSString::from_str(lbl)));
            }

            // Mesh function (required)
            let mesh_fn = shader
                .0
                .newFunctionWithName(&NSString::from_str(mesh_entry))
                .unwrap_or_else(|| {
                    panic!("Mesh function '{mesh_entry}' not found in Metal library")
                });
            desc.setMeshFunction(Some(&mesh_fn));

            // Fragment function
            let frag_fn = shader
                .0
                .newFunctionWithName(&NSString::from_str(fragment_entry))
                .unwrap_or_else(|| {
                    panic!("Fragment function '{fragment_entry}' not found in Metal library")
                });
            desc.setFragmentFunction(Some(&frag_fn));

            // Object function (optional — for GPU-driven culling)
            if let Some(obj_entry) = object_entry {
                let obj_fn = shader
                    .0
                    .newFunctionWithName(&NSString::from_str(obj_entry))
                    .unwrap_or_else(|| {
                        panic!("Object function '{obj_entry}' not found in Metal library")
                    });
                desc.setObjectFunction(Some(&obj_fn));
            }

            // Color attachments
            let color_attachments = desc.colorAttachments();
            for (i, format) in color_formats.iter().enumerate() {
                let attachment = color_attachments.objectAtIndexedSubscript(i);
                attachment.setPixelFormat(to_mtl_pixel_format(*format));

                if let Some(Some(blend)) = color_blends.get(i) {
                    attachment.setBlendingEnabled(true);
                    attachment.setSourceRGBBlendFactor(to_mtl_blend_factor(blend.color.src_factor));
                    attachment
                        .setDestinationRGBBlendFactor(to_mtl_blend_factor(blend.color.dst_factor));
                    attachment.setRgbBlendOperation(to_mtl_blend_op(blend.color.operation));
                    attachment
                        .setSourceAlphaBlendFactor(to_mtl_blend_factor(blend.alpha.src_factor));
                    attachment.setDestinationAlphaBlendFactor(to_mtl_blend_factor(
                        blend.alpha.dst_factor,
                    ));
                    attachment.setAlphaBlendOperation(to_mtl_blend_op(blend.alpha.operation));
                }
            }

            // Depth format
            if let Some(depth_fmt) = depth_format {
                desc.setDepthAttachmentPixelFormat(to_mtl_pixel_format(depth_fmt));
            }

            let pipeline = self
                .device
                .newRenderPipelineStateWithMeshDescriptor_options_reflection_error(
                    &desc,
                    MTLPipelineOption::None,
                    None,
                )
                .expect("Failed to create Metal mesh render pipeline");

            MetalRenderPipeline {
                state: SendSync(pipeline),
                primitive_type: MTLPrimitiveType::Triangle,
            }
        }
    }
}

// ===========================================================================
// WGSL → MSL transpilation via naga
// ===========================================================================

/// Transpile a WGSL shader source to MSL using naga.
///
/// This enables all existing WGSL shaders to run on the Metal backend without
/// maintaining separate MSL shader files. Naga translates the abstract shader
/// IR to valid Metal Shading Language.
fn wgsl_to_msl(wgsl_source: &str) -> String {
    // Parse WGSL → naga IR
    let module = naga::front::wgsl::parse_str(wgsl_source).unwrap_or_else(|e| {
        panic!("Failed to parse WGSL shader: {e}");
    });

    // Validate
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| {
        panic!("WGSL shader validation failed: {e}");
    });

    // Translate naga IR → MSL
    let options = naga::back::msl::Options {
        lang_version: (3, 0), // Metal 3.0 (Apple Silicon)
        per_entry_point_map: Default::default(),
        inline_samplers: Default::default(),
        spirv_cross_compatibility: false,
        fake_missing_bindings: true,
        bounds_check_policies: Default::default(),
        zero_initialize_workgroup_memory: true,
        force_loop_bounding: false,
    };

    let pipeline_options = naga::back::msl::PipelineOptions::default();

    let (msl, _) = naga::back::msl::write_string(&module, &info, &options, &pipeline_options)
        .unwrap_or_else(|e| {
            panic!("Failed to transpile WGSL → MSL: {e}");
        });

    msl
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

fn to_mtl_storage_mode_for_usage(usage: TextureUsages, memoryless: bool) -> MTLStorageMode {
    let is_render_only = usage.contains(TextureUsages::RENDER_ATTACHMENT)
        && !usage.contains(TextureUsages::TEXTURE_BINDING)
        && !usage.contains(TextureUsages::STORAGE_BINDING)
        && !usage.contains(TextureUsages::COPY_SRC);

    if is_render_only && memoryless {
        // Memoryless: texture lives only in tile memory and is never written to
        // system RAM. Saves ~20% memory bandwidth for transient G-buffer
        // attachments (depth, normals, etc.) that are produced and consumed
        // within a single render pass.
        MTLStorageMode::Memoryless
    } else if usage.contains(TextureUsages::RENDER_ATTACHMENT) {
        MTLStorageMode::Private
    } else {
        MTLStorageMode::Shared // Apple Silicon unified memory
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

fn to_mtl_vertex_format(format: VertexFormat) -> MTLVertexFormat {
    match format {
        VertexFormat::Float32 => MTLVertexFormat::Float,
        VertexFormat::Float32x2 => MTLVertexFormat::Float2,
        VertexFormat::Float32x3 => MTLVertexFormat::Float3,
        VertexFormat::Float32x4 => MTLVertexFormat::Float4,
        VertexFormat::Uint32 => MTLVertexFormat::UInt,
        VertexFormat::Uint32x2 => MTLVertexFormat::UInt2,
        VertexFormat::Uint32x3 => MTLVertexFormat::UInt3,
        VertexFormat::Uint32x4 => MTLVertexFormat::UInt4,
        VertexFormat::Sint32 => MTLVertexFormat::Int,
        VertexFormat::Sint32x2 => MTLVertexFormat::Int2,
        VertexFormat::Sint32x3 => MTLVertexFormat::Int3,
        VertexFormat::Sint32x4 => MTLVertexFormat::Int4,
        VertexFormat::Uint8x2 => MTLVertexFormat::UChar2,
        VertexFormat::Uint8x4 => MTLVertexFormat::UChar4,
        VertexFormat::Unorm8x2 => MTLVertexFormat::UChar2Normalized,
        VertexFormat::Unorm8x4 => MTLVertexFormat::UChar4Normalized,
        VertexFormat::Float16x2 => MTLVertexFormat::Half2,
        VertexFormat::Float16x4 => MTLVertexFormat::Half4,
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
            d.setStorageMode(to_mtl_storage_mode_for_usage(
                desc.usage,
                self.capabilities.memoryless_render_targets,
            ));
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
        let msl_source = match &desc.source {
            ShaderSource::Msl(src) => src.to_string(),
            ShaderSource::Wgsl(wgsl) => {
                // Transpile WGSL → MSL via naga. This unlocks all existing WGSL
                // shaders for the Metal backend without maintaining separate MSL files.
                wgsl_to_msl(wgsl)
            }
            ShaderSource::SpirV(_) => {
                panic!("SPIR-V shaders not directly supported by Metal backend — use MSL or WGSL")
            }
        };
        let ns_source = NSString::from_str(&msl_source);
        let options = MTLCompileOptions::new();
        let library = self
            .device
            .newLibraryWithSource_options_error(&ns_source, Some(&options))
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to compile Metal shader '{}': {}",
                    desc.label.unwrap_or("unnamed"),
                    e
                )
            });
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

            // Vertex descriptor — enables hardware vertex fetch + post-transform cache
            if !desc.vertex.buffers.is_empty() {
                let vertex_desc = MTLVertexDescriptor::new();
                let attrs = vertex_desc.attributes();
                let layouts = vertex_desc.layouts();

                for (buf_idx, buf_layout) in desc.vertex.buffers.iter().enumerate() {
                    // Set buffer layout
                    let layout = layouts.objectAtIndexedSubscript(buf_idx);
                    layout.setStride(buf_layout.array_stride as usize);
                    layout.setStepFunction(match buf_layout.step_mode {
                        VertexStepMode::Vertex => MTLVertexStepFunction::PerVertex,
                        VertexStepMode::Instance => MTLVertexStepFunction::PerInstance,
                    });

                    // Set attributes
                    for attr in buf_layout.attributes {
                        let mtl_attr =
                            attrs.objectAtIndexedSubscript(attr.shader_location as usize);
                        mtl_attr.setFormat(to_mtl_vertex_format(attr.format));
                        mtl_attr.setOffset(attr.offset as usize);
                        mtl_attr.setBufferIndex(buf_idx);
                    }
                }

                pipeline_desc.setVertexDescriptor(Some(&vertex_desc));
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

            MetalRenderPipeline {
                state: SendSync(pipeline),
                primitive_type: to_mtl_primitive_type(desc.primitive.topology),
            }
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
        MetalCommandEncoder {
            command_buffer,
            pending_drawable: None,
        }
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
                primitive_type: MTLPrimitiveType::Triangle, // default; overwritten by set_pipeline
                index_buffer: None,
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

    fn clear_buffer(
        &self,
        encoder: &mut MetalCommandEncoder,
        buffer: &MetalBuffer,
        offset: u64,
        size: Option<u64>,
    ) {
        unsafe {
            let blit = encoder
                .command_buffer
                .blitCommandEncoder()
                .expect("Failed to create Metal blit encoder");
            let len = size.unwrap_or(buffer.0.0.length() as u64 - offset);
            let range = objc2_foundation::NSRange {
                location: offset as usize,
                length: len as usize,
            };
            blit.fillBuffer_range_value(&buffer.0.0, range, 0);
            blit.endEncoding();
        }
    }

    fn copy_texture_to_texture(
        &self,
        encoder: &mut MetalCommandEncoder,
        src: &TexelCopyTextureInfo<Self>,
        dst: &TexelCopyTextureInfo<Self>,
        size: Extent3d,
    ) {
        unsafe {
            let blit = encoder
                .command_buffer
                .blitCommandEncoder()
                .expect("Failed to create Metal blit encoder");
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &(src.texture.0).0,
                0, // source slice
                src.mip_level as usize,
                MTLOrigin {
                    x: src.origin.x as usize,
                    y: src.origin.y as usize,
                    z: src.origin.z as usize,
                },
                MTLSize {
                    width: size.width as usize,
                    height: size.height as usize,
                    depth: size.depth_or_array_layers as usize,
                },
                &(dst.texture.0).0,
                0, // destination slice
                dst.mip_level as usize,
                MTLOrigin {
                    x: dst.origin.x as usize,
                    y: dst.origin.y as usize,
                    z: dst.origin.z as usize,
                },
            );
            blit.endEncoding();
        }
    }

    fn submit(&self, encoder: MetalCommandEncoder) {
        // Present any pending drawable BEFORE committing (Metal best practice)
        if let Some(ref drawable) = encoder.pending_drawable {
            let mtl_drawable: &ProtocolObject<dyn MTLDrawable> =
                ProtocolObject::from_ref(&**drawable);
            encoder.command_buffer.presentDrawable(mtl_drawable);
        }

        // Signal the frame semaphore when the GPU finishes this command buffer.
        // This allows the next frame to acquire a drawable without blocking.
        let sem = self.frame_semaphore.0;
        unsafe {
            let block =
                block2::RcBlock::new(move |_buf: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                    dispatch_semaphore_signal(sem);
                });
            let handler: *mut block2::DynBlock<
                dyn Fn(NonNull<ProtocolObject<dyn MTLCommandBuffer>>),
            > = (&*block as *const block2::DynBlock<_>).cast_mut();
            encoder.command_buffer.addCompletedHandler(handler);
        }

        encoder.command_buffer.commit();
    }

    fn submit_multiple(&self, encoders: Vec<MetalCommandEncoder>) {
        let count = encoders.len();
        for (i, encoder) in encoders.into_iter().enumerate() {
            // Present any pending drawable BEFORE committing (Metal best practice)
            if let Some(ref drawable) = encoder.pending_drawable {
                let mtl_drawable: &ProtocolObject<dyn MTLDrawable> =
                    ProtocolObject::from_ref(&**drawable);
                encoder.command_buffer.presentDrawable(mtl_drawable);
            }

            // Only the last command buffer signals the frame semaphore,
            // keeping one signal per frame for correct in-flight tracking.
            let is_last = i + 1 == count;
            if is_last {
                let sem = self.frame_semaphore.0;
                unsafe {
                    let block = block2::RcBlock::new(
                        move |_buf: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                            dispatch_semaphore_signal(sem);
                        },
                    );
                    let handler: *mut block2::DynBlock<
                        dyn Fn(NonNull<ProtocolObject<dyn MTLCommandBuffer>>),
                    > = (&*block as *const block2::DynBlock<_>).cast_mut();
                    encoder.command_buffer.addCompletedHandler(handler);
                }
            }

            encoder.command_buffer.commit();
        }
    }

    fn get_current_texture(&self) -> Result<MetalSurfaceTexture, SurfaceError> {
        // Wait for a frame slot — blocks if MAX_FRAMES_IN_FLIGHT are already
        // in-flight, allowing CPU/GPU overlap without exhausting drawables.
        unsafe {
            dispatch_semaphore_wait(self.frame_semaphore.0, DISPATCH_TIME_FOREVER);
        }
        let drawable = self.layer.nextDrawable().ok_or(SurfaceError::Timeout)?;
        Ok(MetalSurfaceTexture { drawable })
    }

    fn surface_texture_view(&self, surface_texture: &MetalSurfaceTexture) -> MetalTextureView {
        let texture = surface_texture.drawable.texture();
        MetalTextureView(SendSync(texture))
    }

    fn prepare_present(
        &self,
        encoder: &mut MetalCommandEncoder,
        texture: &MetalSurfaceTexture,
    ) {
        encoder.schedule_present(texture);
    }

    fn present(&self, _texture: MetalSurfaceTexture) {
        // Presentation is handled in submit() — the drawable was attached to the
        // command encoder via prepare_present(). This method is a no-op for Metal.
        // The drawable is dropped here, releasing it back to the CAMetalLayer pool.
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
        self.encoder.setRenderPipelineState(&pipeline.state);
        self.primitive_type = pipeline.primitive_type;
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
        buffer: &MetalBuffer,
        format: IndexFormat,
        offset: u64,
        _size: u64,
    ) {
        // Metal passes the index buffer directly to drawIndexedPrimitives,
        // so we store it here for the next draw_indexed call.
        self.index_buffer = Some((
            Retained::clone(&buffer.0.0),
            to_mtl_index_type(format),
            offset,
        ));
    }

    fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        let vertex_count = vertices.end - vertices.start;
        let instance_count = instances.end - instances.start;
        unsafe {
            self.encoder
                .drawPrimitives_vertexStart_vertexCount_instanceCount(
                    self.primitive_type,
                    vertices.start as usize,
                    vertex_count as usize,
                    instance_count as usize,
                );
        }
    }

    fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        let (index_buffer, index_type, base_offset) = self.require_index_buffer("draw_indexed");

        let index_count = (indices.end - indices.start) as usize;
        let instance_count = (instances.end - instances.start) as usize;

        // Compute the byte offset for the first index in the range.
        let index_stride: usize = match *index_type {
            MTLIndexType::UInt16 => 2,
            MTLIndexType::UInt32 => 4,
            _ => unreachable!("unknown MTLIndexType"),
        };
        let index_buffer_offset = *base_offset as usize + indices.start as usize * index_stride;

        unsafe {
            self.encoder
                .drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                    self.primitive_type,
                    index_count,
                    *index_type,
                    index_buffer,
                    index_buffer_offset,
                    instance_count,
                    base_vertex as isize,
                    instances.start as usize,
                );
        }
    }

    fn draw_indexed_indirect(&mut self, indirect_buffer: &MetalBuffer, indirect_offset: u64) {
        let (index_buffer, index_type, index_offset) =
            self.require_index_buffer("draw_indexed_indirect");

        unsafe {
            self.encoder
                .drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                    self.primitive_type,
                    *index_type,
                    index_buffer,
                    *index_offset as usize,
                    &indirect_buffer.0,
                    indirect_offset as usize,
                );
        }
    }

    fn multi_draw_indexed_indirect(
        &mut self,
        indirect_buffer: &MetalBuffer,
        indirect_offset: u64,
        count: u32,
    ) {
        let (index_buffer, index_type, index_offset) =
            self.require_index_buffer("multi_draw_indexed_indirect");

        // Metal has no built-in multi-draw; emit one indirect draw per element.
        // MTLDrawIndexedPrimitivesIndirectArguments is 5 x u32 = 20 bytes.
        const INDIRECT_STRIDE: u64 = 20;
        unsafe {
            for i in 0..count {
                let offset = indirect_offset + i as u64 * INDIRECT_STRIDE;
                self.encoder
                    .drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                        self.primitive_type,
                        *index_type,
                        index_buffer,
                        *index_offset as usize,
                        &indirect_buffer.0,
                        offset as usize,
                    );
            }
        }
    }

    fn multi_draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: &MetalBuffer,
        indirect_offset: u64,
        _count_buffer: &MetalBuffer,
        _count_offset: u64,
        max_count: u32,
    ) {
        // Metal has no native draw-indirect-count. Emit max_count indirect
        // draws; unused slots must have zero instance counts in the indirect
        // buffer. GPU-driven count reduction requires Indirect Command Buffers
        // (ICBs), which will be implemented in a dedicated ICB pass.
        self.multi_draw_indexed_indirect(indirect_buffer, indirect_offset, max_count);
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

impl Drop for MetalDevice {
    fn drop(&mut self) {
        unsafe {
            dispatch_release(self.frame_semaphore.0);
        }
    }
}

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

// ===========================================================================
// MetalFX Temporal Upscaling (C.5)
// ===========================================================================

/// MetalFX temporal upscaler — renders at lower resolution and reconstructs
/// full resolution using temporal accumulation, motion vectors, and depth.
///
/// Similar in concept to NVIDIA DLSS / AMD FSR. Reduces fragment load by
/// rendering at 50% resolution (4x fewer pixels) with high-quality upscaling.
pub struct MetalFXUpscaler {
    scaler: SendSync<Retained<ProtocolObject<dyn objc2_metal_fx::MTLFXTemporalScaler>>>,
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
}

impl MetalDevice {
    /// Create a MetalFX temporal upscaler.
    ///
    /// - `input_width/height`: resolution to render at (e.g., 640x360)
    /// - `output_width/height`: final display resolution (e.g., 1280x720)
    /// - `color_format`: pixel format of the rendered color texture
    /// - `depth_format`: pixel format of the depth texture
    /// - `motion_format`: pixel format of the motion vector texture (typically RG16Float)
    pub fn create_temporal_upscaler(
        &self,
        input_width: u32,
        input_height: u32,
        output_width: u32,
        output_height: u32,
        color_format: TextureFormat,
        depth_format: TextureFormat,
        motion_format: TextureFormat,
    ) -> MetalFXUpscaler {
        use objc2_metal_fx::MTLFXTemporalScalerDescriptor;

        let desc = unsafe { MTLFXTemporalScalerDescriptor::new() };
        unsafe {
            desc.setInputWidth(input_width as usize);
            desc.setInputHeight(input_height as usize);
            desc.setOutputWidth(output_width as usize);
            desc.setOutputHeight(output_height as usize);
            desc.setColorTextureFormat(to_mtl_pixel_format(color_format));
            desc.setDepthTextureFormat(to_mtl_pixel_format(depth_format));
            desc.setMotionTextureFormat(to_mtl_pixel_format(motion_format));
            desc.setOutputTextureFormat(to_mtl_pixel_format(color_format));
            desc.setAutoExposureEnabled(true);
        }

        let scaler = unsafe {
            desc.newTemporalScalerWithDevice(&self.device).expect(
                "Failed to create MetalFX temporal scaler — requires Apple Silicon with Metal 3",
            )
        };

        MetalFXUpscaler {
            scaler: SendSync(scaler),
            input_width,
            input_height,
            output_width,
            output_height,
        }
    }
}

impl MetalFXUpscaler {
    /// Encode the temporal upscale pass into the command buffer.
    ///
    /// Call this after rendering to the low-resolution color/depth/motion textures
    /// and before presenting the output texture.
    ///
    /// - `encoder`: the command encoder (must not have an active render/compute pass)
    /// - `color`: low-resolution rendered color texture
    /// - `depth`: low-resolution depth texture
    /// - `motion`: screen-space motion vectors (RG16Float, in pixels)
    /// - `output`: full-resolution output texture
    /// - `jitter_x/y`: sub-pixel jitter offset used during rendering (for TAA convergence)
    /// - `reset`: true on first frame or after camera cut (resets temporal history)
    pub fn encode(
        &self,
        encoder: &MetalCommandEncoder,
        color: &MetalTexture,
        depth: &MetalTexture,
        motion: &MetalTexture,
        output: &MetalTexture,
        jitter_x: f32,
        jitter_y: f32,
        reset: bool,
    ) {
        use objc2_metal_fx::MTLFXTemporalScalerBase;

        unsafe {
            self.scaler.setColorTexture(Some(&color.0));
            self.scaler.setDepthTexture(Some(&depth.0));
            self.scaler.setMotionTexture(Some(&motion.0));
            self.scaler.setOutputTexture(Some(&output.0));
            self.scaler.setInputContentWidth(self.input_width as usize);
            self.scaler
                .setInputContentHeight(self.input_height as usize);
            self.scaler.setJitterOffsetX(jitter_x);
            self.scaler.setJitterOffsetY(jitter_y);
            self.scaler.setReset(reset);
        }

        use objc2_metal_fx::MTLFXTemporalScaler;
        unsafe {
            self.scaler.encodeToCommandBuffer(&encoder.command_buffer);
        }
    }

    pub fn input_size(&self) -> (u32, u32) {
        (self.input_width, self.input_height)
    }

    pub fn output_size(&self) -> (u32, u32) {
        (self.output_width, self.output_height)
    }
}

// ===========================================================================
// Indirect Command Buffers (C.3)
// ===========================================================================

/// GPU-side command buffer that allows the GPU to build its own draw list.
///
/// A compute shader writes draw commands into the ICB, then the render
/// encoder executes them — eliminating CPU→GPU round trips for draw call
/// submission. Essential for GPU-driven rendering pipelines.
pub struct MetalIndirectCommandBuffer {
    icb: SendSync<Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>>,
    max_count: u32,
}

impl MetalDevice {
    /// Create an indirect command buffer for GPU-driven rendering.
    ///
    /// - `max_count`: maximum number of draw commands the buffer can hold
    /// - `inherit_pipeline`: if true, commands inherit the pipeline state from the encoder
    /// - `max_vertex_buffers`: max vertex buffer bindings per command
    /// - `max_fragment_buffers`: max fragment buffer bindings per command
    pub fn create_indirect_command_buffer(
        &self,
        max_count: u32,
        inherit_pipeline: bool,
        max_vertex_buffers: u32,
        max_fragment_buffers: u32,
    ) -> MetalIndirectCommandBuffer {
        unsafe {
            let desc = MTLIndirectCommandBufferDescriptor::new();
            desc.setCommandTypes(
                MTLIndirectCommandType::Draw | MTLIndirectCommandType::DrawIndexed,
            );
            desc.setInheritPipelineState(inherit_pipeline);
            desc.setInheritBuffers(false);
            desc.setMaxVertexBufferBindCount(max_vertex_buffers as usize);
            desc.setMaxFragmentBufferBindCount(max_fragment_buffers as usize);

            let icb = self
                .device
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                    &desc,
                    max_count as usize,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("Failed to create Metal indirect command buffer");

            MetalIndirectCommandBuffer {
                icb: SendSync(icb),
                max_count,
            }
        }
    }
}

impl MetalIndirectCommandBuffer {
    /// Get the raw ICB for passing to compute shaders via argument buffers.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLIndirectCommandBuffer> {
        &self.icb
    }

    /// Maximum number of commands this buffer can hold.
    pub fn max_count(&self) -> u32 {
        self.max_count
    }

    /// Reset all commands in the buffer (call before re-encoding).
    pub fn reset(&self) {
        unsafe {
            self.icb.resetWithRange(objc2_foundation::NSRange {
                location: 0,
                length: self.max_count as usize,
            });
        }
    }
}

impl MetalRenderPass<'_> {
    /// Execute commands from an indirect command buffer.
    ///
    /// Runs commands at indices `0..count` from the ICB. Commands with zero
    /// instance count are skipped by the GPU.
    pub fn execute_indirect_commands(&self, icb: &MetalIndirectCommandBuffer, count: u32) {
        unsafe {
            self.encoder.executeCommandsInBuffer_withRange(
                &icb.icb,
                objc2_foundation::NSRange {
                    location: 0,
                    length: count as usize,
                },
            );
        }
    }
}

// ===========================================================================
// Tile Shading (C.2)
// ===========================================================================

/// Tile render pipeline for deferred lighting in Apple TBDR tile memory.
///
/// On Apple Silicon, tile memory is fast on-chip SRAM (~64KB per tile). A tile
/// shader reads/writes tile memory directly, enabling single-pass deferred
/// rendering: write G-buffer data in the fragment stage, then run a tile
/// shader to compute lighting without round-tripping through system memory.
pub struct MetalTileRenderPipeline {
    state: SendSync<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
}

impl MetalDevice {
    /// Create a tile render pipeline for deferred lighting in tile memory.
    ///
    /// The tile function runs after the rasterization stage and has access to
    /// the color attachments as imageblock data (tile memory).
    ///
    /// - `shader`: compiled MSL library containing the tile function
    /// - `tile_entry`: name of the `[[kernel]]` function for tile shading
    /// - `color_formats`: pixel formats of the color attachments
    /// - `sample_count`: rasterization sample count (1 for no MSAA)
    /// - `label`: optional debug label
    pub fn create_tile_render_pipeline(
        &self,
        shader: &MetalShaderModule,
        tile_entry: &str,
        color_formats: &[TextureFormat],
        sample_count: u32,
        label: Option<&str>,
    ) -> MetalTileRenderPipeline {
        unsafe {
            let desc = MTLTileRenderPipelineDescriptor::new();

            if let Some(lbl) = label {
                desc.setLabel(Some(&NSString::from_str(lbl)));
            }

            let tile_fn = shader
                .0
                .newFunctionWithName(&NSString::from_str(tile_entry))
                .unwrap_or_else(|| {
                    panic!("Tile function '{tile_entry}' not found in Metal library")
                });
            desc.setTileFunction(&tile_fn);
            desc.setRasterSampleCount(sample_count as usize);
            desc.setThreadgroupSizeMatchesTileSize(true);

            let color_attachments = desc.colorAttachments();
            for (i, format) in color_formats.iter().enumerate() {
                let attachment = color_attachments.objectAtIndexedSubscript(i);
                attachment.setPixelFormat(to_mtl_pixel_format(*format));
            }

            let pipeline = self
                .device
                .newRenderPipelineStateWithTileDescriptor_options_reflection_error(
                    &desc,
                    MTLPipelineOption::None,
                    None,
                )
                .expect("Failed to create Metal tile render pipeline");

            MetalTileRenderPipeline {
                state: SendSync(pipeline),
            }
        }
    }
}

impl MetalRenderPass<'_> {
    /// Set the tile render pipeline for a tile dispatch.
    pub fn set_tile_pipeline(&mut self, pipeline: &MetalTileRenderPipeline) {
        self.encoder.setRenderPipelineState(&pipeline.state);
    }

    /// Bind a buffer to the tile shader at the given slot.
    pub fn set_tile_buffer(&mut self, slot: u32, buffer: &MetalBuffer, offset: u64) {
        unsafe {
            self.encoder.setTileBuffer_offset_atIndex(
                Some(&buffer.0),
                offset as usize,
                slot as usize,
            );
        }
    }

    /// Bind a texture to the tile shader at the given slot.
    pub fn set_tile_texture(&mut self, slot: u32, texture: &MetalTextureView) {
        unsafe {
            self.encoder
                .setTileTexture_atIndex(Some(&texture.0), slot as usize);
        }
    }

    /// Dispatch tile shader threads per tile.
    ///
    /// The tile shader runs once per tile (typically 32x32 pixels). Each thread
    /// in the dispatch can read/write the tile's imageblock data.
    pub fn dispatch_threads_per_tile(&self, threads_per_tile: [u32; 2]) {
        self.encoder.dispatchThreadsPerTile(MTLSize {
            width: threads_per_tile[0] as usize,
            height: threads_per_tile[1] as usize,
            depth: 1,
        });
    }
}
