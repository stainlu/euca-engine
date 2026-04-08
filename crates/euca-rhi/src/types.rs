//! Backend-agnostic GPU descriptor types, enums, and bitflags.
//!
//! These types mirror the subset of `wgpu` descriptors actually used by the
//! engine. Each backend (wgpu, Metal, etc.) provides `From` conversions to
//! translate these into native API calls.

use std::borrow::Cow;
// ---------------------------------------------------------------------------
// Math / size types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Extent3d {
    pub width: u32,
    pub height: u32,
    pub depth_or_array_layers: u32,
}

impl Default for Extent3d {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Origin3d {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl Color {
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
}

// ---------------------------------------------------------------------------
// Texture formats
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    // 8-bit
    R8Unorm,
    R8Snorm,
    R8Uint,

    // 16-bit
    R16Float,
    Rg8Unorm,

    // 32-bit
    R32Float,
    R32Uint,
    Rg16Float,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,

    // 64-bit
    Rg32Float,
    Rgba16Float,

    // 128-bit
    Rgba32Float,

    // Depth / stencil
    Depth16Unorm,
    Depth32Float,
    Depth24Plus,
    Depth24PlusStencil8,
    Depth32FloatStencil8,
}

impl TextureFormat {
    /// Whether this format is an sRGB format.
    pub fn is_srgb(self) -> bool {
        matches!(self, Self::Rgba8UnormSrgb | Self::Bgra8UnormSrgb)
    }

    /// Whether this is a depth or depth-stencil format.
    pub fn is_depth(self) -> bool {
        matches!(
            self,
            Self::Depth16Unorm
                | Self::Depth32Float
                | Self::Depth24Plus
                | Self::Depth24PlusStencil8
                | Self::Depth32FloatStencil8
        )
    }
}

// ---------------------------------------------------------------------------
// Vertex / index formats
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
    Uint8x2,
    Uint8x4,
    Unorm8x2,
    Unorm8x4,
    Float16x2,
    Float16x4,
}

impl VertexFormat {
    /// Size in bytes of this vertex format.
    pub fn size(self) -> u64 {
        match self {
            Self::Uint8x2 | Self::Unorm8x2 => 2,
            Self::Float32
            | Self::Uint32
            | Self::Sint32
            | Self::Uint8x4
            | Self::Unorm8x4
            | Self::Float16x2 => 4,
            Self::Float32x2 | Self::Uint32x2 | Self::Sint32x2 | Self::Float16x4 => 8,
            Self::Float32x3 | Self::Uint32x3 | Self::Sint32x3 => 12,
            Self::Float32x4 | Self::Uint32x4 | Self::Sint32x4 => 16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

// ---------------------------------------------------------------------------
// Texture dimension / view / aspect
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureDimension {
    D1,
    D2,
    D3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureViewDimension {
    D1,
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TextureAspect {
    #[default]
    All,
    StencilOnly,
    DepthOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureSampleType {
    Float { filterable: bool },
    UnfilteredFloat,
    Depth,
    Sint,
    Uint,
}

// ---------------------------------------------------------------------------
// Bitflag types
// ---------------------------------------------------------------------------

macro_rules! bitflags {
    ($(#[$meta:meta])* $vis:vis struct $name:ident($inner:ty); $($flag:ident = $val:expr;)*) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        $vis struct $name($inner);

        impl $name {
            pub const NONE: Self = Self(0);
            $(pub const $flag: Self = Self($val);)*

            pub const fn contains(self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }

            pub const fn bits(self) -> $inner {
                self.0
            }

            pub const fn empty() -> Self {
                Self(0)
            }

            pub const fn is_empty(self) -> bool {
                self.0 == 0
            }
        }

        impl std::ops::BitOr for $name {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self { Self(self.0 | rhs.0) }
        }

        impl std::ops::BitOrAssign for $name {
            fn bitor_assign(&mut self, rhs: Self) { self.0 |= rhs.0; }
        }

        impl std::ops::BitAnd for $name {
            type Output = Self;
            fn bitand(self, rhs: Self) -> Self { Self(self.0 & rhs.0) }
        }

        impl Default for $name {
            fn default() -> Self { Self::NONE }
        }
    };
}

bitflags! {
    pub struct BufferUsages(u32);
    MAP_READ      = 1 << 0;
    MAP_WRITE     = 1 << 1;
    COPY_SRC      = 1 << 2;
    COPY_DST      = 1 << 3;
    INDEX         = 1 << 4;
    VERTEX        = 1 << 5;
    UNIFORM       = 1 << 6;
    STORAGE       = 1 << 7;
    INDIRECT      = 1 << 8;
    QUERY_RESOLVE = 1 << 9;
}

bitflags! {
    pub struct TextureUsages(u32);
    COPY_SRC          = 1 << 0;
    COPY_DST          = 1 << 1;
    TEXTURE_BINDING   = 1 << 2;
    STORAGE_BINDING   = 1 << 3;
    RENDER_ATTACHMENT = 1 << 4;
}

bitflags! {
    pub struct ShaderStages(u32);
    VERTEX   = 1 << 0;
    FRAGMENT = 1 << 1;
    COMPUTE  = 1 << 2;
}

impl ShaderStages {
    pub const VERTEX_FRAGMENT: Self = Self(Self::VERTEX.0 | Self::FRAGMENT.0);
}

bitflags! {
    pub struct ColorWrites(u32);
    RED   = 1 << 0;
    GREEN = 1 << 1;
    BLUE  = 1 << 2;
    ALPHA = 1 << 3;
}

impl ColorWrites {
    pub const ALL: Self = Self(Self::RED.0 | Self::GREEN.0 | Self::BLUE.0 | Self::ALPHA.0);
}

// ---------------------------------------------------------------------------
// Simple enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AddressMode {
    #[default]
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FilterMode {
    #[default]
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Face {
    Front,
    Back,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FrontFace {
    #[default]
    Ccw,
    Cw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    #[default]
    TriangleList,
    TriangleStrip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VertexStepMode {
    #[default]
    Vertex,
    Instance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PolygonMode {
    #[default]
    Fill,
    Line,
    Point,
}

// ---------------------------------------------------------------------------
// Load / store operations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadOp<V> {
    Clear(V),
    Load,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StoreOp {
    #[default]
    Store,
    Discard,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Operations<V> {
    pub load: LoadOp<V>,
    pub store: StoreOp,
}

// ---------------------------------------------------------------------------
// Blend types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    Zero,
    One,
    Src,
    OneMinusSrc,
    SrcAlpha,
    OneMinusSrcAlpha,
    Dst,
    OneMinusDst,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturated,
    Constant,
    OneMinusConstant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BlendOperation {
    #[default]
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendComponent {
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
    pub operation: BlendOperation,
}

impl Default for BlendComponent {
    fn default() -> Self {
        Self {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::Zero,
            operation: BlendOperation::Add,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendState {
    pub color: BlendComponent,
    pub alpha: BlendComponent,
}

impl BlendState {
    pub const REPLACE: Self = Self {
        color: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::Zero,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::Zero,
            operation: BlendOperation::Add,
        },
    };

    pub const ALPHA_BLENDING: Self = Self {
        color: BlendComponent {
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
    };

    pub const PREMULTIPLIED_ALPHA_BLENDING: Self = Self {
        color: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
    };
}

// ---------------------------------------------------------------------------
// State types (pipeline configuration)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrimitiveState {
    pub topology: PrimitiveTopology,
    pub strip_index_format: Option<IndexFormat>,
    pub front_face: FrontFace,
    pub cull_mode: Option<Face>,
    pub polygon_mode: PolygonMode,
    pub unclipped_depth: bool,
    pub conservative: bool,
}

impl Default for PrimitiveState {
    fn default() -> Self {
        Self {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DepthBiasState {
    pub constant: i32,
    pub slope_scale: f32,
    pub clamp: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StencilOperation {
    #[default]
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StencilFaceState {
    pub compare: CompareFunction,
    pub fail_op: StencilOperation,
    pub depth_fail_op: StencilOperation,
    pub pass_op: StencilOperation,
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self {
            compare: CompareFunction::Always,
            fail_op: StencilOperation::Keep,
            depth_fail_op: StencilOperation::Keep,
            pass_op: StencilOperation::Keep,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StencilState {
    pub front: StencilFaceState,
    pub back: StencilFaceState,
    pub read_mask: u32,
    pub write_mask: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthStencilState {
    pub format: TextureFormat,
    pub depth_write_enabled: bool,
    pub depth_compare: CompareFunction,
    pub stencil: StencilState,
    pub bias: DepthBiasState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultisampleState {
    pub count: u32,
    pub mask: u64,
    pub alpha_to_coverage_enabled: bool,
}

impl Default for MultisampleState {
    fn default() -> Self {
        Self {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorTargetState {
    pub format: TextureFormat,
    pub blend: Option<BlendState>,
    pub write_mask: ColorWrites,
}

// ---------------------------------------------------------------------------
// Descriptor types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BufferDesc<'a> {
    pub label: Option<&'a str>,
    pub size: u64,
    pub usage: BufferUsages,
    pub mapped_at_creation: bool,
}

#[derive(Debug, Clone)]
pub struct TextureDesc<'a> {
    pub label: Option<&'a str>,
    pub size: Extent3d,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub dimension: TextureDimension,
    pub format: TextureFormat,
    pub usage: TextureUsages,
    pub view_formats: &'a [TextureFormat],
}

#[derive(Debug, Clone)]
pub struct TextureViewDesc<'a> {
    pub label: Option<&'a str>,
    pub format: Option<TextureFormat>,
    pub dimension: Option<TextureViewDimension>,
    pub aspect: TextureAspect,
    pub base_mip_level: u32,
    pub mip_level_count: Option<u32>,
    pub base_array_layer: u32,
    pub array_layer_count: Option<u32>,
}

impl Default for TextureViewDesc<'_> {
    fn default() -> Self {
        Self {
            label: None,
            format: None,
            dimension: None,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SamplerDesc<'a> {
    pub label: Option<&'a str>,
    pub address_mode_u: AddressMode,
    pub address_mode_v: AddressMode,
    pub address_mode_w: AddressMode,
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub mipmap_filter: FilterMode,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub compare: Option<CompareFunction>,
    pub anisotropy_clamp: u16,
}

impl Default for SamplerDesc<'_> {
    fn default() -> Self {
        Self {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            anisotropy_clamp: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ShaderSource<'a> {
    Wgsl(Cow<'a, str>),
    Msl(Cow<'a, str>),
    SpirV(Cow<'a, [u32]>),
}

#[derive(Debug, Clone)]
pub struct ShaderDesc<'a> {
    pub label: Option<&'a str>,
    pub source: ShaderSource<'a>,
}

// ---------------------------------------------------------------------------
// Binding types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferBindingType {
    Uniform,
    Storage { read_only: bool },
}

impl Default for BufferBindingType {
    fn default() -> Self {
        Self::Uniform
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SamplerBindingType {
    Filtering,
    NonFiltering,
    Comparison,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTextureAccess {
    WriteOnly,
    ReadOnly,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingType {
    Buffer {
        ty: BufferBindingType,
        has_dynamic_offset: bool,
        min_binding_size: Option<u64>,
    },
    Sampler(SamplerBindingType),
    Texture {
        sample_type: TextureSampleType,
        view_dimension: TextureViewDimension,
        multisampled: bool,
    },
    StorageTexture {
        access: StorageTextureAccess,
        format: TextureFormat,
        view_dimension: TextureViewDimension,
    },
}

#[derive(Debug, Clone)]
pub struct BindGroupLayoutEntry {
    pub binding: u32,
    pub visibility: ShaderStages,
    pub ty: BindingType,
    /// For binding arrays (e.g. bindless textures). `None` means not an array.
    pub count: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct BindGroupLayoutDesc<'a> {
    pub label: Option<&'a str>,
    pub entries: &'a [BindGroupLayoutEntry],
}

// ---------------------------------------------------------------------------
// Bind group entry types (generic over RenderDevice)
// ---------------------------------------------------------------------------

use crate::RenderDevice;

pub struct BufferBinding<'a, D: RenderDevice + ?Sized> {
    pub buffer: &'a D::Buffer,
    pub offset: u64,
    pub size: Option<u64>,
}

pub enum BindingResource<'a, D: RenderDevice + ?Sized> {
    Buffer(BufferBinding<'a, D>),
    TextureView(&'a D::TextureView),
    Sampler(&'a D::Sampler),
    TextureViewArray(Vec<&'a D::TextureView>),
}

pub struct BindGroupEntry<'a, D: RenderDevice + ?Sized> {
    pub binding: u32,
    pub resource: BindingResource<'a, D>,
}

pub struct BindGroupDesc<'a, D: RenderDevice + ?Sized> {
    pub label: Option<&'a str>,
    pub layout: &'a D::BindGroupLayout,
    pub entries: &'a [BindGroupEntry<'a, D>],
}

// ---------------------------------------------------------------------------
// Pipeline descriptor types (generic over RenderDevice)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct VertexAttribute {
    pub format: VertexFormat,
    pub offset: u64,
    pub shader_location: u32,
}

#[derive(Debug, Clone)]
pub struct VertexBufferLayout<'a> {
    pub array_stride: u64,
    pub step_mode: VertexStepMode,
    pub attributes: &'a [VertexAttribute],
}

pub struct VertexState<'a, D: RenderDevice + ?Sized> {
    pub module: &'a D::ShaderModule,
    pub entry_point: &'a str,
    pub buffers: &'a [VertexBufferLayout<'a>],
}

pub struct FragmentState<'a, D: RenderDevice + ?Sized> {
    pub module: &'a D::ShaderModule,
    pub entry_point: &'a str,
    pub targets: &'a [Option<ColorTargetState>],
}

pub struct RenderPipelineDesc<'a, D: RenderDevice + ?Sized> {
    pub label: Option<&'a str>,
    pub layout: &'a [&'a D::BindGroupLayout],
    pub vertex: VertexState<'a, D>,
    pub fragment: Option<FragmentState<'a, D>>,
    pub primitive: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub multisample: MultisampleState,
}

pub struct ComputePipelineDesc<'a, D: RenderDevice + ?Sized> {
    pub label: Option<&'a str>,
    pub layout: &'a [&'a D::BindGroupLayout],
    pub module: &'a D::ShaderModule,
    pub entry_point: &'a str,
}

// ---------------------------------------------------------------------------
// Render pass types (generic over RenderDevice)
// ---------------------------------------------------------------------------

pub struct RenderPassColorAttachment<'a, D: RenderDevice + ?Sized> {
    pub view: &'a D::TextureView,
    pub resolve_target: Option<&'a D::TextureView>,
    pub ops: Operations<Color>,
}

pub struct RenderPassDepthStencilAttachment<'a, D: RenderDevice + ?Sized> {
    pub view: &'a D::TextureView,
    pub depth_ops: Option<Operations<f32>>,
    pub stencil_ops: Option<Operations<u32>>,
}

/// Timestamp query writes for a render or compute pass.
///
/// When attached to a pass descriptor, the GPU writes timestamps at the
/// beginning and/or end of the pass into the given query set indices.
pub struct RenderPassTimestampWrites<'a, D: RenderDevice + ?Sized> {
    /// The query set that receives the timestamp values.
    pub query_set: &'a D::QuerySet,
    /// Query index for the beginning-of-pass timestamp, or `None` to skip.
    pub beginning_of_pass_write_index: Option<u32>,
    /// Query index for the end-of-pass timestamp, or `None` to skip.
    pub end_of_pass_write_index: Option<u32>,
}

pub struct RenderPassDesc<'a, D: RenderDevice + ?Sized> {
    pub label: Option<&'a str>,
    pub color_attachments: &'a [Option<RenderPassColorAttachment<'a, D>>],
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachment<'a, D>>,
    /// Optional GPU timestamp writes for per-pass timing.
    pub timestamp_writes: Option<RenderPassTimestampWrites<'a, D>>,
}

// ---------------------------------------------------------------------------
// Texture data layout (for write_texture)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct TextureDataLayout {
    pub offset: u64,
    pub bytes_per_row: Option<u32>,
    pub rows_per_image: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
pub struct TexelCopyTextureInfo<'a, D: RenderDevice + ?Sized> {
    pub texture: &'a D::Texture,
    pub mip_level: u32,
    pub origin: Origin3d,
    pub aspect: TextureAspect,
}

// ---------------------------------------------------------------------------
// Surface error
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum SurfaceError {
    Timeout,
    Outdated,
    Lost,
    OutOfMemory,
}

impl std::fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Timeout => write!(f, "surface acquire timed out"),
            Self::Outdated => write!(f, "surface is outdated"),
            Self::Lost => write!(f, "surface was lost"),
            Self::OutOfMemory => write!(f, "out of GPU memory"),
        }
    }
}

impl std::error::Error for SurfaceError {}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

/// GPU capabilities reported by the backend.
#[derive(Debug, Clone)]
pub struct Capabilities {
    pub unified_memory: bool,
    pub multi_draw_indirect: bool,
    pub multi_draw_indirect_count: bool,
    pub texture_binding_array: bool,
    pub non_uniform_indexing: bool,
    pub max_texture_dimension_2d: u32,
    pub max_bind_groups: u32,
    pub max_bindings_per_bind_group: u32,
    pub max_binding_array_elements: u32,
    /// Human-readable GPU device name (e.g. "Apple M1 Pro").
    pub device_name: String,
    /// Whether this is an Apple Silicon GPU (supports Apple GPU family).
    pub apple_silicon: bool,
    /// Maximum buffer allocation size in bytes.
    pub max_buffer_length: u64,
    /// Whether memoryless render targets are supported (tile memory only,
    /// saves ~20% bandwidth for transient G-buffer attachments).
    pub memoryless_render_targets: bool,
    /// Whether GPU timestamp queries are supported for per-pass timing.
    pub timestamp_query: bool,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            unified_memory: false,
            multi_draw_indirect: false,
            multi_draw_indirect_count: false,
            texture_binding_array: false,
            non_uniform_indexing: false,
            max_texture_dimension_2d: 8192,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 640,
            max_binding_array_elements: 0,
            device_name: String::from("Unknown"),
            apple_silicon: false,
            max_buffer_length: 256 * 1024 * 1024, // 256 MiB conservative default
            memoryless_render_targets: false,
            timestamp_query: false,
        }
    }
}
