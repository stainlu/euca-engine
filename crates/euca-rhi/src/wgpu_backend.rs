//! wgpu backend — implements [`RenderDevice`] by wrapping `wgpu::Device`,
//! `wgpu::Queue`, and `wgpu::Surface`.
//!
//! This backend is the default cross-platform path. Every RHI descriptor type
//! converts to its wgpu equivalent via `From` impls, and the trait methods
//! delegate directly to wgpu calls with zero overhead.

use std::sync::Arc;

use crate::RenderDevice;
use crate::pass::{ComputePassOps, RenderPassOps};
use crate::types::*;

// ===========================================================================
// WgpuDevice
// ===========================================================================

/// Cross-platform GPU backend wrapping wgpu.
pub struct WgpuDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub window: Arc<winit::window::Window>,
    capabilities: Capabilities,
}

impl WgpuDevice {
    /// Create a new `WgpuDevice` from pre-initialized wgpu objects.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: wgpu::Surface<'static>,
        surface_config: wgpu::SurfaceConfiguration,
        window: Arc<winit::window::Window>,
        capabilities: Capabilities,
    ) -> Self {
        Self {
            device,
            queue,
            surface,
            surface_config,
            window,
            capabilities,
        }
    }
}

// ===========================================================================
// From impls: RHI types → wgpu types
// ===========================================================================

impl From<TextureFormat> for wgpu::TextureFormat {
    fn from(f: TextureFormat) -> Self {
        match f {
            TextureFormat::R8Unorm => Self::R8Unorm,
            TextureFormat::R8Snorm => Self::R8Snorm,
            TextureFormat::R8Uint => Self::R8Uint,
            TextureFormat::R16Float => Self::R16Float,
            TextureFormat::Rg8Unorm => Self::Rg8Unorm,
            TextureFormat::R32Float => Self::R32Float,
            TextureFormat::R32Uint => Self::R32Uint,
            TextureFormat::Rg16Float => Self::Rg16Float,
            TextureFormat::Rgba8Unorm => Self::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb => Self::Rgba8UnormSrgb,
            TextureFormat::Bgra8Unorm => Self::Bgra8Unorm,
            TextureFormat::Bgra8UnormSrgb => Self::Bgra8UnormSrgb,
            TextureFormat::Rg32Float => Self::Rg32Float,
            TextureFormat::Rgba16Float => Self::Rgba16Float,
            TextureFormat::Rgba32Float => Self::Rgba32Float,
            TextureFormat::Depth16Unorm => Self::Depth16Unorm,
            TextureFormat::Depth32Float => Self::Depth32Float,
            TextureFormat::Depth24Plus => Self::Depth24Plus,
            TextureFormat::Depth24PlusStencil8 => Self::Depth24PlusStencil8,
            TextureFormat::Depth32FloatStencil8 => Self::Depth32FloatStencil8,
        }
    }
}

impl From<wgpu::TextureFormat> for TextureFormat {
    fn from(f: wgpu::TextureFormat) -> Self {
        match f {
            wgpu::TextureFormat::R8Unorm => Self::R8Unorm,
            wgpu::TextureFormat::R8Snorm => Self::R8Snorm,
            wgpu::TextureFormat::R8Uint => Self::R8Uint,
            wgpu::TextureFormat::R16Float => Self::R16Float,
            wgpu::TextureFormat::Rg8Unorm => Self::Rg8Unorm,
            wgpu::TextureFormat::R32Float => Self::R32Float,
            wgpu::TextureFormat::R32Uint => Self::R32Uint,
            wgpu::TextureFormat::Rg16Float => Self::Rg16Float,
            wgpu::TextureFormat::Rgba8Unorm => Self::Rgba8Unorm,
            wgpu::TextureFormat::Rgba8UnormSrgb => Self::Rgba8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm => Self::Bgra8Unorm,
            wgpu::TextureFormat::Bgra8UnormSrgb => Self::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rg32Float => Self::Rg32Float,
            wgpu::TextureFormat::Rgba16Float => Self::Rgba16Float,
            wgpu::TextureFormat::Rgba32Float => Self::Rgba32Float,
            wgpu::TextureFormat::Depth16Unorm => Self::Depth16Unorm,
            wgpu::TextureFormat::Depth32Float => Self::Depth32Float,
            wgpu::TextureFormat::Depth24Plus => Self::Depth24Plus,
            wgpu::TextureFormat::Depth24PlusStencil8 => Self::Depth24PlusStencil8,
            wgpu::TextureFormat::Depth32FloatStencil8 => Self::Depth32FloatStencil8,
            other => panic!("Unsupported wgpu texture format: {other:?}"),
        }
    }
}

impl From<VertexFormat> for wgpu::VertexFormat {
    fn from(f: VertexFormat) -> Self {
        match f {
            VertexFormat::Float32 => Self::Float32,
            VertexFormat::Float32x2 => Self::Float32x2,
            VertexFormat::Float32x3 => Self::Float32x3,
            VertexFormat::Float32x4 => Self::Float32x4,
            VertexFormat::Uint32 => Self::Uint32,
            VertexFormat::Uint32x2 => Self::Uint32x2,
            VertexFormat::Uint32x3 => Self::Uint32x3,
            VertexFormat::Uint32x4 => Self::Uint32x4,
            VertexFormat::Sint32 => Self::Sint32,
            VertexFormat::Sint32x2 => Self::Sint32x2,
            VertexFormat::Sint32x3 => Self::Sint32x3,
            VertexFormat::Sint32x4 => Self::Sint32x4,
            VertexFormat::Uint8x2 => Self::Uint8x2,
            VertexFormat::Uint8x4 => Self::Uint8x4,
            VertexFormat::Unorm8x2 => Self::Unorm8x2,
            VertexFormat::Unorm8x4 => Self::Unorm8x4,
            VertexFormat::Float16x2 => Self::Float16x2,
            VertexFormat::Float16x4 => Self::Float16x4,
        }
    }
}

impl From<IndexFormat> for wgpu::IndexFormat {
    fn from(f: IndexFormat) -> Self {
        match f {
            IndexFormat::Uint16 => Self::Uint16,
            IndexFormat::Uint32 => Self::Uint32,
        }
    }
}

impl From<TextureDimension> for wgpu::TextureDimension {
    fn from(d: TextureDimension) -> Self {
        match d {
            TextureDimension::D1 => Self::D1,
            TextureDimension::D2 => Self::D2,
            TextureDimension::D3 => Self::D3,
        }
    }
}

impl From<TextureViewDimension> for wgpu::TextureViewDimension {
    fn from(d: TextureViewDimension) -> Self {
        match d {
            TextureViewDimension::D1 => Self::D1,
            TextureViewDimension::D2 => Self::D2,
            TextureViewDimension::D2Array => Self::D2Array,
            TextureViewDimension::Cube => Self::Cube,
            TextureViewDimension::CubeArray => Self::CubeArray,
            TextureViewDimension::D3 => Self::D3,
        }
    }
}

impl From<TextureAspect> for wgpu::TextureAspect {
    fn from(a: TextureAspect) -> Self {
        match a {
            TextureAspect::All => Self::All,
            TextureAspect::StencilOnly => Self::StencilOnly,
            TextureAspect::DepthOnly => Self::DepthOnly,
        }
    }
}

impl From<AddressMode> for wgpu::AddressMode {
    fn from(m: AddressMode) -> Self {
        match m {
            AddressMode::ClampToEdge => Self::ClampToEdge,
            AddressMode::Repeat => Self::Repeat,
            AddressMode::MirrorRepeat => Self::MirrorRepeat,
        }
    }
}

impl From<FilterMode> for wgpu::FilterMode {
    fn from(m: FilterMode) -> Self {
        match m {
            FilterMode::Nearest => Self::Nearest,
            FilterMode::Linear => Self::Linear,
        }
    }
}

impl From<CompareFunction> for wgpu::CompareFunction {
    fn from(c: CompareFunction) -> Self {
        match c {
            CompareFunction::Never => Self::Never,
            CompareFunction::Less => Self::Less,
            CompareFunction::Equal => Self::Equal,
            CompareFunction::LessEqual => Self::LessEqual,
            CompareFunction::Greater => Self::Greater,
            CompareFunction::NotEqual => Self::NotEqual,
            CompareFunction::GreaterEqual => Self::GreaterEqual,
            CompareFunction::Always => Self::Always,
        }
    }
}

impl From<Face> for wgpu::Face {
    fn from(f: Face) -> Self {
        match f {
            Face::Front => Self::Front,
            Face::Back => Self::Back,
        }
    }
}

impl From<FrontFace> for wgpu::FrontFace {
    fn from(f: FrontFace) -> Self {
        match f {
            FrontFace::Ccw => Self::Ccw,
            FrontFace::Cw => Self::Cw,
        }
    }
}

impl From<PrimitiveTopology> for wgpu::PrimitiveTopology {
    fn from(t: PrimitiveTopology) -> Self {
        match t {
            PrimitiveTopology::PointList => Self::PointList,
            PrimitiveTopology::LineList => Self::LineList,
            PrimitiveTopology::LineStrip => Self::LineStrip,
            PrimitiveTopology::TriangleList => Self::TriangleList,
            PrimitiveTopology::TriangleStrip => Self::TriangleStrip,
        }
    }
}

impl From<VertexStepMode> for wgpu::VertexStepMode {
    fn from(m: VertexStepMode) -> Self {
        match m {
            VertexStepMode::Vertex => Self::Vertex,
            VertexStepMode::Instance => Self::Instance,
        }
    }
}

impl From<PolygonMode> for wgpu::PolygonMode {
    fn from(m: PolygonMode) -> Self {
        match m {
            PolygonMode::Fill => Self::Fill,
            PolygonMode::Line => Self::Line,
            PolygonMode::Point => Self::Point,
        }
    }
}

impl From<BlendFactor> for wgpu::BlendFactor {
    fn from(f: BlendFactor) -> Self {
        match f {
            BlendFactor::Zero => Self::Zero,
            BlendFactor::One => Self::One,
            BlendFactor::Src => Self::Src,
            BlendFactor::OneMinusSrc => Self::OneMinusSrc,
            BlendFactor::SrcAlpha => Self::SrcAlpha,
            BlendFactor::OneMinusSrcAlpha => Self::OneMinusSrcAlpha,
            BlendFactor::Dst => Self::Dst,
            BlendFactor::OneMinusDst => Self::OneMinusDst,
            BlendFactor::DstAlpha => Self::DstAlpha,
            BlendFactor::OneMinusDstAlpha => Self::OneMinusDstAlpha,
            BlendFactor::SrcAlphaSaturated => Self::SrcAlphaSaturated,
            BlendFactor::Constant => Self::Constant,
            BlendFactor::OneMinusConstant => Self::OneMinusConstant,
        }
    }
}

impl From<BlendOperation> for wgpu::BlendOperation {
    fn from(o: BlendOperation) -> Self {
        match o {
            BlendOperation::Add => Self::Add,
            BlendOperation::Subtract => Self::Subtract,
            BlendOperation::ReverseSubtract => Self::ReverseSubtract,
            BlendOperation::Min => Self::Min,
            BlendOperation::Max => Self::Max,
        }
    }
}

impl From<BlendComponent> for wgpu::BlendComponent {
    fn from(c: BlendComponent) -> Self {
        Self {
            src_factor: c.src_factor.into(),
            dst_factor: c.dst_factor.into(),
            operation: c.operation.into(),
        }
    }
}

impl From<BlendState> for wgpu::BlendState {
    fn from(s: BlendState) -> Self {
        Self {
            color: s.color.into(),
            alpha: s.alpha.into(),
        }
    }
}

impl From<StencilOperation> for wgpu::StencilOperation {
    fn from(o: StencilOperation) -> Self {
        match o {
            StencilOperation::Keep => Self::Keep,
            StencilOperation::Zero => Self::Zero,
            StencilOperation::Replace => Self::Replace,
            StencilOperation::IncrementClamp => Self::IncrementClamp,
            StencilOperation::DecrementClamp => Self::DecrementClamp,
            StencilOperation::Invert => Self::Invert,
            StencilOperation::IncrementWrap => Self::IncrementWrap,
            StencilOperation::DecrementWrap => Self::DecrementWrap,
        }
    }
}

impl From<StencilFaceState> for wgpu::StencilFaceState {
    fn from(s: StencilFaceState) -> Self {
        Self {
            compare: s.compare.into(),
            fail_op: s.fail_op.into(),
            depth_fail_op: s.depth_fail_op.into(),
            pass_op: s.pass_op.into(),
        }
    }
}

impl From<StencilState> for wgpu::StencilState {
    fn from(s: StencilState) -> Self {
        Self {
            front: s.front.into(),
            back: s.back.into(),
            read_mask: s.read_mask,
            write_mask: s.write_mask,
        }
    }
}

impl From<DepthBiasState> for wgpu::DepthBiasState {
    fn from(d: DepthBiasState) -> Self {
        Self {
            constant: d.constant,
            slope_scale: d.slope_scale,
            clamp: d.clamp,
        }
    }
}

impl From<Extent3d> for wgpu::Extent3d {
    fn from(e: Extent3d) -> Self {
        Self {
            width: e.width,
            height: e.height,
            depth_or_array_layers: e.depth_or_array_layers,
        }
    }
}

impl From<Origin3d> for wgpu::Origin3d {
    fn from(o: Origin3d) -> Self {
        Self {
            x: o.x,
            y: o.y,
            z: o.z,
        }
    }
}

impl From<Color> for wgpu::Color {
    fn from(c: Color) -> Self {
        Self {
            r: c.r,
            g: c.g,
            b: c.b,
            a: c.a,
        }
    }
}

impl From<BufferUsages> for wgpu::BufferUsages {
    fn from(u: BufferUsages) -> Self {
        let mut out = Self::empty();
        if u.contains(BufferUsages::MAP_READ) {
            out |= Self::MAP_READ;
        }
        if u.contains(BufferUsages::MAP_WRITE) {
            out |= Self::MAP_WRITE;
        }
        if u.contains(BufferUsages::COPY_SRC) {
            out |= Self::COPY_SRC;
        }
        if u.contains(BufferUsages::COPY_DST) {
            out |= Self::COPY_DST;
        }
        if u.contains(BufferUsages::INDEX) {
            out |= Self::INDEX;
        }
        if u.contains(BufferUsages::VERTEX) {
            out |= Self::VERTEX;
        }
        if u.contains(BufferUsages::UNIFORM) {
            out |= Self::UNIFORM;
        }
        if u.contains(BufferUsages::STORAGE) {
            out |= Self::STORAGE;
        }
        if u.contains(BufferUsages::INDIRECT) {
            out |= Self::INDIRECT;
        }
        out
    }
}

impl From<TextureUsages> for wgpu::TextureUsages {
    fn from(u: TextureUsages) -> Self {
        let mut out = Self::empty();
        if u.contains(TextureUsages::COPY_SRC) {
            out |= Self::COPY_SRC;
        }
        if u.contains(TextureUsages::COPY_DST) {
            out |= Self::COPY_DST;
        }
        if u.contains(TextureUsages::TEXTURE_BINDING) {
            out |= Self::TEXTURE_BINDING;
        }
        if u.contains(TextureUsages::STORAGE_BINDING) {
            out |= Self::STORAGE_BINDING;
        }
        if u.contains(TextureUsages::RENDER_ATTACHMENT) {
            out |= Self::RENDER_ATTACHMENT;
        }
        out
    }
}

impl From<ShaderStages> for wgpu::ShaderStages {
    fn from(s: ShaderStages) -> Self {
        let mut out = Self::empty();
        if s.contains(ShaderStages::VERTEX) {
            out |= Self::VERTEX;
        }
        if s.contains(ShaderStages::FRAGMENT) {
            out |= Self::FRAGMENT;
        }
        if s.contains(ShaderStages::COMPUTE) {
            out |= Self::COMPUTE;
        }
        out
    }
}

impl From<ColorWrites> for wgpu::ColorWrites {
    fn from(c: ColorWrites) -> Self {
        let mut out = Self::empty();
        if c.contains(ColorWrites::RED) {
            out |= Self::RED;
        }
        if c.contains(ColorWrites::GREEN) {
            out |= Self::GREEN;
        }
        if c.contains(ColorWrites::BLUE) {
            out |= Self::BLUE;
        }
        if c.contains(ColorWrites::ALPHA) {
            out |= Self::ALPHA;
        }
        out
    }
}

impl From<TextureSampleType> for wgpu::TextureSampleType {
    fn from(t: TextureSampleType) -> Self {
        match t {
            TextureSampleType::Float { filterable } => Self::Float { filterable },
            TextureSampleType::UnfilteredFloat => Self::Float { filterable: false },
            TextureSampleType::Depth => Self::Depth,
            TextureSampleType::Sint => Self::Sint,
            TextureSampleType::Uint => Self::Uint,
        }
    }
}

impl From<BufferBindingType> for wgpu::BufferBindingType {
    fn from(t: BufferBindingType) -> Self {
        match t {
            BufferBindingType::Uniform => Self::Uniform,
            BufferBindingType::Storage { read_only } => Self::Storage { read_only },
        }
    }
}

impl From<SamplerBindingType> for wgpu::SamplerBindingType {
    fn from(t: SamplerBindingType) -> Self {
        match t {
            SamplerBindingType::Filtering => Self::Filtering,
            SamplerBindingType::NonFiltering => Self::NonFiltering,
            SamplerBindingType::Comparison => Self::Comparison,
        }
    }
}

impl From<StorageTextureAccess> for wgpu::StorageTextureAccess {
    fn from(a: StorageTextureAccess) -> Self {
        match a {
            StorageTextureAccess::WriteOnly => Self::WriteOnly,
            StorageTextureAccess::ReadOnly => Self::ReadOnly,
            StorageTextureAccess::ReadWrite => Self::ReadWrite,
        }
    }
}

// ===========================================================================
// Conversion helpers for complex descriptor types
// ===========================================================================

fn convert_binding_type(bt: &BindingType) -> wgpu::BindingType {
    match *bt {
        BindingType::Buffer {
            ty,
            has_dynamic_offset,
            min_binding_size,
        } => wgpu::BindingType::Buffer {
            ty: ty.into(),
            has_dynamic_offset,
            min_binding_size: min_binding_size.and_then(std::num::NonZeroU64::new),
        },
        BindingType::Sampler(sbt) => wgpu::BindingType::Sampler(sbt.into()),
        BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled,
        } => wgpu::BindingType::Texture {
            sample_type: sample_type.into(),
            view_dimension: view_dimension.into(),
            multisampled,
        },
        BindingType::StorageTexture {
            access,
            format,
            view_dimension,
        } => wgpu::BindingType::StorageTexture {
            access: access.into(),
            format: format.into(),
            view_dimension: view_dimension.into(),
        },
    }
}

fn convert_bind_group_layout_entry(e: &BindGroupLayoutEntry) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: e.binding,
        visibility: e.visibility.into(),
        ty: convert_binding_type(&e.ty),
        count: e.count.and_then(std::num::NonZeroU32::new),
    }
}

fn convert_color_target_state(s: &ColorTargetState) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format: s.format.into(),
        blend: s.blend.map(|b| b.into()),
        write_mask: s.write_mask.into(),
    }
}

fn convert_depth_stencil_state(d: &DepthStencilState) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: d.format.into(),
        depth_write_enabled: d.depth_write_enabled,
        depth_compare: d.depth_compare.into(),
        stencil: d.stencil.into(),
        bias: d.bias.into(),
    }
}

fn convert_primitive_state(p: &PrimitiveState) -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: p.topology.into(),
        strip_index_format: p.strip_index_format.map(|f| f.into()),
        front_face: p.front_face.into(),
        cull_mode: p.cull_mode.map(|f| f.into()),
        polygon_mode: p.polygon_mode.into(),
        unclipped_depth: p.unclipped_depth,
        conservative: p.conservative,
    }
}

fn convert_multisample_state(m: &MultisampleState) -> wgpu::MultisampleState {
    wgpu::MultisampleState {
        count: m.count,
        mask: m.mask,
        alpha_to_coverage_enabled: m.alpha_to_coverage_enabled,
    }
}

fn convert_load_op_color(op: &LoadOp<Color>) -> wgpu::LoadOp<wgpu::Color> {
    match op {
        LoadOp::Clear(c) => wgpu::LoadOp::Clear((*c).into()),
        LoadOp::Load => wgpu::LoadOp::Load,
    }
}

fn convert_load_op_f32(op: &LoadOp<f32>) -> wgpu::LoadOp<f32> {
    match *op {
        LoadOp::Clear(v) => wgpu::LoadOp::Clear(v),
        LoadOp::Load => wgpu::LoadOp::Load,
    }
}

fn convert_load_op_u32(op: &LoadOp<u32>) -> wgpu::LoadOp<u32> {
    match *op {
        LoadOp::Clear(v) => wgpu::LoadOp::Clear(v),
        LoadOp::Load => wgpu::LoadOp::Load,
    }
}

fn convert_store_op(op: StoreOp) -> wgpu::StoreOp {
    match op {
        StoreOp::Store => wgpu::StoreOp::Store,
        StoreOp::Discard => wgpu::StoreOp::Discard,
    }
}

// ===========================================================================
// RenderDevice implementation
// ===========================================================================

impl RenderDevice for WgpuDevice {
    type Buffer = wgpu::Buffer;
    type Texture = wgpu::Texture;
    type TextureView = wgpu::TextureView;
    type Sampler = wgpu::Sampler;
    type BindGroupLayout = wgpu::BindGroupLayout;
    type BindGroup = wgpu::BindGroup;
    type ShaderModule = wgpu::ShaderModule;
    type RenderPipeline = wgpu::RenderPipeline;
    type ComputePipeline = wgpu::ComputePipeline;
    type CommandEncoder = wgpu::CommandEncoder;
    type RenderPass<'a> = wgpu::RenderPass<'a>;
    type ComputePass<'a> = wgpu::ComputePass<'a>;
    type SurfaceTexture = wgpu::SurfaceTexture;

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn create_buffer(&self, desc: &BufferDesc) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: desc.label,
            size: desc.size,
            usage: desc.usage.into(),
            mapped_at_creation: desc.mapped_at_creation,
        })
    }

    fn create_texture(&self, desc: &TextureDesc) -> wgpu::Texture {
        let view_formats: Vec<wgpu::TextureFormat> =
            desc.view_formats.iter().copied().map(Into::into).collect();
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: desc.label,
            size: desc.size.into(),
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            dimension: desc.dimension.into(),
            format: desc.format.into(),
            usage: desc.usage.into(),
            view_formats: &view_formats,
        })
    }

    fn create_texture_view(
        &self,
        texture: &wgpu::Texture,
        desc: &TextureViewDesc,
    ) -> wgpu::TextureView {
        texture.create_view(&wgpu::TextureViewDescriptor {
            label: desc.label,
            format: desc.format.map(Into::into),
            dimension: desc.dimension.map(Into::into),
            aspect: desc.aspect.into(),
            base_mip_level: desc.base_mip_level,
            mip_level_count: desc.mip_level_count,
            base_array_layer: desc.base_array_layer,
            array_layer_count: desc.array_layer_count,
            usage: None,
        })
    }

    fn create_sampler(&self, desc: &SamplerDesc) -> wgpu::Sampler {
        self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: desc.label,
            address_mode_u: desc.address_mode_u.into(),
            address_mode_v: desc.address_mode_v.into(),
            address_mode_w: desc.address_mode_w.into(),
            mag_filter: desc.mag_filter.into(),
            min_filter: desc.min_filter.into(),
            mipmap_filter: desc.mipmap_filter.into(),
            lod_min_clamp: desc.lod_min_clamp,
            lod_max_clamp: desc.lod_max_clamp,
            compare: desc.compare.map(Into::into),
            anisotropy_clamp: desc.anisotropy_clamp,
            ..Default::default()
        })
    }

    fn create_shader(&self, desc: &ShaderDesc) -> wgpu::ShaderModule {
        let source = match &desc.source {
            ShaderSource::Wgsl(src) => wgpu::ShaderSource::Wgsl(src.clone()),
            ShaderSource::SpirV(_) => panic!("SPIR-V shader source not supported by wgpu backend"),
            ShaderSource::Msl(_) => panic!("MSL shaders not supported by wgpu backend"),
        };
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: desc.label,
                source,
            })
    }

    fn create_bind_group_layout(&self, desc: &BindGroupLayoutDesc) -> wgpu::BindGroupLayout {
        let entries: Vec<wgpu::BindGroupLayoutEntry> = desc
            .entries
            .iter()
            .map(convert_bind_group_layout_entry)
            .collect();
        self.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: desc.label,
                entries: &entries,
            })
    }

    fn create_bind_group(&self, desc: &BindGroupDesc<Self>) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = desc
            .entries
            .iter()
            .map(|e| wgpu::BindGroupEntry {
                binding: e.binding,
                resource: match &e.resource {
                    BindingResource::Buffer(b) => {
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: b.buffer,
                            offset: b.offset,
                            size: b.size.and_then(std::num::NonZeroU64::new),
                        })
                    }
                    BindingResource::TextureView(v) => wgpu::BindingResource::TextureView(v),
                    BindingResource::Sampler(s) => wgpu::BindingResource::Sampler(s),
                    BindingResource::TextureViewArray(views) => {
                        wgpu::BindingResource::TextureViewArray(views)
                    }
                },
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: desc.label,
            layout: desc.layout,
            entries: &entries,
        })
    }

    fn create_render_pipeline(&self, desc: &RenderPipelineDesc<Self>) -> wgpu::RenderPipeline {
        let bgls: Vec<&wgpu::BindGroupLayout> = desc.layout.to_vec();
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &bgls,
                push_constant_ranges: &[],
            });

        let vertex_buffers: Vec<wgpu::VertexBufferLayout> = desc
            .vertex
            .buffers
            .iter()
            .map(|b| {
                let attrs: Vec<wgpu::VertexAttribute> = b
                    .attributes
                    .iter()
                    .map(|a| wgpu::VertexAttribute {
                        format: a.format.into(),
                        offset: a.offset,
                        shader_location: a.shader_location,
                    })
                    .collect();
                // SAFETY: attrs is leaked intentionally since wgpu::VertexBufferLayout
                // requires a 'static reference. In practice, pipeline creation is done
                // at init time and these allocations live for the program lifetime.
                let attrs: &'static [wgpu::VertexAttribute] = attrs.leak();
                wgpu::VertexBufferLayout {
                    array_stride: b.array_stride,
                    step_mode: b.step_mode.into(),
                    attributes: attrs,
                }
            })
            .collect();

        let targets: Vec<Option<wgpu::ColorTargetState>> = desc
            .fragment
            .as_ref()
            .map(|f| {
                f.targets
                    .iter()
                    .map(|t| t.as_ref().map(convert_color_target_state))
                    .collect()
            })
            .unwrap_or_default();

        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: desc.label,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: desc.vertex.module,
                    entry_point: Some(desc.vertex.entry_point),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: desc.fragment.as_ref().map(|f| wgpu::FragmentState {
                    module: f.module,
                    entry_point: Some(f.entry_point),
                    targets: &targets,
                    compilation_options: Default::default(),
                }),
                primitive: convert_primitive_state(&desc.primitive),
                depth_stencil: desc.depth_stencil.as_ref().map(convert_depth_stencil_state),
                multisample: convert_multisample_state(&desc.multisample),
                multiview: None,
                cache: None,
            })
    }

    fn create_compute_pipeline(&self, desc: &ComputePipelineDesc<Self>) -> wgpu::ComputePipeline {
        let bgls: Vec<&wgpu::BindGroupLayout> = desc.layout.to_vec();
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &bgls,
                push_constant_ranges: &[],
            });

        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: desc.label,
                layout: Some(&pipeline_layout),
                module: desc.module,
                entry_point: Some(desc.entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.queue.write_buffer(buffer, offset, data);
    }

    fn write_texture(
        &self,
        dst: &TexelCopyTextureInfo<Self>,
        data: &[u8],
        layout: &TextureDataLayout,
        size: Extent3d,
    ) {
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: dst.texture,
                mip_level: dst.mip_level,
                origin: dst.origin.into(),
                aspect: dst.aspect.into(),
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: layout.offset,
                bytes_per_row: layout.bytes_per_row,
                rows_per_image: layout.rows_per_image,
            },
            size.into(),
        );
    }

    fn create_command_encoder(&self, label: Option<&str>) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label })
    }

    fn begin_render_pass<'a>(
        &self,
        encoder: &'a mut wgpu::CommandEncoder,
        desc: &RenderPassDesc<'_, Self>,
    ) -> wgpu::RenderPass<'a> {
        let color_attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = desc
            .color_attachments
            .iter()
            .map(|a| {
                a.as_ref().map(|a| wgpu::RenderPassColorAttachment {
                    view: a.view,
                    resolve_target: a.resolve_target,
                    ops: wgpu::Operations {
                        load: convert_load_op_color(&a.ops.load),
                        store: convert_store_op(a.ops.store),
                    },
                    depth_slice: None,
                })
            })
            .collect();

        let depth_stencil = desc.depth_stencil_attachment.as_ref().map(|d| {
            wgpu::RenderPassDepthStencilAttachment {
                view: d.view,
                depth_ops: d.depth_ops.as_ref().map(|ops| wgpu::Operations {
                    load: convert_load_op_f32(&ops.load),
                    store: convert_store_op(ops.store),
                }),
                stencil_ops: d.stencil_ops.as_ref().map(|ops| wgpu::Operations {
                    load: convert_load_op_u32(&ops.load),
                    store: convert_store_op(ops.store),
                }),
            }
        });

        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: desc.label,
            color_attachments: &color_attachments,
            depth_stencil_attachment: depth_stencil,
            timestamp_writes: None,
            occlusion_query_set: None,
        })
    }

    fn begin_compute_pass<'a>(
        &self,
        encoder: &'a mut wgpu::CommandEncoder,
        label: Option<&str>,
    ) -> wgpu::ComputePass<'a> {
        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label,
            timestamp_writes: None,
        })
    }

    fn clear_buffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        buffer: &wgpu::Buffer,
        offset: u64,
        size: Option<u64>,
    ) {
        encoder.clear_buffer(buffer, offset, size);
    }

    fn copy_texture_to_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &TexelCopyTextureInfo<Self>,
        dst: &TexelCopyTextureInfo<Self>,
        size: Extent3d,
    ) {
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src.texture,
                mip_level: src.mip_level,
                origin: src.origin.into(),
                aspect: src.aspect.into(),
            },
            wgpu::TexelCopyTextureInfo {
                texture: dst.texture,
                mip_level: dst.mip_level,
                origin: dst.origin.into(),
                aspect: dst.aspect.into(),
            },
            size.into(),
        );
    }

    fn submit(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn submit_multiple(&self, encoders: Vec<wgpu::CommandEncoder>) {
        let command_buffers: Vec<wgpu::CommandBuffer> =
            encoders.into_iter().map(|e| e.finish()).collect();
        self.queue.submit(command_buffers);
    }

    fn get_current_texture(&self) -> Result<wgpu::SurfaceTexture, SurfaceError> {
        self.surface.get_current_texture().map_err(|e| match e {
            wgpu::SurfaceError::Timeout => SurfaceError::Timeout,
            wgpu::SurfaceError::Outdated => SurfaceError::Outdated,
            wgpu::SurfaceError::Lost => SurfaceError::Lost,
            wgpu::SurfaceError::OutOfMemory => SurfaceError::OutOfMemory,
            _ => SurfaceError::Lost,
        })
    }

    fn surface_texture_view(&self, surface_texture: &wgpu::SurfaceTexture) -> wgpu::TextureView {
        surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn present(&self, texture: wgpu::SurfaceTexture) {
        texture.present();
    }

    fn resize_surface(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    fn surface_format(&self) -> TextureFormat {
        self.surface_config.format.into()
    }

    fn surface_size(&self) -> (u32, u32) {
        (self.surface_config.width, self.surface_config.height)
    }
}

// ===========================================================================
// RenderPassOps for wgpu::RenderPass
// ===========================================================================

impl<'rp> RenderPassOps<WgpuDevice> for wgpu::RenderPass<'rp> {
    fn set_pipeline(&mut self, pipeline: &wgpu::RenderPipeline) {
        self.set_pipeline(pipeline);
    }

    fn set_bind_group(&mut self, index: u32, bind_group: &wgpu::BindGroup, offsets: &[u32]) {
        self.set_bind_group(index, bind_group, offsets);
    }

    fn set_vertex_buffer(&mut self, slot: u32, buffer: &wgpu::Buffer, offset: u64, size: u64) {
        self.set_vertex_buffer(slot, buffer.slice(offset..offset + size));
    }

    fn set_index_buffer(
        &mut self,
        buffer: &wgpu::Buffer,
        format: IndexFormat,
        offset: u64,
        size: u64,
    ) {
        self.set_index_buffer(buffer.slice(offset..offset + size), format.into());
    }

    fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        self.draw(vertices, instances);
    }

    fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        self.draw_indexed(indices, base_vertex, instances);
    }

    fn draw_indexed_indirect(&mut self, indirect_buffer: &wgpu::Buffer, indirect_offset: u64) {
        self.draw_indexed_indirect(indirect_buffer, indirect_offset);
    }

    fn multi_draw_indexed_indirect(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
        count: u32,
    ) {
        self.multi_draw_indexed_indirect(indirect_buffer, indirect_offset, count);
    }

    fn multi_draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
        count_buffer: &wgpu::Buffer,
        count_offset: u64,
        max_count: u32,
    ) {
        self.multi_draw_indexed_indirect_count(
            indirect_buffer,
            indirect_offset,
            count_buffer,
            count_offset,
            max_count,
        );
    }

    fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        self.set_viewport(x, y, w, h, min_depth, max_depth);
    }

    fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        self.set_scissor_rect(x, y, w, h);
    }
}

// ===========================================================================
// ComputePassOps for wgpu::ComputePass
// ===========================================================================

impl<'cp> ComputePassOps<WgpuDevice> for wgpu::ComputePass<'cp> {
    fn set_pipeline(&mut self, pipeline: &wgpu::ComputePipeline) {
        self.set_pipeline(pipeline);
    }

    fn set_bind_group(&mut self, index: u32, bind_group: &wgpu::BindGroup, offsets: &[u32]) {
        self.set_bind_group(index, bind_group, offsets);
    }

    fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        self.dispatch_workgroups(x, y, z);
    }

    fn dispatch_workgroups_indirect(
        &mut self,
        indirect_buffer: &wgpu::Buffer,
        indirect_offset: u64,
    ) {
        self.dispatch_workgroups_indirect(indirect_buffer, indirect_offset);
    }
}
