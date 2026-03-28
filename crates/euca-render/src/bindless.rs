//! Bindless material system.
//!
//! Packs all material uniform data into a single GPU storage buffer and all
//! textures into a `binding_array`, eliminating per-batch bind group switches.
//! Each entity carries a `material_id` in its instance data; the shader indexes
//! into the material array and texture binding array at runtime.
//!
//! # GPU Feature Requirements
//!
//! - `TEXTURE_BINDING_ARRAY` — binding arrays of textures
//! - `SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING` —
//!   non-uniform indexing (material_id varies per fragment)
//!
//! When these features are unavailable, the renderer falls back to the
//! traditional per-batch material binding path.

use crate::buffer::{BufferKind, SmartBuffer};
use crate::material::MaterialHandle;
use crate::texture::{TextureHandle, TextureStore};

/// Required wgpu features for the bindless material path.
pub const BINDLESS_FEATURES: wgpu::Features = wgpu::Features::TEXTURE_BINDING_ARRAY
    .union(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);

/// Maximum number of unique textures in the binding array.
/// Each material can reference up to 5 texture slots (albedo, normal,
/// metallic_roughness, ao, emissive). With 256 materials that's 1280 textures
/// worst-case, but most share textures. 512 is a generous default.
const MAX_BINDLESS_TEXTURES: u32 = 512;

// ---------------------------------------------------------------------------
// GPU-side structs (must match WGSL layout exactly)
// ---------------------------------------------------------------------------

/// Per-material uniform data stored in a storage buffer array.
///
/// Extends `MaterialUniforms` with texture indices into the binding array.
/// Size: 96 bytes (aligned to 16 for storage buffer array stride).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BindlessMaterialGpu {
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub has_normal_map: f32,
    pub has_metallic_roughness_tex: f32,
    pub emissive: [f32; 3],
    pub has_emissive_tex: f32,
    pub has_ao_tex: f32,
    pub alpha_mode: f32,
    pub alpha_cutoff: f32,
    pub _pad0: f32,
    // Texture indices into the binding array (u32, reinterpreted as i32 in WGSL).
    // 0xFFFFFFFF = no texture (use fallback white/flat).
    pub albedo_tex_idx: u32,
    pub normal_tex_idx: u32,
    pub metallic_roughness_tex_idx: u32,
    pub ao_tex_idx: u32,
    pub emissive_tex_idx: u32,
    pub _pad1: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<BindlessMaterialGpu>() == 96);
const _: () = assert!(std::mem::size_of::<BindlessMaterialGpu>() % 16 == 0);

// ---------------------------------------------------------------------------
// BindlessMaterialSystem
// ---------------------------------------------------------------------------

/// Manages the GPU storage buffer of all materials and the texture binding array.
///
/// Generic over [`euca_rhi::RenderDevice`] — defaults to [`euca_rhi::wgpu_backend::WgpuDevice`]
/// for backward compatibility. Existing call-sites that use concrete wgpu types
/// continue to compile without changes.
pub struct BindlessMaterialSystem<D: euca_rhi::RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    /// Storage buffer holding `array<BindlessMaterialGpu>`.
    material_buffer: SmartBuffer<D>,
    /// Bind group layout for the bindless material group (group 2).
    pub bind_group_layout: D::BindGroupLayout,
    /// Bind group exposing the material buffer + sampler + texture array.
    pub bind_group: D::BindGroup,
    /// CPU-side copy of material data for incremental updates.
    materials: Vec<BindlessMaterialGpu>,
    /// Texture handles registered in the binding array (resolved to views at bind time).
    texture_handles: Vec<TextureHandle>,
    /// Map from TextureHandle to binding array index.
    texture_index: std::collections::HashMap<TextureHandle, u32>,
    /// Current material buffer capacity in bytes.
    material_buffer_capacity: u64,
    /// Shared sampler for all material textures.
    sampler: D::Sampler,
    /// Fallback 1x1 white texture view (used for empty binding array slots).
    fallback_view: D::TextureView,
    /// The fallback texture (kept alive so view remains valid).
    _fallback_texture: D::Texture,
    /// Whether the system needs to rebuild the bind group (new textures added).
    dirty: bool,
    /// Whether GPU supports bindless features.
    enabled: bool,
    /// Whether the GPU uses unified memory.
    unified_memory: bool,
}

impl BindlessMaterialSystem {
    /// Create a new bindless material system. Returns `None` if the GPU doesn't
    /// support the required features.
    pub fn new(device: &wgpu::Device, features: wgpu::Features, unified_memory: bool) -> Self {
        let enabled = features.contains(BINDLESS_FEATURES);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Bindless Material Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create fallback 1x1 white texture.
        let fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bindless Fallback"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let fallback_view = fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Material storage buffer (start with space for 64 materials).
        let material_buffer = SmartBuffer::from_wgpu(
            device,
            (64 * std::mem::size_of::<BindlessMaterialGpu>()) as u64,
            BufferKind::Storage,
            unified_memory,
            "Bindless Material SSBO",
        );

        let initial_cap = (64 * std::mem::size_of::<BindlessMaterialGpu>()) as u64;
        let bind_group_layout = Self::create_layout(device, enabled);
        // Initial bind group: pad texture views to MAX_BINDLESS_TEXTURES with fallback.
        let initial_views: Vec<&wgpu::TextureView> = if enabled {
            (0..MAX_BINDLESS_TEXTURES as usize)
                .map(|_| &fallback_view)
                .collect()
        } else {
            vec![&fallback_view]
        };
        let bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &material_buffer,
            &sampler,
            &initial_views,
            enabled,
        );

        Self {
            material_buffer,
            bind_group_layout,
            bind_group,
            materials: Vec::new(),
            texture_handles: Vec::new(),
            texture_index: std::collections::HashMap::new(),
            material_buffer_capacity: initial_cap,
            sampler,
            fallback_view,
            _fallback_texture: fallback_texture,
            dirty: false,
            enabled,
            unified_memory,
        }
    }

    /// Whether bindless rendering is active on this GPU.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Register a material and return its index. If the material references
    /// textures not yet in the binding array, they are added.
    pub fn add_material(&mut self, mat: &crate::material::Material) -> MaterialHandle {
        let handle = MaterialHandle(self.materials.len() as u32);

        let albedo_tex_idx = self.resolve_texture(mat.albedo_texture);
        let normal_tex_idx = self.resolve_texture(mat.normal_texture);
        let mr_tex_idx = self.resolve_texture(mat.metallic_roughness_texture);
        let ao_tex_idx = self.resolve_texture(mat.ao_texture);
        let emissive_tex_idx = self.resolve_texture(mat.emissive_texture);

        self.materials.push(BindlessMaterialGpu {
            albedo: mat.albedo,
            metallic: mat.metallic,
            roughness: mat.roughness,
            has_normal_map: if mat.normal_texture.is_some() {
                1.0
            } else {
                0.0
            },
            has_metallic_roughness_tex: if mat.metallic_roughness_texture.is_some() {
                1.0
            } else {
                0.0
            },
            emissive: mat.emissive,
            has_emissive_tex: if mat.emissive_texture.is_some() {
                1.0
            } else {
                0.0
            },
            has_ao_tex: if mat.ao_texture.is_some() { 1.0 } else { 0.0 },
            alpha_mode: match mat.alpha_mode {
                crate::material::AlphaMode::Opaque => 0.0,
                crate::material::AlphaMode::Mask { .. } => 1.0,
                crate::material::AlphaMode::Blend => 2.0,
            },
            alpha_cutoff: match mat.alpha_mode {
                crate::material::AlphaMode::Mask { cutoff } => cutoff,
                _ => 0.5,
            },
            _pad0: 0.0,
            albedo_tex_idx,
            normal_tex_idx,
            metallic_roughness_tex_idx: mr_tex_idx,
            ao_tex_idx,
            emissive_tex_idx,
            _pad1: [0; 3],
        });

        handle
    }

    /// Upload all material data to the GPU and rebuild the bind group if needed.
    /// Pass the `TextureStore` so texture handles can be resolved to views.
    pub fn flush(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, textures: &TextureStore) {
        if self.materials.is_empty() {
            return;
        }

        // Grow material buffer if needed.
        let needed = (self.materials.len() * std::mem::size_of::<BindlessMaterialGpu>()) as u64;
        if needed > self.material_buffer_capacity {
            let new_size = needed.next_power_of_two();
            self.material_buffer = SmartBuffer::from_wgpu(
                device,
                new_size,
                BufferKind::Storage,
                self.unified_memory,
                "Bindless Material SSBO",
            );
            self.material_buffer_capacity = new_size;
            self.dirty = true;
        }

        self.material_buffer.write_wgpu(queue, &self.materials);

        if self.dirty {
            self.rebuild_bind_group(device, textures);
            self.dirty = false;
        }
    }

    /// Number of registered materials.
    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    // ── Private helpers ──

    fn resolve_texture(&mut self, tex: Option<TextureHandle>) -> u32 {
        match tex {
            None => 0xFFFF_FFFF, // sentinel: no texture
            Some(handle) => {
                if let Some(&idx) = self.texture_index.get(&handle) {
                    idx
                } else {
                    let idx = self.texture_handles.len() as u32;
                    self.texture_handles.push(handle);
                    self.texture_index.insert(handle, idx);
                    self.dirty = true;
                    idx
                }
            }
        }
    }

    fn create_layout(device: &wgpu::Device, bindless: bool) -> wgpu::BindGroupLayout {
        if bindless {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bindless Material BGL"),
                entries: &[
                    // Binding 0: material storage buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: shared sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Binding 2: texture binding array
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: std::num::NonZeroU32::new(MAX_BINDLESS_TEXTURES),
                    },
                ],
            })
        } else {
            // Fallback: same layout as the traditional per-material bind group.
            // This allows the renderer to use the same pipeline layout regardless.
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bindless Fallback BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            })
        }
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        material_buffer: &SmartBuffer,
        sampler: &wgpu::Sampler,
        texture_views: &[&wgpu::TextureView],
        bindless: bool,
    ) -> wgpu::BindGroup {
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: material_buffer.raw().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ];
        if bindless {
            entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureViewArray(texture_views),
            });
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bindless Material BG"),
            layout,
            entries: &entries,
        })
    }

    fn rebuild_bind_group(&mut self, device: &wgpu::Device, textures: &TextureStore) {
        // Build the texture view array, padded to MAX_BINDLESS_TEXTURES with fallback.
        let mut views: Vec<&wgpu::TextureView> = self
            .texture_handles
            .iter()
            .map(|h| textures.view(*h))
            .collect();
        if self.enabled {
            while views.len() < MAX_BINDLESS_TEXTURES as usize {
                views.push(&self.fallback_view);
            }
        }
        self.bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            &self.material_buffer,
            &self.sampler,
            &views,
            self.enabled,
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bindless_material_gpu_size() {
        assert_eq!(std::mem::size_of::<BindlessMaterialGpu>(), 96);
        assert_eq!(std::mem::size_of::<BindlessMaterialGpu>() % 16, 0);
    }

    #[test]
    fn max_texture_count_is_power_of_two() {
        assert!(MAX_BINDLESS_TEXTURES.is_power_of_two());
    }
}
