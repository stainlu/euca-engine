//! Traits for GPU render-pass and compute-pass operations.
//!
//! These traits abstract the per-pass commands (set pipeline, bind resources,
//! draw/dispatch). Each backend's associated `RenderPass` and `ComputePass`
//! types implement these traits.

use crate::RenderDevice;
use crate::types::IndexFormat;
use std::ops::Range;

/// Operations available inside a render pass.
pub trait RenderPassOps<D: RenderDevice + ?Sized> {
    fn set_pipeline(&mut self, pipeline: &D::RenderPipeline);
    fn set_bind_group(&mut self, index: u32, bind_group: &D::BindGroup, offsets: &[u32]);
    fn set_vertex_buffer(&mut self, slot: u32, buffer: &D::Buffer, offset: u64, size: u64);
    fn set_index_buffer(&mut self, buffer: &D::Buffer, format: IndexFormat, offset: u64, size: u64);
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>);
    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>);
    fn draw_indexed_indirect(&mut self, indirect_buffer: &D::Buffer, indirect_offset: u64);
    fn multi_draw_indexed_indirect(
        &mut self,
        indirect_buffer: &D::Buffer,
        indirect_offset: u64,
        count: u32,
    );
    fn multi_draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: &D::Buffer,
        indirect_offset: u64,
        count_buffer: &D::Buffer,
        count_offset: u64,
        max_count: u32,
    );
    fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32);
    fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32);
}

/// Operations available inside a compute pass.
pub trait ComputePassOps<D: RenderDevice + ?Sized> {
    fn set_pipeline(&mut self, pipeline: &D::ComputePipeline);
    fn set_bind_group(&mut self, index: u32, bind_group: &D::BindGroup, offsets: &[u32]);
    fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32);
    fn dispatch_workgroups_indirect(&mut self, indirect_buffer: &D::Buffer, indirect_offset: u64);
}
