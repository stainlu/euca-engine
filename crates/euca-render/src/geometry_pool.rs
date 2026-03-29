//! Global geometry pool for GPU-driven rendering.
//!
//! All scene meshes allocate into a single shared vertex buffer and a single
//! shared index buffer.  This enables GPU-driven rendering: the compute
//! culling shader writes [`DrawIndexedIndirect`](super::DrawIndexedIndirectArgs)
//! args referencing offsets into these shared buffers, and a single
//! `multi_draw_indexed_indirect_count` call renders everything.
//!
//! # Growth strategy
//!
//! Buffers start at the caller-specified initial capacity.  When a new
//! allocation would exceed the current capacity the buffer is doubled (with
//! a minimum of 1 024 vertices / 4 096 indices).  Because the RHI does not
//! expose `copy_buffer_to_buffer`, growth re-uploads data from a CPU-side
//! mirror -- acceptable because growth is exponentially rare.

use euca_rhi::{BufferDesc, BufferUsages, RenderDevice};

use crate::vertex::Vertex;

// ---------------------------------------------------------------------------
// MeshAllocation
// ---------------------------------------------------------------------------

/// Tracks a mesh's location within the global geometry pool.
///
/// These fields map directly to [`DrawIndexedIndirect`] arguments, so the
/// compute culling shader can copy them into the indirect buffer as-is.
#[derive(Clone, Copy, Debug)]
pub struct MeshAllocation {
    /// Signed offset added to each index value before indexing into the
    /// vertex buffer (`base_vertex` in `DrawIndexedIndirect`).
    pub vertex_offset: i32,
    /// First index (in indices, not bytes) in the global index buffer.
    pub first_index: u32,
    /// Number of indices for this mesh.
    pub index_count: u32,
    /// Number of vertices for this mesh.
    pub vertex_count: u32,
}

// ---------------------------------------------------------------------------
// GeometryPool
// ---------------------------------------------------------------------------

/// A global vertex + index buffer pair that all meshes allocate into.
///
/// This enables GPU-driven rendering: the compute culling shader writes
/// `DrawIndexedIndirect` args referencing offsets into these shared buffers,
/// and a single `multi_draw_indexed_indirect_count` call renders everything.
pub struct GeometryPool<D: RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    vertex_buffer: D::Buffer,
    index_buffer: D::Buffer,

    /// Next free vertex slot (in vertices, not bytes).
    vertex_cursor: u32,
    /// Next free index slot (in indices, not bytes).
    index_cursor: u32,

    /// Current capacity in vertices.
    vertex_capacity: u32,
    /// Current capacity in indices.
    index_capacity: u32,

    /// All allocations, indexed by mesh handle.
    allocations: Vec<Option<MeshAllocation>>,

    // CPU-side mirrors for re-upload on buffer growth.
    vertex_mirror: Vec<Vertex>,
    index_mirror: Vec<u32>,
}

/// Minimum vertex capacity after growth.
const MIN_VERTEX_CAPACITY: u32 = 1024;
/// Minimum index capacity after growth.
const MIN_INDEX_CAPACITY: u32 = 4096;

/// Size of a single [`Vertex`] in bytes.
const VERTEX_SIZE: u64 = std::mem::size_of::<Vertex>() as u64;
/// Size of a single `u32` index in bytes.
const INDEX_SIZE: u64 = std::mem::size_of::<u32>() as u64;

/// Buffer usage flags for the global vertex buffer:
/// VERTEX (draw), COPY_DST (upload), COPY_SRC (future copy), STORAGE (compute read).
fn vertex_buffer_usage() -> BufferUsages {
    BufferUsages::VERTEX | BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE
}

/// Buffer usage flags for the global index buffer:
/// INDEX (draw), COPY_DST (upload), COPY_SRC (future copy), STORAGE (compute read).
fn index_buffer_usage() -> BufferUsages {
    BufferUsages::INDEX | BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE
}

impl<D: RenderDevice> GeometryPool<D> {
    /// Create a new geometry pool with the given initial capacities.
    pub fn new(device: &D, initial_vertex_capacity: u32, initial_index_capacity: u32) -> Self {
        let vertex_capacity = initial_vertex_capacity.max(MIN_VERTEX_CAPACITY);
        let index_capacity = initial_index_capacity.max(MIN_INDEX_CAPACITY);

        let vertex_buffer = device.create_buffer(&BufferDesc {
            label: Some("geometry_pool_vertex"),
            size: vertex_capacity as u64 * VERTEX_SIZE,
            usage: vertex_buffer_usage(),
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&BufferDesc {
            label: Some("geometry_pool_index"),
            size: index_capacity as u64 * INDEX_SIZE,
            usage: index_buffer_usage(),
            mapped_at_creation: false,
        });

        Self {
            vertex_buffer,
            index_buffer,
            vertex_cursor: 0,
            index_cursor: 0,
            vertex_capacity,
            index_capacity,
            allocations: Vec::new(),
            vertex_mirror: Vec::with_capacity(vertex_capacity as usize),
            index_mirror: Vec::with_capacity(index_capacity as usize),
        }
    }

    /// Allocate space in the pool for a mesh's vertices and indices, uploading
    /// the data to the GPU.
    ///
    /// Returns a [`MeshAllocation`] describing where the mesh lives in the
    /// global buffers.  The allocation is also stored internally and can be
    /// retrieved later via [`get`](Self::get).
    pub fn allocate(&mut self, device: &D, vertices: &[Vertex], indices: &[u32]) -> MeshAllocation {
        let v_count = vertices.len() as u32;
        let i_count = indices.len() as u32;

        // Grow buffers if needed.
        let needed_v = self.vertex_cursor + v_count;
        if needed_v > self.vertex_capacity {
            let new_cap = (self.vertex_capacity * 2)
                .max(needed_v)
                .max(MIN_VERTEX_CAPACITY);
            self.grow_vertex_buffer(device, new_cap);
        }
        let needed_i = self.index_cursor + i_count;
        if needed_i > self.index_capacity {
            let new_cap = (self.index_capacity * 2)
                .max(needed_i)
                .max(MIN_INDEX_CAPACITY);
            self.grow_index_buffer(device, new_cap);
        }

        // Record allocation *before* writing so offsets are known.
        let alloc = MeshAllocation {
            vertex_offset: self.vertex_cursor as i32,
            first_index: self.index_cursor,
            index_count: i_count,
            vertex_count: v_count,
        };

        // Upload vertices.
        let v_byte_offset = self.vertex_cursor as u64 * VERTEX_SIZE;
        device.write_buffer(
            &self.vertex_buffer,
            v_byte_offset,
            bytemuck::cast_slice(vertices),
        );

        // Upload indices.
        let i_byte_offset = self.index_cursor as u64 * INDEX_SIZE;
        device.write_buffer(
            &self.index_buffer,
            i_byte_offset,
            bytemuck::cast_slice(indices),
        );

        // Update CPU mirrors.
        self.vertex_mirror.extend_from_slice(vertices);
        self.index_mirror.extend_from_slice(indices);

        // Advance cursors.
        self.vertex_cursor += v_count;
        self.index_cursor += i_count;

        // Store allocation.
        let handle_index = self.allocations.len();
        self.allocations.push(Some(alloc));
        log::debug!(
            "GeometryPool: allocated mesh {} ({v_count} verts, {i_count} indices) at v_off={}, i_off={}",
            handle_index,
            alloc.vertex_offset,
            alloc.first_index,
        );

        alloc
    }

    /// Grow the vertex buffer to `new_capacity` vertices, re-uploading all
    /// existing vertex data.
    fn grow_vertex_buffer(&mut self, device: &D, new_capacity: u32) {
        debug_assert!(new_capacity > self.vertex_capacity);
        log::info!(
            "GeometryPool: growing vertex buffer {} -> {} vertices ({} -> {} bytes)",
            self.vertex_capacity,
            new_capacity,
            self.vertex_capacity as u64 * VERTEX_SIZE,
            new_capacity as u64 * VERTEX_SIZE,
        );

        self.vertex_buffer = device.create_buffer(&BufferDesc {
            label: Some("geometry_pool_vertex"),
            size: new_capacity as u64 * VERTEX_SIZE,
            usage: vertex_buffer_usage(),
            mapped_at_creation: false,
        });

        // Re-upload existing data from the CPU mirror.
        if !self.vertex_mirror.is_empty() {
            device.write_buffer(
                &self.vertex_buffer,
                0,
                bytemuck::cast_slice(&self.vertex_mirror),
            );
        }

        self.vertex_capacity = new_capacity;
    }

    /// Grow the index buffer to `new_capacity` indices, re-uploading all
    /// existing index data.
    fn grow_index_buffer(&mut self, device: &D, new_capacity: u32) {
        debug_assert!(new_capacity > self.index_capacity);
        log::info!(
            "GeometryPool: growing index buffer {} -> {} indices ({} -> {} bytes)",
            self.index_capacity,
            new_capacity,
            self.index_capacity as u64 * INDEX_SIZE,
            new_capacity as u64 * INDEX_SIZE,
        );

        self.index_buffer = device.create_buffer(&BufferDesc {
            label: Some("geometry_pool_index"),
            size: new_capacity as u64 * INDEX_SIZE,
            usage: index_buffer_usage(),
            mapped_at_creation: false,
        });

        // Re-upload existing data from the CPU mirror.
        if !self.index_mirror.is_empty() {
            device.write_buffer(
                &self.index_buffer,
                0,
                bytemuck::cast_slice(&self.index_mirror),
            );
        }

        self.index_capacity = new_capacity;
    }

    /// Read-only access to the global vertex buffer.
    pub fn vertex_buffer(&self) -> &D::Buffer {
        &self.vertex_buffer
    }

    /// Read-only access to the global index buffer.
    pub fn index_buffer(&self) -> &D::Buffer {
        &self.index_buffer
    }

    /// Size in bytes of the *used* portion of the vertex buffer.
    pub fn vertex_buffer_size(&self) -> u64 {
        self.vertex_cursor as u64 * VERTEX_SIZE
    }

    /// Size in bytes of the *used* portion of the index buffer.
    pub fn index_buffer_size(&self) -> u64 {
        self.index_cursor as u64 * INDEX_SIZE
    }

    /// Total capacity in bytes of the vertex buffer (for binding the full range).
    pub fn vertex_buffer_capacity_bytes(&self) -> u64 {
        self.vertex_capacity as u64 * VERTEX_SIZE
    }

    /// Total capacity in bytes of the index buffer (for binding the full range).
    pub fn index_buffer_capacity_bytes(&self) -> u64 {
        self.index_capacity as u64 * INDEX_SIZE
    }

    /// Look up a mesh allocation by handle index.
    pub fn get(&self, handle_index: u32) -> Option<&MeshAllocation> {
        self.allocations
            .get(handle_index as usize)
            .and_then(|opt| opt.as_ref())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_allocation_is_copy() {
        // MeshAllocation must be Copy so it can live inside DrawCommandGpu
        // and be freely duplicated without overhead.
        fn assert_copy<T: Copy>() {}
        assert_copy::<MeshAllocation>();
    }

    #[test]
    fn mesh_allocation_layout() {
        // 4 fields * 4 bytes = 16 bytes (i32, u32, u32, u32).
        assert_eq!(std::mem::size_of::<MeshAllocation>(), 16);
    }

    #[test]
    fn allocation_tracking_offsets() {
        // Simulate sequential allocations and verify the offsets are correct.
        // We can't create a real RenderDevice in unit tests, so we test the
        // MeshAllocation math directly.

        // First mesh: 24 vertices, 36 indices (a cube).
        let alloc0 = MeshAllocation {
            vertex_offset: 0,
            first_index: 0,
            index_count: 36,
            vertex_count: 24,
        };
        assert_eq!(alloc0.vertex_offset, 0);
        assert_eq!(alloc0.first_index, 0);

        // Second mesh: starts right after the first.
        let alloc1 = MeshAllocation {
            vertex_offset: alloc0.vertex_count as i32,
            first_index: alloc0.first_index + alloc0.index_count,
            index_count: 6,
            vertex_count: 4,
        };
        assert_eq!(alloc1.vertex_offset, 24);
        assert_eq!(alloc1.first_index, 36);
        assert_eq!(alloc1.index_count, 6);
        assert_eq!(alloc1.vertex_count, 4);

        // Third mesh: starts after second.
        let alloc2 = MeshAllocation {
            vertex_offset: alloc1.vertex_offset + alloc1.vertex_count as i32,
            first_index: alloc1.first_index + alloc1.index_count,
            index_count: 12,
            vertex_count: 8,
        };
        assert_eq!(alloc2.vertex_offset, 28);
        assert_eq!(alloc2.first_index, 42);
    }

    #[test]
    fn vertex_and_index_sizes() {
        // Verify our size constants match the actual types.
        assert_eq!(VERTEX_SIZE, 44);
        assert_eq!(INDEX_SIZE, 4);
    }

    #[test]
    fn growth_minimums() {
        assert_eq!(MIN_VERTEX_CAPACITY, 1024);
        assert_eq!(MIN_INDEX_CAPACITY, 4096);
    }

    #[test]
    fn buffer_usage_flags() {
        let vbu = vertex_buffer_usage();
        // Verify vertex buffer has all required flags.
        assert_ne!(vbu & BufferUsages::VERTEX, BufferUsages::NONE);
        assert_ne!(vbu & BufferUsages::COPY_DST, BufferUsages::NONE);
        assert_ne!(vbu & BufferUsages::COPY_SRC, BufferUsages::NONE);
        assert_ne!(vbu & BufferUsages::STORAGE, BufferUsages::NONE);

        let ibu = index_buffer_usage();
        // Verify index buffer has all required flags.
        assert_ne!(ibu & BufferUsages::INDEX, BufferUsages::NONE);
        assert_ne!(ibu & BufferUsages::COPY_DST, BufferUsages::NONE);
        assert_ne!(ibu & BufferUsages::COPY_SRC, BufferUsages::NONE);
        assert_ne!(ibu & BufferUsages::STORAGE, BufferUsages::NONE);
    }
}
