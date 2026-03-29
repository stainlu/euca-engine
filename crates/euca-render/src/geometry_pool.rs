//! Global geometry pool: a single vertex + index buffer for all meshes.
//!
//! Instead of each mesh owning its own vertex/index buffers, the
//! [`GeometryPool`] provides a shared "mega-buffer" pair. Meshes are appended
//! via [`allocate`](GeometryPool::allocate), which returns a
//! [`MeshAllocation`] describing the sub-range. The GPU-driven pipeline then
//! binds these two buffers once and uses `vertex_offset` / `first_index` in
//! the indirect draw arguments to select the correct mesh data.

use crate::vertex::Vertex;
use euca_rhi::RenderDevice;

// ---------------------------------------------------------------------------
// MeshAllocation
// ---------------------------------------------------------------------------

/// Describes where a mesh lives inside the global geometry pool.
#[derive(Clone, Copy, Debug)]
pub struct MeshAllocation {
    /// Offset (in vertices) added to each index before vertex fetch.
    pub vertex_offset: i32,
    /// First index in the global index buffer for this mesh.
    pub first_index: u32,
    /// Number of indices for this mesh.
    pub index_count: u32,
    /// Number of vertices for this mesh.
    pub vertex_count: u32,
}

// ---------------------------------------------------------------------------
// GeometryPool
// ---------------------------------------------------------------------------

/// A global vertex + index buffer pair that all meshes are appended into.
///
/// The pool grows by re-creating the underlying buffers when capacity is
/// exceeded. Previously uploaded mesh data is NOT re-uploaded on grow —
/// the caller is responsible for re-uploading if a grow occurs. In practice,
/// choosing a large enough initial capacity avoids this.
pub struct GeometryPool<D: RenderDevice = euca_rhi::wgpu_backend::WgpuDevice> {
    vertex_buf: D::Buffer,
    index_buf: D::Buffer,
    /// Current number of vertices written.
    vertex_count: u32,
    /// Current number of indices written.
    index_count: u32,
    /// Capacity in vertices.
    vertex_capacity: u32,
    /// Capacity in indices.
    index_capacity: u32,
}

impl<D: RenderDevice> GeometryPool<D> {
    /// Create a new geometry pool with the given initial capacities.
    ///
    /// - `max_vertices`: initial vertex buffer capacity (in vertices).
    /// - `max_indices`: initial index buffer capacity (in indices).
    pub fn new(device: &D, max_vertices: u32, max_indices: u32) -> Self {
        let vertex_buf_size = (max_vertices as u64) * std::mem::size_of::<Vertex>() as u64;
        let index_buf_size = (max_indices as u64) * std::mem::size_of::<u32>() as u64;

        let vertex_buf = device.create_buffer(&euca_rhi::BufferDesc {
            label: Some("geometry_pool_vertices"),
            size: vertex_buf_size,
            usage: euca_rhi::BufferUsages::VERTEX | euca_rhi::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buf = device.create_buffer(&euca_rhi::BufferDesc {
            label: Some("geometry_pool_indices"),
            size: index_buf_size,
            usage: euca_rhi::BufferUsages::INDEX | euca_rhi::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            vertex_buf,
            index_buf,
            vertex_count: 0,
            index_count: 0,
            vertex_capacity: max_vertices,
            index_capacity: max_indices,
        }
    }

    /// Append mesh data to the pool, returning the allocation descriptor.
    ///
    /// # Panics
    ///
    /// Panics if the pool capacity is exceeded. Choose a large enough initial
    /// capacity or implement grow logic.
    pub fn allocate(&mut self, device: &D, vertices: &[Vertex], indices: &[u32]) -> MeshAllocation {
        let v_count = vertices.len() as u32;
        let i_count = indices.len() as u32;

        assert!(
            self.vertex_count + v_count <= self.vertex_capacity,
            "GeometryPool vertex overflow: need {} + {}, capacity {}",
            self.vertex_count,
            v_count,
            self.vertex_capacity,
        );
        assert!(
            self.index_count + i_count <= self.index_capacity,
            "GeometryPool index overflow: need {} + {}, capacity {}",
            self.index_count,
            i_count,
            self.index_capacity,
        );

        let vertex_offset = self.vertex_count as i32;
        let first_index = self.index_count;

        // Write vertex data at the current offset.
        let v_byte_offset = (self.vertex_count as u64) * std::mem::size_of::<Vertex>() as u64;
        device.write_buffer(
            &self.vertex_buf,
            v_byte_offset,
            bytemuck::cast_slice(vertices),
        );

        // Write index data at the current offset.
        let i_byte_offset = (self.index_count as u64) * std::mem::size_of::<u32>() as u64;
        device.write_buffer(
            &self.index_buf,
            i_byte_offset,
            bytemuck::cast_slice(indices),
        );

        self.vertex_count += v_count;
        self.index_count += i_count;

        MeshAllocation {
            vertex_offset,
            first_index,
            index_count: i_count,
            vertex_count: v_count,
        }
    }

    /// The global vertex buffer.
    pub fn vertex_buffer(&self) -> &D::Buffer {
        &self.vertex_buf
    }

    /// The global index buffer.
    pub fn index_buffer(&self) -> &D::Buffer {
        &self.index_buf
    }

    /// Size in bytes of the used portion of the vertex buffer.
    pub fn vertex_buffer_size(&self) -> u64 {
        (self.vertex_count as u64) * std::mem::size_of::<Vertex>() as u64
    }

    /// Size in bytes of the used portion of the index buffer.
    pub fn index_buffer_size(&self) -> u64 {
        (self.index_count as u64) * std::mem::size_of::<u32>() as u64
    }

    /// Total size in bytes of the vertex buffer (capacity).
    pub fn vertex_buffer_capacity_bytes(&self) -> u64 {
        (self.vertex_capacity as u64) * std::mem::size_of::<Vertex>() as u64
    }

    /// Total size in bytes of the index buffer (capacity).
    pub fn index_buffer_capacity_bytes(&self) -> u64 {
        (self.index_capacity as u64) * std::mem::size_of::<u32>() as u64
    }

    /// Number of vertices currently stored.
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// Number of indices currently stored.
    pub fn index_count(&self) -> u32 {
        self.index_count
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_allocation_fields() {
        let alloc = MeshAllocation {
            vertex_offset: 100,
            first_index: 200,
            index_count: 36,
            vertex_count: 24,
        };
        assert_eq!(alloc.vertex_offset, 100);
        assert_eq!(alloc.first_index, 200);
        assert_eq!(alloc.index_count, 36);
        assert_eq!(alloc.vertex_count, 24);
    }

    #[test]
    fn vertex_size_is_expected() {
        // Vertex: 3 + 3 + 3 + 2 = 11 floats = 44 bytes
        assert_eq!(std::mem::size_of::<Vertex>(), 44);
    }
}
