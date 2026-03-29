//! Simple frame graph for tracking render and compute pass dependencies.
//!
//! The frame graph enables splitting a frame's work into multiple command
//! encoders. Independent compute passes can be recorded to a separate
//! encoder and submitted alongside the render encoder, enabling GPU
//! async compute overlap.
//!
//! Phase 1 (this implementation): static pass ordering with explicit
//! compute/render split. No automatic barrier insertion.

use std::collections::{HashMap, HashSet};

/// Unique identifier for a GPU resource tracked by the frame graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceId(pub u32);

/// The kind of GPU pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassKind {
    Compute,
    Render,
}

/// A single pass in the frame graph.
#[derive(Clone, Debug)]
pub struct PassNode {
    /// Human-readable name for debugging.
    pub name: &'static str,
    /// Whether this is a compute or render pass.
    pub kind: PassKind,
    /// Resources read by this pass (must be written by an earlier pass).
    pub reads: Vec<ResourceId>,
    /// Resources written by this pass.
    pub writes: Vec<ResourceId>,
}

/// A simple frame graph that organizes passes into execution groups.
///
/// Passes are topologically sorted based on resource dependencies.
/// Independent compute passes are grouped into a separate "compute batch"
/// that can be recorded to a separate command encoder.
pub struct FrameGraph {
    passes: Vec<PassNode>,
}

/// Result of compiling the frame graph: passes split into encoder groups.
#[derive(Clone, Debug)]
pub struct CompiledGraph {
    /// Passes to record into the compute command encoder (run first).
    pub compute_passes: Vec<usize>,
    /// Passes to record into the render command encoder.
    pub render_passes: Vec<usize>,
}

impl FrameGraph {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Add a pass to the graph. Returns the pass index.
    pub fn add_pass(&mut self, pass: PassNode) -> usize {
        let idx = self.passes.len();
        self.passes.push(pass);
        idx
    }

    /// Get a pass by index.
    pub fn pass(&self, index: usize) -> &PassNode {
        &self.passes[index]
    }

    /// Compile the graph: topologically sort passes and split into
    /// compute and render encoder groups.
    ///
    /// The algorithm:
    /// 1. Build a dependency graph from resource reads/writes
    /// 2. Topological sort
    /// 3. Group: compute passes that don't depend on any render pass
    ///    go into compute_passes. Everything else into render_passes.
    pub fn compile(&self) -> CompiledGraph {
        let n = self.passes.len();
        if n == 0 {
            return CompiledGraph {
                compute_passes: Vec::new(),
                render_passes: Vec::new(),
            };
        }

        // Build dependency edges: if pass B reads a resource that pass A writes,
        // then B depends on A. Similarly for write-after-write.
        let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_degree: Vec<usize> = vec![0; n];

        // Map: resource -> last writer pass index
        let mut last_writer: HashMap<ResourceId, usize> = HashMap::new();

        for (i, pass) in self.passes.iter().enumerate() {
            for &res in &pass.reads {
                if let Some(&writer) = last_writer.get(&res) {
                    deps[writer].push(i);
                    in_degree[i] += 1;
                }
            }
            for &res in &pass.writes {
                if let Some(&prev_writer) = last_writer.get(&res) {
                    // Write-after-write dependency
                    deps[prev_writer].push(i);
                    in_degree[i] += 1;
                }
                last_writer.insert(res, i);
            }
        }

        // Topological sort (Kahn's algorithm)
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut sorted: Vec<usize> = Vec::with_capacity(n);

        while let Some(node) = queue.pop() {
            sorted.push(node);
            for &next in &deps[node] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push(next);
                }
            }
        }

        // If not all passes are in sorted (cycle), fall back to input order
        if sorted.len() < n {
            sorted = (0..n).collect();
        }

        // Split: compute passes that appear before any render pass AND
        // don't read from any render pass output -> compute encoder.
        // Everything else -> render encoder.
        let mut render_outputs: HashSet<ResourceId> = HashSet::new();
        let mut compute_passes = Vec::new();
        let mut render_passes = Vec::new();

        for &idx in &sorted {
            let pass = &self.passes[idx];
            let reads_render_output = pass.reads.iter().any(|r| render_outputs.contains(r));

            if pass.kind == PassKind::Compute && !reads_render_output {
                compute_passes.push(idx);
            } else {
                render_passes.push(idx);
                for &res in &pass.writes {
                    render_outputs.insert(res);
                }
            }
        }

        CompiledGraph {
            compute_passes,
            render_passes,
        }
    }

    /// Reset the graph for the next frame.
    pub fn clear(&mut self) {
        self.passes.clear();
    }
}

impl Default for FrameGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let graph = FrameGraph::new();
        let compiled = graph.compile();
        assert!(compiled.compute_passes.is_empty());
        assert!(compiled.render_passes.is_empty());
    }

    #[test]
    fn test_independent_compute_passes() {
        let mut graph = FrameGraph::new();
        graph.add_pass(PassNode {
            name: "compute_a",
            kind: PassKind::Compute,
            reads: vec![],
            writes: vec![ResourceId(0)],
        });
        graph.add_pass(PassNode {
            name: "compute_b",
            kind: PassKind::Compute,
            reads: vec![],
            writes: vec![ResourceId(1)],
        });
        let compiled = graph.compile();
        assert_eq!(compiled.compute_passes.len(), 2);
        assert!(compiled.render_passes.is_empty());
    }

    #[test]
    fn test_compute_then_render() {
        let mut graph = FrameGraph::new();
        let res = ResourceId(0);
        graph.add_pass(PassNode {
            name: "compute",
            kind: PassKind::Compute,
            reads: vec![],
            writes: vec![res],
        });
        graph.add_pass(PassNode {
            name: "render",
            kind: PassKind::Render,
            reads: vec![res],
            writes: vec![],
        });
        let compiled = graph.compile();
        assert_eq!(compiled.compute_passes, vec![0]);
        assert_eq!(compiled.render_passes, vec![1]);
    }

    #[test]
    fn test_render_dependency_forces_render_group() {
        let mut graph = FrameGraph::new();
        let render_output = ResourceId(0);
        graph.add_pass(PassNode {
            name: "render_pass",
            kind: PassKind::Render,
            reads: vec![],
            writes: vec![render_output],
        });
        // This compute pass reads render output, so it must go into the render group.
        graph.add_pass(PassNode {
            name: "compute_reads_render",
            kind: PassKind::Compute,
            reads: vec![render_output],
            writes: vec![ResourceId(1)],
        });
        let compiled = graph.compile();
        assert!(compiled.compute_passes.is_empty());
        assert_eq!(compiled.render_passes.len(), 2);
        assert!(compiled.render_passes.contains(&0));
        assert!(compiled.render_passes.contains(&1));
    }

    #[test]
    fn test_topological_ordering() {
        let mut graph = FrameGraph::new();
        let res_a = ResourceId(0);
        let res_b = ResourceId(1);
        // Chain: A writes res_a -> B reads res_a, writes res_b -> C reads res_b
        graph.add_pass(PassNode {
            name: "A",
            kind: PassKind::Compute,
            reads: vec![],
            writes: vec![res_a],
        });
        graph.add_pass(PassNode {
            name: "B",
            kind: PassKind::Compute,
            reads: vec![res_a],
            writes: vec![res_b],
        });
        graph.add_pass(PassNode {
            name: "C",
            kind: PassKind::Compute,
            reads: vec![res_b],
            writes: vec![],
        });
        let compiled = graph.compile();
        // All are compute, no render dependencies, so all in compute group.
        assert_eq!(compiled.compute_passes.len(), 3);
        // Verify topological order: A before B, B before C.
        let pos_a = compiled
            .compute_passes
            .iter()
            .position(|&x| x == 0)
            .unwrap();
        let pos_b = compiled
            .compute_passes
            .iter()
            .position(|&x| x == 1)
            .unwrap();
        let pos_c = compiled
            .compute_passes
            .iter()
            .position(|&x| x == 2)
            .unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_cycle_fallback() {
        let mut graph = FrameGraph::new();
        let res_a = ResourceId(0);
        let res_b = ResourceId(1);
        // Create a cycle: A writes res_a, reads res_b; B writes res_b, reads res_a
        graph.add_pass(PassNode {
            name: "A",
            kind: PassKind::Compute,
            reads: vec![res_b],
            writes: vec![res_a],
        });
        graph.add_pass(PassNode {
            name: "B",
            kind: PassKind::Compute,
            reads: vec![res_a],
            writes: vec![res_b],
        });
        let compiled = graph.compile();
        // Cycle detected -> falls back to input order, all still compute
        let all_passes: Vec<usize> = compiled
            .compute_passes
            .iter()
            .chain(compiled.render_passes.iter())
            .copied()
            .collect();
        assert_eq!(all_passes.len(), 2);
        assert!(all_passes.contains(&0));
        assert!(all_passes.contains(&1));
    }
}
