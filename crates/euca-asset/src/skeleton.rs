//! Skeleton data extracted from glTF skins.

use euca_math::{Mat4, Quat, Transform, Vec3};

/// A single joint in a skeleton hierarchy.
#[derive(Clone, Debug)]
pub struct Joint {
    pub name: String,
    /// Index of the parent joint, or None for the root.
    pub parent: Option<usize>,
    /// Local-space rest pose transform.
    pub local_transform: Transform,
}

/// A complete skeleton: joints + inverse bind matrices.
#[derive(Clone, Debug)]
pub struct Skeleton {
    pub joints: Vec<Joint>,
    /// One 4x4 matrix per joint — transforms from mesh space to bone space.
    pub inverse_bind_matrices: Vec<Mat4>,
    /// Node indices from the glTF document (for animation channel mapping).
    pub joint_node_indices: Vec<usize>,
}

impl Skeleton {
    /// Compute the final joint matrices for a given pose.
    ///
    /// `local_poses` is one Transform per joint (same order as `joints`).
    /// Returns one Mat4 per joint: `global_transform * inverse_bind_matrix`.
    pub fn compute_joint_matrices(&self, local_poses: &[Transform]) -> Vec<Mat4> {
        let n = self.joints.len();
        let mut global_mats = vec![Mat4::IDENTITY; n];

        for i in 0..n {
            let local_mat = local_poses[i].to_matrix();
            global_mats[i] = match self.joints[i].parent {
                Some(parent) => global_mats[parent] * local_mat,
                None => local_mat,
            };
        }

        // Multiply by inverse bind matrix to get final skinning matrix
        global_mats
            .iter()
            .zip(self.inverse_bind_matrices.iter())
            .map(|(global, ibm)| *global * *ibm)
            .collect()
    }
}

/// Parse the first skin from a glTF document.
pub fn parse_skeleton(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> Option<Skeleton> {
    let skin = document.skins().next()?;

    let joint_nodes: Vec<gltf::Node> = skin.joints().collect();
    let joint_count = joint_nodes.len();

    // Build node index → joint index map
    let node_to_joint: std::collections::HashMap<usize, usize> = joint_nodes
        .iter()
        .enumerate()
        .map(|(ji, node)| (node.index(), ji))
        .collect();

    let joint_node_indices: Vec<usize> = joint_nodes.iter().map(|n| n.index()).collect();

    // Build parent map by traversing the scene hierarchy.
    // For each node, check if its parent is also a joint.
    let mut node_parent: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    fn visit_node(
        node: &gltf::Node,
        parent_idx: Option<usize>,
        node_parent: &mut std::collections::HashMap<usize, usize>,
    ) {
        if let Some(p) = parent_idx {
            node_parent.insert(node.index(), p);
        }
        for child in node.children() {
            visit_node(&child, Some(node.index()), node_parent);
        }
    }
    for scene in document.scenes() {
        for root in scene.nodes() {
            visit_node(&root, None, &mut node_parent);
        }
    }

    // Parse joints
    let joints: Vec<Joint> = joint_nodes
        .iter()
        .map(|node| {
            // Find parent: look up this node's parent in the hierarchy,
            // then check if that parent is also a joint in this skin.
            let parent = node_parent
                .get(&node.index())
                .and_then(|&parent_node_idx| node_to_joint.get(&parent_node_idx).copied());

            let (translation, rotation, scale) = node.transform().decomposed();
            let local_transform = Transform {
                translation: Vec3::new(translation[0], translation[1], translation[2]),
                rotation: Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]),
                scale: Vec3::new(scale[0], scale[1], scale[2]),
            };

            Joint {
                name: node.name().unwrap_or("joint").to_string(),
                parent,
                local_transform,
            }
        })
        .collect();

    // Parse inverse bind matrices
    let inverse_bind_matrices = if let Some(reader) = skin
        .reader(|buffer| Some(&buffers[buffer.index()]))
        .read_inverse_bind_matrices()
    {
        reader.map(|m| Mat4::from_cols_array_2d(&m)).collect()
    } else {
        vec![Mat4::IDENTITY; joint_count]
    };

    log::info!("Parsed skeleton: {} joints", joints.len(),);

    Some(Skeleton {
        joints,
        inverse_bind_matrices,
        joint_node_indices,
    })
}
