//! Layout solver: resolves UiNode anchors, sizes, and flex layout into
//! screen-space rectangles.
//!
//! The solver operates in two coordinate spaces:
//! - **Reference space**: Authored at 1920x1080 (configurable). Margins and Px
//!   values are in this space.
//! - **Screen space**: Actual viewport pixels. The solver scales from reference
//!   to screen at the end.

use euca_ecs::{Entity, Query, Without, World};
use euca_math::Mat4;
use euca_scene::{Children, GlobalTransform, Parent};

use crate::draw::{UiDrawCommand, UiDrawKind};
use crate::node::{AlignItems, Anchor, FlexDirection, JustifyContent, UiNode, Val};
use crate::widget::{UiButton, UiImage, UiPanel, UiProgressBar, UiText};

/// Configuration for the UI viewport.
#[derive(Clone, Debug)]
pub struct UiViewport {
    /// Reference resolution for authoring (pixels).
    pub reference_width: f32,
    pub reference_height: f32,
    /// Actual viewport size (pixels).
    pub viewport_width: f32,
    pub viewport_height: f32,
    /// View-projection matrix for world-space UI projection.
    pub view_projection: Option<Mat4>,
}

impl Default for UiViewport {
    fn default() -> Self {
        Self {
            reference_width: 1920.0,
            reference_height: 1080.0,
            viewport_width: 1920.0,
            viewport_height: 1080.0,
            view_projection: None,
        }
    }
}

impl UiViewport {
    /// Scale factor from reference to viewport.
    fn scale_x(&self) -> f32 {
        self.viewport_width / self.reference_width
    }

    fn scale_y(&self) -> f32 {
        self.viewport_height / self.reference_height
    }
}

/// Resolved screen-space rectangle for a UI node. Stored as a component.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ResolvedRect {
    /// Top-left corner in screen pixels.
    pub x: f32,
    pub y: f32,
    /// Size in screen pixels.
    pub width: f32,
    pub height: f32,
}

impl ResolvedRect {
    /// Returns true if the point (px, py) in screen pixels is inside this rect.
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.width && py >= self.y && py <= self.y + self.height
    }
}

/// Resolve a single Val against a parent dimension (in reference pixels).
fn resolve_val(val: Val, parent_dimension: f32, default: f32) -> f32 {
    match val {
        Val::Px(px) => px,
        Val::Percent(pct) => parent_dimension * pct / 100.0,
        Val::Auto => default,
    }
}

/// Resolve a UiNode's position and size relative to a parent rect (in reference space).
///
/// Returns the resolved rect in reference pixels (before viewport scaling).
fn resolve_node_rect(node: &UiNode, parent: &ReferenceRect) -> ReferenceRect {
    let pw = parent.width;
    let ph = parent.height;

    if node.anchor == Anchor::Stretch {
        // Stretch: fill parent minus margins.
        return ReferenceRect {
            x: parent.x + node.margin.left,
            y: parent.y + node.margin.top,
            width: (pw - node.margin.left - node.margin.right).max(0.0),
            height: (ph - node.margin.top - node.margin.bottom).max(0.0),
        };
    }

    // Resolve width/height.
    let w = resolve_val(node.size.width, pw, 0.0);
    let h = resolve_val(node.size.height, ph, 0.0);

    // Anchor point on parent (also used as pivot on the node itself).
    let (frac_x, frac_y) = node.anchor.parent_fraction();
    let anchor_x = parent.x + pw * frac_x;
    let anchor_y = parent.y + ph * frac_y;

    let self_offset_x = w * frac_x;
    let self_offset_y = h * frac_y;

    // Position: anchor point minus self pivot, then add margin.
    // Margin direction depends on anchor position:
    //   - left-anchored: margin.left pushes right
    //   - right-anchored: margin.right pushes left
    //   - center-anchored: margin.left pushes right
    let margin_x = match node.anchor {
        Anchor::TopRight | Anchor::CenterRight | Anchor::BottomRight => -node.margin.right,
        Anchor::TopCenter | Anchor::Center | Anchor::BottomCenter => node.margin.left,
        _ => node.margin.left,
    };
    let margin_y = match node.anchor {
        Anchor::BottomLeft | Anchor::BottomCenter | Anchor::BottomRight => -node.margin.bottom,
        Anchor::CenterLeft | Anchor::Center | Anchor::CenterRight => node.margin.top,
        _ => node.margin.top,
    };

    ReferenceRect {
        x: anchor_x - self_offset_x + margin_x,
        y: anchor_y - self_offset_y + margin_y,
        width: w,
        height: h,
    }
}

/// Internal rect in reference coordinates (before viewport scaling).
#[derive(Clone, Copy, Debug, Default)]
struct ReferenceRect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

/// Apply flex layout to a set of children within a parent rect.
///
/// Overrides children's positions (but not sizes) based on the parent's
/// flex_direction, justify_content, and align_items.
fn apply_flex_layout(
    parent_node: &UiNode,
    parent_rect: &ReferenceRect,
    child_rects: &mut [(Entity, ReferenceRect)],
) {
    if child_rects.is_empty() {
        return;
    }

    let is_row = parent_node.flex_direction == FlexDirection::Row;

    // Compute total main-axis size of children.
    let total_main: f32 = child_rects
        .iter()
        .map(|(_, r)| if is_row { r.width } else { r.height })
        .sum();

    let parent_main = if is_row {
        parent_rect.width
    } else {
        parent_rect.height
    };
    let parent_cross = if is_row {
        parent_rect.height
    } else {
        parent_rect.width
    };

    let free_space = (parent_main - total_main).max(0.0);
    let n = child_rects.len();

    // Compute starting offset and gap based on justify_content.
    let (mut cursor, gap) = match parent_node.justify_content {
        JustifyContent::Start => (0.0, 0.0),
        JustifyContent::End => (free_space, 0.0),
        JustifyContent::Center => (free_space / 2.0, 0.0),
        JustifyContent::SpaceBetween => {
            if n > 1 {
                (0.0, free_space / (n - 1) as f32)
            } else {
                (0.0, 0.0)
            }
        }
        JustifyContent::SpaceAround => {
            let g = free_space / n as f32;
            (g / 2.0, g)
        }
    };

    let stretch = parent_node.align_items == AlignItems::Stretch;

    for (_, rect) in child_rects.iter_mut() {
        let child_main = if is_row { rect.width } else { rect.height };
        let child_cross = if is_row { rect.height } else { rect.width };

        // Cross-axis alignment and sizing.
        let (cross_pos, final_cross_size) = if stretch {
            (0.0, parent_cross)
        } else {
            let pos = match parent_node.align_items {
                AlignItems::Start => 0.0,
                AlignItems::End => parent_cross - child_cross,
                AlignItems::Center => (parent_cross - child_cross) / 2.0,
                AlignItems::Stretch => unreachable!(),
            };
            (pos, child_cross)
        };

        if is_row {
            rect.x = parent_rect.x + cursor;
            rect.y = parent_rect.y + cross_pos;
            rect.height = final_cross_size;
        } else {
            rect.y = parent_rect.y + cursor;
            rect.x = parent_rect.x + cross_pos;
            rect.width = final_cross_size;
        }

        cursor += child_main + gap;
    }
}

/// Project a 3D world position to screen coordinates using the view-projection matrix.
///
/// Returns `Some((screen_x, screen_y))` if the point is in front of the camera,
/// `None` if behind.
fn project_to_screen(
    world_pos: euca_math::Vec3,
    view_proj: &Mat4,
    viewport_width: f32,
    viewport_height: f32,
) -> Option<(f32, f32)> {
    let clip = *view_proj * euca_math::Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);

    // Behind camera check.
    if clip.w <= 0.0 {
        return None;
    }

    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;

    // NDC [-1, 1] to screen pixels. Y is flipped (screen Y goes down).
    let screen_x = (ndc_x + 1.0) * 0.5 * viewport_width;
    let screen_y = (1.0 - ndc_y) * 0.5 * viewport_height;

    Some((screen_x, screen_y))
}

/// Main layout system. Resolves all UiNode positions into ResolvedRect components.
///
/// Call this each frame before collecting draw data or running input routing.
pub fn ui_layout_system(world: &mut World) {
    let viewport = world.resource::<UiViewport>().cloned().unwrap_or_default();

    let scale_x = viewport.scale_x();
    let scale_y = viewport.scale_y();

    let screen_rect = ReferenceRect {
        x: 0.0,
        y: 0.0,
        width: viewport.reference_width,
        height: viewport.reference_height,
    };

    // Collect root UI nodes (no Parent component).
    let roots: Vec<(Entity, UiNode)> = {
        let query = Query::<(Entity, &UiNode), Without<Parent>>::new(world);
        query.iter().map(|(e, n)| (e, n.clone())).collect()
    };

    // Process each root and its descendants via BFS.
    struct LayoutEntry {
        entity: Entity,
        node: UiNode,
        parent_rect: ReferenceRect,
    }

    let mut stack: Vec<LayoutEntry> = Vec::new();

    // Enqueue roots.
    for (entity, node) in roots {
        stack.push(LayoutEntry {
            entity,
            node,
            parent_rect: screen_rect,
        });
    }

    while let Some(entry) = stack.pop() {
        if !entry.node.visible {
            // Invisible nodes: set zero rect and skip children.
            world.insert(
                entry.entity,
                ResolvedRect {
                    x: 0.0,
                    y: 0.0,
                    width: 0.0,
                    height: 0.0,
                },
            );
            continue;
        }

        // Check for world-space UI: if entity has a GlobalTransform and we have
        // a view-projection matrix, override the position by projecting 3D to screen.
        let world_screen_pos = if let (Some(gt), Some(vp)) = (
            world.get::<GlobalTransform>(entry.entity).map(|g| g.0),
            viewport.view_projection.as_ref(),
        ) {
            project_to_screen(
                gt.translation,
                vp,
                viewport.viewport_width,
                viewport.viewport_height,
            )
        } else {
            None
        };

        let mut ref_rect = resolve_node_rect(&entry.node, &entry.parent_rect);

        // For world-space UI, override position with projected screen coords.
        // The node is centered on the projected point.
        if let Some((sx, sy)) = world_screen_pos {
            // Convert screen pixels back to reference space for consistency.
            ref_rect.x = sx / scale_x - ref_rect.width / 2.0;
            ref_rect.y = sy / scale_y - ref_rect.height / 2.0;
        }

        // Collect children for flex layout.
        let children_entities: Vec<Entity> = world
            .get::<Children>(entry.entity)
            .map(|c| c.0.clone())
            .unwrap_or_default();

        // Gather child nodes and their initial rects.
        let mut child_entries: Vec<(Entity, UiNode, ReferenceRect)> = Vec::new();
        for &child_entity in &children_entities {
            if let Some(child_node) = world.get::<UiNode>(child_entity) {
                let child_node = child_node.clone();
                let child_rect = resolve_node_rect(&child_node, &ref_rect);
                child_entries.push((child_entity, child_node, child_rect));
            }
        }

        // Apply flex layout if there are children with UiNode.
        if !child_entries.is_empty() {
            let mut flex_rects: Vec<(Entity, ReferenceRect)> =
                child_entries.iter().map(|(e, _, r)| (*e, *r)).collect();

            apply_flex_layout(&entry.node, &ref_rect, &mut flex_rects);

            // Update child_entries with flex-adjusted rects and enqueue them.
            for ((entity, node, _), (_, flex_rect)) in
                child_entries.into_iter().zip(flex_rects.into_iter())
            {
                stack.push(LayoutEntry {
                    entity,
                    node,
                    parent_rect: flex_rect,
                });
                // Note: the child's own rect will be re-resolved against its
                // (possibly flex-adjusted) parent when it's processed from the stack.
                // But for flex children, we store the flex result directly.
                let scaled = ResolvedRect {
                    x: flex_rect.x * scale_x,
                    y: flex_rect.y * scale_y,
                    width: flex_rect.width * scale_x,
                    height: flex_rect.height * scale_y,
                };
                world.insert(entity, scaled);
            }
        }

        // Store resolved rect for this node.
        let resolved = ResolvedRect {
            x: ref_rect.x * scale_x,
            y: ref_rect.y * scale_y,
            width: ref_rect.width * scale_x,
            height: ref_rect.height * scale_y,
        };
        world.insert(entry.entity, resolved);
    }
}

/// Collect draw commands from all visible UI entities.
///
/// Returns draw commands sorted by z_index (lowest first = drawn first).
pub fn collect_ui_draw_data(world: &World) -> Vec<UiDrawCommand> {
    let mut commands = Vec::new();

    // Panels.
    {
        let query = Query::<(&UiNode, &ResolvedRect, &UiPanel)>::new(world);
        for (node, rect, panel) in query.iter() {
            if !node.visible {
                continue;
            }
            // Border (if any): draw a slightly larger colored quad behind.
            if panel.border_width > 0.0 {
                commands.push(UiDrawCommand {
                    position: [rect.x - panel.border_width, rect.y - panel.border_width],
                    size: [
                        rect.width + panel.border_width * 2.0,
                        rect.height + panel.border_width * 2.0,
                    ],
                    color: panel.border_color,
                    kind: UiDrawKind::Colored,
                    z_index: node.z_index,
                    clip_rect: None,
                });
            }
            commands.push(UiDrawCommand {
                position: [rect.x, rect.y],
                size: [rect.width, rect.height],
                color: panel.bg_color,
                kind: UiDrawKind::Colored,
                z_index: node.z_index,
                clip_rect: None,
            });
        }
    }

    // Buttons.
    {
        let query = Query::<(&UiNode, &ResolvedRect, &UiButton)>::new(world);
        for (node, rect, button) in query.iter() {
            if !node.visible {
                continue;
            }
            commands.push(UiDrawCommand {
                position: [rect.x, rect.y],
                size: [rect.width, rect.height],
                color: button.current_color(),
                kind: UiDrawKind::Colored,
                z_index: node.z_index,
                clip_rect: None,
            });
        }
    }

    // Progress bars.
    {
        let query = Query::<(&UiNode, &ResolvedRect, &UiProgressBar)>::new(world);
        for (node, rect, bar) in query.iter() {
            if !node.visible {
                continue;
            }
            // Background.
            commands.push(UiDrawCommand {
                position: [rect.x, rect.y],
                size: [rect.width, rect.height],
                color: bar.bg_color,
                kind: UiDrawKind::Colored,
                z_index: node.z_index,
                clip_rect: None,
            });
            // Fill.
            let fill_width = rect.width * bar.fraction();
            if fill_width > 0.0 {
                commands.push(UiDrawCommand {
                    position: [rect.x, rect.y],
                    size: [fill_width, rect.height],
                    color: bar.fill_color,
                    kind: UiDrawKind::Colored,
                    z_index: node.z_index + 1,
                    clip_rect: None,
                });
            }
        }
    }

    // Images.
    {
        let query = Query::<(&UiNode, &ResolvedRect, &UiImage)>::new(world);
        for (node, rect, image) in query.iter() {
            if !node.visible {
                continue;
            }
            let kind = match image.texture {
                Some(tex) => UiDrawKind::Textured {
                    texture: tex,
                    uv: image.uv,
                },
                None => UiDrawKind::Colored,
            };
            commands.push(UiDrawCommand {
                position: [rect.x, rect.y],
                size: [rect.width, rect.height],
                color: image.color,
                kind,
                z_index: node.z_index,
                clip_rect: None,
            });
        }
    }

    // Text.
    {
        let query = Query::<(&UiNode, &ResolvedRect, &UiText)>::new(world);
        for (node, rect, text) in query.iter() {
            if !node.visible || text.text.is_empty() {
                continue;
            }
            commands.push(UiDrawCommand {
                position: [rect.x, rect.y],
                size: [rect.width, rect.height],
                color: text.color,
                kind: UiDrawKind::Text {
                    text: text.text.clone(),
                    font_size: text.font_size,
                },
                z_index: node.z_index,
                clip_rect: None,
            });
        }
    }

    // Sort by z_index so the renderer can draw in order.
    commands.sort_by_key(|cmd| cmd.z_index);
    commands
}

/// Convenience to check if there are any UiNode entities in the world.
pub fn has_ui_nodes(world: &World) -> bool {
    let query = Query::<&UiNode>::new(world);
    query.count() > 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{UiNode, UiRect, UiSize, Val};

    fn setup_world() -> World {
        let mut world = World::new();
        world.insert_resource(UiViewport::default());
        world
    }

    #[test]
    fn resolve_top_left_anchor() {
        let mut world = setup_world();
        let e = world.spawn(UiNode {
            anchor: Anchor::TopLeft,
            margin: UiRect {
                left: 10.0,
                top: 20.0,
                ..Default::default()
            },
            size: UiSize {
                width: Val::Px(200.0),
                height: Val::Px(100.0),
            },
            ..Default::default()
        });

        ui_layout_system(&mut world);

        let rect = world.get::<ResolvedRect>(e).unwrap();
        assert!((rect.x - 10.0).abs() < 1e-3);
        assert!((rect.y - 20.0).abs() < 1e-3);
        assert!((rect.width - 200.0).abs() < 1e-3);
        assert!((rect.height - 100.0).abs() < 1e-3);
    }

    #[test]
    fn resolve_center_anchor() {
        let mut world = setup_world();
        let e = world.spawn(UiNode {
            anchor: Anchor::Center,
            size: UiSize {
                width: Val::Px(100.0),
                height: Val::Px(50.0),
            },
            ..Default::default()
        });

        ui_layout_system(&mut world);

        let rect = world.get::<ResolvedRect>(e).unwrap();
        // Center of 1920x1080 minus half of 100x50.
        assert!((rect.x - (960.0 - 50.0)).abs() < 1e-3);
        assert!((rect.y - (540.0 - 25.0)).abs() < 1e-3);
    }

    #[test]
    fn resolve_stretch_anchor() {
        let mut world = setup_world();
        let e = world.spawn(UiNode {
            anchor: Anchor::Stretch,
            margin: UiRect::all(50.0),
            ..Default::default()
        });

        ui_layout_system(&mut world);

        let rect = world.get::<ResolvedRect>(e).unwrap();
        assert!((rect.x - 50.0).abs() < 1e-3);
        assert!((rect.y - 50.0).abs() < 1e-3);
        assert!((rect.width - (1920.0 - 100.0)).abs() < 1e-3);
        assert!((rect.height - (1080.0 - 100.0)).abs() < 1e-3);
    }

    #[test]
    fn resolve_percent_size() {
        let mut world = setup_world();
        let e = world.spawn(UiNode {
            anchor: Anchor::TopLeft,
            size: UiSize {
                width: Val::Percent(50.0),
                height: Val::Percent(25.0),
            },
            ..Default::default()
        });

        ui_layout_system(&mut world);

        let rect = world.get::<ResolvedRect>(e).unwrap();
        assert!((rect.width - 960.0).abs() < 1e-3);
        assert!((rect.height - 270.0).abs() < 1e-3);
    }

    #[test]
    fn invisible_node_gets_zero_rect() {
        let mut world = setup_world();
        let e = world.spawn(UiNode {
            visible: false,
            size: UiSize {
                width: Val::Px(200.0),
                height: Val::Px(100.0),
            },
            ..Default::default()
        });

        ui_layout_system(&mut world);

        let rect = world.get::<ResolvedRect>(e).unwrap();
        assert_eq!(rect.width, 0.0);
        assert_eq!(rect.height, 0.0);
    }

    #[test]
    fn viewport_scaling() {
        let mut world = World::new();
        world.insert_resource(UiViewport {
            reference_width: 1920.0,
            reference_height: 1080.0,
            viewport_width: 3840.0,
            viewport_height: 2160.0,
            view_projection: None,
        });

        let e = world.spawn(UiNode {
            anchor: Anchor::TopLeft,
            margin: UiRect {
                left: 100.0,
                top: 50.0,
                ..Default::default()
            },
            size: UiSize {
                width: Val::Px(200.0),
                height: Val::Px(100.0),
            },
            ..Default::default()
        });

        ui_layout_system(&mut world);

        let rect = world.get::<ResolvedRect>(e).unwrap();
        // 2x scaling: everything doubles.
        assert!((rect.x - 200.0).abs() < 1e-3);
        assert!((rect.y - 100.0).abs() < 1e-3);
        assert!((rect.width - 400.0).abs() < 1e-3);
        assert!((rect.height - 200.0).abs() < 1e-3);
    }

    #[test]
    fn resolved_rect_contains() {
        let rect = ResolvedRect {
            x: 100.0,
            y: 200.0,
            width: 300.0,
            height: 150.0,
        };
        assert!(rect.contains(100.0, 200.0));
        assert!(rect.contains(250.0, 275.0));
        assert!(rect.contains(400.0, 350.0));
        assert!(!rect.contains(99.0, 200.0));
        assert!(!rect.contains(100.0, 351.0));
    }

    #[test]
    fn collect_draw_data_empty() {
        let world = setup_world();
        let commands = collect_ui_draw_data(&world);
        assert!(commands.is_empty());
    }

    #[test]
    fn collect_draw_data_panel() {
        let mut world = setup_world();
        let e = world.spawn(UiNode {
            anchor: Anchor::TopLeft,
            size: UiSize {
                width: Val::Px(200.0),
                height: Val::Px(100.0),
            },
            ..Default::default()
        });
        world.insert(e, UiPanel::default());

        ui_layout_system(&mut world);
        let commands = collect_ui_draw_data(&world);

        // Panel with border_width > 0 produces 2 commands (border + bg).
        assert_eq!(commands.len(), 2);
    }

    #[test]
    fn collect_draw_data_progress_bar() {
        let mut world = setup_world();
        let e = world.spawn(UiNode {
            anchor: Anchor::TopLeft,
            size: UiSize {
                width: Val::Px(200.0),
                height: Val::Px(20.0),
            },
            ..Default::default()
        });
        world.insert(
            e,
            UiProgressBar {
                value: 50.0,
                max: 100.0,
                ..Default::default()
            },
        );

        ui_layout_system(&mut world);
        let commands = collect_ui_draw_data(&world);

        // Background + fill = 2 commands.
        assert_eq!(commands.len(), 2);
    }

    #[test]
    fn draw_commands_sorted_by_z_index() {
        let mut world = setup_world();

        let e1 = world.spawn(UiNode {
            z_index: 10,
            size: UiSize {
                width: Val::Px(100.0),
                height: Val::Px(100.0),
            },
            ..Default::default()
        });
        world.insert(
            e1,
            UiPanel {
                border_width: 0.0,
                ..Default::default()
            },
        );

        let e2 = world.spawn(UiNode {
            z_index: 5,
            size: UiSize {
                width: Val::Px(100.0),
                height: Val::Px(100.0),
            },
            ..Default::default()
        });
        world.insert(
            e2,
            UiPanel {
                border_width: 0.0,
                ..Default::default()
            },
        );

        ui_layout_system(&mut world);
        let commands = collect_ui_draw_data(&world);

        assert!(commands.len() >= 2);
        assert!(commands[0].z_index <= commands[1].z_index);
    }

    #[test]
    fn flex_row_layout() {
        let mut world = setup_world();

        // Parent with row flex.
        let parent = world.spawn(UiNode {
            anchor: Anchor::TopLeft,
            size: UiSize {
                width: Val::Px(600.0),
                height: Val::Px(100.0),
            },
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::Start,
            ..Default::default()
        });

        // Two children of 200px each.
        let c1 = world.spawn(UiNode {
            size: UiSize {
                width: Val::Px(200.0),
                height: Val::Px(50.0),
            },
            ..Default::default()
        });
        let c2 = world.spawn(UiNode {
            size: UiSize {
                width: Val::Px(200.0),
                height: Val::Px(50.0),
            },
            ..Default::default()
        });

        world.insert(c1, Parent(parent));
        world.insert(c2, Parent(parent));
        world.insert(parent, Children(vec![c1, c2]));

        ui_layout_system(&mut world);

        let r1 = world.get::<ResolvedRect>(c1).unwrap();
        let r2 = world.get::<ResolvedRect>(c2).unwrap();

        // c1 starts at x=0, c2 starts at x=200.
        assert!((r1.x - 0.0).abs() < 1e-3);
        assert!((r2.x - 200.0).abs() < 1e-3);
    }

    #[test]
    fn flex_column_center() {
        let mut world = setup_world();

        let parent = world.spawn(UiNode {
            anchor: Anchor::TopLeft,
            size: UiSize {
                width: Val::Px(200.0),
                height: Val::Px(400.0),
            },
            flex_direction: FlexDirection::Column,
            justify_content: JustifyContent::Center,
            ..Default::default()
        });

        let child = world.spawn(UiNode {
            size: UiSize {
                width: Val::Px(100.0),
                height: Val::Px(100.0),
            },
            ..Default::default()
        });

        world.insert(child, Parent(parent));
        world.insert(parent, Children(vec![child]));

        ui_layout_system(&mut world);

        let r = world.get::<ResolvedRect>(child).unwrap();
        // Column centered: y should be at (400 - 100) / 2 = 150.
        assert!((r.y - 150.0).abs() < 1e-3);
    }

    #[test]
    fn world_space_ui_projection() {
        let mut world = World::new();

        // Set up a simple orthographic view-projection where world Z maps to screen.
        // We use identity for simplicity: world (0,0,0) maps to screen center.
        let vp = Mat4::orthographic_lh(-10.0, 10.0, -10.0, 10.0, 0.0, 100.0);

        world.insert_resource(UiViewport {
            reference_width: 1920.0,
            reference_height: 1080.0,
            viewport_width: 1920.0,
            viewport_height: 1080.0,
            view_projection: Some(vp),
        });

        // Entity with UiNode + GlobalTransform at world origin.
        let e = world.spawn(UiNode {
            anchor: Anchor::TopLeft,
            size: UiSize {
                width: Val::Px(100.0),
                height: Val::Px(50.0),
            },
            ..Default::default()
        });
        world.insert(
            e,
            GlobalTransform(euca_math::Transform::from_translation(
                euca_math::Vec3::ZERO,
            )),
        );

        ui_layout_system(&mut world);

        let rect = world.get::<ResolvedRect>(e).unwrap();
        // World origin projects to screen center (960, 540), then node is centered on it.
        // x = 960 - 50 = 910, y = 540 - 25 = 515
        assert!((rect.x - 910.0).abs() < 2.0);
        assert!((rect.y - 515.0).abs() < 2.0);
    }
}
