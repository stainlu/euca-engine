//! Core UI node component and associated layout types.

/// How a UI node anchors to its parent's bounds (or the screen if root).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Anchor {
    #[default]
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
    /// Node stretches to fill its parent, respecting margins.
    Stretch,
}

impl Anchor {
    /// Returns the fractional offset (0..1) within the parent for this anchor.
    /// (0, 0) = top-left, (1, 1) = bottom-right.
    pub fn parent_fraction(self) -> (f32, f32) {
        match self {
            Anchor::TopLeft => (0.0, 0.0),
            Anchor::TopCenter => (0.5, 0.0),
            Anchor::TopRight => (1.0, 0.0),
            Anchor::CenterLeft => (0.0, 0.5),
            Anchor::Center => (0.5, 0.5),
            Anchor::CenterRight => (1.0, 0.5),
            Anchor::BottomLeft => (0.0, 1.0),
            Anchor::BottomCenter => (0.5, 1.0),
            Anchor::BottomRight => (1.0, 1.0),
            Anchor::Stretch => (0.0, 0.0),
        }
    }

}

/// A dimensional value for UI sizing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Val {
    /// Absolute pixels (at reference resolution).
    Px(f32),
    /// Percentage of parent dimension.
    Percent(f32),
    /// Auto-size (determined by content or flex layout).
    #[default]
    Auto,
}

/// Width and height specification for a UI node.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct UiSize {
    pub width: Val,
    pub height: Val,
}

/// Edge insets in pixels (at reference resolution).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct UiRect {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
}

impl UiRect {
    pub const ZERO: Self = Self {
        left: 0.0,
        right: 0.0,
        top: 0.0,
        bottom: 0.0,
    };

    /// Create a UiRect with equal insets on all sides.
    pub fn all(value: f32) -> Self {
        Self {
            left: value,
            right: value,
            top: value,
            bottom: value,
        }
    }
}

/// Direction for flex layout of children.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FlexDirection {
    #[default]
    Row,
    Column,
}

/// How children are distributed along the main axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum JustifyContent {
    #[default]
    Start,
    Center,
    End,
    SpaceBetween,
    SpaceAround,
}

/// How children are aligned along the cross axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AlignItems {
    #[default]
    Start,
    Center,
    End,
    Stretch,
}

/// Core UI element with layout properties.
///
/// Any entity with a `UiNode` participates in the UI layout system.
/// Additional widget components (UiText, UiButton, etc.) add behavior on top.
#[derive(Clone, Debug)]
pub struct UiNode {
    /// How this node anchors within its parent bounds.
    pub anchor: Anchor,
    /// Pixel margins from the anchor point (at reference resolution).
    pub margin: UiRect,
    /// Width and height of the node.
    pub size: UiSize,
    /// Draw ordering. Higher z_index draws on top.
    pub z_index: i32,
    /// Whether this node (and its children) are visible.
    pub visible: bool,
    /// Flex direction for laying out children.
    pub flex_direction: FlexDirection,
    /// How children are distributed along the main axis.
    pub justify_content: JustifyContent,
    /// How children are aligned along the cross axis.
    pub align_items: AlignItems,
}

impl Default for UiNode {
    fn default() -> Self {
        Self {
            anchor: Anchor::default(),
            margin: UiRect::default(),
            size: UiSize::default(),
            z_index: 0,
            visible: true,
            flex_direction: FlexDirection::default(),
            justify_content: JustifyContent::default(),
            align_items: AlignItems::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_fractions() {
        assert_eq!(Anchor::TopLeft.parent_fraction(), (0.0, 0.0));
        assert_eq!(Anchor::Center.parent_fraction(), (0.5, 0.5));
        assert_eq!(Anchor::BottomRight.parent_fraction(), (1.0, 1.0));
        assert_eq!(Anchor::Stretch.parent_fraction(), (0.0, 0.0));
    }

    #[test]
    fn val_defaults_to_auto() {
        assert_eq!(Val::default(), Val::Auto);
    }

    #[test]
    fn ui_rect_all() {
        let r = UiRect::all(10.0);
        assert_eq!(r.left, 10.0);
        assert_eq!(r.right, 10.0);
        assert_eq!(r.top, 10.0);
        assert_eq!(r.bottom, 10.0);
    }

    #[test]
    fn ui_node_default() {
        let node = UiNode::default();
        assert_eq!(node.anchor, Anchor::TopLeft);
        assert!(node.visible);
        assert_eq!(node.z_index, 0);
    }
}
