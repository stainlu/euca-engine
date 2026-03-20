//! Renderer-agnostic draw command output.
//!
//! The layout system resolves positions and sizes, then `collect_ui_draw_data`
//! produces a flat list of `UiDrawCommand`s sorted by z-index. The renderer
//! batches these as textured/colored quads.

use crate::widget::TextureHandle;

/// The type of UI element to render.
#[derive(Clone, Debug, PartialEq)]
pub enum UiDrawKind {
    /// Solid colored quad (panel background, button, progress bar segment).
    Colored,
    /// Textured quad (image widget).
    Textured {
        texture: TextureHandle,
        uv: [f32; 4],
    },
    /// Text rendering.
    Text {
        text: String,
        font_size: f32,
    },
}

/// A single draw command for the renderer.
///
/// All coordinates are in final screen pixels (after viewport scaling).
#[derive(Clone, Debug)]
pub struct UiDrawCommand {
    /// Screen-space position of the quad's top-left corner.
    pub position: [f32; 2],
    /// Width and height of the quad in screen pixels.
    pub size: [f32; 2],
    /// RGBA color [0..1].
    pub color: [f32; 4],
    /// What to draw.
    pub kind: UiDrawKind,
    /// Z-index for sorting. Higher values draw on top.
    pub z_index: i32,
    /// Clip rectangle (min_x, min_y, max_x, max_y) in screen pixels.
    /// If None, no clipping.
    pub clip_rect: Option<[f32; 4]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_command_construction() {
        let cmd = UiDrawCommand {
            position: [100.0, 200.0],
            size: [300.0, 50.0],
            color: [1.0, 0.0, 0.0, 1.0],
            kind: UiDrawKind::Colored,
            z_index: 5,
            clip_rect: None,
        };
        assert_eq!(cmd.position, [100.0, 200.0]);
        assert_eq!(cmd.z_index, 5);
    }
}
