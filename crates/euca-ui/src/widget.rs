//! Widget components that add behavior to a UiNode entity.
//!
//! Each widget component is attached alongside a `UiNode` on the same entity.
//! The layout system resolves the node's position/size, and the draw system
//! uses widget data to emit draw commands.

/// Opaque handle to a texture resource. The renderer defines the actual type;
/// this is just a numeric ID so euca-ui stays renderer-agnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct TextureHandle(pub u64);

/// Text display widget.
#[derive(Clone, Debug)]
pub struct UiText {
    pub text: String,
    pub font_size: f32,
    /// RGBA color [0..1].
    pub color: [f32; 4],
}

impl Default for UiText {
    fn default() -> Self {
        Self {
            text: String::new(),
            font_size: 16.0,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

/// Image display widget.
#[derive(Clone, Debug)]
pub struct UiImage {
    /// Texture to display. `None` means no texture (solid color).
    pub texture: Option<TextureHandle>,
    /// RGBA tint color [0..1]. White = no tinting.
    pub color: [f32; 4],
    /// UV coordinates [u_min, v_min, u_max, v_max] for sprite sheets.
    pub uv: [f32; 4],
}

impl Default for UiImage {
    fn default() -> Self {
        Self {
            texture: None,
            color: [1.0, 1.0, 1.0, 1.0],
            uv: [0.0, 0.0, 1.0, 1.0],
        }
    }
}

/// Interactive button widget.
///
/// The `ui_input_system` sets `is_hovered`, `is_pressed`, and `just_clicked`
/// based on mouse position and button state.
#[derive(Clone, Debug)]
pub struct UiButton {
    /// Color when idle.
    pub normal_color: [f32; 4],
    /// Color when hovered.
    pub hover_color: [f32; 4],
    /// Color when pressed.
    pub pressed_color: [f32; 4],
    /// Set by the input system: mouse is over this button.
    pub is_hovered: bool,
    /// Set by the input system: mouse is pressing this button.
    pub is_pressed: bool,
    /// Set by the input system: mouse click completed this frame.
    pub just_clicked: bool,
}

impl Default for UiButton {
    fn default() -> Self {
        Self {
            normal_color: [0.3, 0.3, 0.3, 1.0],
            hover_color: [0.4, 0.4, 0.4, 1.0],
            pressed_color: [0.2, 0.2, 0.2, 1.0],
            is_hovered: false,
            is_pressed: false,
            just_clicked: false,
        }
    }
}

impl UiButton {
    /// Returns the current color based on interaction state.
    pub fn current_color(&self) -> [f32; 4] {
        if self.is_pressed {
            self.pressed_color
        } else if self.is_hovered {
            self.hover_color
        } else {
            self.normal_color
        }
    }
}

/// Progress bar widget.
#[derive(Clone, Debug)]
pub struct UiProgressBar {
    /// Current value.
    pub value: f32,
    /// Maximum value. Fill fraction = value / max.
    pub max: f32,
    /// RGBA color of the filled portion.
    pub fill_color: [f32; 4],
    /// RGBA color of the background.
    pub bg_color: [f32; 4],
}

impl Default for UiProgressBar {
    fn default() -> Self {
        Self {
            value: 0.0,
            max: 1.0,
            fill_color: [0.2, 0.8, 0.2, 1.0],
            bg_color: [0.2, 0.2, 0.2, 1.0],
        }
    }
}

impl UiProgressBar {
    /// Returns the fill fraction clamped to [0, 1].
    pub fn fraction(&self) -> f32 {
        if self.max <= 0.0 {
            0.0
        } else {
            (self.value / self.max).clamp(0.0, 1.0)
        }
    }
}

/// Background panel widget.
#[derive(Clone, Debug)]
pub struct UiPanel {
    /// RGBA background color.
    pub bg_color: [f32; 4],
    /// RGBA border color.
    pub border_color: [f32; 4],
    /// Border width in pixels (at reference resolution).
    pub border_width: f32,
}

impl Default for UiPanel {
    fn default() -> Self {
        Self {
            bg_color: [0.1, 0.1, 0.1, 0.8],
            border_color: [0.5, 0.5, 0.5, 1.0],
            border_width: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn button_current_color() {
        let mut btn = UiButton::default();
        assert_eq!(btn.current_color(), btn.normal_color);

        btn.is_hovered = true;
        assert_eq!(btn.current_color(), btn.hover_color);

        btn.is_pressed = true;
        assert_eq!(btn.current_color(), btn.pressed_color);
    }

    #[test]
    fn progress_bar_fraction() {
        let mut bar = UiProgressBar {
            value: 75.0,
            max: 100.0,
            ..Default::default()
        };
        assert!((bar.fraction() - 0.75).abs() < 1e-6);

        bar.value = 200.0;
        assert!((bar.fraction() - 1.0).abs() < 1e-6);

        bar.value = -10.0;
        assert!((bar.fraction() - 0.0).abs() < 1e-6);

        bar.max = 0.0;
        assert!((bar.fraction() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn texture_handle_default() {
        assert_eq!(TextureHandle::default(), TextureHandle(0));
    }
}
