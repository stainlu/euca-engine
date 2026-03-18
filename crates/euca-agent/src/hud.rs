//! In-game HUD elements controlled by agents via CLI/HTTP.
//!
//! Stored as a World resource. The editor's egui pass reads and renders them.

use serde::{Deserialize, Serialize};

/// A single HUD element on screen.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum HudElement {
    /// Text label at a screen position.
    #[serde(rename = "text")]
    Text {
        text: String,
        x: f32,
        y: f32,
        #[serde(default = "default_size")]
        size: f32,
        #[serde(default = "default_color")]
        color: String,
    },
    /// Filled bar (health bar, progress bar).
    #[serde(rename = "bar")]
    Bar {
        x: f32,
        y: f32,
        #[serde(default = "default_bar_width")]
        width: f32,
        #[serde(default = "default_bar_height")]
        height: f32,
        fill: f32,
        #[serde(default = "default_color")]
        color: String,
    },
    /// Colored rectangle.
    #[serde(rename = "rect")]
    Rect {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        #[serde(default = "default_color")]
        color: String,
    },
}

fn default_size() -> f32 {
    20.0
}
fn default_color() -> String {
    "white".to_string()
}
fn default_bar_width() -> f32 {
    0.2
}
fn default_bar_height() -> f32 {
    0.03
}

/// Collection of HUD elements to render each frame. Stored as World resource.
#[derive(Clone, Debug, Default)]
pub struct HudCanvas {
    pub elements: Vec<HudElement>,
}

impl HudCanvas {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.elements.clear();
    }

    pub fn add(&mut self, element: HudElement) {
        self.elements.push(element);
    }
}

/// Parse a color name to RGBA [0.0-1.0].
pub fn parse_color(name: &str) -> [f32; 4] {
    match name {
        "red" => [1.0, 0.2, 0.2, 1.0],
        "green" => [0.2, 1.0, 0.2, 1.0],
        "blue" => [0.3, 0.5, 1.0, 1.0],
        "yellow" => [1.0, 1.0, 0.2, 1.0],
        "cyan" => [0.2, 1.0, 1.0, 1.0],
        "magenta" => [1.0, 0.2, 1.0, 1.0],
        "orange" => [1.0, 0.6, 0.1, 1.0],
        "white" => [1.0, 1.0, 1.0, 1.0],
        "black" => [0.0, 0.0, 0.0, 1.0],
        "gray" => [0.5, 0.5, 0.5, 1.0],
        "gold" => [1.0, 0.84, 0.0, 1.0],
        _ => [1.0, 1.0, 1.0, 1.0],
    }
}
