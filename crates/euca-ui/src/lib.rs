//! Runtime UI framework for in-game HUD, menus, and overlays.
//!
//! This crate is renderer-agnostic: it resolves UI layout into screen-space
//! rectangles and emits `UiDrawCommand`s that the renderer consumes as
//! textured/colored quads.
//!
//! # Architecture
//!
//! - **`UiNode`**: Core component with anchor, margin, size, visibility, and
//!   flex layout properties. Any entity with a `UiNode` participates in layout.
//!
//! - **Widget components** (`UiText`, `UiImage`, `UiButton`, `UiProgressBar`,
//!   `UiPanel`): Attached alongside `UiNode` to define rendering behavior.
//!
//! - **Layout solver**: `ui_layout_system` resolves all `UiNode` trees into
//!   `ResolvedRect` components in screen pixels, handling anchoring, margins,
//!   percentage sizing, flex layout, and viewport scaling.
//!
//! - **Input routing**: `ui_input_system` hit-tests mouse position against
//!   resolved rects, updating `UiButton` interaction state. Sets
//!   `UiInputConsumed` so game systems can skip consumed input.
//!
//! - **Draw output**: `collect_ui_draw_data` produces sorted `UiDrawCommand`s
//!   ready for the renderer to batch.
//!
//! - **World-space UI**: Entities with both `UiNode` and `GlobalTransform`
//!   project their 3D position to screen coordinates (floating health bars,
//!   name plates, etc.).

mod draw;
mod input;
mod layout;
mod node;
mod widget;

// Re-export public API.
pub use draw::{UiDrawCommand, UiDrawKind};
pub use input::{UiInputConsumed, ui_input_system};
pub use layout::{ResolvedRect, UiViewport, collect_ui_draw_data, has_ui_nodes, ui_layout_system};
pub use node::{AlignItems, Anchor, FlexDirection, JustifyContent, UiNode, UiRect, UiSize, Val};
pub use widget::{TextureHandle, UiButton, UiImage, UiPanel, UiProgressBar, UiText};
