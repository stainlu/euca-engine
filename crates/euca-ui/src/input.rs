//! UI input routing system.
//!
//! Checks mouse position against resolved UI rects and updates button
//! interaction state (hover, pressed, clicked). When UI consumes input,
//! it sets a flag so game systems can skip processing.

use euca_ecs::{Entity, Query, World};
use euca_input::{InputKey, InputState};

use crate::layout::ResolvedRect;
use crate::node::UiNode;
use crate::widget::UiButton;

/// Resource flag indicating the UI consumed input this frame.
///
/// Game systems should check this and skip mouse input when true.
#[derive(Clone, Debug, Default)]
pub struct UiInputConsumed {
    /// True if the mouse is over any interactive UI element.
    pub mouse_over_ui: bool,
}

/// UI input routing system.
///
/// Must run after `ui_layout_system` so that `ResolvedRect` components are up to date.
///
/// For each entity with `UiNode + UiButton + ResolvedRect`:
/// - Sets `is_hovered` based on mouse position.
/// - Sets `is_pressed` if hovered and mouse button is held.
/// - Sets `just_clicked` if hovered and mouse button was just released.
///
/// Sets the `UiInputConsumed` resource if any button is hovered.
pub fn ui_input_system(world: &mut World) {
    let (mouse_x, mouse_y, mouse_held, mouse_just_released) = {
        let input = match world.resource::<InputState>() {
            Some(input) => input,
            None => return,
        };
        let held = input.is_pressed(&InputKey::MouseLeft);
        let just_released = input.is_just_released(&InputKey::MouseLeft);
        (
            input.mouse_position[0],
            input.mouse_position[1],
            held,
            just_released,
        )
    };

    // Collect button entities and their rects, sorted by z_index (highest first)
    // so that topmost buttons get priority.
    let mut button_entities: Vec<(Entity, i32)> = {
        let query = Query::<(Entity, &UiNode, &ResolvedRect), euca_ecs::With<UiButton>>::new(world);
        query
            .iter()
            .filter(|(_, node, _)| node.visible)
            .map(|(e, node, _)| (e, node.z_index))
            .collect()
    };
    button_entities.sort_by(|a, b| b.1.cmp(&a.1)); // Highest z_index first.

    let mut any_hovered = false;
    let mut consumed_by: Option<Entity> = None;

    for (entity, _) in &button_entities {
        let rect = match world.get::<ResolvedRect>(*entity) {
            Some(r) => *r,
            None => continue,
        };

        let hovered = rect.contains(mouse_x, mouse_y) && consumed_by.is_none();
        let pressed = hovered && mouse_held;
        let just_clicked = hovered && mouse_just_released;

        if hovered {
            any_hovered = true;
            consumed_by = Some(*entity);
        }

        if let Some(button) = world.get_mut::<UiButton>(*entity) {
            button.is_hovered = hovered;
            button.is_pressed = pressed;
            button.just_clicked = just_clicked;
        }
    }

    // Update or insert the consumed flag.
    if let Some(consumed) = world.resource_mut::<UiInputConsumed>() {
        consumed.mouse_over_ui = any_hovered;
    } else {
        world.insert_resource(UiInputConsumed {
            mouse_over_ui: any_hovered,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::UiViewport;
    use crate::node::{Anchor, UiNode, UiSize, Val};

    fn setup_world_with_button() -> (World, Entity) {
        let mut world = World::new();
        world.insert_resource(UiViewport::default());
        world.insert_resource(InputState::new());

        // Create a button at (100, 100) with size 200x50.
        let e = world.spawn(UiNode {
            anchor: Anchor::TopLeft,
            size: UiSize {
                width: Val::Px(200.0),
                height: Val::Px(50.0),
            },
            margin: crate::node::UiRect {
                left: 100.0,
                top: 100.0,
                ..Default::default()
            },
            ..Default::default()
        });
        world.insert(e, UiButton::default());

        // Run layout to get ResolvedRect.
        crate::layout::ui_layout_system(&mut world);

        (world, e)
    }

    #[test]
    fn button_hover_detection() {
        let (mut world, e) = setup_world_with_button();

        // Mouse inside button.
        world.resource_mut::<InputState>().unwrap().set_mouse_position(200.0, 125.0);

        ui_input_system(&mut world);

        let button = world.get::<UiButton>(e).unwrap();
        assert!(button.is_hovered);
        assert!(!button.is_pressed);
        assert!(!button.just_clicked);

        let consumed = world.resource::<UiInputConsumed>().unwrap();
        assert!(consumed.mouse_over_ui);
    }

    #[test]
    fn button_not_hovered_when_outside() {
        let (mut world, e) = setup_world_with_button();

        // Mouse outside button.
        world.resource_mut::<InputState>().unwrap().set_mouse_position(50.0, 50.0);

        ui_input_system(&mut world);

        let button = world.get::<UiButton>(e).unwrap();
        assert!(!button.is_hovered);
        assert!(!button.is_pressed);

        let consumed = world.resource::<UiInputConsumed>().unwrap();
        assert!(!consumed.mouse_over_ui);
    }

    #[test]
    fn button_press_detection() {
        let (mut world, e) = setup_world_with_button();

        // Mouse inside, button held.
        {
            let input = world.resource_mut::<InputState>().unwrap();
            input.set_mouse_position(200.0, 125.0);
            input.press(InputKey::MouseLeft);
        }

        ui_input_system(&mut world);

        let button = world.get::<UiButton>(e).unwrap();
        assert!(button.is_hovered);
        assert!(button.is_pressed);
        assert!(!button.just_clicked);
    }

    #[test]
    fn button_click_detection() {
        let (mut world, e) = setup_world_with_button();

        // First frame: press the button.
        {
            let input = world.resource_mut::<InputState>().unwrap();
            input.set_mouse_position(200.0, 125.0);
            input.press(InputKey::MouseLeft);
        }
        ui_input_system(&mut world);

        // Second frame: release.
        {
            let input = world.resource_mut::<InputState>().unwrap();
            input.begin_frame();
            input.release(InputKey::MouseLeft);
        }
        ui_input_system(&mut world);

        let button = world.get::<UiButton>(e).unwrap();
        assert!(button.is_hovered);
        assert!(!button.is_pressed);
        assert!(button.just_clicked);
    }

    #[test]
    fn z_index_priority() {
        let mut world = World::new();
        world.insert_resource(UiViewport::default());
        world.insert_resource(InputState::new());

        // Two overlapping buttons.
        let bottom = world.spawn(UiNode {
            anchor: Anchor::TopLeft,
            size: UiSize {
                width: Val::Px(200.0),
                height: Val::Px(200.0),
            },
            z_index: 0,
            ..Default::default()
        });
        world.insert(bottom, UiButton::default());

        let top = world.spawn(UiNode {
            anchor: Anchor::TopLeft,
            size: UiSize {
                width: Val::Px(200.0),
                height: Val::Px(200.0),
            },
            z_index: 10,
            ..Default::default()
        });
        world.insert(top, UiButton::default());

        crate::layout::ui_layout_system(&mut world);

        world.resource_mut::<InputState>().unwrap().set_mouse_position(100.0, 100.0);

        ui_input_system(&mut world);

        // Top button should be hovered, bottom should not (consumed by top).
        let top_btn = world.get::<UiButton>(top).unwrap();
        assert!(top_btn.is_hovered);

        let bottom_btn = world.get::<UiButton>(bottom).unwrap();
        assert!(!bottom_btn.is_hovered);
    }
}
