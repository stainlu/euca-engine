//! Active item mechanics — items with click-to-use abilities, cooldowns,
//! charges, backpack slots, and neutral item slot.
//!
//! Types: `ItemActive`, `ItemCharges`, `Backpack`, `NeutralItemSlot`, `ItemState`.
//! Functions: `use_item_active`, `tick_cooldowns`, `tick_charges`,
//!            `swap_to_backpack`, `can_use_active`, `consume_charge`.
//!
//! This module extends the inventory system with MOBA-style active item
//! mechanics. Each entity's `ItemState` tracks active abilities, charges,
//! backpack contents, and a dedicated neutral item slot.

use serde::{Deserialize, Serialize};

// ── Error ──

/// Errors that can occur during active item operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ItemError {
    /// The specified slot is empty — no active ability to use.
    EmptySlot,
    /// The item's active ability is still on cooldown.
    OnCooldown,
    /// Not enough mana to activate this item.
    InsufficientMana,
    /// The entity is muted and this item cannot be used while muted.
    Muted,
    /// Slot index is out of bounds.
    SlotOutOfBounds,
    /// The item in this slot has no charges.
    NoCharges,
    /// The item's charges are depleted.
    ChargesDepleted,
    /// The backpack slot index is out of bounds (must be 0..3).
    BackpackSlotOutOfBounds,
}

impl std::fmt::Display for ItemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySlot => write!(f, "slot is empty"),
            Self::OnCooldown => write!(f, "item is on cooldown"),
            Self::InsufficientMana => write!(f, "insufficient mana"),
            Self::Muted => write!(f, "cannot use item while muted"),
            Self::SlotOutOfBounds => write!(f, "slot index out of bounds"),
            Self::NoCharges => write!(f, "item has no charges"),
            Self::ChargesDepleted => write!(f, "charges depleted"),
            Self::BackpackSlotOutOfBounds => write!(f, "backpack slot index out of bounds"),
        }
    }
}

impl std::error::Error for ItemError {}

// ── Data types ──

/// An active ability attached to an item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemActive {
    /// Human-readable name (e.g., "Blink", "Black King Bar").
    pub name: String,
    /// Cooldown duration in seconds.
    pub cooldown: f32,
    /// Current remaining cooldown (0 = ready).
    pub remaining_cooldown: f32,
    /// Mana cost to use (0 = no mana cost).
    pub mana_cost: f32,
    /// Effect identifier — game code maps this to actual behavior.
    pub effect_id: String,
    /// Cast range (0 = self-cast).
    pub cast_range: f32,
    /// Whether this active can be used while the entity is muted.
    pub usable_while_muted: bool,
    /// Shared cooldown group. Items in the same group share cooldown.
    pub cooldown_group: Option<CooldownGroup>,
}

/// Consumable charges on an item (e.g., TP Scroll, Wand).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemCharges {
    /// Current number of charges.
    pub current: u32,
    /// Maximum charges this item can hold.
    pub max: u32,
    /// Seconds between automatic charge gains (0 = no auto-recharge).
    pub recharge_time: f32,
    /// Timer counting toward the next recharge.
    pub recharge_timer: f32,
}

/// Backpack: 3 extra item slots with swap delay.
///
/// Items in the backpack have their passives disabled. When swapped into
/// a main slot, they enter an activation cooldown before they can be used.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Backpack {
    /// Items in backpack slots (up to 3).
    pub slots: [Option<BackpackItem>; 3],
    /// Cooldown applied to items swapped from backpack to main inventory.
    pub swap_cooldown: f32,
}

/// An item stored in a backpack slot, preserving its full active-item state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackpackItem {
    /// Item identifier (matches `ItemDef::id`).
    pub item_id: u32,
    /// Preserved active ability state (if this item has one).
    pub active: Option<ItemActive>,
    /// Preserved charge state (if this item has charges).
    pub charges: Option<ItemCharges>,
    /// Time remaining before this item becomes active after being swapped in.
    /// While > 0, the item cannot be used. Only meaningful during/after a swap.
    pub activation_cooldown: f32,
}

/// Neutral item slot — one dedicated slot for neutral drops.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NeutralItemSlot {
    /// Item ID of the equipped neutral item, if any.
    pub item: Option<u32>,
}

/// Shared cooldown group identifier.
///
/// Items with the same `CooldownGroup` share a single cooldown timer.
/// When any item in the group goes on cooldown, all items in that group
/// start their cooldowns simultaneously.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CooldownGroup(pub u32);

/// Complete active-item state for one entity.
///
/// Indices into `actives` and `charges` correspond to main inventory slots.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ItemState {
    /// Active abilities on equipped items, indexed by inventory slot.
    pub actives: Vec<Option<ItemActive>>,
    /// Charge state per inventory slot.
    pub charges: Vec<Option<ItemCharges>>,
    /// Backpack (3 extra storage slots with swap delay).
    pub backpack: Backpack,
    /// Neutral item slot.
    pub neutral_slot: NeutralItemSlot,
}

impl Default for Backpack {
    fn default() -> Self {
        Self {
            slots: [None, None, None],
            swap_cooldown: 6.0,
        }
    }
}

// ── Operations ──

/// Check whether an active item in the given slot can be used.
///
/// Returns `true` if the slot has an active ability that is off cooldown,
/// the entity has enough mana, and the entity is not muted (or the item
/// is usable while muted).
pub fn can_use_active(state: &ItemState, slot: usize, current_mana: f32, is_muted: bool) -> bool {
    let Some(active) = state.actives.get(slot).and_then(|a| a.as_ref()) else {
        return false;
    };

    if active.remaining_cooldown > 0.0 {
        return false;
    }
    if current_mana < active.mana_cost {
        return false;
    }
    if is_muted && !active.usable_while_muted {
        return false;
    }

    true
}

/// Use an active item ability. Returns the `effect_id` on success so the
/// caller can dispatch the actual game effect.
///
/// On success, puts the item (and its cooldown group) on cooldown.
pub fn use_item_active(
    state: &mut ItemState,
    slot: usize,
    current_mana: f32,
    is_muted: bool,
) -> Result<String, ItemError> {
    let active = state
        .actives
        .get(slot)
        .ok_or(ItemError::SlotOutOfBounds)?
        .as_ref()
        .ok_or(ItemError::EmptySlot)?;

    if active.remaining_cooldown > 0.0 {
        return Err(ItemError::OnCooldown);
    }
    if current_mana < active.mana_cost {
        return Err(ItemError::InsufficientMana);
    }
    if is_muted && !active.usable_while_muted {
        return Err(ItemError::Muted);
    }

    let effect_id = active.effect_id.clone();
    let cooldown = active.cooldown;
    let cooldown_group = active.cooldown_group;

    // Put this item on cooldown.
    let active_mut = state.actives[slot].as_mut().unwrap();
    active_mut.remaining_cooldown = cooldown;

    // If this item belongs to a cooldown group, trigger cooldown on all
    // other items in the same group.
    if let Some(group) = cooldown_group {
        for (i, maybe_active) in state.actives.iter_mut().enumerate() {
            if i == slot {
                continue;
            }
            if let Some(other) = maybe_active.as_mut()
                && other.cooldown_group == Some(group)
            {
                other.remaining_cooldown = other.cooldown;
            }
        }
    }

    Ok(effect_id)
}

/// Advance all active-item cooldowns by `dt` seconds.
///
/// Also ticks backpack activation cooldowns so swapped items eventually
/// become usable.
pub fn tick_cooldowns(state: &mut ItemState, dt: f32) {
    for maybe_active in &mut state.actives {
        if let Some(active) = maybe_active.as_mut() {
            active.remaining_cooldown = (active.remaining_cooldown - dt).max(0.0);
        }
    }

    // Tick backpack activation cooldowns.
    for maybe_item in &mut state.backpack.slots {
        if let Some(item) = maybe_item.as_mut() {
            item.activation_cooldown = (item.activation_cooldown - dt).max(0.0);
        }
    }
}

/// Advance charge recharge timers by `dt` seconds, granting charges when ready.
pub fn tick_charges(state: &mut ItemState, dt: f32) {
    for maybe_charges in &mut state.charges {
        if let Some(charges) = maybe_charges.as_mut() {
            if charges.recharge_time <= 0.0 || charges.current >= charges.max {
                continue;
            }

            charges.recharge_timer += dt;

            // Grant charges for each full recharge period elapsed.
            while charges.recharge_timer >= charges.recharge_time && charges.current < charges.max {
                charges.recharge_timer -= charges.recharge_time;
                charges.current += 1;
            }

            // If at max, reset timer so we don't accumulate drift.
            if charges.current >= charges.max {
                charges.recharge_timer = 0.0;
            }
        }
    }
}

/// Consume one charge from the item in the given slot.
pub fn consume_charge(state: &mut ItemState, slot: usize) -> Result<(), ItemError> {
    let charges = state
        .charges
        .get_mut(slot)
        .ok_or(ItemError::SlotOutOfBounds)?
        .as_mut()
        .ok_or(ItemError::NoCharges)?;

    if charges.current == 0 {
        return Err(ItemError::ChargesDepleted);
    }

    charges.current -= 1;
    Ok(())
}

/// Swap an item between a main inventory slot and a backpack slot.
///
/// The main-slot item (active + charges) moves into the backpack.
/// The backpack item (if any) moves into the main slot with an activation
/// cooldown — its active ability cannot be used until the cooldown expires.
///
/// The caller is responsible for also swapping the actual `Inventory` /
/// `ItemStack` entries at the inventory layer.
pub fn swap_to_backpack(
    state: &mut ItemState,
    main_slot: usize,
    backpack_slot: usize,
) -> Result<(), ItemError> {
    if main_slot >= state.actives.len() {
        return Err(ItemError::SlotOutOfBounds);
    }
    if backpack_slot >= 3 {
        return Err(ItemError::BackpackSlotOutOfBounds);
    }

    let swap_cooldown = state.backpack.swap_cooldown;

    // Take the active-item state from the main slot.
    let main_active = state.actives[main_slot].take();
    let main_charges = if main_slot < state.charges.len() {
        state.charges[main_slot].take()
    } else {
        None
    };

    // Take the backpack item (if any).
    let from_backpack = state.backpack.slots[backpack_slot].take();

    // Move the main-slot state into the backpack (no activation cooldown
    // needed — items in backpack are simply inactive).
    if main_active.is_some() || main_charges.is_some() {
        state.backpack.slots[backpack_slot] = Some(BackpackItem {
            item_id: 0, // Caller sets this from the Inventory layer.
            active: main_active,
            charges: main_charges,
            activation_cooldown: 0.0,
        });
    }

    // Move the backpack item into the main slot with activation cooldown.
    if let Some(bp_item) = from_backpack {
        // Restore the active ability, but impose activation cooldown.
        if let Some(mut active) = bp_item.active {
            // The activation cooldown prevents use — model it as extra
            // remaining cooldown (the *higher* of existing CD and swap CD).
            active.remaining_cooldown = active.remaining_cooldown.max(swap_cooldown);
            state.actives[main_slot] = Some(active);
        }

        // Restore charges.
        if main_slot < state.charges.len() {
            state.charges[main_slot] = bp_item.charges;
        }
    }

    Ok(())
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create an `ItemState` with `n` main slots, all empty.
    fn empty_state(n: usize) -> ItemState {
        ItemState {
            actives: vec![None; n],
            charges: vec![None; n],
            backpack: Backpack::default(),
            neutral_slot: NeutralItemSlot::default(),
        }
    }

    /// Helper: create a basic active ability.
    fn basic_active(name: &str, cooldown: f32, mana_cost: f32) -> ItemActive {
        ItemActive {
            name: name.to_string(),
            cooldown,
            remaining_cooldown: 0.0,
            mana_cost,
            effect_id: format!("effect_{name}"),
            cast_range: 0.0,
            usable_while_muted: false,
            cooldown_group: None,
        }
    }

    // ── 1. test_use_active_success ──

    #[test]
    fn test_use_active_success() {
        let mut state = empty_state(6);
        state.actives[0] = Some(basic_active("blink", 12.0, 0.0));

        let result = use_item_active(&mut state, 0, 100.0, false);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "effect_blink");

        // Cooldown should now be set.
        let active = state.actives[0].as_ref().unwrap();
        assert_eq!(active.remaining_cooldown, 12.0);
    }

    // ── 2. test_use_active_on_cooldown ──

    #[test]
    fn test_use_active_on_cooldown() {
        let mut state = empty_state(6);
        let mut active = basic_active("blink", 12.0, 0.0);
        active.remaining_cooldown = 5.0;
        state.actives[0] = Some(active);

        let result = use_item_active(&mut state, 0, 100.0, false);
        assert_eq!(result, Err(ItemError::OnCooldown));
    }

    // ── 3. test_use_active_no_mana ──

    #[test]
    fn test_use_active_no_mana() {
        let mut state = empty_state(6);
        state.actives[0] = Some(basic_active("bkb", 75.0, 50.0));

        let result = use_item_active(&mut state, 0, 30.0, false);
        assert_eq!(result, Err(ItemError::InsufficientMana));
    }

    // ── 4. test_tick_cooldowns ──

    #[test]
    fn test_tick_cooldowns() {
        let mut state = empty_state(6);
        let mut active = basic_active("blink", 12.0, 0.0);
        active.remaining_cooldown = 10.0;
        state.actives[0] = Some(active);

        tick_cooldowns(&mut state, 3.0);
        assert!((state.actives[0].as_ref().unwrap().remaining_cooldown - 7.0).abs() < f32::EPSILON);

        // Tick past zero — should clamp at 0.
        tick_cooldowns(&mut state, 10.0);
        assert_eq!(state.actives[0].as_ref().unwrap().remaining_cooldown, 0.0);
    }

    // ── 5. test_charge_consume ──

    #[test]
    fn test_charge_consume() {
        let mut state = empty_state(6);
        state.charges[0] = Some(ItemCharges {
            current: 3,
            max: 5,
            recharge_time: 0.0,
            recharge_timer: 0.0,
        });

        assert!(consume_charge(&mut state, 0).is_ok());
        assert_eq!(state.charges[0].as_ref().unwrap().current, 2);

        assert!(consume_charge(&mut state, 0).is_ok());
        assert!(consume_charge(&mut state, 0).is_ok());
        // Now at 0 charges.
        assert_eq!(
            consume_charge(&mut state, 0),
            Err(ItemError::ChargesDepleted)
        );
    }

    // ── 6. test_charge_recharge ──

    #[test]
    fn test_charge_recharge() {
        let mut state = empty_state(6);
        state.charges[0] = Some(ItemCharges {
            current: 1,
            max: 3,
            recharge_time: 10.0,
            recharge_timer: 0.0,
        });

        // 10 seconds = 1 charge gained.
        tick_charges(&mut state, 10.0);
        assert_eq!(state.charges[0].as_ref().unwrap().current, 2);

        // 20 more seconds = 1 more charge (capped at max 3).
        tick_charges(&mut state, 20.0);
        assert_eq!(state.charges[0].as_ref().unwrap().current, 3);

        // At max — no further gains and timer resets.
        tick_charges(&mut state, 100.0);
        assert_eq!(state.charges[0].as_ref().unwrap().current, 3);
        assert_eq!(state.charges[0].as_ref().unwrap().recharge_timer, 0.0);
    }

    // ── 7. test_backpack_swap ──

    #[test]
    fn test_backpack_swap() {
        let mut state = empty_state(6);
        state.actives[0] = Some(basic_active("blink", 12.0, 0.0));
        state.backpack.slots[0] = Some(BackpackItem {
            item_id: 42,
            active: Some(basic_active("force_staff", 20.0, 25.0)),
            charges: None,
            activation_cooldown: 0.0,
        });

        let result = swap_to_backpack(&mut state, 0, 0);
        assert!(result.is_ok());

        // The blink dagger should now be in the backpack.
        let bp = state.backpack.slots[0].as_ref().unwrap();
        assert_eq!(bp.active.as_ref().unwrap().name, "blink");

        // The force staff should now be in the main slot with activation cooldown.
        let main = state.actives[0].as_ref().unwrap();
        assert_eq!(main.name, "force_staff");
        assert_eq!(main.remaining_cooldown, 6.0); // swap_cooldown
    }

    // ── 8. test_backpack_activation_delay ──

    #[test]
    fn test_backpack_activation_delay() {
        let mut state = empty_state(6);

        // Swap a backpack item in — it gets activation cooldown.
        state.backpack.slots[0] = Some(BackpackItem {
            item_id: 10,
            active: Some(basic_active("pipe", 60.0, 0.0)),
            charges: None,
            activation_cooldown: 0.0,
        });

        swap_to_backpack(&mut state, 0, 0).unwrap();

        // Item should be in main slot with activation cooldown preventing use.
        assert!(!can_use_active(&state, 0, 100.0, false));
        assert_eq!(state.actives[0].as_ref().unwrap().remaining_cooldown, 6.0);

        // Tick down the cooldown.
        tick_cooldowns(&mut state, 6.0);
        assert!(can_use_active(&state, 0, 100.0, false));
    }

    // ── 9. test_neutral_slot ──

    #[test]
    fn test_neutral_slot() {
        let mut state = empty_state(6);

        // Equip a neutral item.
        assert!(state.neutral_slot.item.is_none());
        state.neutral_slot.item = Some(99);
        assert_eq!(state.neutral_slot.item, Some(99));

        // Unequip.
        state.neutral_slot.item = None;
        assert!(state.neutral_slot.item.is_none());
    }

    // ── 10. test_cooldown_group ──

    #[test]
    fn test_cooldown_group() {
        let mut state = empty_state(6);

        // Two items in the same cooldown group.
        let mut blink1 = basic_active("blink_dagger", 12.0, 0.0);
        blink1.cooldown_group = Some(CooldownGroup(1));
        let mut blink2 = basic_active("overwhelming_blink", 15.0, 0.0);
        blink2.cooldown_group = Some(CooldownGroup(1));

        // A third item in a different group.
        let mut unrelated = basic_active("bkb", 75.0, 0.0);
        unrelated.cooldown_group = Some(CooldownGroup(2));

        state.actives[0] = Some(blink1);
        state.actives[1] = Some(blink2);
        state.actives[2] = Some(unrelated);

        // Use blink_dagger (slot 0).
        let result = use_item_active(&mut state, 0, 100.0, false);
        assert!(result.is_ok());

        // Slot 0: on its own cooldown (12s).
        assert_eq!(state.actives[0].as_ref().unwrap().remaining_cooldown, 12.0);
        // Slot 1: also on cooldown (its own duration, 15s) because same group.
        assert_eq!(state.actives[1].as_ref().unwrap().remaining_cooldown, 15.0);
        // Slot 2: unaffected — different group.
        assert_eq!(state.actives[2].as_ref().unwrap().remaining_cooldown, 0.0);
    }

    // ── 11. test_muted_blocks_active ──

    #[test]
    fn test_muted_blocks_active() {
        let mut state = empty_state(6);
        state.actives[0] = Some(basic_active("bkb", 75.0, 0.0));

        // Muted — should fail.
        let result = use_item_active(&mut state, 0, 100.0, true);
        assert_eq!(result, Err(ItemError::Muted));

        // can_use_active agrees.
        assert!(!can_use_active(&state, 0, 100.0, true));

        // But if the item is usable while muted, it should work.
        state.actives[0].as_mut().unwrap().usable_while_muted = true;
        assert!(can_use_active(&state, 0, 100.0, true));
        let result = use_item_active(&mut state, 0, 100.0, true);
        assert!(result.is_ok());
    }

    // ── 12. test_empty_slot_error ──

    #[test]
    fn test_empty_slot_error() {
        let mut state = empty_state(6);

        // All slots are empty.
        assert_eq!(
            use_item_active(&mut state, 0, 100.0, false),
            Err(ItemError::EmptySlot)
        );

        // Out of bounds slot.
        assert_eq!(
            use_item_active(&mut state, 99, 100.0, false),
            Err(ItemError::SlotOutOfBounds)
        );

        // Consume charge on empty slot.
        assert_eq!(consume_charge(&mut state, 0), Err(ItemError::NoCharges));
        assert_eq!(
            consume_charge(&mut state, 99),
            Err(ItemError::SlotOutOfBounds)
        );
    }

    // ── Extra: comprehensive can_use_active ──

    #[test]
    fn test_can_use_active_comprehensive() {
        let mut state = empty_state(6);
        state.actives[0] = Some(basic_active("blink", 12.0, 50.0));

        // Ready with enough mana.
        assert!(can_use_active(&state, 0, 100.0, false));

        // Not enough mana.
        assert!(!can_use_active(&state, 0, 30.0, false));

        // On cooldown.
        state.actives[0].as_mut().unwrap().remaining_cooldown = 5.0;
        assert!(!can_use_active(&state, 0, 100.0, false));

        // Empty slot.
        assert!(!can_use_active(&state, 3, 100.0, false));

        // Out of bounds.
        assert!(!can_use_active(&state, 99, 100.0, false));
    }

    // ── Extra: tick cooldowns across multiple slots ──

    #[test]
    fn test_tick_cooldowns_multiple_slots() {
        let mut state = empty_state(6);

        let mut a1 = basic_active("item_a", 10.0, 0.0);
        a1.remaining_cooldown = 5.0;
        let mut a2 = basic_active("item_b", 20.0, 0.0);
        a2.remaining_cooldown = 3.0;
        state.actives[0] = Some(a1);
        state.actives[2] = Some(a2);

        tick_cooldowns(&mut state, 2.0);
        assert!((state.actives[0].as_ref().unwrap().remaining_cooldown - 3.0).abs() < f32::EPSILON);
        assert!((state.actives[2].as_ref().unwrap().remaining_cooldown - 1.0).abs() < f32::EPSILON);
    }

    // ── Extra: backpack out of bounds ──

    #[test]
    fn test_backpack_out_of_bounds() {
        let mut state = empty_state(6);
        assert_eq!(
            swap_to_backpack(&mut state, 0, 5),
            Err(ItemError::BackpackSlotOutOfBounds)
        );
        assert_eq!(
            swap_to_backpack(&mut state, 99, 0),
            Err(ItemError::SlotOutOfBounds)
        );
    }

    // ── Extra: charges with no auto-recharge ──

    #[test]
    fn test_charge_no_auto_recharge() {
        let mut state = empty_state(6);
        state.charges[0] = Some(ItemCharges {
            current: 1,
            max: 5,
            recharge_time: 0.0,
            recharge_timer: 0.0,
        });

        tick_charges(&mut state, 100.0);
        assert_eq!(state.charges[0].as_ref().unwrap().current, 1);
    }

    // ── Extra: backpack activation cooldown ticks down ──

    #[test]
    fn test_backpack_item_activation_cooldown_ticks() {
        let mut state = empty_state(6);

        state.backpack.slots[1] = Some(BackpackItem {
            item_id: 10,
            active: None,
            charges: None,
            activation_cooldown: 6.0,
        });

        tick_cooldowns(&mut state, 2.0);
        assert!(
            (state.backpack.slots[1]
                .as_ref()
                .unwrap()
                .activation_cooldown
                - 4.0)
                .abs()
                < f32::EPSILON
        );

        tick_cooldowns(&mut state, 5.0);
        assert_eq!(
            state.backpack.slots[1]
                .as_ref()
                .unwrap()
                .activation_cooldown,
            0.0
        );
    }
}
