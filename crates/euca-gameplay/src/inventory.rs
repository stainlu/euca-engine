//! Data-driven inventory and equipment system.
//!
//! Components: `Inventory`, `Equipment`, `StatModifiers`.
//! Resources: `ItemRegistry`.
//! Systems: `equipment_stat_system`.
//!
//! Items are defined entirely through data — no hardcoded item types. Each
//! `ItemDef` carries a `properties` map with arbitrary key-value pairs
//! (e.g. `"damage": 50.0`, `"armor": 10.0`). Games decide their own keys.
//!
//! Equipment slots are arbitrary strings (`"weapon"`, `"head"`, `"ring1"`) —
//! the engine imposes no schema.

use std::collections::HashMap;

use euca_ecs::World;
use serde::{Deserialize, Serialize};

// ── Data types ──

/// Definition of an item type, loaded from data files.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ItemDef {
    /// Unique item identifier.
    pub id: u32,
    /// Human-readable name.
    pub name: String,
    /// Arbitrary numeric properties — games define their own keys.
    #[serde(default)]
    pub properties: HashMap<String, f64>,
}

/// A stack of identical items in an inventory slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ItemStack {
    pub item_id: u32,
    pub count: u32,
}

// ── Components ──

/// Slotted inventory: a fixed-size bag of item stacks.
#[derive(Clone, Debug)]
pub struct Inventory {
    pub slots: Vec<Option<ItemStack>>,
    pub max_slots: u16,
}

impl Inventory {
    pub fn new(max_slots: u16) -> Self {
        Self {
            slots: vec![None; max_slots as usize],
            max_slots,
        }
    }
}

/// Equipment: maps named slots to item IDs.
///
/// Slot names are arbitrary strings — the game defines its own schema
/// (e.g. `"weapon"`, `"head"`, `"ring1"`).
#[derive(Clone, Debug, Default)]
pub struct Equipment {
    pub equipped: HashMap<String, u32>,
}

/// Aggregated stat modifiers from equipped items.
///
/// Other systems read this to adjust behavior (damage, move speed, etc.).
/// Written by `equipment_stat_system`.
#[derive(Clone, Debug, Default)]
pub struct StatModifiers {
    pub values: HashMap<String, f64>,
}

// ── Resources ──

/// Registry of all item definitions, keyed by item ID.
///
/// Stored as a World resource. Typically loaded from JSON data files.
#[derive(Clone, Debug, Default)]
pub struct ItemRegistry {
    pub items: HashMap<u32, ItemDef>,
}

impl ItemRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an item definition.
    pub fn register(&mut self, def: ItemDef) {
        self.items.insert(def.id, def);
    }

    /// Look up an item definition by ID.
    pub fn get(&self, id: u32) -> Option<&ItemDef> {
        self.items.get(&id)
    }
}

// ── Inventory operations ──

/// Add items to an inventory, stacking where possible.
///
/// Returns the number of items that could not be added (0 = all fit).
pub fn add_item(inventory: &mut Inventory, item_id: u32, count: u32) -> u32 {
    // First pass: stack onto existing stacks of the same item.
    for slot in inventory.slots.iter_mut() {
        if count == 0 {
            return 0;
        }
        if let Some(stack) = slot
            && stack.item_id == item_id
        {
            stack.count += count;
            return 0;
        }
    }

    // Second pass: fill empty slots.
    for slot in inventory.slots.iter_mut() {
        if count == 0 {
            return 0;
        }
        if slot.is_none() {
            *slot = Some(ItemStack { item_id, count });
            return 0;
        }
    }

    count
}

/// Remove items from an inventory.
///
/// Returns the number of items that could not be removed (0 = all removed).
pub fn remove_item(inventory: &mut Inventory, item_id: u32, mut count: u32) -> u32 {
    for slot in inventory.slots.iter_mut() {
        if count == 0 {
            return 0;
        }
        if let Some(stack) = slot
            && stack.item_id == item_id
        {
            if stack.count <= count {
                count -= stack.count;
                *slot = None;
            } else {
                stack.count -= count;
                return 0;
            }
        }
    }
    count
}

/// Check whether the inventory has room for at least one more item.
pub fn has_space(inventory: &Inventory) -> bool {
    inventory.slots.iter().any(|s| s.is_none())
}

/// Find the first slot index containing the given item ID.
pub fn find_item(inventory: &Inventory, item_id: u32) -> Option<usize> {
    inventory
        .slots
        .iter()
        .position(|s| s.is_some_and(|stack| stack.item_id == item_id))
}

/// Equip an item into a named slot, removing it from inventory.
///
/// Returns `true` if successful (item was in inventory and was equipped).
pub fn equip(
    inventory: &mut Inventory,
    equipment: &mut Equipment,
    slot: &str,
    item_id: u32,
) -> bool {
    let idx = match find_item(inventory, item_id) {
        Some(i) => i,
        None => return false,
    };

    // Remove one from inventory.
    if let Some(stack) = &mut inventory.slots[idx] {
        if stack.count <= 1 {
            inventory.slots[idx] = None;
        } else {
            stack.count -= 1;
        }
    }

    // If something was already in that slot, return it to inventory.
    if let Some(old_id) = equipment.equipped.insert(slot.to_string(), item_id) {
        add_item(inventory, old_id, 1);
    }

    true
}

/// Unequip an item from a named slot, returning it to inventory.
///
/// Returns `true` if something was unequipped.
pub fn unequip(inventory: &mut Inventory, equipment: &mut Equipment, slot: &str) -> bool {
    match equipment.equipped.remove(slot) {
        Some(item_id) => {
            add_item(inventory, item_id, 1);
            true
        }
        None => false,
    }
}

// ── System ──

/// Recompute `StatModifiers` from equipped items each tick.
///
/// For every entity with `Equipment` and `StatModifiers`, sums the
/// `properties` of all equipped items from the `ItemRegistry`.
pub fn equipment_stat_system(world: &mut World) {
    let registry: Option<ItemRegistry> = world.resource::<ItemRegistry>().cloned();
    let registry = match registry {
        Some(r) => r,
        None => return,
    };

    let entities: Vec<euca_ecs::Entity> = {
        let query = euca_ecs::Query::<(euca_ecs::Entity, &Equipment)>::new(world);
        query.iter().map(|(e, _)| e).collect()
    };

    for entity in entities {
        let equipped = match world.get::<Equipment>(entity) {
            Some(eq) => eq.equipped.clone(),
            None => continue,
        };

        let mut modifiers = HashMap::new();
        for item_id in equipped.values() {
            if let Some(def) = registry.get(*item_id) {
                for (key, value) in &def.properties {
                    *modifiers.entry(key.clone()).or_insert(0.0) += value;
                }
            }
        }

        if let Some(stats) = world.get_mut::<StatModifiers>(entity) {
            stats.values = modifiers;
        }
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn make_registry() -> ItemRegistry {
        let mut reg = ItemRegistry::new();
        reg.register(ItemDef {
            id: 1,
            name: "Sword".into(),
            properties: [("damage".into(), 50.0), ("speed".into(), -5.0)]
                .into_iter()
                .collect(),
        });
        reg.register(ItemDef {
            id: 2,
            name: "Shield".into(),
            properties: [("armor".into(), 30.0)].into_iter().collect(),
        });
        reg.register(ItemDef {
            id: 3,
            name: "Potion".into(),
            properties: HashMap::new(),
        });
        reg
    }

    #[test]
    fn add_and_find_item() {
        let mut inv = Inventory::new(4);
        let leftover = add_item(&mut inv, 1, 3);
        assert_eq!(leftover, 0);
        assert_eq!(find_item(&inv, 1), Some(0));
        assert_eq!(inv.slots[0].unwrap().count, 3);
    }

    #[test]
    fn add_stacks_onto_existing() {
        let mut inv = Inventory::new(4);
        add_item(&mut inv, 1, 2);
        add_item(&mut inv, 1, 3);
        // Should stack, not use a second slot.
        assert_eq!(inv.slots[0].unwrap().count, 5);
        assert!(inv.slots[1].is_none());
    }

    #[test]
    fn add_item_no_space() {
        let mut inv = Inventory::new(1);
        add_item(&mut inv, 1, 1);
        let leftover = add_item(&mut inv, 2, 5);
        assert_eq!(leftover, 5);
    }

    #[test]
    fn remove_partial_stack() {
        let mut inv = Inventory::new(4);
        add_item(&mut inv, 1, 10);
        let leftover = remove_item(&mut inv, 1, 3);
        assert_eq!(leftover, 0);
        assert_eq!(inv.slots[0].unwrap().count, 7);
    }

    #[test]
    fn remove_entire_stack() {
        let mut inv = Inventory::new(4);
        add_item(&mut inv, 1, 5);
        let leftover = remove_item(&mut inv, 1, 5);
        assert_eq!(leftover, 0);
        assert!(inv.slots[0].is_none());
    }

    #[test]
    fn remove_more_than_available() {
        let mut inv = Inventory::new(4);
        add_item(&mut inv, 1, 3);
        let leftover = remove_item(&mut inv, 1, 10);
        assert_eq!(leftover, 7);
        assert!(inv.slots[0].is_none());
    }

    #[test]
    fn has_space_empty_and_full() {
        let mut inv = Inventory::new(1);
        assert!(has_space(&inv));
        add_item(&mut inv, 1, 1);
        assert!(!has_space(&inv));
    }

    #[test]
    fn equip_and_unequip() {
        let mut inv = Inventory::new(4);
        let mut eq = Equipment::default();

        add_item(&mut inv, 1, 1);
        assert!(equip(&mut inv, &mut eq, "weapon", 1));
        assert_eq!(eq.equipped.get("weapon"), Some(&1));
        assert!(find_item(&inv, 1).is_none()); // removed from inventory

        assert!(unequip(&mut inv, &mut eq, "weapon"));
        assert!(eq.equipped.get("weapon").is_none());
        assert!(find_item(&inv, 1).is_some()); // returned to inventory
    }

    #[test]
    fn equip_replaces_old_item() {
        let mut inv = Inventory::new(4);
        let mut eq = Equipment::default();

        add_item(&mut inv, 1, 1);
        add_item(&mut inv, 2, 1);

        equip(&mut inv, &mut eq, "weapon", 1);
        equip(&mut inv, &mut eq, "weapon", 2);

        // Sword (1) should be back in inventory, Shield (2) equipped.
        assert_eq!(eq.equipped.get("weapon"), Some(&2));
        assert!(find_item(&inv, 1).is_some());
        assert!(find_item(&inv, 2).is_none());
    }

    #[test]
    fn equip_fails_without_item() {
        let mut inv = Inventory::new(4);
        let mut eq = Equipment::default();
        assert!(!equip(&mut inv, &mut eq, "weapon", 99));
    }

    #[test]
    fn equipment_stat_system_aggregates() {
        let mut world = World::new();
        let registry = make_registry();
        world.insert_resource(registry);

        let entity = world.spawn(Equipment {
            equipped: [
                ("weapon".into(), 1),  // Sword: damage=50, speed=-5
                ("offhand".into(), 2), // Shield: armor=30
            ]
            .into_iter()
            .collect(),
        });
        world.insert(entity, StatModifiers::default());

        equipment_stat_system(&mut world);

        let stats = world.get::<StatModifiers>(entity).unwrap();
        assert_eq!(stats.values.get("damage"), Some(&50.0));
        assert_eq!(stats.values.get("speed"), Some(&-5.0));
        assert_eq!(stats.values.get("armor"), Some(&30.0));
    }

    #[test]
    fn equipment_stat_system_no_registry() {
        let mut world = World::new();
        // No ItemRegistry resource — system should not panic.
        let entity = world.spawn(Equipment::default());
        world.insert(entity, StatModifiers::default());
        equipment_stat_system(&mut world);
    }

    #[test]
    fn item_registry_operations() {
        let reg = make_registry();
        assert_eq!(reg.get(1).unwrap().name, "Sword");
        assert_eq!(reg.get(2).unwrap().name, "Shield");
        assert!(reg.get(999).is_none());
    }

    #[test]
    fn capacity_boundary() {
        let mut inv = Inventory::new(0);
        assert!(!has_space(&inv));
        let leftover = add_item(&mut inv, 1, 1);
        assert_eq!(leftover, 1);
    }
}
