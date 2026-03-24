//! Inventory, equipment, and item definitions — fully data-driven.
//!
//! Components: `Inventory`, `Equipment`, `StatModifiers`.
//! Resources: `ItemRegistry`.
//! Systems: `equipment_stat_system`.

use std::collections::HashMap;

use euca_ecs::{Entity, Query, World};

/// A data-driven item definition. Properties are arbitrary key-value pairs
/// (e.g. "damage": 50.0, "armor": 10.0, "cooldown_reduction": 0.1).
#[derive(Clone, Debug)]
pub struct ItemDef {
    pub id: u32,
    pub name: String,
    pub properties: HashMap<String, f64>,
}

impl ItemDef {
    pub fn new(id: u32, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            properties: HashMap::new(),
        }
    }

    /// Builder: set a property on this item definition.
    pub fn with_property(mut self, key: impl Into<String>, value: f64) -> Self {
        self.properties.insert(key.into(), value);
        self
    }
}

/// A stack of identical items in an inventory slot.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemStack {
    pub item_id: u32,
    pub count: u32,
}

/// Entity component: a fixed-size bag of item slots.
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

/// Entity component: equipped items mapped by slot name.
/// Slot names are arbitrary strings ("weapon", "head", "ring1", etc.).
#[derive(Clone, Debug, Default)]
pub struct Equipment {
    pub equipped: HashMap<String, u32>,
}

/// Entity component: stat modifiers derived from equipped items.
/// Recomputed each tick by `equipment_stat_system`.
#[derive(Clone, Debug, Default)]
pub struct StatModifiers {
    pub modifiers: HashMap<String, f64>,
}

/// World resource: global registry of all item definitions.
#[derive(Clone, Debug, Default)]
pub struct ItemRegistry {
    pub items: HashMap<u32, ItemDef>,
}

impl ItemRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an item definition. Overwrites if the id already exists.
    pub fn register(&mut self, item: ItemDef) {
        self.items.insert(item.id, item);
    }

    /// Look up an item definition by id.
    pub fn get(&self, id: u32) -> Option<&ItemDef> {
        self.items.get(&id)
    }
}

// ── Inventory operations ──

/// Returns `true` if the inventory has at least one empty slot.
pub fn has_space(inventory: &Inventory) -> bool {
    inventory.slots.iter().any(|s| s.is_none())
}

/// Find the first slot index containing the given item id, if any.
pub fn find_item(inventory: &Inventory, item_id: u32) -> Option<usize> {
    inventory
        .slots
        .iter()
        .position(|s| matches!(s, Some(stack) if stack.item_id == item_id))
}

/// Add an item to the inventory. Stacks with an existing stack of the same
/// item if one exists, otherwise uses the first empty slot.
/// Returns `true` if the item was added, `false` if the inventory is full.
pub fn add_item(world: &mut World, entity: Entity, item_id: u32, count: u32) -> bool {
    let inventory = match world.get_mut::<Inventory>(entity) {
        Some(inv) => inv,
        None => return false,
    };

    // Try to stack with an existing slot
    if let Some(idx) = find_item(inventory, item_id) {
        inventory.slots[idx].as_mut().unwrap().count += count;
        return true;
    }

    // Try to use an empty slot
    if let Some(idx) = inventory.slots.iter().position(|s| s.is_none()) {
        inventory.slots[idx] = Some(ItemStack { item_id, count });
        return true;
    }

    false
}

/// Remove `count` of an item from the inventory. If the stack count reaches
/// zero, the slot is cleared. Returns `true` if the removal succeeded.
pub fn remove_item(world: &mut World, entity: Entity, item_id: u32, count: u32) -> bool {
    let inventory = match world.get_mut::<Inventory>(entity) {
        Some(inv) => inv,
        None => return false,
    };

    let idx = match find_item(inventory, item_id) {
        Some(i) => i,
        None => return false,
    };

    let stack = inventory.slots[idx].as_mut().unwrap();
    if stack.count < count {
        return false;
    }
    stack.count -= count;
    if stack.count == 0 {
        inventory.slots[idx] = None;
    }
    true
}

/// Equip an item into a named slot. The item must exist in the inventory.
/// Moves 1 unit from the inventory into the equipment slot.
/// If the slot already has an item, it is returned to the inventory first.
pub fn equip(world: &mut World, entity: Entity, slot_name: &str, item_id: u32) -> bool {
    // Verify the item is in the inventory
    {
        let inventory = match world.get::<Inventory>(entity) {
            Some(inv) => inv,
            None => return false,
        };
        if find_item(inventory, item_id).is_none() {
            return false;
        }
    }

    // Unequip whatever is currently in that slot (put it back in inventory)
    let prev_item = {
        let equipment = match world.get::<Equipment>(entity) {
            Some(eq) => eq,
            None => return false,
        };
        equipment.equipped.get(slot_name).copied()
    };

    if let Some(prev_id) = prev_item {
        // Return previous item to inventory
        if !add_item(world, entity, prev_id, 1) {
            return false; // inventory full, can't swap
        }
        // Remove from equipment
        if let Some(equipment) = world.get_mut::<Equipment>(entity) {
            equipment.equipped.remove(slot_name);
        }
    }

    // Remove item from inventory
    if !remove_item(world, entity, item_id, 1) {
        return false;
    }

    // Place in equipment slot
    if let Some(equipment) = world.get_mut::<Equipment>(entity) {
        equipment.equipped.insert(slot_name.to_string(), item_id);
    }

    true
}

/// Unequip an item from a named slot, returning it to the inventory.
pub fn unequip(world: &mut World, entity: Entity, slot_name: &str) -> bool {
    let item_id = {
        let equipment = match world.get::<Equipment>(entity) {
            Some(eq) => eq,
            None => return false,
        };
        match equipment.equipped.get(slot_name) {
            Some(&id) => id,
            None => return false,
        }
    };

    // Add back to inventory
    if !add_item(world, entity, item_id, 1) {
        return false; // inventory full
    }

    // Remove from equipment
    if let Some(equipment) = world.get_mut::<Equipment>(entity) {
        equipment.equipped.remove(slot_name);
    }

    true
}

// ── Systems ──

/// Iterate all entities with `Equipment`, look up each equipped item's
/// properties in the `ItemRegistry`, and write the aggregated stat totals
/// into a `StatModifiers` component.
pub fn equipment_stat_system(world: &mut World) {
    let registry = match world.resource::<ItemRegistry>() {
        Some(r) => r.clone(),
        None => return,
    };

    let updates: Vec<(Entity, HashMap<String, f64>)> = {
        let query = Query::<(Entity, &Equipment)>::new(world);
        query
            .iter()
            .map(|(entity, equipment)| {
                let mut totals: HashMap<String, f64> = HashMap::new();
                for item_id in equipment.equipped.values() {
                    if let Some(def) = registry.get(*item_id) {
                        for (key, value) in &def.properties {
                            *totals.entry(key.clone()).or_default() += value;
                        }
                    }
                }
                (entity, totals)
            })
            .collect()
    };

    for (entity, modifiers) in updates {
        if world.get::<StatModifiers>(entity).is_some() {
            if let Some(sm) = world.get_mut::<StatModifiers>(entity) {
                sm.modifiers = modifiers;
            }
        } else {
            world.insert(entity, StatModifiers { modifiers });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_world() -> (World, Entity) {
        let mut world = World::new();
        let mut registry = ItemRegistry::new();
        registry.register(
            ItemDef::new(1, "Sword")
                .with_property("damage", 50.0)
                .with_property("attack_speed", 1.2),
        );
        registry.register(
            ItemDef::new(2, "Shield")
                .with_property("armor", 30.0)
                .with_property("health", 100.0),
        );
        registry.register(ItemDef::new(3, "Potion").with_property("heal", 75.0));
        world.insert_resource(registry);
        let entity = world.spawn(Inventory::new(4));
        world.insert(entity, Equipment::default());
        world.insert(entity, StatModifiers::default());

        (world, entity)
    }

    #[test]
    fn add_item_to_empty_inventory() {
        let (mut world, entity) = setup_world();
        assert!(add_item(&mut world, entity, 1, 1));

        let inv = world.get::<Inventory>(entity).unwrap();
        assert_eq!(
            inv.slots[0],
            Some(ItemStack {
                item_id: 1,
                count: 1
            })
        );
    }

    #[test]
    fn add_item_stacks() {
        let (mut world, entity) = setup_world();
        add_item(&mut world, entity, 3, 2);
        add_item(&mut world, entity, 3, 3);

        let inv = world.get::<Inventory>(entity).unwrap();
        assert_eq!(
            inv.slots[0],
            Some(ItemStack {
                item_id: 3,
                count: 5
            })
        );
        // Only one slot used
        assert!(inv.slots[1].is_none());
    }

    #[test]
    fn add_item_fails_when_full() {
        let (mut world, entity) = setup_world();
        // Fill all 4 slots with different items
        add_item(&mut world, entity, 1, 1);
        add_item(&mut world, entity, 2, 1);
        add_item(&mut world, entity, 3, 1);
        // Use a fake item_id=99 for the 4th slot
        add_item(&mut world, entity, 99, 1);

        // 5th different item should fail
        assert!(!add_item(&mut world, entity, 100, 1));
    }

    #[test]
    fn remove_item_decrements_count() {
        let (mut world, entity) = setup_world();
        add_item(&mut world, entity, 3, 5);
        assert!(remove_item(&mut world, entity, 3, 2));

        let inv = world.get::<Inventory>(entity).unwrap();
        assert_eq!(
            inv.slots[0],
            Some(ItemStack {
                item_id: 3,
                count: 3
            })
        );
    }

    #[test]
    fn remove_item_clears_slot_at_zero() {
        let (mut world, entity) = setup_world();
        add_item(&mut world, entity, 1, 1);
        assert!(remove_item(&mut world, entity, 1, 1));

        let inv = world.get::<Inventory>(entity).unwrap();
        assert!(inv.slots[0].is_none());
    }

    #[test]
    fn remove_item_fails_insufficient_count() {
        let (mut world, entity) = setup_world();
        add_item(&mut world, entity, 1, 2);
        assert!(!remove_item(&mut world, entity, 1, 5));
    }

    #[test]
    fn remove_item_fails_not_found() {
        let (mut world, entity) = setup_world();
        assert!(!remove_item(&mut world, entity, 99, 1));
    }

    #[test]
    fn equip_moves_from_inventory() {
        let (mut world, entity) = setup_world();
        add_item(&mut world, entity, 1, 1);
        assert!(equip(&mut world, entity, "weapon", 1));

        // Item removed from inventory
        let inv = world.get::<Inventory>(entity).unwrap();
        assert!(find_item(inv, 1).is_none());

        // Item in equipment
        let eq = world.get::<Equipment>(entity).unwrap();
        assert_eq!(eq.equipped.get("weapon"), Some(&1));
    }

    #[test]
    fn equip_swaps_existing() {
        let (mut world, entity) = setup_world();
        add_item(&mut world, entity, 1, 1);
        add_item(&mut world, entity, 2, 1);

        equip(&mut world, entity, "weapon", 1);
        // Now equip item 2 in the same slot — item 1 should return to inventory
        assert!(equip(&mut world, entity, "weapon", 2));

        let eq = world.get::<Equipment>(entity).unwrap();
        assert_eq!(eq.equipped.get("weapon"), Some(&2));

        let inv = world.get::<Inventory>(entity).unwrap();
        assert!(find_item(inv, 1).is_some()); // item 1 returned
    }

    #[test]
    fn equip_fails_without_item() {
        let (mut world, entity) = setup_world();
        assert!(!equip(&mut world, entity, "weapon", 1));
    }

    #[test]
    fn unequip_returns_to_inventory() {
        let (mut world, entity) = setup_world();
        add_item(&mut world, entity, 1, 1);
        equip(&mut world, entity, "weapon", 1);
        assert!(unequip(&mut world, entity, "weapon"));

        let eq = world.get::<Equipment>(entity).unwrap();
        assert!(eq.equipped.get("weapon").is_none());

        let inv = world.get::<Inventory>(entity).unwrap();
        assert!(find_item(inv, 1).is_some());
    }

    #[test]
    fn unequip_fails_empty_slot() {
        let (mut world, entity) = setup_world();
        assert!(!unequip(&mut world, entity, "weapon"));
    }

    #[test]
    fn has_space_works() {
        let (mut world, entity) = setup_world();
        let inv = world.get::<Inventory>(entity).unwrap();
        assert!(has_space(inv));

        // Fill all slots
        add_item(&mut world, entity, 1, 1);
        add_item(&mut world, entity, 2, 1);
        add_item(&mut world, entity, 3, 1);
        add_item(&mut world, entity, 99, 1);

        let inv = world.get::<Inventory>(entity).unwrap();
        assert!(!has_space(inv));
    }

    #[test]
    fn find_item_works() {
        let (mut world, entity) = setup_world();
        add_item(&mut world, entity, 2, 1);

        let inv = world.get::<Inventory>(entity).unwrap();
        assert_eq!(find_item(inv, 2), Some(0));
        assert_eq!(find_item(inv, 99), None);
    }

    #[test]
    fn equipment_stat_system_aggregates_properties() {
        let (mut world, entity) = setup_world();
        add_item(&mut world, entity, 1, 1); // Sword: damage=50, attack_speed=1.2
        add_item(&mut world, entity, 2, 1); // Shield: armor=30, health=100
        equip(&mut world, entity, "weapon", 1);
        equip(&mut world, entity, "offhand", 2);

        equipment_stat_system(&mut world);

        let mods = world.get::<StatModifiers>(entity).unwrap();
        assert_eq!(mods.modifiers.get("damage"), Some(&50.0));
        assert_eq!(mods.modifiers.get("attack_speed"), Some(&1.2));
        assert_eq!(mods.modifiers.get("armor"), Some(&30.0));
        assert_eq!(mods.modifiers.get("health"), Some(&100.0));
    }

    #[test]
    fn equipment_stat_system_clears_on_unequip() {
        let (mut world, entity) = setup_world();
        add_item(&mut world, entity, 1, 1);
        equip(&mut world, entity, "weapon", 1);
        equipment_stat_system(&mut world);

        // Verify stats present
        let mods = world.get::<StatModifiers>(entity).unwrap();
        assert_eq!(mods.modifiers.get("damage"), Some(&50.0));

        // Unequip and re-run system
        unequip(&mut world, entity, "weapon");
        equipment_stat_system(&mut world);

        let mods = world.get::<StatModifiers>(entity).unwrap();
        assert!(mods.modifiers.is_empty());
    }

    #[test]
    fn item_registry_from_json() {
        let json = r#"[
            {"id": 10, "name": "Bow", "properties": {"damage": 35.0, "range": 8.0}},
            {"id": 11, "name": "Arrow", "properties": {"damage": 5.0}}
        ]"#;

        let defs: Vec<serde_json::Value> = serde_json::from_str(json).unwrap();
        let mut registry = ItemRegistry::new();
        for def in defs {
            let id = def["id"].as_u64().unwrap() as u32;
            let name = def["name"].as_str().unwrap().to_string();
            let mut item = ItemDef::new(id, name);
            if let Some(props) = def["properties"].as_object() {
                for (k, v) in props {
                    if let Some(val) = v.as_f64() {
                        item.properties.insert(k.clone(), val);
                    }
                }
            }
            registry.register(item);
        }

        assert_eq!(registry.items.len(), 2);
        assert_eq!(registry.get(10).unwrap().name, "Bow");
        assert_eq!(
            registry.get(10).unwrap().properties.get("range"),
            Some(&8.0)
        );
    }
}
