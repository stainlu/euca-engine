//! Shop system — buy and sell items using gold.
//!
//! Functions: `buy_item`, `sell_item`.
//! Resources: `RecipeRegistry`.
//!
//! Connects gold economy with inventory — the missing economy loop for a
//! MOBA-style game. Triggered by player action, not per-tick.

use std::collections::HashMap;

use euca_ecs::{Entity, World};

use euca_gameplay::economy::Gold;
use euca_gameplay::health::Dead;
use euca_gameplay::inventory::{self, Inventory, ItemRegistry};

// ── Error ──

/// Errors that can occur during shop transactions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShopError {
    /// Entity is dead and cannot use the shop.
    NotAlive,
    /// Entity does not have enough gold for this purchase.
    InsufficientGold,
    /// Entity's inventory is full.
    InventoryFull,
    /// The requested item does not exist in the registry.
    ItemNotFound,
    /// Entity is missing one or more recipe component items.
    MissingRecipeComponents,
    /// Entity does not have a Gold component.
    NoGold,
    /// Entity does not have an Inventory component.
    NoInventory,
}

impl std::fmt::Display for ShopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAlive => write!(f, "entity is dead"),
            Self::InsufficientGold => write!(f, "insufficient gold"),
            Self::InventoryFull => write!(f, "inventory is full"),
            Self::ItemNotFound => write!(f, "item not found in registry"),
            Self::MissingRecipeComponents => write!(f, "missing recipe component items"),
            Self::NoGold => write!(f, "entity has no Gold component"),
            Self::NoInventory => write!(f, "entity has no Inventory component"),
        }
    }
}

impl std::error::Error for ShopError {}

// ── Recipe registry ──

/// Definition of a crafting recipe: combine component items into a result item.
#[derive(Clone, Debug)]
pub struct RecipeDef {
    /// Item ID produced by this recipe.
    pub result_id: u32,
    /// Item IDs that must be consumed from inventory.
    pub components: Vec<u32>,
    /// Additional gold cost beyond component item costs.
    pub extra_cost: i32,
}

/// Registry of all crafting recipes, keyed by result item ID.
///
/// Stored as a World resource.
#[derive(Clone, Debug, Default)]
pub struct RecipeRegistry {
    pub recipes: HashMap<u32, RecipeDef>,
}

impl RecipeRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a recipe definition.
    pub fn register(&mut self, def: RecipeDef) {
        self.recipes.insert(def.result_id, def);
    }

    /// Look up a recipe by the result item ID.
    pub fn get(&self, result_id: u32) -> Option<&RecipeDef> {
        self.recipes.get(&result_id)
    }
}

// ── Shop operations ──

/// Get the gold cost of an item from its properties.
fn item_cost(registry: &ItemRegistry, item_id: u32) -> Option<i32> {
    registry
        .get(item_id)
        .and_then(|def| def.properties.get("cost"))
        .map(|c| *c as i32)
}

/// Buy an item from the shop.
///
/// If a recipe exists for the item, the entity must own all component items
/// (which are consumed) and pay only the `extra_cost`. Otherwise, the full
/// item cost is deducted from gold.
pub fn buy_item(world: &mut World, entity: Entity, item_id: u32) -> Result<(), ShopError> {
    // 1. Check entity is alive.
    if world.get::<Dead>(entity).is_some() {
        return Err(ShopError::NotAlive);
    }

    // 2. Look up item in registry.
    let item_registry = world
        .resource::<ItemRegistry>()
        .cloned()
        .ok_or(ShopError::ItemNotFound)?;
    let cost = item_cost(&item_registry, item_id).ok_or(ShopError::ItemNotFound)?;

    // 3. Check recipe path.
    let recipe = world
        .resource::<RecipeRegistry>()
        .and_then(|reg| reg.get(item_id).cloned());

    let gold_needed;
    let components_to_consume;

    if let Some(ref recipe) = recipe {
        // Recipe path: verify entity owns all component items.
        let inv = world
            .get::<Inventory>(entity)
            .ok_or(ShopError::NoInventory)?;
        for &comp_id in &recipe.components {
            if inventory::find_item(inv, comp_id).is_none() {
                return Err(ShopError::MissingRecipeComponents);
            }
        }
        gold_needed = recipe.extra_cost;
        components_to_consume = recipe.components.clone();
    } else {
        // Direct purchase: pay full cost.
        gold_needed = cost;
        components_to_consume = Vec::new();
    };

    // 4. Check gold.
    let current_gold = world.get::<Gold>(entity).ok_or(ShopError::NoGold)?.0;
    if current_gold < gold_needed {
        return Err(ShopError::InsufficientGold);
    }

    // 5. Check inventory space (recipe consumes items so space may open up,
    //    but we need at least one slot after removals — check conservatively
    //    by seeing if removal frees a slot or one already exists).
    {
        let inv = world
            .get::<Inventory>(entity)
            .ok_or(ShopError::NoInventory)?;
        if !inventory::has_space(inv) && components_to_consume.is_empty() {
            return Err(ShopError::InventoryFull);
        }
        // If recipe has components, removing them will free slots, so we are
        // guaranteed space for the result item.
    }

    // 6. Execute transaction: remove components.
    if !components_to_consume.is_empty() {
        let inv = world
            .get_mut::<Inventory>(entity)
            .ok_or(ShopError::NoInventory)?;
        for comp_id in &components_to_consume {
            inventory::remove_item(inv, *comp_id, 1);
        }
    }

    // 7. Deduct gold.
    world.get_mut::<Gold>(entity).ok_or(ShopError::NoGold)?.0 -= gold_needed;

    // 8. Add result item to inventory.
    let inv = world
        .get_mut::<Inventory>(entity)
        .ok_or(ShopError::NoInventory)?;
    let leftover = inventory::add_item(inv, item_id, 1);
    if leftover > 0 {
        // Should not happen given the space check above, but handle gracefully.
        return Err(ShopError::InventoryFull);
    }

    Ok(())
}

/// Sell an item from inventory, receiving 50% of its cost (rounded down).
pub fn sell_item(world: &mut World, entity: Entity, item_id: u32) -> Result<(), ShopError> {
    // 1. Look up item cost.
    let item_registry = world
        .resource::<ItemRegistry>()
        .cloned()
        .ok_or(ShopError::ItemNotFound)?;
    let cost = item_cost(&item_registry, item_id).ok_or(ShopError::ItemNotFound)?;

    // 2. Remove item from inventory.
    let inv = world
        .get_mut::<Inventory>(entity)
        .ok_or(ShopError::NoInventory)?;
    let leftover = inventory::remove_item(inv, item_id, 1);
    if leftover > 0 {
        return Err(ShopError::ItemNotFound);
    }

    // 3. Credit 50% gold (rounded down).
    let refund = cost / 2;
    world.get_mut::<Gold>(entity).ok_or(ShopError::NoGold)?.0 += refund;

    Ok(())
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use euca_gameplay::inventory::ItemDef;

    fn setup_world() -> (World, Entity) {
        let mut world = World::new();

        // Register items.
        let mut registry = ItemRegistry::new();
        registry.register(ItemDef {
            id: 1,
            name: "Broadsword".into(),
            properties: [("cost".into(), 1200.0), ("damage".into(), 40.0)]
                .into_iter()
                .collect(),
        });
        registry.register(ItemDef {
            id: 2,
            name: "Claymore".into(),
            properties: [("cost".into(), 1400.0), ("damage".into(), 20.0)]
                .into_iter()
                .collect(),
        });
        registry.register(ItemDef {
            id: 3,
            name: "Demon Edge".into(),
            properties: [("cost".into(), 2200.0), ("damage".into(), 42.0)]
                .into_iter()
                .collect(),
        });
        // Recipe item: Daedalus = Broadsword + Claymore + Demon Edge + 1000 extra gold
        registry.register(ItemDef {
            id: 100,
            name: "Daedalus".into(),
            properties: [("cost".into(), 5800.0), ("damage".into(), 88.0)]
                .into_iter()
                .collect(),
        });
        world.insert_resource(registry);

        // Register recipe.
        let mut recipes = RecipeRegistry::new();
        recipes.register(RecipeDef {
            result_id: 100,
            components: vec![1, 2, 3],
            extra_cost: 1000,
        });
        world.insert_resource(recipes);

        // Create hero entity with gold and inventory.
        let hero = world.spawn(Gold::new(3000));
        world.insert(hero, Inventory::new(6));

        (world, hero)
    }

    #[test]
    fn buy_item_success() {
        let (mut world, hero) = setup_world();

        // Buy Broadsword (cost 1200).
        let result = buy_item(&mut world, hero, 1);
        assert!(result.is_ok());

        assert_eq!(world.get::<Gold>(hero).unwrap().0, 1800);
        assert!(inventory::find_item(world.get::<Inventory>(hero).unwrap(), 1).is_some());
    }

    #[test]
    fn buy_item_insufficient_gold() {
        let (mut world, hero) = setup_world();
        world.get_mut::<Gold>(hero).unwrap().0 = 100;

        let result = buy_item(&mut world, hero, 1);
        assert_eq!(result, Err(ShopError::InsufficientGold));
    }

    #[test]
    fn buy_item_inventory_full() {
        let (mut world, hero) = setup_world();
        world.get_mut::<Gold>(hero).unwrap().0 = 99999;

        // Fill inventory completely (6 slots with different items).
        let inv = world.get_mut::<Inventory>(hero).unwrap();
        for i in 0..6 {
            inventory::add_item(inv, 200 + i, 1);
        }

        let result = buy_item(&mut world, hero, 1);
        assert_eq!(result, Err(ShopError::InventoryFull));
    }

    #[test]
    fn buy_item_entity_is_dead() {
        let (mut world, hero) = setup_world();
        world.insert(hero, Dead);

        let result = buy_item(&mut world, hero, 1);
        assert_eq!(result, Err(ShopError::NotAlive));
    }

    #[test]
    fn sell_item_returns_half_gold() {
        let (mut world, hero) = setup_world();

        // Give hero a Broadsword.
        let inv = world.get_mut::<Inventory>(hero).unwrap();
        inventory::add_item(inv, 1, 1);

        let starting_gold = world.get::<Gold>(hero).unwrap().0;
        let result = sell_item(&mut world, hero, 1);
        assert!(result.is_ok());

        // Broadsword cost 1200, sell for 600.
        assert_eq!(world.get::<Gold>(hero).unwrap().0, starting_gold + 600);
        assert!(inventory::find_item(world.get::<Inventory>(hero).unwrap(), 1).is_none());
    }

    #[test]
    fn recipe_combine_success() {
        let (mut world, hero) = setup_world();

        // Give hero all recipe components.
        let inv = world.get_mut::<Inventory>(hero).unwrap();
        inventory::add_item(inv, 1, 1); // Broadsword
        inventory::add_item(inv, 2, 1); // Claymore
        inventory::add_item(inv, 3, 1); // Demon Edge

        // Buy Daedalus (recipe: extra_cost = 1000).
        let result = buy_item(&mut world, hero, 100);
        assert!(result.is_ok());

        // Gold deducted by extra_cost only.
        assert_eq!(world.get::<Gold>(hero).unwrap().0, 2000);

        // Component items consumed.
        let inv = world.get::<Inventory>(hero).unwrap();
        assert!(inventory::find_item(inv, 1).is_none());
        assert!(inventory::find_item(inv, 2).is_none());
        assert!(inventory::find_item(inv, 3).is_none());

        // Result item added.
        assert!(inventory::find_item(inv, 100).is_some());
    }

    #[test]
    fn recipe_missing_components_fails() {
        let (mut world, hero) = setup_world();

        // Give hero only one of three required components.
        let inv = world.get_mut::<Inventory>(hero).unwrap();
        inventory::add_item(inv, 1, 1); // Broadsword only

        let result = buy_item(&mut world, hero, 100);
        assert_eq!(result, Err(ShopError::MissingRecipeComponents));

        // Nothing should have changed.
        assert_eq!(world.get::<Gold>(hero).unwrap().0, 3000);
        assert!(inventory::find_item(world.get::<Inventory>(hero).unwrap(), 1).is_some());
    }
}
