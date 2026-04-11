use serde::{Deserialize, Serialize};

/// A unique identifier for an entity in the world.
///
/// Uses generational indices to detect stale references:
/// - `index`: slot in the entity array (reused after despawn)
/// - `generation`: incremented each time the slot is recycled
///
/// # Examples
///
/// ```
/// # use euca_ecs::{World, Entity};
/// let mut world = World::new();
/// let entity = world.spawn(42u32);
///
/// assert!(world.is_alive(entity));
/// assert_eq!(entity.generation(), 0);
///
/// world.despawn(entity);
/// assert!(!world.is_alive(entity));
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Entity {
    pub(crate) index: u32,
    pub(crate) generation: u32,
}

impl Entity {
    /// Create an entity from raw parts (for deserialization/testing).
    #[inline]
    pub fn from_raw(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// The slot index.
    #[inline]
    pub fn index(self) -> u32 {
        self.index
    }

    /// The generation (increments each time the slot is reused).
    #[inline]
    pub fn generation(self) -> u32 {
        self.generation
    }
}

impl std::fmt::Display for Entity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}v{}", self.index, self.generation)
    }
}

/// Allocates and recycles entity IDs with generational safety.
#[derive(Clone, Debug)]
pub(crate) struct EntityAllocator {
    /// For each slot: the current generation.
    generations: Vec<u32>,
    /// Free list of available slot indices.
    free_list: Vec<u32>,
    /// Number of currently alive entities.
    alive_count: u32,
}

impl EntityAllocator {
    /// Creates an allocator with no entities.
    pub fn new() -> Self {
        Self {
            generations: Vec::new(),
            free_list: Vec::new(),
            alive_count: 0,
        }
    }

    /// Allocate a new entity ID.
    pub fn allocate(&mut self) -> Entity {
        self.alive_count += 1;
        while let Some(index) = self.free_list.pop() {
            // Skip slots whose generation has reached u32::MAX — they can
            // never be incremented again and must be permanently retired.
            if self.generations[index as usize] == u32::MAX {
                continue;
            }
            return Entity {
                index,
                generation: self.generations[index as usize],
            };
        }
        {
            let index = self.generations.len() as u32;
            self.generations.push(0);
            Entity {
                index,
                generation: 0,
            }
        }
    }

    /// Deallocate an entity, incrementing its generation.
    /// Returns `true` if the entity was valid and is now despawned.
    pub fn deallocate(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        let idx = entity.index as usize;
        self.generations[idx] = self.generations[idx].saturating_add(1);
        // Only recycle the slot if the generation hasn't saturated.
        // Slots at u32::MAX are permanently retired — they can never
        // produce a distinguishable new generation.
        if self.generations[idx] < u32::MAX {
            self.free_list.push(entity.index);
        }
        self.alive_count -= 1;
        true
    }

    /// Check if an entity handle is still valid.
    #[inline]
    pub fn is_alive(&self, entity: Entity) -> bool {
        (entity.index as usize) < self.generations.len()
            && self.generations[entity.index as usize] == entity.generation
    }

    /// Number of currently alive entities.
    #[inline]
    pub fn alive_count(&self) -> u32 {
        self.alive_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_check() {
        let mut alloc = EntityAllocator::new();
        let e1 = alloc.allocate();
        let e2 = alloc.allocate();

        assert_eq!(e1.index(), 0);
        assert_eq!(e2.index(), 1);
        assert_eq!(e1.generation(), 0);
        assert!(alloc.is_alive(e1));
        assert!(alloc.is_alive(e2));
        assert_eq!(alloc.alive_count(), 2);
    }

    #[test]
    fn deallocate_and_reuse() {
        let mut alloc = EntityAllocator::new();
        let e1 = alloc.allocate();
        assert!(alloc.deallocate(e1));
        assert!(!alloc.is_alive(e1));

        // Reuse the slot
        let e2 = alloc.allocate();
        assert_eq!(e2.index(), e1.index()); // Same slot
        assert_eq!(e2.generation(), 1); // New generation
        assert!(!alloc.is_alive(e1)); // Old handle is stale
        assert!(alloc.is_alive(e2));
    }

    #[test]
    fn double_deallocate_fails() {
        let mut alloc = EntityAllocator::new();
        let e = alloc.allocate();
        assert!(alloc.deallocate(e));
        assert!(!alloc.deallocate(e)); // Already dead
    }

    #[test]
    fn display() {
        let e = Entity::from_raw(42, 3);
        assert_eq!(format!("{e}"), "42v3");
    }

    #[test]
    fn generation_overflow_retires_slot() {
        let mut alloc = EntityAllocator::new();
        let e = alloc.allocate(); // slot 0, gen 0
        assert_eq!(e.index(), 0);

        // Artificially set generation to one below MAX
        alloc.generations[e.index() as usize] = u32::MAX - 1;

        // Deallocate at gen u32::MAX-1 → generation saturates to u32::MAX
        assert!(alloc.deallocate(Entity::from_raw(0, u32::MAX - 1)));
        assert_eq!(
            alloc.generations[0],
            u32::MAX,
            "Generation should saturate to u32::MAX, not wrap to zero"
        );

        // Slot 0 is now permanently retired (gen == MAX, not on the free
        // list). The next allocation must create a brand-new slot.
        let e2 = alloc.allocate();
        assert_ne!(
            e2.index(),
            0,
            "Slot 0 should be permanently retired, got index {}",
            e2.index()
        );
        assert_eq!(e2.generation(), 0, "New slot should start at generation 0");
    }
}
