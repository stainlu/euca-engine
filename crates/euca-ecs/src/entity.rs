use serde::{Deserialize, Serialize};

/// A unique identifier for an entity in the world.
///
/// Uses generational indices to detect stale references:
/// - `index`: slot in the entity array (reused after despawn)
/// - `generation`: incremented each time the slot is recycled
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
        if let Some(index) = self.free_list.pop() {
            Entity {
                index,
                generation: self.generations[index as usize],
            }
        } else {
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
        self.generations[entity.index as usize] += 1;
        self.free_list.push(entity.index);
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
}
