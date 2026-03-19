//! Bandwidth budgeting: prioritize entity replication within a byte budget.

use std::collections::BinaryHeap;

/// Priority entry for bandwidth allocation.
#[derive(Clone, Debug, PartialEq)]
struct PriorityEntry {
    entity_index: u32,
    /// Higher = more urgent to replicate.
    priority: f32,
    /// Estimated bytes to replicate this entity.
    estimated_bytes: u32,
}

impl Eq for PriorityEntry {}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority
            .partial_cmp(&other.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Bandwidth budget for a single client per tick.
#[derive(Clone, Debug)]
pub struct BandwidthBudget {
    /// Maximum bytes per tick for this client.
    pub bytes_per_tick: u32,
    /// Bytes used so far this tick.
    pub bytes_used: u32,
}

impl BandwidthBudget {
    pub fn new(bytes_per_tick: u32) -> Self {
        Self {
            bytes_per_tick,
            bytes_used: 0,
        }
    }

    /// Reset budget for a new tick.
    pub fn reset(&mut self) {
        self.bytes_used = 0;
    }

    /// Try to allocate bytes. Returns true if budget allows.
    pub fn try_allocate(&mut self, bytes: u32) -> bool {
        if self.bytes_used + bytes <= self.bytes_per_tick {
            self.bytes_used += bytes;
            true
        } else {
            false
        }
    }

    /// Remaining bytes in budget.
    pub fn remaining(&self) -> u32 {
        self.bytes_per_tick.saturating_sub(self.bytes_used)
    }
}

impl Default for BandwidthBudget {
    fn default() -> Self {
        // 64KB per tick (at 60Hz = ~3.8 MB/s)
        Self::new(65536)
    }
}

/// Entity replication priority calculator.
pub struct PriorityCalculator;

impl PriorityCalculator {
    /// Calculate replication priority for an entity.
    ///
    /// Factors:
    /// - Distance (closer = higher priority)
    /// - Velocity (faster = higher priority)
    /// - Time since last sent (stale = higher priority)
    pub fn calculate(distance: f32, velocity_magnitude: f32, ticks_since_last_sent: u32) -> f32 {
        let distance_factor = 1.0 / (1.0 + distance * 0.1);
        let velocity_factor = 1.0 + velocity_magnitude * 0.5;
        let staleness_factor = 1.0 + ticks_since_last_sent as f32 * 0.2;
        distance_factor * velocity_factor * staleness_factor
    }
}

/// Select which entities to replicate within the bandwidth budget.
///
/// Returns indices of entities to replicate, ordered by priority.
pub fn select_entities_for_replication(
    entities: &[(u32, f32, f32, u32)], // (index, distance, velocity, ticks_stale)
    budget: &mut BandwidthBudget,
    bytes_per_entity: u32,
) -> Vec<u32> {
    let mut heap = BinaryHeap::new();

    for &(idx, dist, vel, stale) in entities {
        let priority = PriorityCalculator::calculate(dist, vel, stale);
        heap.push(PriorityEntry {
            entity_index: idx,
            priority,
            estimated_bytes: bytes_per_entity,
        });
    }

    let mut selected = Vec::new();
    while let Some(entry) = heap.pop() {
        if budget.try_allocate(entry.estimated_bytes) {
            selected.push(entry.entity_index);
        } else {
            break; // Budget exhausted
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_allocation() {
        let mut budget = BandwidthBudget::new(100);
        assert!(budget.try_allocate(50));
        assert_eq!(budget.remaining(), 50);
        assert!(budget.try_allocate(50));
        assert!(!budget.try_allocate(1)); // exhausted
        budget.reset();
        assert_eq!(budget.remaining(), 100);
    }

    #[test]
    fn priority_calculation() {
        // Closer entity should have higher priority
        let close = PriorityCalculator::calculate(1.0, 0.0, 0);
        let far = PriorityCalculator::calculate(100.0, 0.0, 0);
        assert!(close > far);

        // Faster entity should have higher priority
        let still = PriorityCalculator::calculate(10.0, 0.0, 0);
        let fast = PriorityCalculator::calculate(10.0, 20.0, 0);
        assert!(fast > still);

        // Stale entity should have higher priority
        let fresh = PriorityCalculator::calculate(10.0, 0.0, 0);
        let stale = PriorityCalculator::calculate(10.0, 0.0, 30);
        assert!(stale > fresh);
    }

    #[test]
    fn entity_selection_within_budget() {
        let mut budget = BandwidthBudget::new(200);
        let entities = vec![
            (0, 5.0, 0.0, 0),   // close, high priority
            (1, 50.0, 0.0, 0),  // far, low priority
            (2, 10.0, 10.0, 5), // medium distance, moving, stale
        ];

        let selected = select_entities_for_replication(&entities, &mut budget, 100);
        // Budget=200, 100/entity = max 2 entities
        assert_eq!(selected.len(), 2);
        // Should select highest priority first
    }
}
