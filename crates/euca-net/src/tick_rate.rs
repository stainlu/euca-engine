//! Tick rate control: decouple server simulation rate from network send rate.

use serde::{Deserialize, Serialize};

/// Controls the rate at which network updates are sent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TickRateConfig {
    /// Simulation ticks per second (physics + gameplay).
    pub sim_rate: u32,
    /// Network updates per second (how often state is sent to clients).
    pub net_rate: u32,
}

impl Default for TickRateConfig {
    fn default() -> Self {
        Self {
            sim_rate: 60,
            net_rate: 20,
        }
    }
}

impl TickRateConfig {
    /// How many simulation ticks between each network send.
    pub fn ticks_per_send(&self) -> u32 {
        if self.net_rate == 0 {
            return u32::MAX;
        }
        (self.sim_rate / self.net_rate).max(1)
    }
}

/// Accumulator for fixed-rate network sends.
#[derive(Clone, Debug, Default)]
pub struct NetworkTickAccumulator {
    /// Ticks since last network send.
    pub ticks_since_send: u32,
}

impl NetworkTickAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Tick the accumulator. Returns true if it's time to send a network update.
    pub fn tick(&mut self, config: &TickRateConfig) -> bool {
        self.ticks_since_send += 1;
        if self.ticks_since_send >= config.ticks_per_send() {
            self.ticks_since_send = 0;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tick_rate() {
        let config = TickRateConfig::default();
        assert_eq!(config.sim_rate, 60);
        assert_eq!(config.net_rate, 20);
        assert_eq!(config.ticks_per_send(), 3);
    }

    #[test]
    fn accumulator_sends_at_correct_rate() {
        let config = TickRateConfig {
            sim_rate: 60,
            net_rate: 20,
        };
        let mut acc = NetworkTickAccumulator::new();

        // Should send every 3 ticks
        assert!(!acc.tick(&config)); // tick 1
        assert!(!acc.tick(&config)); // tick 2
        assert!(acc.tick(&config)); // tick 3 → send!
        assert!(!acc.tick(&config)); // tick 1 again
        assert!(!acc.tick(&config)); // tick 2
        assert!(acc.tick(&config)); // tick 3 → send!
    }

    #[test]
    fn same_sim_and_net_rate() {
        let config = TickRateConfig {
            sim_rate: 60,
            net_rate: 60,
        };
        let mut acc = NetworkTickAccumulator::new();
        // Should send every tick
        assert!(acc.tick(&config));
        assert!(acc.tick(&config));
    }

    #[test]
    fn zero_net_rate_never_sends() {
        let config = TickRateConfig {
            sim_rate: 60,
            net_rate: 0,
        };
        let mut acc = NetworkTickAccumulator::new();
        for _ in 0..1000 {
            assert!(!acc.tick(&config));
        }
    }
}
