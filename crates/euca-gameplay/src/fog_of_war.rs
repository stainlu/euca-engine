//! Fog of war, day/night cycle, and ward system.
//!
//! Provides grid-based team vision maps, a Dota-style day/night cycle that
//! modulates vision ranges, and a ward placement/stock system with observer
//! and sentry wards.
//!
//! Types: [`VisionMap`], [`DayNightCycle`], [`Ward`], [`WardStock`], [`VisionSource`].
//! Free functions: [`update_vision`], [`tick_wards`], [`tick_ward_stock`],
//! [`place_ward`], [`hero_vision_radius`], [`is_unit_visible`].

use serde::{Deserialize, Serialize};

// ── Cell visibility ─────────────────────────────────────────────────────

/// Visibility state of a single map cell from one team's perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellVisibility {
    /// Never been seen — completely black.
    Unseen,
    /// Previously seen but no current vision — grey/fogged.
    Fogged,
    /// Currently visible by at least one allied vision source.
    Visible,
}

// ── Vision map ──────────────────────────────────────────────────────────

/// Per-team grid of [`CellVisibility`] states, mapping world space to a
/// discrete grid for efficient fog-of-war queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionMap {
    pub team: u32,
    pub width: u32,
    pub height: u32,
    /// World units per cell (e.g. 64.0).
    pub cell_size: f32,
    pub cells: Vec<CellVisibility>,
}

impl VisionMap {
    /// Create a new vision map with all cells starting as [`CellVisibility::Unseen`].
    pub fn new(team: u32, width: u32, height: u32, cell_size: f32) -> Self {
        Self {
            team,
            width,
            height,
            cell_size,
            cells: vec![CellVisibility::Unseen; (width * height) as usize],
        }
    }

    /// Get the visibility state of a cell. Returns [`CellVisibility::Unseen`]
    /// for out-of-bounds coordinates.
    pub fn get(&self, x: u32, y: u32) -> CellVisibility {
        if x >= self.width || y >= self.height {
            return CellVisibility::Unseen;
        }
        self.cells[(y * self.width + x) as usize]
    }

    /// Set the visibility state of a cell. No-op for out-of-bounds coordinates.
    pub fn set(&mut self, x: u32, y: u32, state: CellVisibility) {
        if x < self.width && y < self.height {
            self.cells[(y * self.width + x) as usize] = state;
        }
    }

    /// Fade all [`CellVisibility::Visible`] cells to [`CellVisibility::Fogged`].
    /// Called at the start of each vision update before revealing new circles.
    pub fn fade_vision(&mut self) {
        for cell in &mut self.cells {
            if *cell == CellVisibility::Visible {
                *cell = CellVisibility::Fogged;
            }
        }
    }

    /// Mark all cells within `radius` world units of (`world_x`, `world_y`)
    /// as [`CellVisibility::Visible`].
    pub fn reveal_circle(&mut self, world_x: f32, world_y: f32, radius: f32) {
        let (cx, cy) = self.world_to_cell(world_x, world_y);
        let cell_radius = (radius / self.cell_size).ceil() as i64;

        let radius_sq = radius * radius;

        let min_x = (cx as i64 - cell_radius).max(0) as u32;
        let max_x = ((cx as i64 + cell_radius) as u32).min(self.width - 1);
        let min_y = (cy as i64 - cell_radius).max(0) as u32;
        let max_y = ((cy as i64 + cell_radius) as u32).min(self.height - 1);

        for gy in min_y..=max_y {
            for gx in min_x..=max_x {
                // Centre of cell in world space.
                let cell_world_x = (gx as f32 + 0.5) * self.cell_size;
                let cell_world_y = (gy as f32 + 0.5) * self.cell_size;
                let dx = cell_world_x - world_x;
                let dy = cell_world_y - world_y;
                if dx * dx + dy * dy <= radius_sq {
                    self.set(gx, gy, CellVisibility::Visible);
                }
            }
        }
    }

    /// Check if a world position is currently visible to this team.
    pub fn is_visible(&self, world_x: f32, world_y: f32) -> bool {
        let (cx, cy) = self.world_to_cell(world_x, world_y);
        self.get(cx, cy) == CellVisibility::Visible
    }

    /// Convert world coordinates to cell coordinates, clamped to grid bounds.
    pub fn world_to_cell(&self, world_x: f32, world_y: f32) -> (u32, u32) {
        let cx = (world_x / self.cell_size).floor().max(0.0) as u32;
        let cy = (world_y / self.cell_size).floor().max(0.0) as u32;
        (
            cx.min(self.width.saturating_sub(1)),
            cy.min(self.height.saturating_sub(1)),
        )
    }
}

// ── Day / night cycle ───────────────────────────────────────────────────

/// Day/night cycle that modulates vision ranges.
///
/// Default: 5-minute day + 5-minute night = 10-minute full cycle.
/// During night, vision ranges are reduced to ~44.4% of day values
/// (800 / 1800 — Dota 2 night-vision ratio).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DayNightCycle {
    /// Total cycle length in seconds (day + night).
    pub cycle_duration: f32,
    /// Current time within the cycle, in `0..cycle_duration`.
    pub current_time: f32,
    /// Duration of the day portion in seconds.
    pub day_duration: f32,
    /// Duration of the night portion in seconds.
    pub night_duration: f32,
}

/// Dota 2 night-vision ratio: 800 / 1800 ≈ 0.4444.
const NIGHT_VISION_MULTIPLIER: f32 = 800.0 / 1800.0;

impl DayNightCycle {
    /// Create a default day/night cycle: 300s day + 300s night.
    pub fn new() -> Self {
        Self {
            cycle_duration: 600.0,
            current_time: 0.0,
            day_duration: 300.0,
            night_duration: 300.0,
        }
    }

    /// `true` during the day portion of the cycle.
    pub fn is_day(&self) -> bool {
        self.current_time < self.day_duration
    }

    /// `true` during the night portion of the cycle.
    pub fn is_night(&self) -> bool {
        !self.is_day()
    }

    /// Advance the cycle by `dt` seconds (wraps around).
    pub fn tick(&mut self, dt: f32) {
        self.current_time = (self.current_time + dt) % self.cycle_duration;
    }

    /// Vision multiplier: 1.0 during day, reduced at night.
    pub fn vision_multiplier(&self) -> f32 {
        if self.is_day() {
            1.0
        } else {
            NIGHT_VISION_MULTIPLIER
        }
    }
}

impl Default for DayNightCycle {
    fn default() -> Self {
        Self::new()
    }
}

// ── Wards ───────────────────────────────────────────────────────────────

/// Ward classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WardType {
    /// Observer ward: provides regular vision but no True Sight.
    Observer,
    /// Sentry ward: provides True Sight (reveals invisible units) but no
    /// regular vision.
    Sentry,
}

/// A placed ward on the map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ward {
    pub ward_type: WardType,
    pub team: u32,
    pub position: [f32; 2],
    pub remaining_duration: f32,
    /// Regular vision radius (0 for sentries).
    pub vision_radius: f32,
    /// True-sight radius (0 for observers, 900 for sentries).
    pub true_sight_radius: f32,
    /// Whether enemies can see this ward without True Sight.
    pub is_visible_to_enemies: bool,
}

impl Ward {
    /// Create an observer ward: 1600 vision radius, 6-minute duration.
    pub fn observer(team: u32, position: [f32; 2]) -> Self {
        Self {
            ward_type: WardType::Observer,
            team,
            position,
            remaining_duration: 360.0,
            vision_radius: 1600.0,
            true_sight_radius: 0.0,
            is_visible_to_enemies: false,
        }
    }

    /// Create a sentry ward: 900 true-sight radius, 4-minute duration.
    pub fn sentry(team: u32, position: [f32; 2]) -> Self {
        Self {
            ward_type: WardType::Sentry,
            team,
            position,
            remaining_duration: 240.0,
            vision_radius: 0.0,
            true_sight_radius: 900.0,
            is_visible_to_enemies: false,
        }
    }

    /// Whether this ward has expired (duration <= 0).
    pub fn is_expired(&self) -> bool {
        self.remaining_duration <= 0.0
    }
}

// ── Ward stock ──────────────────────────────────────────────────────────

/// Ward stock management for a team. Tracks available ward counts and
/// automatic restock timers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WardStock {
    pub observer_count: u32,
    pub observer_max: u32,
    pub observer_restock_time: f32,
    pub observer_restock_timer: f32,

    pub sentry_count: u32,
    pub sentry_max: u32,
    pub sentry_restock_time: f32,
    pub sentry_restock_timer: f32,
}

impl WardStock {
    /// Default stock: 2 observers (max 4, 135s restock), 3 sentries (max 5, 85s restock).
    pub fn new() -> Self {
        Self {
            observer_count: 2,
            observer_max: 4,
            observer_restock_time: 135.0,
            observer_restock_timer: 0.0,

            sentry_count: 3,
            sentry_max: 5,
            sentry_restock_time: 85.0,
            sentry_restock_timer: 0.0,
        }
    }
}

impl Default for WardStock {
    fn default() -> Self {
        Self::new()
    }
}

// ── Vision source ───────────────────────────────────────────────────────

/// A generic vision provider — heroes, buildings, wards, and summons all
/// map to this before feeding into [`update_vision`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionSource {
    pub team: u32,
    pub position: [f32; 2],
    pub radius: f32,
    pub provides_true_sight: bool,
}

// ── Free functions ──────────────────────────────────────────────────────

/// Perform a full vision update for one team: fade existing vision, then
/// reveal circles for every source.
pub fn update_vision(map: &mut VisionMap, sources: &[VisionSource]) {
    map.fade_vision();
    for src in sources {
        if src.team == map.team && src.radius > 0.0 {
            map.reveal_circle(src.position[0], src.position[1], src.radius);
        }
    }
}

/// Tick all wards: reduce remaining durations and remove expired ones.
/// Returns the number of wards removed.
pub fn tick_wards(wards: &mut Vec<Ward>, dt: f32) -> usize {
    for w in wards.iter_mut() {
        w.remaining_duration -= dt;
    }
    let before = wards.len();
    wards.retain(|w| !w.is_expired());
    before - wards.len()
}

/// Tick ward stock timers and restock when ready.
pub fn tick_ward_stock(stock: &mut WardStock, dt: f32) {
    // Observer restock.
    if stock.observer_count < stock.observer_max {
        stock.observer_restock_timer += dt;
        if stock.observer_restock_timer >= stock.observer_restock_time {
            stock.observer_restock_timer -= stock.observer_restock_time;
            stock.observer_count += 1;
        }
    } else {
        // Reset timer when at max so restocking starts fresh after next use.
        stock.observer_restock_timer = 0.0;
    }

    // Sentry restock.
    if stock.sentry_count < stock.sentry_max {
        stock.sentry_restock_timer += dt;
        if stock.sentry_restock_timer >= stock.sentry_restock_time {
            stock.sentry_restock_timer -= stock.sentry_restock_time;
            stock.sentry_count += 1;
        }
    } else {
        stock.sentry_restock_timer = 0.0;
    }
}

/// Place a ward, consuming one from stock. Returns the new ward or an
/// error if stock is depleted.
pub fn place_ward(
    stock: &mut WardStock,
    ward_type: WardType,
    team: u32,
    pos: [f32; 2],
) -> Result<Ward, &'static str> {
    match ward_type {
        WardType::Observer => {
            if stock.observer_count == 0 {
                return Err("no observer wards in stock");
            }
            stock.observer_count -= 1;
            // Start restock timer if it was idle (count was at max before).
            if stock.observer_restock_timer == 0.0 && stock.observer_count < stock.observer_max {
                // Timer already at 0, will begin counting next tick.
            }
            Ok(Ward::observer(team, pos))
        }
        WardType::Sentry => {
            if stock.sentry_count == 0 {
                return Err("no sentry wards in stock");
            }
            stock.sentry_count -= 1;
            Ok(Ward::sentry(team, pos))
        }
    }
}

/// Compute the effective hero vision radius given the day/night cycle.
///
/// `base_day` is the hero's daytime vision (typically 1800).
/// `base_night` is the hero's nighttime vision (typically 800).
/// The cycle determines which base value to use, and `vision_multiplier`
/// is *not* applied on top — day/night bases already encode the reduction.
pub fn hero_vision_radius(base_day: f32, base_night: f32, cycle: &DayNightCycle) -> f32 {
    if cycle.is_day() { base_day } else { base_night }
}

/// Check whether a unit at `unit_pos` is visible on the given team's
/// vision map.
pub fn is_unit_visible(unit_pos: [f32; 2], _unit_team: u32, observer_team_map: &VisionMap) -> bool {
    observer_team_map.is_visible(unit_pos[0], unit_pos[1])
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── VisionMap ──

    #[test]
    fn test_vision_map_new() {
        let map = VisionMap::new(1, 16, 16, 64.0);
        assert_eq!(map.cells.len(), 256);
        assert!(map.cells.iter().all(|c| *c == CellVisibility::Unseen));
    }

    #[test]
    fn test_reveal_circle() {
        let mut map = VisionMap::new(1, 32, 32, 64.0);
        // Reveal a circle at the centre of the map.
        map.reveal_circle(1024.0, 1024.0, 128.0);
        // The cell containing the centre should be visible.
        let (cx, cy) = map.world_to_cell(1024.0, 1024.0);
        assert_eq!(map.get(cx, cy), CellVisibility::Visible);
        // A cell far away should remain unseen.
        assert_eq!(map.get(0, 0), CellVisibility::Unseen);
    }

    #[test]
    fn test_fade_vision() {
        let mut map = VisionMap::new(1, 8, 8, 64.0);
        map.set(2, 2, CellVisibility::Visible);
        map.set(3, 3, CellVisibility::Visible);
        map.fade_vision();
        assert_eq!(map.get(2, 2), CellVisibility::Fogged);
        assert_eq!(map.get(3, 3), CellVisibility::Fogged);
    }

    #[test]
    fn test_unseen_stays_unseen() {
        let mut map = VisionMap::new(1, 8, 8, 64.0);
        // Only mark one cell visible; the rest stay unseen.
        map.set(0, 0, CellVisibility::Visible);
        map.fade_vision();
        // (0,0) fades to fogged, but (1,1) was never seen.
        assert_eq!(map.get(0, 0), CellVisibility::Fogged);
        assert_eq!(map.get(1, 1), CellVisibility::Unseen);
    }

    #[test]
    fn test_world_to_cell() {
        let map = VisionMap::new(1, 16, 16, 64.0);
        assert_eq!(map.world_to_cell(0.0, 0.0), (0, 0));
        assert_eq!(map.world_to_cell(64.0, 64.0), (1, 1));
        assert_eq!(map.world_to_cell(128.0, 192.0), (2, 3));
        // Negative clamps to 0.
        assert_eq!(map.world_to_cell(-100.0, -100.0), (0, 0));
        // Beyond grid clamps to max.
        assert_eq!(map.world_to_cell(99999.0, 99999.0), (15, 15));
    }

    // ── Day/night cycle ──

    #[test]
    fn test_day_night_cycle() {
        let mut cycle = DayNightCycle::new();
        assert!(cycle.is_day());
        // Advance past day duration.
        cycle.tick(300.0);
        assert!(cycle.is_night());
        // Wrap back to day.
        cycle.tick(300.0);
        assert!(cycle.is_day());
    }

    #[test]
    fn test_is_day() {
        let mut cycle = DayNightCycle::new();
        // At t=0 it should be day.
        assert!(cycle.is_day());
        // At t=150 still day.
        cycle.tick(150.0);
        assert!(cycle.is_day());
        // At t=299 still day.
        cycle.tick(149.0);
        assert!(cycle.is_day());
    }

    #[test]
    fn test_is_night() {
        let mut cycle = DayNightCycle::new();
        // Advance into night.
        cycle.tick(300.0);
        assert!(cycle.is_night());
        cycle.tick(100.0);
        assert!(cycle.is_night());
    }

    #[test]
    fn test_vision_multiplier() {
        let mut cycle = DayNightCycle::new();
        assert!((cycle.vision_multiplier() - 1.0).abs() < f32::EPSILON);
        cycle.tick(300.0);
        let expected = 800.0_f32 / 1800.0;
        assert!((cycle.vision_multiplier() - expected).abs() < 1e-5);
    }

    // ── Wards ──

    #[test]
    fn test_observer_ward_stats() {
        let w = Ward::observer(1, [100.0, 200.0]);
        assert_eq!(w.ward_type, WardType::Observer);
        assert!((w.vision_radius - 1600.0).abs() < f32::EPSILON);
        assert!((w.remaining_duration - 360.0).abs() < f32::EPSILON);
        assert!((w.true_sight_radius - 0.0).abs() < f32::EPSILON);
        assert!(!w.is_visible_to_enemies);
    }

    #[test]
    fn test_sentry_ward_stats() {
        let w = Ward::sentry(2, [300.0, 400.0]);
        assert_eq!(w.ward_type, WardType::Sentry);
        assert!((w.true_sight_radius - 900.0).abs() < f32::EPSILON);
        assert!((w.remaining_duration - 240.0).abs() < f32::EPSILON);
        assert!((w.vision_radius - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ward_expiry() {
        let mut wards = vec![
            Ward::observer(1, [0.0, 0.0]),   // 360s duration
            Ward::sentry(1, [100.0, 100.0]), // 240s duration
        ];
        // Tick past sentry duration (240s) but not observer (360s).
        let removed = tick_wards(&mut wards, 241.0);
        assert_eq!(removed, 1);
        assert_eq!(wards.len(), 1);
        assert_eq!(wards[0].ward_type, WardType::Observer);

        // Tick the remaining observer past its leftover duration (360 - 241 = 119s).
        let removed = tick_wards(&mut wards, 120.0);
        assert_eq!(removed, 1);
        assert!(wards.is_empty());
    }

    #[test]
    fn test_ward_stock_restock() {
        let mut stock = WardStock::new();
        // Deplete one observer.
        stock.observer_count = 1;
        // Tick exactly the restock time.
        tick_ward_stock(&mut stock, 135.0);
        assert_eq!(stock.observer_count, 2);
    }

    #[test]
    fn test_place_ward_depletes_stock() {
        let mut stock = WardStock::new();
        let initial = stock.observer_count;
        let result = place_ward(&mut stock, WardType::Observer, 1, [0.0, 0.0]);
        assert!(result.is_ok());
        assert_eq!(stock.observer_count, initial - 1);
    }

    #[test]
    fn test_place_ward_no_stock() {
        let mut stock = WardStock::new();
        stock.observer_count = 0;
        let result = place_ward(&mut stock, WardType::Observer, 1, [0.0, 0.0]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "no observer wards in stock");
    }

    #[test]
    fn test_update_vision_multiple_sources() {
        let mut map = VisionMap::new(1, 64, 64, 32.0);
        let sources = vec![
            VisionSource {
                team: 1,
                position: [256.0, 256.0],
                radius: 128.0,
                provides_true_sight: false,
            },
            VisionSource {
                team: 1,
                position: [1024.0, 1024.0],
                radius: 128.0,
                provides_true_sight: false,
            },
            // Different team — should be ignored.
            VisionSource {
                team: 2,
                position: [512.0, 512.0],
                radius: 128.0,
                provides_true_sight: false,
            },
        ];

        update_vision(&mut map, &sources);

        // Both team-1 source centres should be visible.
        assert!(map.is_visible(256.0, 256.0));
        assert!(map.is_visible(1024.0, 1024.0));
        // Team-2 source should NOT have revealed anything for team 1.
        assert!(!map.is_visible(512.0, 512.0));
    }

    #[test]
    fn test_hero_vision_radius_day_and_night() {
        let mut cycle = DayNightCycle::new();
        assert!((hero_vision_radius(1800.0, 800.0, &cycle) - 1800.0).abs() < f32::EPSILON);
        cycle.tick(300.0); // night
        assert!((hero_vision_radius(1800.0, 800.0, &cycle) - 800.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_is_unit_visible() {
        let mut map = VisionMap::new(1, 16, 16, 64.0);
        map.reveal_circle(512.0, 512.0, 200.0);
        assert!(is_unit_visible([512.0, 512.0], 2, &map));
        assert!(!is_unit_visible([0.0, 0.0], 2, &map));
    }

    #[test]
    fn test_sentry_stock_restock() {
        let mut stock = WardStock::new();
        stock.sentry_count = 0;
        tick_ward_stock(&mut stock, 85.0);
        assert_eq!(stock.sentry_count, 1);
    }

    #[test]
    fn test_place_sentry_no_stock() {
        let mut stock = WardStock::new();
        stock.sentry_count = 0;
        let result = place_ward(&mut stock, WardType::Sentry, 1, [0.0, 0.0]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "no sentry wards in stock");
    }

    #[test]
    fn test_fogged_not_visible() {
        let mut map = VisionMap::new(1, 8, 8, 64.0);
        // Reveal then fade — cell should be fogged, not visible.
        map.set(4, 4, CellVisibility::Visible);
        map.fade_vision();
        assert!(!map.is_visible(4.0 * 64.0 + 32.0, 4.0 * 64.0 + 32.0));
        assert_eq!(map.get(4, 4), CellVisibility::Fogged);
    }
}
