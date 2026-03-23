use std::time::Instant;

/// Tracks time progression across frames.
pub struct Time {
    /// Time since the last frame (seconds).
    pub delta: f32,
    /// Total elapsed time since app start (seconds).
    pub elapsed: f64,
    /// Frame counter.
    pub frame_count: u64,
    /// Instant of the last frame start.
    last_frame: Instant,
    /// Instant of app start.
    start: Instant,
}

impl Time {
    /// Create a new time tracker starting from the current instant.
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            delta: 0.0,
            elapsed: 0.0,
            frame_count: 0,
            last_frame: now,
            start: now,
        }
    }

    /// Call at the start of each frame to update delta and elapsed.
    pub fn update(&mut self) {
        let now = Instant::now();
        self.delta = now.duration_since(self.last_frame).as_secs_f32();
        self.elapsed = now.duration_since(self.start).as_secs_f64();
        self.last_frame = now;
        self.frame_count += 1;
    }
}

impl Default for Time {
    fn default() -> Self {
        Self::new()
    }
}
