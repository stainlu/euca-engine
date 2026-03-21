use std::collections::VecDeque;
use std::time::Instant;

/// Maximum number of frame times retained for averaging.
const MAX_FRAME_HISTORY: usize = 60;

/// A recorded profile section within a single frame.
pub struct ProfileSection {
    pub name: &'static str,
    pub duration_us: f64,
}

/// Built-in frame profiler that tracks per-section timings and rolling frame statistics.
///
/// Usage: call [`profiler_begin`] at the start of a section and [`profiler_end`] to
/// finish it. At the end of each frame, read the summary via [`Profiler::frame_summary`]
/// and then call [`Profiler::end_frame`] to archive the total frame time.
pub struct Profiler {
    sections: Vec<ProfileSection>,
    frame_times: VecDeque<f64>,
    /// Stack of in-progress section timings (name + start instant).
    pending: Vec<(&'static str, Instant)>,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            sections: Vec::new(),
            frame_times: VecDeque::with_capacity(MAX_FRAME_HISTORY),
            pending: Vec::new(),
        }
    }

    /// Return section names and durations (in microseconds) for the current frame.
    pub fn frame_summary(&self) -> Vec<(&str, f64)> {
        self.sections
            .iter()
            .map(|s| (s.name, s.duration_us))
            .collect()
    }

    /// Average frame time in milliseconds over the last [`MAX_FRAME_HISTORY`] frames.
    ///
    /// Returns `0.0` when no frame times have been recorded yet.
    pub fn avg_frame_time_ms(&self) -> f64 {
        if self.frame_times.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.frame_times.iter().sum();
        sum / self.frame_times.len() as f64
    }

    /// Estimated frames per second derived from the rolling average frame time.
    ///
    /// Returns `0.0` when no frame data is available (avoids division by zero).
    pub fn fps(&self) -> f64 {
        let avg = self.avg_frame_time_ms();
        if avg == 0.0 {
            return 0.0;
        }
        1000.0 / avg
    }

    /// Finish the current frame: record total frame time from all sections and reset
    /// the section list for the next frame.
    pub fn end_frame(&mut self) {
        let total_us: f64 = self.sections.iter().map(|s| s.duration_us).sum();
        let total_ms = total_us / 1000.0;

        if self.frame_times.len() == MAX_FRAME_HISTORY {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(total_ms);
        self.sections.clear();
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Begin timing a named section. Pushes onto an internal stack so sections can nest.
pub fn profiler_begin(profiler: &mut Profiler, name: &'static str) {
    profiler.pending.push((name, Instant::now()));
}

/// End the most recently begun section and record its duration.
///
/// # Panics
///
/// Panics if there is no matching `profiler_begin` call.
pub fn profiler_end(profiler: &mut Profiler) {
    let (name, start) = profiler
        .pending
        .pop()
        .expect("profiler_end called without a matching profiler_begin");
    let elapsed = start.elapsed();
    profiler.sections.push(ProfileSection {
        name,
        duration_us: elapsed.as_secs_f64() * 1_000_000.0,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn begin_end_records_section() {
        let mut profiler = Profiler::new();
        profiler_begin(&mut profiler, "test_section");
        thread::sleep(Duration::from_micros(100));
        profiler_end(&mut profiler);

        let summary = profiler.frame_summary();
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].0, "test_section");
        // Duration should be at least 100 us (we slept 100 us).
        assert!(summary[0].1 >= 100.0, "duration_us = {}", summary[0].1);
    }

    #[test]
    fn multiple_sections_in_one_frame() {
        let mut profiler = Profiler::new();

        profiler_begin(&mut profiler, "physics");
        thread::sleep(Duration::from_micros(50));
        profiler_end(&mut profiler);

        profiler_begin(&mut profiler, "render");
        thread::sleep(Duration::from_micros(50));
        profiler_end(&mut profiler);

        let summary = profiler.frame_summary();
        assert_eq!(summary.len(), 2);
        assert_eq!(summary[0].0, "physics");
        assert_eq!(summary[1].0, "render");
    }

    #[test]
    fn end_frame_archives_and_clears() {
        let mut profiler = Profiler::new();

        profiler_begin(&mut profiler, "work");
        thread::sleep(Duration::from_micros(50));
        profiler_end(&mut profiler);

        assert_eq!(profiler.frame_summary().len(), 1);

        profiler.end_frame();

        // Sections should be cleared after end_frame.
        assert!(profiler.frame_summary().is_empty());
        // One frame time should now be recorded.
        assert_eq!(profiler.frame_times.len(), 1);
    }

    #[test]
    fn avg_frame_time_and_fps() {
        let mut profiler = Profiler::new();

        // Simulate 3 frames, each with 1000 us (= 1 ms) of work.
        for _ in 0..3 {
            profiler_begin(&mut profiler, "work");
            thread::sleep(Duration::from_millis(1));
            profiler_end(&mut profiler);
            profiler.end_frame();
        }

        let avg = profiler.avg_frame_time_ms();
        // Each frame is ~1 ms; allow generous tolerance for CI.
        assert!(avg >= 0.5, "avg_frame_time_ms = {avg}");
        assert!(avg < 50.0, "avg_frame_time_ms unexpectedly large: {avg}");

        let fps = profiler.fps();
        assert!(fps > 0.0, "fps = {fps}");
    }

    #[test]
    fn frame_history_caps_at_max() {
        let mut profiler = Profiler::new();

        for _ in 0..100 {
            profiler_begin(&mut profiler, "tick");
            profiler_end(&mut profiler);
            profiler.end_frame();
        }

        assert_eq!(profiler.frame_times.len(), MAX_FRAME_HISTORY);
    }

    #[test]
    fn empty_profiler_returns_safe_defaults() {
        let profiler = Profiler::new();
        assert!(profiler.frame_summary().is_empty());
        assert_eq!(profiler.avg_frame_time_ms(), 0.0);
        assert_eq!(profiler.fps(), 0.0);
    }

    #[test]
    #[should_panic(expected = "profiler_end called without a matching profiler_begin")]
    fn end_without_begin_panics() {
        let mut profiler = Profiler::new();
        profiler_end(&mut profiler);
    }

    #[test]
    fn nested_sections() {
        let mut profiler = Profiler::new();

        profiler_begin(&mut profiler, "outer");
        profiler_begin(&mut profiler, "inner");
        thread::sleep(Duration::from_micros(50));
        profiler_end(&mut profiler); // ends "inner"
        profiler_end(&mut profiler); // ends "outer"

        let summary = profiler.frame_summary();
        assert_eq!(summary.len(), 2);
        // Inner finishes first (stack order).
        assert_eq!(summary[0].0, "inner");
        assert_eq!(summary[1].0, "outer");
        // Outer should be >= inner since it wraps it.
        assert!(summary[1].1 >= summary[0].1);
    }
}
