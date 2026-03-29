//! Game replay recording and playback.
//!
//! Records per-tick input snapshots during gameplay and replays them
//! deterministically. Replays are self-contained files that can be
//! saved, loaded, and shared.

use euca_input::InputSnapshot;
use serde::{Deserialize, Serialize};

/// Metadata about a recorded replay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayHeader {
    /// Engine version that recorded this replay.
    pub engine_version: String,
    /// Map/level identifier.
    pub map_id: String,
    /// Unix timestamp when recording started.
    pub recorded_at: u64,
    /// Total number of ticks in the replay.
    pub tick_count: u64,
    /// Tick rate (ticks per second) used during recording.
    pub tick_rate: u32,
    /// Player count at recording start.
    pub player_count: u32,
    /// Optional display name for the replay.
    pub name: String,
}

/// A complete recorded replay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Replay {
    /// Header with metadata about the recording session.
    pub header: ReplayHeader,
    /// Per-tick input snapshots for all players, ordered by tick.
    pub frames: Vec<ReplayFrame>,
}

/// One frame of replay data (one tick, all players).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayFrame {
    /// The tick number for this frame.
    pub tick: u64,
    /// Input snapshots for each player this tick.
    pub inputs: Vec<PlayerInput>,
}

/// Input from one player in one tick.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlayerInput {
    /// Index of the player who produced this input.
    pub player_index: u32,
    /// The captured input snapshot.
    pub snapshot: InputSnapshot,
}

/// Records a game session tick-by-tick.
pub struct ReplayRecorder {
    header: ReplayHeader,
    frames: Vec<ReplayFrame>,
    current_tick: u64,
    recording: bool,
}

impl ReplayRecorder {
    /// Create a new recorder.
    pub fn new(map_id: impl Into<String>, tick_rate: u32, player_count: u32) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            header: ReplayHeader {
                engine_version: env!("CARGO_PKG_VERSION").to_string(),
                map_id: map_id.into(),
                recorded_at: now,
                tick_count: 0,
                tick_rate,
                player_count,
                name: String::new(),
            },
            frames: Vec::new(),
            current_tick: 0,
            recording: true,
        }
    }

    /// Record one tick's worth of input. No-op if recording has been stopped.
    pub fn record_tick(&mut self, inputs: Vec<PlayerInput>) {
        if !self.recording {
            return;
        }
        self.frames.push(ReplayFrame {
            tick: self.current_tick,
            inputs,
        });
        self.current_tick += 1;
        self.header.tick_count = self.current_tick;
    }

    /// Stop recording without consuming the recorder.
    ///
    /// After this call, [`record_tick`](Self::record_tick) becomes a no-op.
    /// Use [`finish`](Self::finish) when you are ready to extract the replay.
    pub fn stop(&mut self) {
        self.recording = false;
    }

    /// Stop recording and produce the final replay.
    pub fn finish(mut self) -> Replay {
        self.recording = false;
        Replay {
            header: self.header,
            frames: self.frames,
        }
    }

    /// Whether recording is active.
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Current tick number.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }
}

/// Plays back a recorded replay tick-by-tick.
pub struct ReplayPlayer {
    replay: Replay,
    current_tick: u64,
    playing: bool,
    speed: f32,
    /// Accumulated fractional ticks from speed != 1.0.
    accumulator: f32,
}

impl ReplayPlayer {
    /// Create a new player for the given replay.
    pub fn new(replay: Replay) -> Self {
        Self {
            replay,
            current_tick: 0,
            playing: false,
            speed: 1.0,
            accumulator: 0.0,
        }
    }

    /// Start or resume playback.
    pub fn play(&mut self) {
        self.playing = true;
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Set playback speed (1.0 = normal, 2.0 = double, 0.5 = half).
    ///
    /// Clamped to the range `[0.1, 16.0]`.
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed.clamp(0.1, 16.0);
    }

    /// Seek to a specific tick.
    pub fn seek(&mut self, tick: u64) {
        self.current_tick = tick.min(self.total_ticks().saturating_sub(1));
        self.accumulator = 0.0;
    }

    /// Advance playback by one engine tick and return the inputs for that frame.
    ///
    /// Returns `None` if playback is paused or finished.
    pub fn advance(&mut self) -> Option<&ReplayFrame> {
        if !self.playing {
            return None;
        }

        self.accumulator += self.speed;
        let ticks_to_advance = self.accumulator as u64;
        self.accumulator -= ticks_to_advance as f32;

        self.current_tick += ticks_to_advance;
        if self.current_tick >= self.replay.frames.len() as u64 {
            self.current_tick = self.replay.frames.len() as u64;
            self.playing = false;
            return None;
        }

        self.replay.frames.get(self.current_tick as usize)
    }

    /// Get the frame at a specific tick without advancing.
    pub fn frame_at(&self, tick: u64) -> Option<&ReplayFrame> {
        self.replay.frames.get(tick as usize)
    }

    /// Total ticks in the replay.
    pub fn total_ticks(&self) -> u64 {
        self.replay.header.tick_count
    }

    /// Current playback position.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Whether playback is active.
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Whether playback has reached the end.
    pub fn is_finished(&self) -> bool {
        self.current_tick >= self.replay.frames.len() as u64
    }

    /// Playback speed.
    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// Progress as a fraction (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        if self.total_ticks() == 0 {
            return 0.0;
        }
        self.current_tick as f32 / self.total_ticks() as f32
    }

    /// Access the replay header.
    pub fn header(&self) -> &ReplayHeader {
        &self.replay.header
    }
}

/// Serialize a replay to bytes (for saving to file).
pub fn serialize_replay(replay: &Replay) -> Result<Vec<u8>, String> {
    bincode::serialize(replay).map_err(|e| e.to_string())
}

/// Deserialize a replay from bytes (for loading from file).
pub fn deserialize_replay(data: &[u8]) -> Result<Replay, String> {
    bincode::deserialize(data).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use euca_input::InputKey;

    /// Helper: build a `PlayerInput` for a single player at a given tick.
    fn make_player_input(player_index: u32, tick: u64) -> PlayerInput {
        PlayerInput {
            player_index,
            snapshot: InputSnapshot {
                tick,
                pressed_keys: vec![InputKey::Key("W".into())],
                mouse_position: [100.0, 200.0],
                mouse_delta: [1.0, -1.0],
            },
        }
    }

    /// Helper: record N ticks with one player and return the finished replay.
    fn record_n_ticks(n: u64) -> Replay {
        let mut recorder = ReplayRecorder::new("test_map", 60, 1);
        for tick in 0..n {
            recorder.record_tick(vec![make_player_input(0, tick)]);
        }
        recorder.finish()
    }

    #[test]
    fn test_recorder_basic() {
        let replay = record_n_ticks(10);

        assert_eq!(replay.header.map_id, "test_map");
        assert_eq!(replay.header.tick_rate, 60);
        assert_eq!(replay.header.player_count, 1);
        assert_eq!(replay.header.tick_count, 10);
        assert_eq!(replay.frames.len(), 10);
        assert!(replay.header.recorded_at > 0);

        // Verify tick ordering.
        for (i, frame) in replay.frames.iter().enumerate() {
            assert_eq!(frame.tick, i as u64);
            assert_eq!(frame.inputs.len(), 1);
            assert_eq!(frame.inputs[0].player_index, 0);
        }
    }

    #[test]
    fn test_recorder_not_recording() {
        let mut recorder = ReplayRecorder::new("test_map", 60, 1);
        recorder.record_tick(vec![make_player_input(0, 0)]);
        recorder.record_tick(vec![make_player_input(0, 1)]);

        // Stop recording — subsequent record_tick calls should be no-ops.
        recorder.stop();
        assert!(!recorder.is_recording());

        recorder.record_tick(vec![make_player_input(0, 2)]);
        recorder.record_tick(vec![make_player_input(0, 3)]);

        let replay = recorder.finish();
        assert_eq!(replay.header.tick_count, 2);
        assert_eq!(replay.frames.len(), 2);
    }

    #[test]
    fn test_player_basic() {
        let replay = record_n_ticks(5);
        let mut player = ReplayPlayer::new(replay);

        assert!(!player.is_playing());
        assert_eq!(player.current_tick(), 0);
        assert_eq!(player.total_ticks(), 5);

        player.play();
        assert!(player.is_playing());

        // Advance through all frames.
        let mut seen_ticks = Vec::new();
        while let Some(frame) = player.advance() {
            seen_ticks.push(frame.tick);
        }

        assert_eq!(seen_ticks, vec![1, 2, 3, 4]);
        assert!(player.is_finished());
        assert!(!player.is_playing());
    }

    #[test]
    fn test_player_pause_resume() {
        let replay = record_n_ticks(5);
        let mut player = ReplayPlayer::new(replay);
        player.play();

        // Advance once.
        let frame = player.advance();
        assert!(frame.is_some());
        let tick_after_first = player.current_tick();

        // Pause — advance should return None.
        player.pause();
        assert!(!player.is_playing());
        assert!(player.advance().is_none());
        assert_eq!(player.current_tick(), tick_after_first);

        // Resume — should continue from where we left off.
        player.play();
        let frame = player.advance();
        assert!(frame.is_some());
        assert!(player.current_tick() > tick_after_first);
    }

    #[test]
    fn test_player_speed() {
        let replay = record_n_ticks(10);
        let mut player = ReplayPlayer::new(replay);
        player.set_speed(2.0);
        player.play();

        // At 2x speed, each advance moves 2 ticks.
        let frame = player.advance().unwrap();
        assert_eq!(frame.tick, 2); // jumped from 0 to tick index 2
        assert_eq!(player.current_tick(), 2);

        let frame = player.advance().unwrap();
        assert_eq!(frame.tick, 4);
        assert_eq!(player.current_tick(), 4);
    }

    #[test]
    fn test_player_seek() {
        let replay = record_n_ticks(10);
        let mut player = ReplayPlayer::new(replay);

        player.seek(5);
        assert_eq!(player.current_tick(), 5);

        let frame = player.frame_at(5).unwrap();
        assert_eq!(frame.tick, 5);

        // Seek past end should clamp.
        player.seek(100);
        assert_eq!(player.current_tick(), 9);
    }

    #[test]
    fn test_player_progress() {
        let replay = record_n_ticks(10);
        let mut player = ReplayPlayer::new(replay);

        assert!((player.progress() - 0.0).abs() < f32::EPSILON);

        player.seek(5);
        assert!((player.progress() - 0.5).abs() < f32::EPSILON);

        player.seek(10);
        // Clamped to 9, so progress = 9/10 = 0.9.
        assert!((player.progress() - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_serialize_roundtrip() {
        let replay = record_n_ticks(5);
        let bytes = serialize_replay(&replay).expect("serialization should succeed");
        let restored = deserialize_replay(&bytes).expect("deserialization should succeed");

        assert_eq!(replay, restored);
    }

    #[test]
    fn test_empty_replay() {
        let replay = record_n_ticks(0);
        assert_eq!(replay.header.tick_count, 0);
        assert!(replay.frames.is_empty());

        let mut player = ReplayPlayer::new(replay);
        assert!(player.is_finished());
        assert!((player.progress() - 0.0).abs() < f32::EPSILON);

        player.play();
        assert!(player.advance().is_none());
        assert!(player.is_finished());
    }
}
