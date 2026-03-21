//! Shared engine control state for cross-thread communication.
//!
//! Stored as World resources so both the editor (main thread) and the
//! HTTP handler (tokio threads) can access them through SharedWorld.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

/// Simulation play/pause control. Stored as a World resource.
#[derive(Clone)]
pub struct EngineControl {
    playing: Arc<AtomicBool>,
    step_requested: Arc<AtomicBool>,
}

impl EngineControl {
    pub fn new() -> Self {
        Self {
            playing: Arc::new(AtomicBool::new(false)),
            step_requested: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn is_playing(&self) -> bool {
        self.playing.load(Ordering::Relaxed)
    }

    pub fn set_playing(&self, playing: bool) {
        self.playing.store(playing, Ordering::Relaxed);
    }

    pub fn request_step(&self) {
        self.step_requested.store(true, Ordering::Relaxed);
    }

    pub fn take_step_request(&self) -> bool {
        self.step_requested.swap(false, Ordering::Relaxed)
    }
}

impl Default for EngineControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Cross-thread screenshot request channel. Stored as a World resource.
///
/// The HTTP handler places a oneshot sender here. The render loop checks
/// each frame, captures the viewport if a request is pending, and sends
/// the PNG bytes back through the channel.
#[derive(Clone)]
pub struct ScreenshotChannel {
    pending: Arc<Mutex<Option<oneshot::Sender<Vec<u8>>>>>,
}

impl ScreenshotChannel {
    pub fn new() -> Self {
        Self {
            pending: Arc::new(Mutex::new(None)),
        }
    }

    /// Place a screenshot request. Returns the receiver for PNG bytes.
    pub fn request(&self) -> oneshot::Receiver<Vec<u8>> {
        let (tx, rx) = oneshot::channel();
        *self
            .pending
            .lock()
            .expect("ScreenshotChannel lock poisoned") = Some(tx);
        rx
    }

    /// Take the pending request (called by the render loop).
    pub fn take(&self) -> Option<oneshot::Sender<Vec<u8>>> {
        self.pending
            .lock()
            .expect("ScreenshotChannel lock poisoned")
            .take()
    }
}

impl Default for ScreenshotChannel {
    fn default() -> Self {
        Self::new()
    }
}

/// Flag indicating the camera was set programmatically this frame.
/// When set, the editor skips mouse-based camera orbit/pan to avoid
/// overwriting the agent's camera position.
#[derive(Clone, Default)]
pub struct CameraOverride {
    active: Arc<AtomicBool>,
}

impl CameraOverride {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&self) {
        self.active.store(true, Ordering::Relaxed);
    }

    /// Take the flag (returns true once, then false until set again).
    pub fn take(&self) -> bool {
        self.active.swap(false, Ordering::Relaxed)
    }
}
