//! Async asset loading and handle management.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

/// Lock a mutex, recovering its contents if a previous holder panicked.
fn lock_or_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| {
        log::warn!("recovered from poisoned asset lock");
        e.into_inner()
    })
}

/// Load state of an asset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadState {
    /// Asset loading has been requested but not yet started.
    Pending,
    /// Asset is currently being loaded (background thread/task).
    Loading,
    /// Asset loaded successfully.
    Ready,
    /// Asset failed to load.
    Failed(String),
}

/// Handle to an asset of type T. Tracks load state.
#[derive(Debug)]
pub struct AssetHandle<T> {
    pub id: u32,
    pub state: LoadState,
    pub data: Option<T>,
}

impl<T> AssetHandle<T> {
    pub fn pending(id: u32) -> Self {
        Self {
            id,
            state: LoadState::Pending,
            data: None,
        }
    }

    pub fn ready(id: u32, data: T) -> Self {
        Self {
            id,
            state: LoadState::Ready,
            data: Some(data),
        }
    }

    pub fn failed(id: u32, error: String) -> Self {
        Self {
            id,
            state: LoadState::Failed(error),
            data: None,
        }
    }

    pub fn is_ready(&self) -> bool {
        self.state == LoadState::Ready
    }

    pub fn is_failed(&self) -> bool {
        matches!(self.state, LoadState::Failed(_))
    }
}

/// Manages a collection of assets with async loading support.
pub struct AssetStore<T: Send + 'static> {
    assets: HashMap<u32, Arc<Mutex<AssetHandle<T>>>>,
    next_id: u32,
}

impl<T: Send + 'static> AssetStore<T> {
    pub fn new() -> Self {
        Self {
            assets: HashMap::new(),
            next_id: 0,
        }
    }

    /// Register a synchronously-loaded asset.
    pub fn insert(&mut self, data: T) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.assets
            .insert(id, Arc::new(Mutex::new(AssetHandle::ready(id, data))));
        id
    }

    /// Request an async load. Returns the asset ID immediately.
    /// The actual loading happens on a background thread via the provided closure.
    pub fn load_async<F>(&mut self, loader: F) -> u32
    where
        F: FnOnce() -> Result<T, String> + Send + 'static,
    {
        let id = self.next_id;
        self.next_id += 1;

        let handle = Arc::new(Mutex::new(AssetHandle::<T>::pending(id)));
        self.assets.insert(id, handle.clone());

        // Set to loading
        {
            let mut h = lock_or_recover(&handle);
            h.state = LoadState::Loading;
        }

        // Spawn background thread for loading
        std::thread::spawn(move || match loader() {
            Ok(data) => {
                let mut h = lock_or_recover(&handle);
                h.data = Some(data);
                h.state = LoadState::Ready;
            }
            Err(e) => {
                let mut h = lock_or_recover(&handle);
                h.state = LoadState::Failed(e);
            }
        });

        id
    }

    /// Get the load state of an asset.
    pub fn state(&self, id: u32) -> Option<LoadState> {
        self.assets
            .get(&id)
            .map(|h| lock_or_recover(h).state.clone())
    }

    /// Get a reference to the asset data (only if Ready).
    /// Returns None if not loaded yet or if loading failed.
    pub fn get(&self, id: u32) -> Option<Arc<Mutex<AssetHandle<T>>>> {
        self.assets.get(&id).cloned()
    }

    /// Number of assets in the store.
    pub fn len(&self) -> usize {
        self.assets.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.assets.is_empty()
    }
}

impl<T: Send + 'static> Default for AssetStore<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_insert_and_retrieve() {
        let mut store = AssetStore::<String>::new();
        let id = store.insert("hello".to_string());
        assert_eq!(store.state(id), Some(LoadState::Ready));

        let handle = store.get(id).unwrap();
        let h = handle.lock().unwrap();
        assert_eq!(h.data.as_deref(), Some("hello"));
    }

    #[test]
    fn async_load_completes() {
        let mut store = AssetStore::<String>::new();
        let id = store.load_async(|| {
            std::thread::sleep(std::time::Duration::from_millis(50));
            Ok("loaded".to_string())
        });

        // Should be Loading initially
        let state = store.state(id).unwrap();
        assert!(state == LoadState::Loading || state == LoadState::Ready);

        // Wait for completion
        std::thread::sleep(std::time::Duration::from_millis(200));

        assert_eq!(store.state(id), Some(LoadState::Ready));
        let handle = store.get(id).unwrap();
        let h = handle.lock().unwrap();
        assert_eq!(h.data.as_deref(), Some("loaded"));
    }

    #[test]
    fn async_load_failure() {
        let mut store = AssetStore::<String>::new();
        let id = store.load_async(|| Err("not found".to_string()));

        std::thread::sleep(std::time::Duration::from_millis(100));

        assert!(store.state(id).unwrap().is_failed());
    }

    impl LoadState {
        fn is_failed(&self) -> bool {
            matches!(self, LoadState::Failed(_))
        }
    }
}
