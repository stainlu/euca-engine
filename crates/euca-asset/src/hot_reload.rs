//! Hot-reload: watch asset directories for changes and trigger reloads.
//!
//! Uses polling (no external dependency) — checks file modification times.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Watches a directory for file changes by polling modification times.
pub struct FileWatcher {
    /// Directories to watch.
    watch_dirs: Vec<PathBuf>,
    /// Last known modification time per file.
    mod_times: HashMap<PathBuf, SystemTime>,
    /// Files that changed since last poll.
    changed: Vec<PathBuf>,
}

impl FileWatcher {
    pub fn new() -> Self {
        Self {
            watch_dirs: Vec::new(),
            mod_times: HashMap::new(),
            changed: Vec::new(),
        }
    }

    /// Add a directory to watch.
    pub fn watch(&mut self, dir: impl AsRef<Path>) {
        self.watch_dirs.push(dir.as_ref().to_path_buf());
    }

    /// Poll for changes. Returns paths of files that were modified since last poll.
    pub fn poll(&mut self) -> &[PathBuf] {
        self.changed.clear();

        for dir in &self.watch_dirs.clone() {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if !path.is_file() {
                        continue;
                    }

                    let mod_time = match path.metadata().and_then(|m| m.modified()) {
                        Ok(t) => t,
                        Err(_) => continue,
                    };

                    let prev = self.mod_times.get(&path);
                    if prev != Some(&mod_time) {
                        if prev.is_some() {
                            // File was modified (not first scan)
                            self.changed.push(path.clone());
                            log::info!("File changed: {}", path.display());
                        }
                        self.mod_times.insert(path, mod_time);
                    }
                }
            }
        }

        &self.changed
    }

    /// Get the list of watched directories.
    pub fn watched_dirs(&self) -> &[PathBuf] {
        &self.watch_dirs
    }
}

impl Default for FileWatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_watcher_no_changes() {
        let mut watcher = FileWatcher::new();
        let changes = watcher.poll();
        assert!(changes.is_empty());
    }

    #[test]
    fn watch_nonexistent_dir() {
        let mut watcher = FileWatcher::new();
        watcher.watch("/nonexistent/path/123456");
        let changes = watcher.poll();
        assert!(changes.is_empty()); // Should not panic
    }
}
