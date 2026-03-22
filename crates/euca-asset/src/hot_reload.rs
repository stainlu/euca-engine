//! Hot-reload: watch asset directories and individual files for changes.
//!
//! Uses polling (no external dependency) — checks file modification times.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Watches directories and individual files for changes by polling modification times.
pub struct FileWatcher {
    /// Directories to watch (all files inside are scanned).
    watch_dirs: Vec<PathBuf>,
    /// Individual files to watch.
    watch_files: Vec<PathBuf>,
    /// Last known modification time per file.
    mod_times: HashMap<PathBuf, SystemTime>,
    /// Files that changed since last poll.
    changed: Vec<PathBuf>,
}

impl FileWatcher {
    pub fn new() -> Self {
        Self {
            watch_dirs: Vec::new(),
            watch_files: Vec::new(),
            mod_times: HashMap::new(),
            changed: Vec::new(),
        }
    }

    /// Add a directory to watch (all files inside are scanned each poll).
    pub fn watch(&mut self, dir: impl AsRef<Path>) {
        self.watch_dirs.push(dir.as_ref().to_path_buf());
    }

    /// Add an individual file to watch.
    pub fn watch_file(&mut self, file: impl AsRef<Path>) {
        self.watch_files.push(file.as_ref().to_path_buf());
    }

    /// Poll for changes. Returns paths of files that were modified since last poll.
    pub fn poll(&mut self) -> &[PathBuf] {
        self.changed.clear();

        // Scan directories
        for dir in &self.watch_dirs.clone() {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if !path.is_file() {
                        continue;
                    }
                    self.check_file(path);
                }
            }
        }

        // Scan individual files
        for file in &self.watch_files.clone() {
            if file.is_file() {
                self.check_file(file.clone());
            }
        }

        &self.changed
    }

    /// Check a single file for modification. Adds to `changed` if modified.
    fn check_file(&mut self, path: PathBuf) {
        let mod_time = match path.metadata().and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => return,
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

    /// Get the list of watched directories.
    pub fn watched_dirs(&self) -> &[PathBuf] {
        &self.watch_dirs
    }

    /// Get the list of watched individual files.
    pub fn watched_files(&self) -> &[PathBuf] {
        &self.watch_files
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

    #[test]
    fn detect_file_modification() {
        let dir = std::env::temp_dir().join("euca_watcher_test");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("test_file.txt");
        std::fs::write(&file, "initial").unwrap();

        let mut watcher = FileWatcher::new();
        watcher.watch(&dir);

        // First poll seeds the mod times — no changes reported
        let changes = watcher.poll();
        assert!(changes.is_empty());

        // Modify the file
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::write(&file, "modified").unwrap();

        // Second poll should detect the change
        let changes = watcher.poll();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0], file);

        // Third poll with no changes
        let changes = watcher.poll();
        assert!(changes.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn watch_individual_file() {
        let file = std::env::temp_dir().join("euca_watcher_single.txt");
        std::fs::write(&file, "v1").unwrap();

        let mut watcher = FileWatcher::new();
        watcher.watch_file(&file);

        // Seed
        let changes = watcher.poll();
        assert!(changes.is_empty());

        // Modify
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::write(&file, "v2").unwrap();

        let changes = watcher.poll();
        assert_eq!(changes.len(), 1);

        std::fs::remove_file(&file).ok();
    }
}
