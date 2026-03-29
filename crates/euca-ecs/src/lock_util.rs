//! Helpers for recovering from poisoned `RwLock`/`Mutex` guards.
//!
//! Our locks protect caches and pools — not transactional state — so it is
//! safe to recover the inner data after another thread panicked.

use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Acquire a read guard, recovering from poison.
pub(crate) fn read_or_recover<'a, T>(lock: &'a RwLock<T>, context: &str) -> RwLockReadGuard<'a, T> {
    lock.read().unwrap_or_else(|e| {
        log::warn!("recovered from poisoned lock in {context}");
        e.into_inner()
    })
}

/// Acquire a write guard, recovering from poison.
pub(crate) fn write_or_recover<'a, T>(
    lock: &'a RwLock<T>,
    context: &str,
) -> RwLockWriteGuard<'a, T> {
    lock.write().unwrap_or_else(|e| {
        log::warn!("recovered from poisoned lock in {context}");
        e.into_inner()
    })
}
