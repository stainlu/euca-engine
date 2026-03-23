//! System access tracking for future parallel scheduling.
//!
//! Each system can declare what it reads/writes so the scheduler can
//! determine which systems can run in parallel.

use std::any::TypeId;

use crate::component::ComponentId;

/// Describes a single read or write access by a system.
///
/// Used by the scheduler to determine which systems can run in parallel.
/// Two systems conflict if one writes a component or resource that the other
/// reads or writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SystemAccess {
    /// Shared (immutable) access to a component type.
    ComponentRead(ComponentId),
    /// Exclusive (mutable) access to a component type.
    ComponentWrite(ComponentId),
    /// Shared (immutable) access to a resource type.
    ResourceRead(TypeId),
    /// Exclusive (mutable) access to a resource type.
    ResourceWrite(TypeId),
}

/// Validates that two sets of system accesses have no conflicts.
///
/// Returns `true` if the two systems can safely run in parallel. A conflict
/// exists when one system writes a component or resource that the other
/// reads or writes. Two read-only accesses to the same item are not a conflict.
#[allow(dead_code)] // Used by future parallel scheduler (#3)
pub fn validate_no_conflicts(a: &[SystemAccess], b: &[SystemAccess]) -> bool {
    for access_a in a {
        for access_b in b {
            let conflict = match (access_a, access_b) {
                (SystemAccess::ComponentWrite(id_a), SystemAccess::ComponentRead(id_b))
                | (SystemAccess::ComponentRead(id_a), SystemAccess::ComponentWrite(id_b))
                | (SystemAccess::ComponentWrite(id_a), SystemAccess::ComponentWrite(id_b)) => {
                    id_a == id_b
                }
                (SystemAccess::ResourceWrite(id_a), SystemAccess::ResourceRead(id_b))
                | (SystemAccess::ResourceRead(id_a), SystemAccess::ResourceWrite(id_b))
                | (SystemAccess::ResourceWrite(id_a), SystemAccess::ResourceWrite(id_b)) => {
                    id_a == id_b
                }
                _ => false,
            };
            if conflict {
                return false;
            }
        }
    }
    true
}

/// Immutable reference wrapper for an ECS resource.
///
/// Provides `Deref` to `T` for ergonomic access. Wraps a shared reference
/// obtained from [`World::resource`](crate::World::resource).
pub struct Res<'w, T: Send + Sync + 'static>(pub &'w T);

impl<T: Send + Sync + 'static> std::ops::Deref for Res<'_, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        self.0
    }
}

/// Mutable reference wrapper for an ECS resource.
///
/// Provides `Deref` and `DerefMut` to `T` for ergonomic access. Wraps an
/// exclusive reference obtained from [`World::resource_mut`](crate::World::resource_mut).
pub struct ResMut<'w, T: Send + Sync + 'static>(pub &'w mut T);

impl<T: Send + Sync + 'static> std::ops::Deref for ResMut<'_, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        self.0
    }
}

impl<T: Send + Sync + 'static> std::ops::DerefMut for ResMut<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_conflict_different_components() {
        let a = vec![SystemAccess::ComponentRead(ComponentId::from_raw(0))];
        let b = vec![SystemAccess::ComponentWrite(ComponentId::from_raw(1))];
        assert!(validate_no_conflicts(&a, &b));
    }

    #[test]
    fn conflict_same_component_read_write() {
        let a = vec![SystemAccess::ComponentRead(ComponentId::from_raw(0))];
        let b = vec![SystemAccess::ComponentWrite(ComponentId::from_raw(0))];
        assert!(!validate_no_conflicts(&a, &b));
    }

    #[test]
    fn no_conflict_both_read() {
        let a = vec![SystemAccess::ComponentRead(ComponentId::from_raw(0))];
        let b = vec![SystemAccess::ComponentRead(ComponentId::from_raw(0))];
        assert!(validate_no_conflicts(&a, &b));
    }

    #[test]
    fn conflict_same_resource_write_write() {
        let a = vec![SystemAccess::ResourceWrite(TypeId::of::<u32>())];
        let b = vec![SystemAccess::ResourceWrite(TypeId::of::<u32>())];
        assert!(!validate_no_conflicts(&a, &b));
    }

    #[test]
    fn no_conflict_different_resource_types() {
        let a = vec![SystemAccess::ResourceRead(TypeId::of::<u32>())];
        let b = vec![SystemAccess::ResourceWrite(TypeId::of::<f32>())];
        assert!(validate_no_conflicts(&a, &b));
    }
}
