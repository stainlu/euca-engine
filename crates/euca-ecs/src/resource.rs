use std::any::{Any, TypeId};
use std::collections::HashMap;

/// Type-erased storage for singleton resources.
///
/// Resources are global data accessible by systems — things like Time, InputState, AssetServer.
pub struct Resources {
    data: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Resources {
    /// Creates an empty resource store.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Insert a resource. Overwrites if already present.
    pub fn insert<T: Send + Sync + 'static>(&mut self, value: T) {
        self.data.insert(TypeId::of::<T>(), Box::new(value));
    }

    /// Get an immutable reference to a resource.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.data
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref())
    }

    /// Get a mutable reference to a resource.
    pub fn get_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.data
            .get_mut(&TypeId::of::<T>())
            .and_then(|b| b.downcast_mut())
    }

    /// Remove a resource, returning it.
    pub fn remove<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.data
            .remove(&TypeId::of::<T>())
            .and_then(|b| b.downcast().ok())
            .map(|b| *b)
    }

    /// Check if a resource exists.
    pub fn contains<T: Send + Sync + 'static>(&self) -> bool {
        self.data.contains_key(&TypeId::of::<T>())
    }
}

impl Default for Resources {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Time {
        elapsed: f64,
        delta: f32,
    }

    #[derive(Debug, PartialEq)]
    struct Score(u32);

    #[test]
    fn insert_and_get() {
        let mut res = Resources::new();
        res.insert(Time {
            elapsed: 1.0,
            delta: 0.016,
        });
        res.insert(Score(42));

        assert_eq!(res.get::<Time>().unwrap().elapsed, 1.0);
        assert_eq!(res.get::<Score>().unwrap().0, 42);
    }

    #[test]
    fn get_mut() {
        let mut res = Resources::new();
        res.insert(Score(0));
        res.get_mut::<Score>().unwrap().0 += 10;
        assert_eq!(res.get::<Score>().unwrap().0, 10);
    }

    #[test]
    fn remove() {
        let mut res = Resources::new();
        res.insert(Score(42));
        let removed = res.remove::<Score>();
        assert_eq!(removed, Some(Score(42)));
        assert!(!res.contains::<Score>());
    }

    #[test]
    fn missing_resource() {
        let res = Resources::new();
        assert!(res.get::<Score>().is_none());
    }

    #[test]
    fn overwrite() {
        let mut res = Resources::new();
        res.insert(Score(1));
        res.insert(Score(2));
        assert_eq!(res.get::<Score>().unwrap().0, 2);
    }
}
