use std::any::{Any, TypeId};
use std::collections::HashMap;

/// Type-erased storage for singleton resources.
///
/// Resources are global data accessible by systems — things like Time,
/// InputState, AssetServer. Every resource must implement `Clone` so that
/// [`World::clone`](crate::World::clone) can deep-copy the full world
/// state for forking. Resources that represent shared infrastructure
/// (GPU devices, network handles, asset caches) should be wrapped in
/// `Arc<T>` at the insertion site — `Arc<T>: Clone` shares the handle
/// cheaply, while actual state resources (Time, Score, GameState) are
/// duplicated deeply.
pub struct Resources {
    /// The actual resource values, keyed by `TypeId`. Stored through the
    /// `Any` trait so that `get`/`remove` can downcast back to the
    /// concrete type.
    data: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
    /// Parallel map of type-erased clone functions. Populated on `insert`
    /// and used by `Clone for Resources` to produce an independent copy
    /// without knowing the concrete type of each resource at clone time.
    clone_fns: HashMap<TypeId, fn(&(dyn Any + Send + Sync)) -> Box<dyn Any + Send + Sync>>,
}

/// Type-erased helper: clone `value` (known to be a `T`) into a fresh `Box`.
fn clone_as<T: Any + Clone + Send + Sync>(
    value: &(dyn Any + Send + Sync),
) -> Box<dyn Any + Send + Sync> {
    let concrete = value
        .downcast_ref::<T>()
        .expect("clone_fn called on wrong concrete type");
    Box::new(concrete.clone())
}

impl Resources {
    /// Creates an empty resource store.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            clone_fns: HashMap::new(),
        }
    }

    /// Insert a resource. Overwrites if already present. `T` must implement
    /// `Clone` so the resource can be carried into [`World::clone`] forks.
    pub fn insert<T: Send + Sync + Clone + 'static>(&mut self, value: T) {
        let type_id = TypeId::of::<T>();
        self.data.insert(type_id, Box::new(value));
        self.clone_fns.insert(type_id, clone_as::<T>);
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
        let type_id = TypeId::of::<T>();
        self.clone_fns.remove(&type_id);
        self.data
            .remove(&type_id)
            .and_then(|b| b.downcast().ok())
            .map(|b| *b)
    }

    /// Check if a resource exists.
    pub fn contains<T: Send + Sync + 'static>(&self) -> bool {
        self.data.contains_key(&TypeId::of::<T>())
    }
}

impl Clone for Resources {
    fn clone(&self) -> Self {
        let mut data: HashMap<TypeId, Box<dyn Any + Send + Sync>> =
            HashMap::with_capacity(self.data.len());
        for (type_id, value) in &self.data {
            let clone_fn = self
                .clone_fns
                .get(type_id)
                .expect("resource was inserted without a clone_fn — this is an invariant bug");
            let cloned = clone_fn(value.as_ref());
            data.insert(*type_id, cloned);
        }
        Resources {
            data,
            clone_fns: self.clone_fns.clone(),
        }
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

    #[derive(Clone, Debug, PartialEq)]
    struct Time {
        elapsed: f64,
        delta: f32,
    }

    #[derive(Clone, Debug, PartialEq)]
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

    #[test]
    fn clone_resources_deep() {
        let mut res = Resources::new();
        res.insert(Score(10));
        let mut cloned = res.clone();
        cloned.get_mut::<Score>().unwrap().0 = 99;
        assert_eq!(res.get::<Score>().unwrap().0, 10);
        assert_eq!(cloned.get::<Score>().unwrap().0, 99);
    }
}
