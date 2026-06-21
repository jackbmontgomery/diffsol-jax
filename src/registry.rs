//! Solver objects live in Rust but must be reachable from inside the XLA FFI
//! kernel, which can only carry plain bytes (an `int64` attr) across the
//! boundary. Rather than smuggle a raw address -- where a stale or freed object
//! deref is undefined behaviour -- we hand out an opaque `u64` id. A lookup miss
//! returns `None`, which the caller turns into a clean error.
//!
//! One generic `Registry<T>` is reused per object kind by instantiating a
//! separate static.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

pub struct Registry<T> {
    inner: Mutex<HashMap<u64, T>>,
    next: AtomicU64,
}

impl<T: Clone> Registry<T> {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            // Start at 1 so 0 is always an invalid ("null") id.
            next: AtomicU64::new(1),
        }
    }

    /// Store `value`, returning the id that retrieves it.
    pub fn insert(&self, value: T) -> u64 {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        self.inner.lock().unwrap().insert(id, value);
        id
    }

    /// Clone out the value for `id`, releasing the registry lock before the
    /// caller uses it. For `T = OdeWrapper` (an `Arc` handle) the clone is a
    /// refcount bump, so a long solve never holds the global lock.
    pub fn get(&self, id: u64) -> Option<T> {
        self.inner.lock().unwrap().get(&id).cloned()
    }

    /// Drop the entry for `id`. Called when the owning Python object is freed.
    pub fn remove(&self, id: u64) {
        self.inner.lock().unwrap().remove(&id);
    }
}

impl<T: Clone> Default for Registry<T> {
    fn default() -> Self {
        Self::new()
    }
}
