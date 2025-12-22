// ForkUnion wrapper module for low-latency fork-join parallelism
// Uses ForkUnion for tight numerical loops where Rayon's overhead is noticeable
//
// This module provides ergonomic wrappers that maintain a global thread pool
// and expose simple parallel map operations optimized for PKBoost's workloads.

use fork_union as fu;
use std::cell::{RefCell, UnsafeCell};

/// Thread-local pool for ForkUnion operations
/// Uses RefCell instead of Mutex for zero synchronization overhead
/// SAFETY: ForkUnion dispatches from the calling thread, workers are internal
thread_local! {
    static POOL: RefCell<Option<fu::ThreadPool>> = RefCell::new(None);
}

/// Get or initialize the thread-local ForkUnion pool
fn with_pool<F, R>(f: F) -> R
where
    F: FnOnce(&mut fu::ThreadPool) -> R,
{
    POOL.with(|pool_cell| {
        let mut pool_opt = pool_cell.borrow_mut();

        // Lazily initialize the pool on first use
        if pool_opt.is_none() {
            let num_threads = num_cpus::get();
            *pool_opt = Some(
                fu::ThreadPool::try_spawn(num_threads)
                    .expect("Failed to create ForkUnion thread pool"),
            );
        }

        f(pool_opt.as_mut().unwrap())
    })
}

/// Wrapper for safe parallel writes - allows each thread to write to disjoint indices
/// This is the key primitive that makes parallel map safe
#[repr(transparent)]
struct ParallelVec<T> {
    inner: UnsafeCell<Vec<T>>,
}

// Safe because we ensure each thread only writes to its own index
unsafe impl<T: Send> Sync for ParallelVec<T> {}

impl<T> ParallelVec<T> {
    fn new(vec: Vec<T>) -> Self {
        Self {
            inner: UnsafeCell::new(vec),
        }
    }

    /// Write to a specific index - caller must ensure no concurrent writes to same index
    unsafe fn write(&self, idx: usize, val: T) {
        let ptr = self.inner.get();
        (&mut *ptr)[idx] = val;
    }

    fn into_inner(self) -> Vec<T> {
        self.inner.into_inner()
    }
}

/// Parallel map over a slice using ForkUnion's static scheduling
/// Best for uniform-cost workloads (e.g., histogram building, predictions)
///
/// # Example
/// ```ignore
/// let results = pfor_map(&data, |item| item * 2);
/// ```
pub fn pfor_map<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send + Default + Clone,
    F: Fn(&T) -> R + Sync,
{
    let n = items.len();

    if n == 0 {
        return Vec::new();
    }

    // For small workloads, avoid parallelism overhead
    if n < 64 {
        return items.iter().map(|item| f(item)).collect();
    }

    let results = ParallelVec::new(vec![R::default(); n]);

    // Use ForkUnion's for_n for static scheduling
    with_pool(|pool| {
        pool.for_n(n, |prong| {
            let idx = prong.task_index;
            let val = f(&items[idx]);
            // Safety: each task writes to a unique index (task_index is unique per task)
            unsafe {
                results.write(idx, val);
            }
        });
    });

    results.into_inner()
}

/// Parallel map over a range using ForkUnion
/// Best for index-based workloads where you need the index
pub fn pfor_range_map<R, F>(range: std::ops::Range<usize>, f: F) -> Vec<R>
where
    R: Send + Default + Clone,
    F: Fn(usize) -> R + Sync,
{
    let n = range.len();
    let start = range.start;

    if n == 0 {
        return Vec::new();
    }

    // For small workloads, avoid parallelism overhead
    if n < 64 {
        return range.map(|i| f(i)).collect();
    }

    let results = ParallelVec::new(vec![R::default(); n]);

    with_pool(|pool| {
        pool.for_n(n, |prong| {
            let local_idx = prong.task_index;
            let global_idx = start + local_idx;
            let val = f(global_idx);
            // Safety: each task writes to a unique index
            unsafe {
                results.write(local_idx, val);
            }
        });
    });

    results.into_inner()
}

/// Parallel map using dynamic work-stealing
/// Best for variable-cost workloads (e.g., feature binning with variable data sizes)
pub fn pfor_dynamic_map<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send + Default + Clone,
    F: Fn(&T) -> R + Sync,
{
    let n = items.len();

    if n == 0 {
        return Vec::new();
    }

    // For small workloads, avoid parallelism overhead
    if n < 64 {
        return items.iter().map(|item| f(item)).collect();
    }

    let results = ParallelVec::new(vec![R::default(); n]);

    // Use ForkUnion's for_n_dynamic for work-stealing (variable workloads)
    with_pool(|pool| {
        pool.for_n_dynamic(n, |prong| {
            let idx = prong.task_index;
            let val = f(&items[idx]);
            // Safety: each task writes to a unique index
            unsafe {
                results.write(idx, val);
            }
        });
    });

    results.into_inner()
}

/// Parallel zip-map over two slices using ForkUnion
/// Best for operations like histogram subtraction
pub fn pfor_zip_map<T, U, R, F>(a: &[T], b: &[U], f: F) -> Vec<R>
where
    T: Sync,
    U: Sync,
    R: Send + Default + Clone,
    F: Fn(&T, &U) -> R + Sync,
{
    let n = a.len().min(b.len());

    if n == 0 {
        return Vec::new();
    }

    // For small workloads, avoid parallelism overhead
    if n < 64 {
        return a.iter().zip(b.iter()).map(|(x, y)| f(x, y)).collect();
    }

    let results = ParallelVec::new(vec![R::default(); n]);

    with_pool(|pool| {
        pool.for_n(n, |prong| {
            let idx = prong.task_index;
            let val = f(&a[idx], &b[idx]);
            // Safety: each task writes to a unique index
            unsafe {
                results.write(idx, val);
            }
        });
    });

    results.into_inner()
}

/// Parallel for-each using ForkUnion with index
/// Used when we need to mutate external state via index
pub fn pfor_indexed<F>(n: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    if n == 0 {
        return;
    }

    // For small workloads, avoid parallelism overhead
    if n < 64 {
        for i in 0..n {
            f(i);
        }
        return;
    }

    with_pool(|pool| {
        pool.for_n(n, |prong| {
            f(prong.task_index);
        });
    });
}

/// Cache-aligned wrapper to prevent false sharing
/// Use this when accumulating results across threads
#[repr(align(64))]
#[derive(Debug, Clone, Default)]
pub struct CacheAligned<T>(pub T);

impl<T> CacheAligned<T> {
    pub fn new(val: T) -> Self {
        CacheAligned(val)
    }

    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> std::ops::Deref for CacheAligned<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for CacheAligned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pfor_map() {
        let data: Vec<i32> = (0..1000).collect();
        let results = pfor_map(&data, |x| x * 2);
        assert_eq!(results.len(), 1000);
        assert_eq!(results[0], 0);
        assert_eq!(results[500], 1000);
        assert_eq!(results[999], 1998);
    }

    #[test]
    fn test_pfor_range_map() {
        let results = pfor_range_map(0..1000, |i| i * 3);
        assert_eq!(results.len(), 1000);
        assert_eq!(results[0], 0);
        assert_eq!(results[333], 999);
    }

    #[test]
    fn test_pfor_zip_map() {
        let a: Vec<i32> = (0..500).collect();
        let b: Vec<i32> = (500..1000).collect();
        let results = pfor_zip_map(&a, &b, |x, y| x + y);
        assert_eq!(results.len(), 500);
        assert_eq!(results[0], 500);
        assert_eq!(results[100], 700);
    }

    #[test]
    fn test_pfor_dynamic_map() {
        let data: Vec<i32> = (0..1000).collect();
        let results = pfor_dynamic_map(&data, |x| x * 2);
        assert_eq!(results.len(), 1000);
        assert_eq!(results[0], 0);
        assert_eq!(results[999], 1998);
    }

    #[test]
    fn test_cache_aligned() {
        let aligned = CacheAligned::new(42i32);
        assert_eq!(*aligned, 42);
        assert_eq!(std::mem::align_of_val(&aligned), 64);
    }
}
