// ForkUnion wrapper module for low-latency fork-join parallelism
// Uses ForkUnion for tight numerical loops where Rayon's overhead is noticeable
//
// This module provides ergonomic wrappers that maintain a global thread pool
// and expose simple parallel map/mutate operations optimized for PKBoost's workloads.

use fork_union as fu;
use std::cell::UnsafeCell;
use std::sync::{Mutex, OnceLock};

// Global thread pool — wrapped in Mutex because fork_union's for_n/for_n_dynamic
// take &mut self (they modify internal state to dispatch work).
static GLOBAL_POOL: OnceLock<Mutex<fu::ThreadPool>> = OnceLock::new();

fn with_pool<F, R>(f: F) -> R
where
    F: FnOnce(&mut fu::ThreadPool) -> R,
{
    let mutex = GLOBAL_POOL.get_or_init(|| {
        let num_threads = num_cpus::get();
        let pool =
            fu::ThreadPool::try_spawn(num_threads).expect("Failed to create ForkUnion thread pool");
        Mutex::new(pool)
    });
    let mut guard = mutex.lock().expect("ForkUnion pool mutex poisoned");
    f(&mut *guard)
}

// ---------------------------------------------------------------------------
// ParallelVec: safe parallel writes to disjoint indices via UnsafeCell
// ---------------------------------------------------------------------------

/// Wrapper that allows parallel writes to a Vec when each thread owns disjoint indices.
#[repr(transparent)]
struct ParallelVec<T> {
    inner: UnsafeCell<Vec<T>>,
}

// SAFETY: We uphold the invariant that no two threads write to the same index.
unsafe impl<T: Send> Send for ParallelVec<T> {}
unsafe impl<T: Send> Sync for ParallelVec<T> {}

impl<T> ParallelVec<T> {
    fn new(vec: Vec<T>) -> Self {
        Self {
            inner: UnsafeCell::new(vec),
        }
    }

    /// Write `val` to `idx`. Caller must ensure no other thread writes to `idx`.
    unsafe fn write(&self, idx: usize, val: T) {
        let vec = &mut *self.inner.get();
        vec[idx] = val;
    }

    /// Mutate element at `idx` via closure. Caller must ensure exclusive access to `idx`.
    unsafe fn with_mut<F: FnOnce(&mut T)>(&self, idx: usize, f: F) {
        let vec = &mut *self.inner.get();
        f(&mut vec[idx]);
    }

    fn into_inner(self) -> Vec<T> {
        self.inner.into_inner()
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parallel map over a slice using ForkUnion's static scheduling.
/// Best for uniform-cost workloads (e.g., histogram building, predictions).
///
/// # Example
/// ```ignore
/// let results = pfor_map(&data, |item| item * 2);
/// ```
/// Parallel map over a slice using ForkUnion's static scheduling.
pub fn pfor_map<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync,
{
    let n = items.len();
    if n == 0 {
        return Vec::new();
    }
    if n < 64 {
        return items.iter().map(f).collect();
    }

    use std::mem::MaybeUninit;
    let mut results = Vec::with_capacity(n);
    unsafe { results.set_len(n) };
    let results_ptr = SendPtr(results.as_mut_ptr() as *mut MaybeUninit<R>);

    with_pool(|pool| {
        let num_threads = pool.threads().min(n);
        let chunk_size = (n + num_threads - 1) / num_threads;
        let results_ptr = results_ptr; // Shadow OUTSIDE the closure to ensure Sync capture
        pool.for_n(num_threads, |prong| {
            let start = prong.task_index * chunk_size;
            let end = (start + chunk_size).min(n);
            for idx in start..end {
                unsafe {
                    (*results_ptr.addr().add(idx)).write(f(&items[idx]));
                }
            }
        });
    });

    results
}

/// Parallel map over a range using ForkUnion.
pub fn pfor_range_map<R, F>(range: std::ops::Range<usize>, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync,
{
    let n = range.len();
    let start = range.start;
    if n == 0 {
        return Vec::new();
    }
    if n < 64 {
        return range.map(f).collect();
    }

    use std::mem::MaybeUninit;
    let mut results = Vec::with_capacity(n);
    unsafe { results.set_len(n) };
    let results_ptr = SendPtr(results.as_mut_ptr() as *mut MaybeUninit<R>);

    with_pool(|pool| {
        let num_threads = pool.threads().min(n);
        let chunk_size = (n + num_threads - 1) / num_threads;
        let results_ptr = results_ptr; // Shadow OUTSIDE the closure to ensure Sync capture
        pool.for_n(num_threads, |prong| {
            let local_start = prong.task_index * chunk_size;
            let local_end = (local_start + chunk_size).min(n);
            for local_idx in local_start..local_end {
                let global_idx = start + local_idx;
                unsafe {
                    (*results_ptr.addr().add(local_idx)).write(f(global_idx));
                }
            }
        });
    });

    results
}

/// Parallel map using dynamic work-stealing.
/// Best for variable-cost workloads (e.g., feature binning with variable data sizes).
pub fn pfor_dynamic_map<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync,
{
    let n = items.len();
    if n == 0 {
        return Vec::new();
    }
    if n < 64 {
        return items.iter().map(f).collect();
    }

    use std::mem::MaybeUninit;
    let mut results = Vec::with_capacity(n);
    unsafe { results.set_len(n) };
    let results_ptr = SendPtr(results.as_mut_ptr() as *mut MaybeUninit<R>);

    with_pool(|pool| {
        let num_threads = pool.threads().min(n);
        let chunk_size = (n + num_threads - 1) / num_threads;
        let results_ptr = results_ptr; // Shadow OUTSIDE
        pool.for_n_dynamic(num_threads, |prong| {
            let start = prong.task_index * chunk_size;
            let end = (start + chunk_size).min(n);
            for idx in start..end {
                unsafe {
                    (*results_ptr.addr().add(idx)).write(f(&items[idx]));
                }
            }
        });
    });

    results
}

/// Parallel zip-map over two slices using ForkUnion.
pub fn pfor_zip_map<T, U, R, F>(a: &[T], b: &[U], f: F) -> Vec<R>
where
    T: Sync,
    U: Sync,
    R: Send,
    F: Fn(&T, &U) -> R + Sync,
{
    let n = a.len().min(b.len());
    if n == 0 {
        return Vec::new();
    }
    if n < 64 {
        return a.iter().zip(b.iter()).map(|(x, y)| f(x, y)).collect();
    }

    use std::mem::MaybeUninit;
    let mut results = Vec::with_capacity(n);
    unsafe { results.set_len(n) };
    let results_ptr = SendPtr(results.as_mut_ptr() as *mut MaybeUninit<R>);

    with_pool(|pool| {
        let num_threads = pool.threads().min(n);
        let chunk_size = (n + num_threads - 1) / num_threads;
        let results_ptr = results_ptr; // Shadow OUTSIDE
        pool.for_n(num_threads, |prong| {
            let start = prong.task_index * chunk_size;
            let end = (start + chunk_size).min(n);
            for idx in start..end {
                unsafe {
                    (*results_ptr.addr().add(idx)).write(f(&a[idx], &b[idx]));
                }
            }
        });
    });

    results
}

/// Parallel for-each using ForkUnion with index.
/// Used when we need to mutate external state via index (e.g. atomic counters).
pub fn pfor_indexed<F>(n: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    if n == 0 {
        return;
    }
    if n < 64 {
        for i in 0..n {
            f(i);
        }
        return;
    }

    with_pool(|pool| {
        let num_threads = pool.threads().min(n);
        let chunk_size = (n + num_threads - 1) / num_threads;
        pool.for_n(num_threads, |prong| {
            let start = prong.task_index * chunk_size;
            let end = (start + chunk_size).min(n);
            for i in start..end {
                f(i);
            }
        });
    });
}

/// Parallel for-each on a mutable slice.
/// Replaces: `slice.par_iter_mut().for_each(|x| { ... })`
pub fn pfor_for_each_mut<T, F>(data: &mut [T], f: F)
where
    T: Send,
    F: Fn(&mut T) + Sync,
{
    let n = data.len();
    if n == 0 {
        return;
    }
    if n < 64 {
        data.iter_mut().for_each(|x| f(x));
        return;
    }

    // Build a non-owning Vec view so we can wrap it in ParallelVec.
    // SAFETY: `data` outlives the parallel region; we forget the Vec before it drops.
    let par = ParallelVec {
        inner: UnsafeCell::new(unsafe { Vec::from_raw_parts(data.as_mut_ptr(), n, n) }),
    };

    with_pool(|pool| {
        let num_threads = pool.threads().min(n);
        let chunk_size = (n + num_threads - 1) / num_threads;
        pool.for_n(num_threads, |prong| {
            let start = prong.task_index * chunk_size;
            let end = (start + chunk_size).min(n);
            for idx in start..end {
                // SAFETY: disjoint chunks, no aliasing.
                unsafe { par.with_mut(idx, |x| f(x)) };
            }
        });
    });

    // Prevent the inner Vec from freeing memory it doesn't own.
    std::mem::forget(par.inner.into_inner());
}

/// Parallel for-each on a mutable slice with an immutable zip.
/// Replaces: `a.par_iter_mut().zip(b.par_iter()).for_each(|(x, y)| { ... })`
pub fn pfor_zip_for_each_mut<T, U, F>(data: &mut [T], other: &[U], f: F)
where
    T: Send,
    U: Sync,
    F: Fn(&mut T, &U) + Sync,
{
    let n = data.len().min(other.len());
    if n == 0 {
        return;
    }
    if n < 64 {
        data.iter_mut().zip(other.iter()).for_each(|(x, y)| f(x, y));
        return;
    }

    let par = ParallelVec {
        inner: UnsafeCell::new(unsafe { Vec::from_raw_parts(data.as_mut_ptr(), n, n) }),
    };

    with_pool(|pool| {
        let num_threads = pool.threads();
        let chunk_size = (n + num_threads - 1) / num_threads;
        pool.for_n(num_threads, |prong| {
            let start = prong.task_index * chunk_size;
            let end = (start + chunk_size).min(n);
            for idx in start..end {
                // SAFETY: disjoint chunks; `other` is read-only.
                unsafe { par.with_mut(idx, |x| f(x, &other[idx])) };
            }
        });
    });

    std::mem::forget(par.inner.into_inner());
}

/// Parallel for-each on 2 mutable slices zipped with 2 immutable slices.
/// Replaces: `grad.par_iter_mut().zip(hess).zip(y_pred).zip(y_true).for_each(...)`
pub fn pfor_zip4_for_each_mut<A, B, C, D, F>(a: &mut [A], b: &mut [B], c: &[C], d: &[D], f: F)
where
    A: Send,
    B: Send,
    C: Sync,
    D: Sync,
    F: Fn(&mut A, &mut B, &C, &D) + Sync,
{
    let n = a.len().min(b.len()).min(c.len()).min(d.len());
    if n == 0 {
        return;
    }
    if n < 64 {
        for i in 0..n {
            f(&mut a[i], &mut b[i], &c[i], &d[i]);
        }
        return;
    }

    // Wrap both mutable slices in ParallelVec — this gives us `Sync` via our impl,
    // avoiding the bare UnsafeCell<Vec<T>> which is explicitly !Sync.
    let par_a = ParallelVec {
        inner: UnsafeCell::new(unsafe { Vec::from_raw_parts(a.as_mut_ptr(), n, n) }),
    };
    let par_b = ParallelVec {
        inner: UnsafeCell::new(unsafe { Vec::from_raw_parts(b.as_mut_ptr(), n, n) }),
    };

    with_pool(|pool| {
        let num_threads = pool.threads();
        let chunk_size = (n + num_threads - 1) / num_threads;
        pool.for_n(num_threads, |prong| {
            let start = prong.task_index * chunk_size;
            let end = (start + chunk_size).min(n);
            for idx in start..end {
                // SAFETY: disjoint chunks across both mutable slices.
                // We go through `with_mut` on par_a/par_b (which are &ParallelVec,
                // not &UnsafeCell) so the closure is Sync.
                unsafe {
                    par_a.with_mut(idx, |pa| {
                        par_b.with_mut(idx, |pb| {
                            f(pa, pb, &c[idx], &d[idx]);
                        });
                    });
                }
            }
        });
    });

    std::mem::forget(par_a.inner.into_inner());
    std::mem::forget(par_b.inner.into_inner());
}

/// Parallel reduce: compute per-thread partial results and combine.
/// Replaces: `par_iter().map(...).sum()` and similar patterns.
pub fn pfor_reduce<T, R, MapF, ReduceF>(
    items: &[T],
    identity: R,
    map_f: MapF,
    reduce_f: ReduceF,
) -> R
where
    T: Sync,
    R: Send + Clone,
    MapF: Fn(&T) -> R + Sync,
    ReduceF: Fn(R, R) -> R + Sync, // needs Sync so it can be captured in the for_n closure
{
    let n = items.len();
    if n == 0 {
        return identity;
    }
    if n < 64 {
        return items
            .iter()
            .map(|item| map_f(item))
            .fold(identity, |acc, x| reduce_f(acc, x));
    }

    let num_threads = with_pool(|pool| pool.threads());
    let chunk_size = (n + num_threads - 1) / num_threads;

    // One partial-result slot per thread, cache-line padded to avoid false sharing.
    let partials: ParallelVec<CacheAligned<Option<R>>> =
        ParallelVec::new((0..num_threads).map(|_| CacheAligned::new(None)).collect());

    with_pool(|pool| {
        pool.for_n(num_threads, |prong| {
            let start = prong.task_index * chunk_size;
            let end = (start + chunk_size).min(n);
            if start >= n {
                return;
            }
            let local =
                items[start..end]
                    .iter()
                    .map(|item| map_f(item))
                    .fold(None::<R>, |acc, x| {
                        Some(match acc {
                            None => x,
                            Some(a) => reduce_f(a, x),
                        })
                    });
            // SAFETY: each thread writes to its own unique slot.
            unsafe { partials.write(prong.task_index, CacheAligned::new(local)) };
        });
    });

    partials
        .into_inner()
        .into_iter()
        .filter_map(|ca| ca.into_inner())
        .fold(identity, |acc, x| reduce_f(acc, x))
}

// ---------------------------------------------------------------------------
// CacheAligned: prevents false sharing when accumulating across threads
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// SendPtr: bypass Send/Sync for raw pointers (SAFETY REQUIRED)
// ---------------------------------------------------------------------------

#[repr(transparent)]
#[derive(Debug)]
pub struct SendPtr<T>(pub *mut T);

impl<T> SendPtr<T> {
    #[inline(always)]
    pub fn addr(self) -> *mut T {
        self.0
    }
}

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SendPtr<T> {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
    fn test_pfor_for_each_mut() {
        let mut data: Vec<i32> = (0..100).collect();
        pfor_for_each_mut(&mut data, |x| *x *= 2);
        assert_eq!(data[0], 0);
        assert_eq!(data[50], 100);
        assert_eq!(data[99], 198);
    }

    #[test]
    fn test_pfor_zip_for_each_mut() {
        let mut data: Vec<i32> = vec![1; 100];
        let other: Vec<i32> = (0..100).collect();
        pfor_zip_for_each_mut(&mut data, &other, |x, y| *x += y);
        assert_eq!(data[0], 1);
        assert_eq!(data[50], 51);
        assert_eq!(data[99], 100);
    }

    #[test]
    fn test_pfor_zip4_for_each_mut() {
        let mut a: Vec<i32> = vec![0; 100];
        let mut b: Vec<i32> = vec![0; 100];
        let c: Vec<i32> = (0..100).collect();
        let d: Vec<i32> = (100..200).collect();
        pfor_zip4_for_each_mut(&mut a, &mut b, &c, &d, |ai, bi, ci, di| {
            *ai = *ci;
            *bi = *di;
        });
        assert_eq!(a[0], 0);
        assert_eq!(a[99], 99);
        assert_eq!(b[0], 100);
        assert_eq!(b[99], 199);
    }

    #[test]
    fn test_pfor_reduce() {
        let data: Vec<i32> = (0..100).collect();
        let sum = pfor_reduce(&data, 0, |x| *x, |a, b| a + b);
        assert_eq!(sum, 4950); // 0+1+...+99 = 4950
    }

    #[test]
    fn test_cache_aligned() {
        let aligned = CacheAligned::new(42i32);
        assert_eq!(*aligned, 42);
        assert_eq!(std::mem::align_of_val(&aligned), 64);
    }
}
