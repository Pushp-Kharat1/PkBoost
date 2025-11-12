/// Parallel utilities using fork_union
use fork_union::prelude::*;

/// Parallel map over a range - CORRECTED VERSION
pub fn par_map_range<F, R>(range: std::ops::Range<usize>, f: F) -> Vec<R>
where
    F: Fn(usize) -> R + Send + Sync,
    R: Send + std::fmt::Debug,
{
    let mut pool = fork_union::spawn(num_cpus::get());
    
    // ✅ CORRECT: Pre-allocate with SpinMutex
    let results: Vec<fork_union::SpinMutex<Option<R>>> = 
        (0..range.len()).map(|_| fork_union::SpinMutex::new(None)).collect();
    
    let range_start = range.start;
    range.into_par_iter()
        .with_pool(&mut pool)
        .for_each(|i| {
            let idx = i - range_start;  // Adjust for range start
            *results[idx].lock() = Some(f(i));
        });
    
    // Extract results
    results.into_iter()
        .map(|m| m.into_inner().unwrap())
        .collect()
}

/// Parallel zip map - CORRECTED VERSION
pub fn par_zip_map<T, U, F, R>(a: &[T], b: &[U], f: F) -> Vec<R>
where
    T: Sync,
    U: Sync,
    F: Fn(&T, &U) -> R + Send + Sync,
    R: Send + std::fmt::Debug,
{
    let mut pool = fork_union::spawn(num_cpus::get());
    let len = a.len().min(b.len());
    
    // ✅ CORRECT: Pre-allocate with SpinMutex
    let results: Vec<fork_union::SpinMutex<Option<R>>> = 
        (0..len).map(|_| fork_union::SpinMutex::new(None)).collect();
    
    (0..len)
        .into_par_iter()
        .with_pool(&mut pool)
        .for_each(|i| {
            *results[i].lock() = Some(f(&a[i], &b[i]));
        });
    
    // Extract results
    results.into_iter()
        .map(|m| m.into_inner().unwrap())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_par_map_range() {
        let result = par_map_range(0..10, |x| x * 2);
        assert_eq!(result, vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18]);
    }

    #[test]
    fn test_par_zip_map() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![10, 20, 30, 40, 50];
        let result = par_zip_map(&a, &b, |x, y| x + y);
        assert_eq!(result, vec![11, 22, 33, 44, 55]);
    }
}
