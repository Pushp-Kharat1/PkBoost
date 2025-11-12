/// Parallel utilities using fork_union
use fork_union::prelude::*;
use fork_union::SpinMutex;

pub fn par_map_range<F, R>(range: std::ops::Range<usize>, f: F) -> Vec<R>
where
    F: Fn(usize) -> R + Send + Sync,
    R: Send + Default,
{
    let mut pool = fork_union::spawn(num_cpus::get());
    
    // Create vector directly without Arc
    let results: Vec<SpinMutex<R>> = (0..range.len())
        .map(|_| SpinMutex::new(R::default()))
        .collect();
    
    let range_start = range.start;
    
    // Process in parallel - results is borrowed, not moved
    range.clone().into_par_iter().with_pool(&mut pool).for_each(|i| {
        let idx = i - range_start;
        *results[idx].lock() = f(i);
    });
    
    // Extract results
    results.into_iter().map(|m| m.into_inner()).collect()
}

pub fn par_zip_map<T, U, F, R>(a: &[T], b: &[U], f: F) -> Vec<R>
where
    T: Sync,
    U: Sync,
    F: Fn(&T, &U) -> R + Send + Sync,
    R: Send,
{
    a.iter().zip(b.iter()).map(|(x, y)| f(x, y)).collect()
}
