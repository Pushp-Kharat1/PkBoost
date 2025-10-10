use rayon::prelude::*;
use std::sync::OnceLock;
#[derive(Debug, Clone)]
pub struct AdaptiveParallelConfig {
    pub num_threads: usize,
    pub parallel_threshold_small: usize,
    pub parallel_threshold_medium: usize,
    pub parallel_threshold_large: usize,
    pub batch_size: usize,
    pub chunk_size: usize,
    pub memory_efficient_mode: bool,
}

impl AdaptiveParallelConfig {
    pub fn detect_hardware() -> Self {
        let num_threads = rayon::current_num_threads();
        let num_cores = num_cpus::get_physical();
        
        let total_memory_gb = Self::estimate_memory_gb();
        let is_low_core_count = num_cores <= 4;
        let is_low_memory = total_memory_gb < 8.0;
        
        let (small_thresh, med_thresh, large_thresh, batch_sz, chunk_sz) = match (num_cores, is_low_memory) {
            (1..=4, _) => (2000, 8000, 20000, 5000, 1000),
            (5..=8, false) => (1000, 4000, 10000, 10000, 2000),
            (5..=8, true) => (1500, 6000, 15000, 8000, 1500),
            (9..=16, false) => (500, 2000, 5000, 20000, 4000),
            (9..=16, true) => (800, 3000, 8000, 15000, 3000),
            (17.., false) => (200, 1000, 3000, 50000, 8000),
            (17.., true) => (400, 1500, 4000, 30000, 6000),
            _ => (1000, 4000, 10000, 10000, 2000),
        };
        
        println!("Detected: {} cores, {} threads, ~{:.1}GB RAM", 
                 num_cores, num_threads, total_memory_gb);
        println!("Adaptive thresholds: small={}, med={}, large={}", 
                 small_thresh, med_thresh, large_thresh);
        
        Self {
            num_threads,
            parallel_threshold_small: small_thresh,
            parallel_threshold_medium: med_thresh,
            parallel_threshold_large: large_thresh,
            batch_size: batch_sz,
            chunk_size: chunk_sz,
            memory_efficient_mode: is_low_memory || is_low_core_count,
        }
    }
    
    fn estimate_memory_gb() -> f64 {
        match std::env::var("MEMORY_GB") {
            Ok(mem_str) => mem_str.parse().unwrap_or(8.0),
            Err(_) => {
                let threads = rayon::current_num_threads();
                match threads {
                    1..=4 => 4.0,
                    5..=8 => 8.0,
                    9..=16 => 16.0,
                    _ => 32.0,
                }
            }
        }
    }
    
    pub fn should_parallelize(&self, complexity: ParallelComplexity, size: usize) -> bool {
        let threshold = match complexity {
            ParallelComplexity::Simple => self.parallel_threshold_small,
            ParallelComplexity::Medium => self.parallel_threshold_medium,
            ParallelComplexity::Complex => self.parallel_threshold_large,
        };
        size >= threshold
    }
    
    pub fn get_chunk_size(&self, total_size: usize, complexity: ParallelComplexity) -> usize {
        let base_chunk = match complexity {
            ParallelComplexity::Simple => self.chunk_size * 2,
            ParallelComplexity::Medium => self.chunk_size,
            ParallelComplexity::Complex => self.chunk_size / 2,
        };
        
        (total_size / self.num_threads).max(base_chunk.min(total_size))
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ParallelComplexity {
    Simple,
    Medium,
    Complex,
}

static PARALLEL_CONFIG: OnceLock<AdaptiveParallelConfig> = OnceLock::new();

pub fn get_parallel_config() -> &'static AdaptiveParallelConfig {
    PARALLEL_CONFIG.get_or_init(|| AdaptiveParallelConfig::detect_hardware())
}

pub fn adaptive_par_map<T, F, R>(
    slice: &[T], 
    complexity: ParallelComplexity,
    f: F
) -> Vec<R> 
where 
    F: Fn(&T) -> R + Sync + Send, 
    T: Sync, 
    R: Send 
{
    let config = get_parallel_config();
    
    if config.should_parallelize(complexity, slice.len()) {
        let chunk_size = config.get_chunk_size(slice.len(), complexity);
        slice.par_chunks(chunk_size).flat_map(|chunk| {
            chunk.iter().map(&f).collect::<Vec<_>>()
        }).collect()
    } else {
        slice.iter().map(f).collect()
    }
}

pub struct MemoryMonitor;

impl MemoryMonitor {
    pub fn new() -> Self {
        Self
    }
    
    pub fn log_memory_usage(&mut self, label: &str) {
        if get_parallel_config().memory_efficient_mode {
            println!("Memory checkpoint: {}", label);
        }
    }
}

pub fn process_large_dataset_batches<T, F, R>(
    data: &[T],
    batch_size: usize,
    complexity: ParallelComplexity,
    processor: F
) -> Vec<R>
where
    F: Fn(&[T]) -> Vec<R> + Sync + Send,
    T: Sync,
    R: Send,
{
    let config = get_parallel_config();
    
    if config.memory_efficient_mode && data.len() > batch_size {
        let mut results = Vec::new();
        for chunk in data.chunks(batch_size) {
            let mut batch_result = processor(chunk);
            results.append(&mut batch_result);
        }
        results
    } else if config.should_parallelize(complexity, data.len()) {
        data.par_chunks(batch_size)
            .flat_map(|chunk| processor(chunk))
            .collect()
    } else {
        processor(data)
    }
}