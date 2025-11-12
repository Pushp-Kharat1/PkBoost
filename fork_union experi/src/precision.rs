use half::{bf16, f16};

const OVERFLOW_THRESHOLD_F16: f32 = 65000.0;
const UNDERFLOW_THRESHOLD: f32 = 1e-7;
const GRADIENT_PROMOTION_THRESHOLD: f64 = 1e-4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrecisionLevel {
    F16,
    BF16,
    F32,
    F64,
}

pub trait ProgressivePrecision: Sized {
    fn from_f64(val: f64) -> Self;
    fn to_f64(&self) -> f64;
    fn precision_level() -> PrecisionLevel;
    fn should_promote(&self) -> bool;
    fn promote<T: ProgressivePrecision>(&self) -> T;
}

impl ProgressivePrecision for f16 {
    fn from_f64(val: f64) -> Self { f16::from_f64(val) }
    fn to_f64(&self) -> f64 { f16::to_f64(*self) }
    fn precision_level() -> PrecisionLevel { PrecisionLevel::F16 }
    fn should_promote(&self) -> bool {
        let val = self.to_f32().abs();
        val > OVERFLOW_THRESHOLD_F16 || (val > 0.0 && val < UNDERFLOW_THRESHOLD)
    }
    fn promote<T: ProgressivePrecision>(&self) -> T { T::from_f64(f16::to_f64(*self)) }
}

impl ProgressivePrecision for bf16 {
    fn from_f64(val: f64) -> Self { bf16::from_f64(val) }
    fn to_f64(&self) -> f64 { bf16::to_f64(*self) }
    fn precision_level() -> PrecisionLevel { PrecisionLevel::BF16 }
    fn should_promote(&self) -> bool {
        let val = self.to_f32().abs();
        val > 1e10 || (val > 0.0 && val < UNDERFLOW_THRESHOLD)
    }
    fn promote<T: ProgressivePrecision>(&self) -> T { T::from_f64(bf16::to_f64(*self)) }
}

impl ProgressivePrecision for f32 {
    fn from_f64(val: f64) -> Self { val as f32 }
    fn to_f64(&self) -> f64 { *self as f64 }
    fn precision_level() -> PrecisionLevel { PrecisionLevel::F32 }
    fn should_promote(&self) -> bool {
        !self.is_finite() || self.abs() > 1e30
    }
    fn promote<T: ProgressivePrecision>(&self) -> T { T::from_f64(self.to_f64()) }
}

impl ProgressivePrecision for f64 {
    fn from_f64(val: f64) -> Self { val }
    fn to_f64(&self) -> f64 { *self }
    fn precision_level() -> PrecisionLevel { PrecisionLevel::F64 }
    fn should_promote(&self) -> bool { false }
    fn promote<T: ProgressivePrecision>(&self) -> T { T::from_f64(*self) }
}

pub struct AdaptiveCompute;

impl AdaptiveCompute {
    #[inline]
    pub fn dot_product_f16(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += f16::from_f64(a[i]).to_f32() * f16::from_f64(b[i]).to_f32();
        }
        sum as f64
    }

    #[inline]
    pub fn dot_product_bf16(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            sum += bf16::from_f64(a[i]).to_f32() * bf16::from_f64(b[i]).to_f32();
        }
        sum as f64
    }

    #[inline]
    pub fn dot_product_adaptive(a: &[f64], b: &[f64]) -> f64 {
        let max_val = a.iter().chain(b.iter())
            .map(|x| x.abs())
            .fold(0.0f64, f64::max);

        if max_val < 100.0 {
            Self::dot_product_f16(a, b)
        } else if max_val < 10000.0 {
            Self::dot_product_bf16(a, b)
        } else {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        }
    }

    // NOTE: For distance calculations, use SimSIMD instead - it's already SIMD-optimized
    // Progressive precision is best for:
    // - Gradient accumulation (many small values)
    // - Weight updates (numerical stability)
    // - Entropy calculations (precision-sensitive)

    pub fn gradient_accumulate_progressive(gradients: &[f64]) -> f64 {
        let max_grad = gradients.iter().map(|g| g.abs()).fold(0.0, f64::max);

        if max_grad < GRADIENT_PROMOTION_THRESHOLD {
            gradients.iter()
                .map(|&g| f16::from_f64(g).to_f64())
                .sum()
        } else if max_grad < 1.0 {
            gradients.iter()
                .map(|&g| bf16::from_f64(g).to_f64())
                .sum()
        } else if max_grad < 1000.0 {
            gradients.iter().copied().sum()
        } else {
            gradients.iter().copied().sum()
        }
    }

    pub fn weighted_sum_f16(values: &[f64], weights: &[f64]) -> f64 {
        values.iter().zip(weights.iter())
            .map(|(&v, &w)| (f16::from_f64(v).to_f32() * f16::from_f64(w).to_f32()) as f64)
            .sum()
    }

    pub fn weighted_sum_bf16(values: &[f64], weights: &[f64]) -> f64 {
        values.iter().zip(weights.iter())
            .map(|(&v, &w)| (bf16::from_f64(v).to_f32() * bf16::from_f64(w).to_f32()) as f64)
            .sum()
    }

    pub fn weighted_sum_adaptive(values: &[f64], weights: &[f64]) -> f64 {
        let max_val = values.iter().chain(weights.iter())
            .map(|x| x.abs())
            .fold(0.0, f64::max);

        if max_val < 50.0 {
            Self::weighted_sum_f16(values, weights)
        } else if max_val < 5000.0 {
            Self::weighted_sum_bf16(values, weights)
        } else {
            values.iter().zip(weights.iter()).map(|(v, w)| v * w).sum()
        }
    }
}

pub struct ProgressiveBuffer<T: ProgressivePrecision> {
    data: Vec<T>,
    promoted: bool,
}

impl<T: ProgressivePrecision> ProgressiveBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            promoted: false,
        }
    }

    pub fn push(&mut self, val: f64) {
        self.data.push(T::from_f64(val));
    }

    pub fn should_promote(&self) -> bool {
        !self.promoted && self.data.iter().any(|x| x.should_promote())
    }

    pub fn promote_to<U: ProgressivePrecision>(self) -> ProgressiveBuffer<U> {
        ProgressiveBuffer {
            data: self.data.iter().map(|x| x.promote()).collect(),
            promoted: true,
        }
    }

    pub fn to_f64_vec(&self) -> Vec<f64> {
        self.data.iter().map(|x| x.to_f64()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_levels() {
        let val = 42.5;
        assert_eq!(f16::from_f64(val).to_f64().round(), val.round());
        assert_eq!(bf16::from_f64(val).to_f64().round(), val.round());
    }

    #[test]
    fn test_adaptive_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = AdaptiveCompute::dot_product_adaptive(&a, &b);
        assert!((result - 32.0).abs() < 0.1);
    }

    #[test]
    fn test_progressive_buffer() {
        let mut buf = ProgressiveBuffer::<f16>::new(10);
        buf.push(1.0);
        buf.push(2.0);
        assert_eq!(buf.to_f64_vec().len(), 2);
    }
}
