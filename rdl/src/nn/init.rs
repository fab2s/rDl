use crate::tensor::{Device, Result, Tensor, TensorOptions};

/// Kaiming uniform initialization (for ReLU networks).
/// bound = sqrt(6 / fan_in), uniform(-bound, bound)
pub fn kaiming_uniform(shape: &[i64], fan_in: i64, device: Device) -> Result<Tensor> {
    let bound = (6.0 / fan_in as f64).sqrt();
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    Tensor::rand(shape, opts)?.mul_scalar(2.0 * bound)?.add_scalar(-bound)
}

/// Kaiming normal initialization (for ReLU networks).
/// std = sqrt(2 / fan_in), normal(0, std)
pub fn kaiming_normal(shape: &[i64], fan_in: i64, device: Device) -> Result<Tensor> {
    let std = (2.0 / fan_in as f64).sqrt();
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    Tensor::randn(shape, opts)?.mul_scalar(std)
}

/// Xavier (Glorot) uniform initialization (for sigmoid/tanh networks).
/// bound = sqrt(6 / (fan_in + fan_out)), uniform(-bound, bound)
pub fn xavier_uniform(shape: &[i64], fan_in: i64, fan_out: i64, device: Device) -> Result<Tensor> {
    let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    Tensor::rand(shape, opts)?.mul_scalar(2.0 * bound)?.add_scalar(-bound)
}

/// Xavier (Glorot) normal initialization (for sigmoid/tanh networks).
/// std = sqrt(2 / (fan_in + fan_out)), normal(0, std)
pub fn xavier_normal(shape: &[i64], fan_in: i64, fan_out: i64, device: Device) -> Result<Tensor> {
    let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    Tensor::randn(shape, opts)?.mul_scalar(std)
}

/// Uniform bias initialization.
/// bound = 1 / sqrt(fan_in), uniform(-bound, bound)
pub fn uniform_bias(fan_in: i64, shape: &[i64], device: Device) -> Result<Tensor> {
    let bound = 1.0 / (fan_in as f64).sqrt();
    let opts = TensorOptions {
        dtype: crate::tensor::DType::Float32,
        device,
    };
    Tensor::rand(shape, opts)?.mul_scalar(2.0 * bound)?.add_scalar(-bound)
}
