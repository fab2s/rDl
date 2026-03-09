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
