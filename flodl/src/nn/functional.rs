use crate::autograd::{Variable, conv2d};
use crate::tensor::{Result, Tensor};

/// Apply Gaussian blur to a 4-D input `[B, C, H, W]`.
///
/// Uses separable depthwise convolution (two 1-D passes) for efficiency.
/// Kernel size is `2 * ceil(3 * sigma) + 1`, matching OpenCV's default.
///
/// ```ignore
/// let blurred = gaussian_blur_2d(&input, 1.5)?;
/// assert_eq!(blurred.shape(), input.shape());
/// ```
pub fn gaussian_blur_2d(input: &Variable, sigma: f64) -> Result<Variable> {
    let shape = input.shape();
    assert!(shape.len() == 4, "gaussian_blur_2d expects [B, C, H, W], got {:?}", shape);
    let channels = shape[1];
    let device = input.device();

    // Build 1-D Gaussian kernel
    let radius = (3.0 * sigma).ceil() as i64;
    let ksize = 2 * radius + 1;
    let mut weights = vec![0.0f32; ksize as usize];
    let mut sum = 0.0f64;
    for i in 0..ksize {
        let x = (i - radius) as f64;
        let w = (-x * x / (2.0 * sigma * sigma)).exp();
        weights[i as usize] = w as f32;
        sum += w;
    }
    for w in &mut weights {
        *w /= sum as f32;
    }

    // Horizontal kernel: [C, 1, 1, K]
    let kernel_1d = Tensor::from_f32(&weights, &[1, 1, 1, ksize], device)?;
    let h_kernel = kernel_1d.expand(&[channels, 1, 1, ksize])?;
    let h_weight = Variable::new(h_kernel, false);

    // Vertical kernel: [C, 1, K, 1]
    let kernel_1d_v = Tensor::from_f32(&weights, &[1, 1, ksize, 1], device)?;
    let v_kernel = kernel_1d_v.expand(&[channels, 1, ksize, 1])?;
    let v_weight = Variable::new(v_kernel, false);

    // Pad input to maintain spatial dims
    let pad_h = &[radius, radius, 0, 0];  // left, right, top, bottom
    let pad_v = &[0, 0, radius, radius];

    // Horizontal pass (depthwise: groups = C)
    let padded_h = input.pad(pad_h, 0.0)?;
    let h_out = conv2d(
        &padded_h, &h_weight, None,
        [1, 1], [0, 0], [1, 1], channels,
    )?;

    // Vertical pass
    let padded_v = h_out.pad(pad_v, 0.0)?;
    conv2d(
        &padded_v, &v_weight, None,
        [1, 1], [0, 0], [1, 1], channels,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Device, TensorOptions};

    #[test]
    fn test_gaussian_blur_preserves_shape() {
        let opts = TensorOptions { dtype: crate::tensor::DType::Float32, device: Device::CPU };
        let input = Variable::new(Tensor::randn(&[1, 3, 16, 16], opts).unwrap(), false);
        let output = gaussian_blur_2d(&input, 1.0).unwrap();
        assert_eq!(output.shape(), vec![1, 3, 16, 16]);
    }

    #[test]
    fn test_gaussian_blur_smooths() {
        let opts = TensorOptions { dtype: crate::tensor::DType::Float32, device: Device::CPU };
        let input = Variable::new(Tensor::randn(&[1, 1, 8, 8], opts).unwrap(), false);
        let output = gaussian_blur_2d(&input, 2.0).unwrap();

        // Blurred output should have lower variance than input
        let in_var = input.var().unwrap().item().unwrap();
        let out_var = output.var().unwrap().item().unwrap();
        assert!(out_var < in_var, "blur should reduce variance: in={} out={}", in_var, out_var);
    }
}
