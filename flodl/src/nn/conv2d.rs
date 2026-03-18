use crate::autograd::{Variable, conv2d};
use crate::tensor::{Result, Device};

use super::init::{kaiming_uniform, uniform_bias};
use super::parameter::Parameter;
use super::Module;

/// 2D convolution layer.
///
/// Weight shape: `[out_channels, in_channels / groups, kernel_h, kernel_w]`.
/// Bias shape: `[out_channels]` (optional).
///
/// Input: `[batch, in_channels, H, W]`.
/// Output: `[batch, out_channels, H_out, W_out]` where
/// `H_out = (H + 2*padding - dilation*(kernel-1) - 1) / stride + 1`.
///
/// ```ignore
/// let conv = Conv2d::new(3, 16, 3)?; // 3→16 channels, 3x3 kernel
/// let x = Variable::new(Tensor::randn(&[1, 3, 32, 32], opts)?, false);
/// let y = conv.forward(&x)?; // [1, 16, 30, 30]
/// ```
pub struct Conv2d {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    pub stride: [i64; 2],
    pub padding: [i64; 2],
    pub dilation: [i64; 2],
    pub groups: i64,
}

/// Builder for configuring Conv2d layers with a fluent API.
///
/// ```ignore
/// let conv = Conv2d::configure(3, 16, 3)
///     .with_stride(2)
///     .with_padding(1)
///     .on_device(Device::CUDA(0))
///     .done()?;
/// ```
pub struct Conv2dBuilder {
    in_channels: i64,
    out_channels: i64,
    kernel_size: i64,
    with_bias: bool,
    stride: [i64; 2],
    padding: [i64; 2],
    dilation: [i64; 2],
    groups: i64,
    device: Device,
}

impl Conv2dBuilder {
    /// Set stride (applied to both dimensions).
    pub fn with_stride(mut self, stride: i64) -> Self {
        self.stride = [stride, stride];
        self
    }

    /// Set padding (applied to both dimensions).
    pub fn with_padding(mut self, padding: i64) -> Self {
        self.padding = [padding, padding];
        self
    }

    /// Set dilation (applied to both dimensions).
    pub fn with_dilation(mut self, dilation: i64) -> Self {
        self.dilation = [dilation, dilation];
        self
    }

    /// Set the number of groups for grouped convolution.
    pub fn with_groups(mut self, groups: i64) -> Self {
        self.groups = groups;
        self
    }

    /// Disable bias.
    pub fn without_bias(mut self) -> Self {
        self.with_bias = false;
        self
    }

    /// Set the device for parameter allocation.
    pub fn on_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Finalize and create the Conv2d layer.
    pub fn done(self) -> Result<Conv2d> {
        Conv2d::build(
            self.in_channels, self.out_channels, self.kernel_size,
            self.with_bias, self.stride, self.padding, self.dilation,
            self.groups, self.device,
        )
    }
}

impl Conv2d {
    /// Create a Conv2d layer with default stride=1, padding=0, dilation=1, groups=1, with bias.
    pub fn new(
        in_channels: i64, out_channels: i64, kernel_size: i64,
    ) -> Result<Self> {
        Self::build(in_channels, out_channels, kernel_size, true, [1, 1], [0, 0], [1, 1], 1, Device::CPU)
    }

    /// Create a Conv2d layer without bias.
    pub fn no_bias(
        in_channels: i64, out_channels: i64, kernel_size: i64,
    ) -> Result<Self> {
        Self::build(in_channels, out_channels, kernel_size, false, [1, 1], [0, 0], [1, 1], 1, Device::CPU)
    }

    /// Create a Conv2d layer on a specific device.
    pub fn on_device(
        in_channels: i64, out_channels: i64, kernel_size: i64, device: Device,
    ) -> Result<Self> {
        Self::build(in_channels, out_channels, kernel_size, true, [1, 1], [0, 0], [1, 1], 1, device)
    }

    /// Start a fluent builder for full configuration.
    ///
    /// ```ignore
    /// let conv = Conv2d::configure(3, 16, 3)
    ///     .with_stride(2)
    ///     .with_padding(1)
    ///     .done()?;
    /// ```
    pub fn configure(in_channels: i64, out_channels: i64, kernel_size: i64) -> Conv2dBuilder {
        Conv2dBuilder {
            in_channels,
            out_channels,
            kernel_size,
            with_bias: true,
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
            device: Device::CPU,
        }
    }

    /// Fully configurable Conv2d constructor.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        in_channels: i64, out_channels: i64, kernel_size: i64,
        with_bias: bool,
        stride: [i64; 2], padding: [i64; 2], dilation: [i64; 2],
        groups: i64, device: Device,
    ) -> Result<Self> {
        let shape = [out_channels, in_channels / groups, kernel_size, kernel_size];
        let fan_in = (in_channels / groups) * kernel_size * kernel_size;

        let weight_data = kaiming_uniform(&shape, fan_in, 5.0_f64.sqrt(), device)?;
        let weight = Variable::new(weight_data, true);

        let bias = if with_bias {
            let bias_data = uniform_bias(fan_in, &[out_channels], device)?;
            Some(Parameter {
                variable: Variable::new(bias_data, true),
                name: "bias".into(),
            })
        } else {
            None
        };

        Ok(Conv2d {
            weight: Parameter { variable: weight, name: "weight".into() },
            bias,
            stride,
            padding,
            dilation,
            groups,
        })
    }
}

impl Module for Conv2d {
    fn name(&self) -> &str { "conv2d" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        conv2d(
            input,
            &self.weight.variable,
            self.bias.as_ref().map(|b| &b.variable),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
}
