//! ConvNet benchmark: Conv2d → BatchNorm2d → ReLU → MaxPool2d → Linear.
//!
//! Tests convolutional pipeline throughput on image-shaped data.

use flodl::*;
use crate::harness::{BenchConfig, BenchResult, run_benchmark};

/// Classification head: adaptive_avg_pool2d → flatten → linear.
struct ClassifyHead {
    linear: Linear,
}

impl ClassifyHead {
    fn new(in_channels: i64, num_classes: i64, device: Device) -> Result<Self> {
        Ok(Self {
            linear: Linear::on_device(in_channels, num_classes, device)?,
        })
    }
}

impl Module for ClassifyHead {
    fn name(&self) -> &str { "classify_head" }

    fn forward(&self, input: &Variable) -> Result<Variable> {
        let pooled = adaptive_avg_pool2d(input, [1, 1])?;
        let flat = pooled.flatten(1, -1)?;
        self.linear.forward(&flat)
    }

    fn parameters(&self) -> Vec<flodl::nn::parameter::Parameter> {
        self.linear.parameters()
    }
}

pub fn run(device: Device) -> Result<BenchResult> {
    let config = BenchConfig {
        name: "convnet".into(),
        batch_size: 64,
        batches_per_epoch: 100,
        ..Default::default()
    };

    let opts = TensorOptions { dtype: DType::Float32, device };

    // Conv → BN → ReLU → MaxPool, 3 blocks, then global avg pool → classify
    let model = FlowBuilder::from(
        Conv2d::configure(3, 32, 3).with_padding(1).on_device(device).done()?
    )
        .through(BatchNorm2d::on_device(32, device)?)
        .through(ReLU)
        .through(MaxPool2d::new(2))
        .through(
            Conv2d::configure(32, 64, 3).with_padding(1).on_device(device).done()?
        )
        .through(BatchNorm2d::on_device(64, device)?)
        .through(ReLU)
        .through(MaxPool2d::new(2))
        .through(
            Conv2d::configure(64, 128, 3).with_padding(1).on_device(device).done()?
        )
        .through(BatchNorm2d::on_device(128, device)?)
        .through(ReLU)
        .through(ClassifyHead::new(128, 10, device)?)
        .build()?;

    let params = model.parameters();
    let param_count = params.iter().map(|p| p.variable.numel()).sum::<i64>() as usize;
    let mut optimizer = Adam::new(&params, 1e-3);
    model.train();

    // Synthetic image data: [B, 3, 32, 32] → class labels [B, 10]
    let batches: Vec<(Tensor, Tensor)> = (0..config.batches_per_epoch)
        .map(|_| {
            let x = Tensor::randn(&[config.batch_size as i64, 3, 32, 32], opts).unwrap();
            let y = Tensor::randn(&[config.batch_size as i64, 10], opts).unwrap();
            (x, y)
        })
        .collect();

    run_benchmark(&config, param_count, |_epoch, _warmup| {
        let mut total_loss = 0.0;
        for (x, y) in &batches {
            let input = Variable::new(x.clone(), true);
            let target = Variable::new(y.clone(), false);
            let pred = model.forward(&input)?;
            let loss = mse_loss(&pred, &target)?;

            optimizer.zero_grad();
            loss.backward()?;
            optimizer.step()?;

            total_loss += loss.item()?;
        }
        Ok(total_loss / batches.len() as f64)
    })
}
