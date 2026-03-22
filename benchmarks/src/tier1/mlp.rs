//! MLP benchmark: Linear → GELU → LayerNorm, 3 layers.
//!
//! Tests raw matmul + activation throughput — the simplest possible model.

use flodl::*;
use crate::harness::{BenchConfig, BenchResult, run_benchmark};

pub fn run(device: Device) -> Result<BenchResult> {
    let config = BenchConfig {
        name: "mlp".into(),
        batch_size: 128,
        batches_per_epoch: 100,
        ..Default::default()
    };

    let opts = TensorOptions { dtype: DType::Float32, device };

    // Model: 256 → 512 → 512 → 256
    let model = FlowBuilder::from(Linear::on_device(256, 512, device)?)
        .through(GELU)
        .through(LayerNorm::on_device(512, device)?)
        .through(Linear::on_device(512, 512, device)?)
        .through(GELU)
        .through(LayerNorm::on_device(512, device)?)
        .through(Linear::on_device(512, 256, device)?)
        .build()?;

    let params = model.parameters();
    let param_count = params.iter().map(|p| p.variable.numel()).sum::<i64>() as usize;
    let mut optimizer = Adam::new(&params, 1e-3);
    model.train();

    // Pre-generate synthetic data
    let batches: Vec<(Tensor, Tensor)> = (0..config.batches_per_epoch)
        .map(|_| {
            let x = Tensor::randn(&[config.batch_size as i64, 256], opts).unwrap();
            let y = Tensor::randn(&[config.batch_size as i64, 256], opts).unwrap();
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
