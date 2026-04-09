//! Synthetic dataset generation for reproducible benchmarks.

use std::sync::Arc;

use flodl::data::BatchDataSet;
use flodl::tensor::{Result, Tensor, TensorOptions};

/// A pre-generated synthetic dataset stored as flat Vec of tensors.
///
/// Each "sample" is a set of tensors (e.g., input + target).
/// `get_batch` stacks the requested indices along dim 0.
pub struct SyntheticDataSet {
    /// tensors[tensor_idx][sample_idx] = [per-sample shape]
    tensors: Vec<Vec<Tensor>>,
    len: usize,
}

impl SyntheticDataSet {
    /// Generate a regression dataset: input [input_dim], target [output_dim].
    ///
    /// Uses a fixed linear transform + noise so models can actually converge.
    pub fn regression(
        seed: u64,
        total_samples: usize,
        input_dim: i64,
        output_dim: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let opts = TensorOptions::default();

        // Fixed weight matrix for generating targets
        let w = Tensor::randn(&[input_dim, output_dim], opts)?;

        let mut inputs = Vec::with_capacity(total_samples);
        let mut targets = Vec::with_capacity(total_samples);

        for _ in 0..total_samples {
            let x = Tensor::randn(&[input_dim], opts)?;
            let y = x.matmul(&w)?;
            inputs.push(x);
            targets.push(y);
        }

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            len: total_samples,
        }))
    }

    /// Generate a classification dataset: input [dims...], target [1] (class index).
    pub fn classification(
        seed: u64,
        total_samples: usize,
        input_shape: &[i64],
        num_classes: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let opts = TensorOptions::default();

        let mut inputs = Vec::with_capacity(total_samples);
        let mut targets = Vec::with_capacity(total_samples);

        for _ in 0..total_samples {
            let x = Tensor::randn(input_shape, opts)?;
            let y = Tensor::randint(0, num_classes, &[], opts)?;
            inputs.push(x);
            targets.push(y);
        }

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            len: total_samples,
        }))
    }

    /// Generate a reconstruction dataset: input [dims...], target = input.
    pub fn reconstruction(
        seed: u64,
        total_samples: usize,
        shape: &[i64],
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let opts = TensorOptions::default();

        let mut inputs = Vec::with_capacity(total_samples);

        for _ in 0..total_samples {
            let x = Tensor::randn(shape, opts)?;
            inputs.push(x);
        }

        // Target = input (clone the tensors)
        let targets: Vec<Tensor> = inputs.iter().map(|t| t.clone()).collect();

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            len: total_samples,
        }))
    }

    /// Generate a sequence dataset: input [seq_len, input_dim], target [output_dim].
    pub fn sequence(
        seed: u64,
        total_samples: usize,
        seq_len: i64,
        input_dim: i64,
        output_dim: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let opts = TensorOptions::default();

        let w = Tensor::randn(&[input_dim, output_dim], opts)?;

        let mut inputs = Vec::with_capacity(total_samples);
        let mut targets = Vec::with_capacity(total_samples);

        for _ in 0..total_samples {
            let x = Tensor::randn(&[seq_len, input_dim], opts)?;
            // Target from last timestep
            let last = x.select(0, seq_len - 1)?;
            let y = last.matmul(&w)?;
            inputs.push(x);
            targets.push(y);
        }

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            len: total_samples,
        }))
    }

    /// Generate a token sequence dataset: input [seq_len] (i64 tokens), target [seq_len] (shifted).
    pub fn token_sequence(
        seed: u64,
        total_samples: usize,
        seq_len: i64,
        vocab_size: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let opts = TensorOptions::default();

        let mut inputs = Vec::with_capacity(total_samples);
        let mut targets = Vec::with_capacity(total_samples);

        for _ in 0..total_samples {
            let x = Tensor::randint(0, vocab_size, &[seq_len], opts)?;
            // Target: shifted by 1 (simple language modeling)
            let y = Tensor::randint(0, vocab_size, &[seq_len], opts)?;
            inputs.push(x);
            targets.push(y);
        }

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            len: total_samples,
        }))
    }
}

impl BatchDataSet for SyntheticDataSet {
    fn len(&self) -> usize {
        self.len
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
        let mut result = Vec::with_capacity(self.tensors.len());
        for tensor_group in &self.tensors {
            let batch: Vec<&Tensor> = indices
                .iter()
                .map(|&i| &tensor_group[i % self.len])
                .collect();
            let stacked = Tensor::stack(&batch, 0)?;
            result.push(stacked);
        }
        Ok(result)
    }
}
