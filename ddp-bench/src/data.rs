//! Synthetic dataset generation for reproducible benchmarks.
//!
//! Uses bulk tensor operations for fast pool generation.
//! Physical pool is kept small (a few thousand samples) while `len()` reports
//! the virtual size the DataLoader / training loop expects.  `get_batch`
//! wraps indices via modulo so the pool is silently recycled.

use std::sync::Arc;

use flodl::data::BatchDataSet;
use flodl::tensor::{Device, Result, Tensor, TensorOptions};

/// Default pool multiplier: physical pool = batch_size * POOL_MUL.
/// 8x is enough to prevent GPU-cache distortion without eating RAM.
pub const POOL_MUL: usize = 8;

/// A pre-generated synthetic dataset stored as bulk tensors.
///
/// `pool_size` samples live in memory.  `virtual_len` (returned by `len()`)
/// can be larger; indices wrap via modulo in `get_batch`.
pub struct SyntheticDataSet {
    /// tensors[group_idx] = [pool_size, per-sample dims...]
    tensors: Vec<Tensor>,
    pool_size: usize,
    virtual_len: usize,
}

impl SyntheticDataSet {
    /// Generate a regression dataset: input [input_dim], target [output_dim].
    ///
    /// `virtual_len` is what `len()` reports (determines epoch size).
    /// `pool_size` is the actual number of samples allocated.
    pub fn regression(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        input_dim: i64,
        output_dim: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();

        let inputs = Tensor::randn(&[n, input_dim], opts)?;
        let targets = Tensor::randn(&[n, output_dim], opts)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            pool_size,
            virtual_len,
        }))
    }

    /// Generate a classification dataset: input [dims...], target [] (class index).
    pub fn classification(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        input_shape: &[i64],
        num_classes: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();

        let mut shape = vec![n];
        shape.extend_from_slice(input_shape);
        let inputs = Tensor::randn(&shape, opts)?;
        let targets = Tensor::randint(0, num_classes, &[n], opts)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            pool_size,
            virtual_len,
        }))
    }

    /// Generate a reconstruction dataset: input [dims...], target = input.
    pub fn reconstruction(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        shape: &[i64],
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();

        let mut full_shape = vec![n];
        full_shape.extend_from_slice(shape);
        let inputs = Tensor::randn(&full_shape, opts)?;
        let targets = inputs.clone();

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            pool_size,
            virtual_len,
        }))
    }

    /// Generate a sequence dataset: input [seq_len, input_dim], target [output_dim].
    pub fn sequence(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        seq_len: i64,
        input_dim: i64,
        output_dim: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();

        let inputs = Tensor::randn(&[n, seq_len, input_dim], opts)?;
        let targets = Tensor::randn(&[n, output_dim], opts)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            pool_size,
            virtual_len,
        }))
    }

    /// Generate a token sequence dataset: input [seq_len] (i64 tokens), target [seq_len].
    pub fn token_sequence(
        seed: u64,
        virtual_len: usize,
        pool_size: usize,
        seq_len: i64,
        vocab_size: i64,
    ) -> Result<Arc<dyn BatchDataSet>> {
        flodl::manual_seed(seed);
        let n = pool_size as i64;
        let opts = TensorOptions::default();

        let inputs = Tensor::randint(0, vocab_size, &[n, seq_len], opts)?;
        let targets = Tensor::randint(0, vocab_size, &[n, seq_len], opts)?;

        Ok(Arc::new(SyntheticDataSet {
            tensors: vec![inputs, targets],
            pool_size,
            virtual_len,
        }))
    }
}

impl BatchDataSet for SyntheticDataSet {
    fn len(&self) -> usize {
        self.virtual_len
    }

    fn get_batch(&self, indices: &[usize]) -> Result<Vec<Tensor>> {
        // Wrap indices into the physical pool via modulo.
        let idx: Vec<i64> = indices
            .iter()
            .map(|&i| (i % self.pool_size) as i64)
            .collect();
        let idx_tensor = Tensor::from_i64(&idx, &[idx.len() as i64], Device::CPU)?;

        let mut result = Vec::with_capacity(self.tensors.len());
        for bulk in &self.tensors {
            result.push(bulk.index_select(0, &idx_tensor)?);
        }
        Ok(result)
    }
}
