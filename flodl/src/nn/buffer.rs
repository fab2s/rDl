use std::sync::{Arc, RwLock};

use crate::tensor::{Device, Result, Tensor};

/// A named non-learnable tensor buffer (e.g., running statistics).
///
/// Unlike Parameters, buffers are not tracked by optimizers but are
/// persisted in checkpoints alongside parameters. `Clone` shares the
/// underlying `Arc`, so the checkpoint system can write through to the
/// same cell the owning module holds.
#[derive(Clone)]
pub struct Buffer {
    pub(crate) inner: Arc<RwLock<Tensor>>,
    pub name: String,
}

impl Buffer {
    /// Create a named buffer from a tensor.
    pub fn new(tensor: Tensor, name: &str) -> Self {
        Buffer {
            inner: Arc::new(RwLock::new(tensor)),
            name: name.to_string(),
        }
    }

    /// Get a shallow clone of the underlying tensor.
    pub fn get(&self) -> Tensor {
        self.inner.read().unwrap().clone()
    }

    /// Replace the underlying tensor.
    pub fn set(&self, tensor: Tensor) {
        *self.inner.write().unwrap() = tensor;
    }

    /// Shape of the underlying tensor.
    pub fn shape(&self) -> Vec<i64> {
        self.inner.read().unwrap().shape()
    }

    /// Device of the underlying tensor.
    pub fn device(&self) -> Device {
        self.inner.read().unwrap().device()
    }

    /// Move buffer to a device (writes through the Arc).
    pub fn to_device(&self, device: Device) -> Result<()> {
        if self.device() != device {
            let moved = self.inner.read().unwrap().to_device(device)?;
            *self.inner.write().unwrap() = moved;
        }
        Ok(())
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Buffer({}, {:?})", self.name, self.inner.read().unwrap().shape())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_shared_via_clone() {
        let buf = Buffer::new(
            Tensor::zeros(&[3], crate::tensor::test_opts()).unwrap(),
            "running_mean",
        );
        let clone = buf.clone();

        // Write through one handle, read through the other
        let new_data = Tensor::ones(&[3], crate::tensor::test_opts()).unwrap();
        clone.set(new_data);

        let vals = buf.get().to_f32_vec().unwrap();
        assert_eq!(vals, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_buffer_to_device() {
        let buf = Buffer::new(
            Tensor::zeros(&[4], crate::tensor::test_opts()).unwrap(),
            "stats",
        );
        assert_eq!(buf.device(), crate::tensor::test_device());
        // Moving to same device is a no-op
        buf.to_device(crate::tensor::test_device()).unwrap();
        assert_eq!(buf.shape(), vec![4]);
    }
}
