use crate::autograd::Variable;
use crate::tensor::Result;

/// Mean Squared Error loss: mean((pred - target)²)
pub fn mse_loss(pred: &Variable, target: &Variable) -> Result<Variable> {
    let diff = pred.sub(target)?;
    let sq = diff.mul(&diff)?;
    let total = sq.sum()?;
    let n = pred.numel() as f64;
    total.mul_scalar(1.0 / n)
}
