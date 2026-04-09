//! Benchmark model definitions.
//!
//! Each model provides a factory for building on a given device,
//! a synthetic dataset generator, a training closure, and default config.

pub mod autoencoder;
pub mod convnet;
pub mod feedback;
pub mod linear;
pub mod lstm;
pub mod mlp;
pub mod moe;
pub mod residual;
pub mod transformer;

use std::sync::Arc;

use flodl::autograd::Variable;
use flodl::data::BatchDataSet;
use flodl::nn::Module;
use flodl::tensor::{Device, Result, Tensor};

use crate::config::ModelDefaults;

/// A benchmark model definition.
pub struct ModelDef {
    /// Short name (used in CLI and output paths).
    pub name: &'static str,
    /// What this model tests in DDP context.
    pub description: &'static str,
    /// Build the model on a specific device.
    pub build: fn(Device) -> Result<Box<dyn Module>>,
    /// Create a synthetic dataset with the given seed and total samples.
    pub dataset: fn(u64, usize) -> Result<Arc<dyn BatchDataSet>>,
    /// Training step: forward + loss. Returns the loss Variable.
    pub train_fn: fn(&dyn Module, &[Tensor]) -> Result<Variable>,
    /// Default configuration.
    pub defaults: ModelDefaults,
}

/// All registered benchmark models.
pub fn all_models() -> Vec<ModelDef> {
    vec![
        linear::def(),
        mlp::def(),
        convnet::def(),
        lstm::def(),
        transformer::def(),
        autoencoder::def(),
        residual::def(),
        moe::def(),
        feedback::def(),
    ]
}

/// Find a model by name.
pub fn find_model(name: &str) -> Option<ModelDef> {
    all_models().into_iter().find(|m| m.name == name)
}

/// All model names.
pub fn model_names() -> Vec<&'static str> {
    vec![
        "linear",
        "mlp",
        "convnet",
        "lstm",
        "transformer",
        "autoencoder",
        "residual",
        "moe",
        "feedback",
    ]
}
