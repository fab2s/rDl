use crate::autograd::Variable;
use crate::tensor::{Device, Result};

use super::init;
use super::parameter::Parameter;
use super::Module;

/// Fully connected layer: y = x @ W^T + b
pub struct Linear {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
}

impl Linear {
    pub fn new(in_features: i64, out_features: i64) -> Result<Self> {
        Self::on_device(in_features, out_features, Device::CPU)
    }

    pub fn on_device(in_features: i64, out_features: i64, device: Device) -> Result<Self> {
        let w = init::kaiming_uniform(&[out_features, in_features], in_features, device)?;
        let b = init::uniform_bias(in_features, &[out_features], device)?;
        Ok(Linear {
            weight: Parameter::new(w, "weight"),
            bias: Some(Parameter::new(b, "bias")),
        })
    }

    pub fn no_bias(in_features: i64, out_features: i64) -> Result<Self> {
        let w = init::kaiming_uniform(&[out_features, in_features], in_features, Device::CPU)?;
        Ok(Linear {
            weight: Parameter::new(w, "weight"),
            bias: None,
        })
    }
}

impl Module for Linear {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        let wt = self.weight.variable.transpose(0, 1)?;
        let out = input.matmul(&wt)?;
        match self.bias {
            Some(ref b) => out.add(&b.variable),
            None => Ok(out),
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
}
