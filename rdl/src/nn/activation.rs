use crate::autograd::Variable;
use crate::tensor::Result;

use super::parameter::Parameter;
use super::Module;

/// ReLU activation module.
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.relu()
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Sigmoid activation module.
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.sigmoid()
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

/// Tanh activation module.
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Tanh
    }
}

impl Module for Tanh {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        input.tanh_act()
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}
