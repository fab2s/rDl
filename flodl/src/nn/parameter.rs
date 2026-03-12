use crate::autograd::Variable;
use crate::tensor::{Device, Result, Tensor};

/// A named learnable parameter wrapping a Variable with requires_grad=true.
#[derive(Clone)]
pub struct Parameter {
    pub variable: Variable,
    pub name: String,
}

impl Parameter {
    /// Create a named parameter from a tensor (always requires_grad=true).
    pub fn new(data: Tensor, name: &str) -> Self {
        Parameter {
            variable: Variable::new(data, true),
            name: name.to_string(),
        }
    }

    /// Move this parameter to a different device. Returns a new Parameter.
    pub fn to_device(&self, device: Device) -> Result<Parameter> {
        let moved = self.variable.data().to_device(device)?;
        Ok(Parameter {
            variable: Variable::new(moved, true),
            name: self.name.clone(),
        })
    }
}

impl std::fmt::Debug for Parameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parameter({}, {:?})", self.name, self.variable)
    }
}
