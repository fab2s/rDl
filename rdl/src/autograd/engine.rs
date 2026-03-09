use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::tensor::{Result, Tensor, TensorError};

use super::variable::{Variable, VariableInner};

/// Run reverse-mode autodiff from a scalar output.
///
/// Computes gradients for all leaf variables that have `requires_grad = true`.
/// Gradients are accumulated into each variable's `grad` field (additive —
/// call `zero_grad()` before backward if you need fresh gradients).
///
/// After processing each node, its `grad_fn` is set to `None`. This drops
/// the closure and all saved tensors — deterministic VRAM release. This is
/// equivalent to goDl's Phase 1 (nil gradFn) + Phase 3 (Release saved tensors)
/// combined into a single Rust `Option::take()`.
pub fn backward(output: &Variable) -> Result<()> {
    // Must be scalar
    {
        let inner = output.inner.borrow();
        if inner.data.numel() != 1 {
            return Err(TensorError::new(
                "backward requires scalar output (numel=1)",
            ));
        }
    }

    let order = topo_sort(output);

    // Gradient map: variable identity → accumulated gradient tensor.
    let mut grads: HashMap<*const RefCell<VariableInner>, Tensor> = HashMap::new();

    // Seed: ones_like(output)
    let seed = {
        let inner = output.inner.borrow();
        Tensor::ones_like(&inner.data)?
    };
    grads.insert(Rc::as_ptr(&output.inner), seed);

    // Reverse walk: output → leaves.
    // Leaf accumulation is deferred to after the full sweep because a leaf
    // variable used at multiple depths (e.g., loop body bias) may appear in
    // the topo sort between nodes that contribute to its gradient. Processing
    // it inline would miss later contributions.
    for var in order.iter().rev() {
        let ptr = Rc::as_ptr(&var.inner);

        let grad = match grads.get(&ptr) {
            Some(g) => g.clone(),
            None => continue,
        };

        let (is_leaf, has_grad_fn) = {
            let inner = var.inner.borrow();
            (inner.is_leaf, inner.grad_fn.is_some())
        };

        // Compute input gradients via the backward function
        if has_grad_fn {
            // Take the GradFn out → sets to None → drops closure → frees saved tensors
            let grad_fn = {
                let mut inner = var.inner.borrow_mut();
                inner.grad_fn.take()
            };

            if let Some(gf) = grad_fn {
                let input_grads = (gf.apply)(&grad)?;

                // Accumulate gradients for each input in the grads map
                for (input, ig) in gf.inputs.iter().zip(input_grads) {
                    let input_ptr = Rc::as_ptr(&input.inner);

                    if let Some(existing) = grads.get(&input_ptr) {
                        let new_grad = existing.add(&ig)?;
                        grads.insert(input_ptr, new_grad);
                    } else {
                        grads.insert(input_ptr, ig);
                    }
                }
            }
        }

        // Clean up non-leaf gradients from map (they're only needed transiently)
        if !is_leaf {
            grads.remove(&ptr);
        }
    }

    // Final pass: accumulate all leaf gradients from the completed map
    for var in &order {
        let inner = var.inner.borrow();
        if inner.is_leaf && inner.requires_grad {
            let ptr = Rc::as_ptr(&var.inner);
            if let Some(grad) = grads.get(&ptr) {
                let grad = grad.clone();
                drop(inner);
                var.accumulate_grad(&grad)?;
            }
        }
    }

    Ok(())
}

/// Topological sort via iterative post-order DFS.
///
/// Returns variables in post-order: leaves first, root last.
/// The backward pass reverses this: root first, leaves last.
fn topo_sort(root: &Variable) -> Vec<Variable> {
    let mut visited: HashSet<*const RefCell<VariableInner>> = HashSet::new();
    let mut order = Vec::new();
    let mut stack: Vec<(Variable, bool)> = vec![(root.clone(), false)];

    while let Some((var, processed)) = stack.pop() {
        let ptr = Rc::as_ptr(&var.inner);

        if processed {
            order.push(var);
            continue;
        }

        if visited.contains(&ptr) {
            continue;
        }
        visited.insert(ptr);

        // Push self back as "processed" (will be added to order on next pop)
        stack.push((var.clone(), true));

        // Push children (inputs to this node's operation)
        let inputs: Option<Vec<Variable>> = {
            let inner = var.inner.borrow();
            inner.grad_fn.as_ref().map(|gf| gf.inputs.clone())
        };

        if let Some(inputs) = inputs {
            for input in inputs.into_iter().rev() {
                if !visited.contains(&Rc::as_ptr(&input.inner)) {
                    stack.push((input, false));
                }
            }
        }
    }

    order
}
