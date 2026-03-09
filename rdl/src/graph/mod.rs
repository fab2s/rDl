pub mod node;
pub mod flow;
pub mod loop_node;

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use node::*;
use crate::autograd::Variable;
use crate::nn::{Module, Parameter};
use crate::tensor::{Result, TensorError};

pub use flow::FlowBuilder;
pub use loop_node::LoopBuilder;

/// Merge operation for combining split branches.
pub enum MergeOp {
    Add,
    Mean,
}

/// An executable computation graph. Implements Module for composability.
pub struct Graph {
    nodes: Vec<Node>,
    node_index: HashMap<String, usize>,
    levels: Vec<Vec<usize>>,
    edges: Vec<Edge>,
    edges_from: HashMap<usize, Vec<usize>>,
    inputs: Vec<ExposedPort>,
    outputs: Vec<ExposedPort>,
    order: Vec<usize>,
}

impl Graph {
    pub(crate) fn build(
        node_map: HashMap<String, Node>,
        edges: Vec<Edge>,
        inputs: Vec<ExposedPort>,
        outputs: Vec<ExposedPort>,
        _tags: HashMap<String, NodeRef>,
    ) -> Result<Self> {
        // Convert to indexed storage
        let mut nodes = Vec::with_capacity(node_map.len());
        let mut node_index = HashMap::with_capacity(node_map.len());

        for (_key, node) in node_map {
            let idx = nodes.len();
            node_index.insert(node.id.clone(), idx);
            nodes.push(node);
        }

        // Validate edges
        for edge in &edges {
            if !node_index.contains_key(&edge.from_node) {
                return Err(TensorError::new(&format!(
                    "unknown source node: {}",
                    edge.from_node
                )));
            }
            if !node_index.contains_key(&edge.to_node) {
                return Err(TensorError::new(&format!(
                    "unknown target node: {}",
                    edge.to_node
                )));
            }
        }

        // Build edges_from lookup
        let mut edges_from: HashMap<usize, Vec<usize>> = HashMap::new();
        for (ei, edge) in edges.iter().enumerate() {
            let from_idx = node_index[&edge.from_node];
            edges_from.entry(from_idx).or_default().push(ei);
        }

        // Topological levels (Kahn's algorithm)
        let levels = topological_levels(&nodes, &node_index, &edges)?;
        let order: Vec<usize> = levels.iter().flat_map(|l| l.iter().copied()).collect();

        Ok(Graph {
            nodes,
            node_index,
            levels,
            edges,
            edges_from,
            inputs,
            outputs,
            order,
        })
    }

    fn forward_impl(&self, graph_inputs: &[Variable]) -> Result<Variable> {
        if graph_inputs.len() != self.inputs.len() {
            return Err(TensorError::new(&format!(
                "expected {} inputs, got {}",
                self.inputs.len(),
                graph_inputs.len()
            )));
        }

        let n = self.nodes.len();
        let mut input_slots: Vec<HashMap<String, Variable>> =
            (0..n).map(|_| HashMap::new()).collect();
        let mut output_values: Vec<Option<Vec<Variable>>> = vec![None; n];

        // Route graph inputs to node input ports
        for (i, ep) in self.inputs.iter().enumerate() {
            let ni = self.node_index[&ep.node_id];
            input_slots[ni].insert(ep.port.clone(), graph_inputs[i].clone());
        }

        // Execute levels sequentially
        for level in &self.levels {
            for &ni in level {
                let node = &self.nodes[ni];

                // Collect inputs in port order
                let inputs: Vec<Variable> = node
                    .input_ports
                    .iter()
                    .map(|port| {
                        input_slots[ni].get(port).cloned().unwrap_or_else(|| {
                            panic!("missing input '{}' for node '{}'", port, node.id)
                        })
                    })
                    .collect();

                // Execute
                let outputs = (node.run)(&inputs)?;
                output_values[ni] = Some(outputs);

                // Route outputs to downstream nodes
                if let Some(edge_indices) = self.edges_from.get(&ni) {
                    for &ei in edge_indices {
                        let edge = &self.edges[ei];
                        let from_port_idx = node
                            .output_ports
                            .iter()
                            .position(|p| p == &edge.from_port)
                            .expect("bad output port");
                        let to_ni = self.node_index[&edge.to_node];
                        let value =
                            output_values[ni].as_ref().unwrap()[from_port_idx].clone();
                        input_slots[to_ni].insert(edge.to_port.clone(), value);
                    }
                }
            }
        }

        // Collect graph output
        let out = &self.outputs[0];
        let out_ni = self.node_index[&out.node_id];
        let out_port_idx = self.nodes[out_ni]
            .output_ports
            .iter()
            .position(|p| p == &out.port)
            .expect("bad output port");

        output_values[out_ni]
            .as_ref()
            .and_then(|o| o.get(out_port_idx).cloned())
            .ok_or_else(|| TensorError::new("graph produced no output"))
    }
}

impl Module for Graph {
    fn forward(&self, input: &Variable) -> Result<Variable> {
        self.forward_impl(&[input.clone()])
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        let mut seen = HashSet::new();

        for &ni in &self.order {
            if let Some(ref module) = self.nodes[ni].module {
                for p in module.parameters() {
                    let ptr = Rc::as_ptr(&p.variable.inner) as usize;
                    if seen.insert(ptr) {
                        params.push(p);
                    }
                }
            }
        }

        params
    }
}

/// Kahn's algorithm with level grouping for parallel execution.
fn topological_levels(
    nodes: &[Node],
    node_index: &HashMap<String, usize>,
    edges: &[Edge],
) -> Result<Vec<Vec<usize>>> {
    let n = nodes.len();

    // Build unique dependency sets (node-level, not edge-level)
    let mut deps: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut dependents: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    for edge in edges {
        let from_ni = node_index[&edge.from_node];
        let to_ni = node_index[&edge.to_node];
        deps[to_ni].insert(from_ni);
        dependents[from_ni].insert(to_ni);
    }

    let mut in_degree: Vec<usize> = deps.iter().map(|d| d.len()).collect();

    // Seed with zero in-degree nodes
    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut levels = Vec::new();
    let mut visited = 0;

    while !queue.is_empty() {
        levels.push(queue.clone());
        visited += queue.len();

        let mut next_queue = Vec::new();
        for &ni in &queue {
            for &dep in &dependents[ni] {
                in_degree[dep] -= 1;
                if in_degree[dep] == 0 {
                    next_queue.push(dep);
                }
            }
        }
        queue = next_queue;
    }

    if visited != n {
        return Err(TensorError::new("cycle detected in graph"));
    }

    Ok(levels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Variable;
    use crate::nn::{Linear, NamedInputModule, ReLU, Sigmoid, mse_loss, Optimizer, SGD};
    use crate::tensor::{Device, Tensor};
    use std::collections::HashMap;

    fn from_f32(data: &[f32], shape: &[i64]) -> Tensor {
        Tensor::from_f32(data, shape, Device::CPU).unwrap()
    }

    // --- Helper modules for testing ---

    /// Doubles the input: forward(x) = 2*x
    struct Doubler;
    impl Module for Doubler {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            input.add(input)
        }
        fn parameters(&self) -> Vec<Parameter> {
            vec![]
        }
    }

    /// Adds a learnable bias at each step (for gradient accumulation testing).
    struct BiasStep {
        bias: Parameter,
    }
    impl BiasStep {
        fn new(size: i64) -> Result<Self> {
            let data = Tensor::zeros(&[size], crate::tensor::TensorOptions {
                dtype: crate::tensor::DType::Float32,
                device: Device::CPU,
            })?;
            let var = Variable::new(data, true);
            Ok(BiasStep {
                bias: Parameter {
                    variable: var,
                    name: "loop_bias".to_string(),
                },
            })
        }
    }
    impl Module for BiasStep {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            input.add(&self.bias.variable)
        }
        fn parameters(&self) -> Vec<Parameter> {
            vec![self.bias.clone()]
        }
    }

    /// Halt when max(state) > threshold.
    struct ThresholdHalt(f64);
    impl Module for ThresholdHalt {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            let data = input.data().to_f32_vec()?;
            let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let halt = if max_val as f64 > self.0 { 1.0f32 } else { -1.0f32 };
            Ok(Variable::new(
                Tensor::from_f32(&[halt], &[1], Device::CPU)?,
                false,
            ))
        }
        fn parameters(&self) -> Vec<Parameter> {
            vec![]
        }
    }

    /// Module that adds a tagged ref to the stream (for Using tests).
    struct AddRefModule;
    impl Module for AddRefModule {
        fn forward(&self, input: &Variable) -> Result<Variable> {
            Ok(input.clone())
        }
        fn parameters(&self) -> Vec<Parameter> {
            vec![]
        }
    }
    impl NamedInputModule for AddRefModule {
        fn forward_named(
            &self,
            input: &Variable,
            refs: &HashMap<String, Variable>,
        ) -> Result<Variable> {
            if let Some(ctx) = refs.get("ctx") {
                input.add(ctx)
            } else {
                Ok(input.clone())
            }
        }
    }

    // --- Core graph tests (from before) ---

    #[test]
    fn test_single_module() {
        let l = Linear::new(3, 2).unwrap();
        let graph = FlowBuilder::from(l).build().unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2]);
    }

    #[test]
    fn test_linear_chain() {
        let graph = FlowBuilder::from(Linear::new(3, 4).unwrap())
            .through(ReLU::new())
            .through(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2]);
    }

    #[test]
    fn test_also_residual() {
        let l1 = Linear::new(3, 3).unwrap();
        l1.weight.variable.set_data(from_f32(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
        ));
        l1.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[0.0, 0.0, 0.0], &[3]));

        let l2 = Linear::new(3, 3).unwrap();
        l2.weight.variable.set_data(from_f32(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
        ));
        l2.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[1.0, 1.0, 1.0], &[3]));

        // l1(x) + l2(l1(x)) = x + (x + 1) = 2x + 1
        let graph = FlowBuilder::from(l1).also(l2).build().unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[1] - 5.0).abs() < 1e-5);
        assert!((data[2] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_split_merge_add() {
        let graph = FlowBuilder::from(Linear::new(3, 3).unwrap())
            .split(vec![Box::new(ReLU::new()), Box::new(Sigmoid::new())])
            .merge(MergeOp::Add)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, -1.0, 2.0], &[1, 3]), false);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 3]);
    }

    #[test]
    fn test_split_merge_mean() {
        let l = Linear::new(2, 2).unwrap();
        l.weight
            .variable
            .set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        l.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[0.0, 0.0], &[2]));

        let b1 = Linear::new(2, 2).unwrap();
        b1.weight
            .variable
            .set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        b1.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[0.0, 0.0], &[2]));
        let b2 = Linear::new(2, 2).unwrap();
        b2.weight
            .variable
            .set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        b2.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[0.0, 0.0], &[2]));

        let graph = FlowBuilder::from(l)
            .split(vec![Box::new(b1), Box::new(b2)])
            .merge(MergeOp::Mean)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[3.0, 7.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_parameters() {
        let graph = FlowBuilder::from(Linear::new(3, 4).unwrap())
            .through(ReLU::new())
            .through(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();

        let params = graph.parameters();
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_graph_backward() {
        let l1 = Linear::new(3, 2).unwrap();
        let l2 = Linear::new(2, 1).unwrap();

        let graph = FlowBuilder::from(l1)
            .through(ReLU::new())
            .through(l2)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_graph_as_module() {
        let inner = FlowBuilder::from(Linear::new(3, 4).unwrap())
            .through(ReLU::new())
            .build()
            .unwrap();

        let outer = FlowBuilder::from(inner)
            .through(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let y = outer.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2]);
        assert_eq!(outer.parameters().len(), 4);
    }

    #[test]
    fn test_training_loop() {
        let graph = FlowBuilder::from(Linear::new(1, 1).unwrap())
            .build()
            .unwrap();

        let params = graph.parameters();
        let mut optim = SGD::new(&params, 0.01, 0.0);

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0, 4.0], &[4, 1]), false);
        let target = Variable::new(from_f32(&[3.0, 5.0, 7.0, 9.0], &[4, 1]), false);

        let mut last_loss = f64::MAX;
        for _ in 0..800 {
            optim.zero_grad();
            let pred = graph.forward(&x).unwrap();
            let loss = mse_loss(&pred, &target).unwrap();
            last_loss = loss.item().unwrap();
            loss.backward().unwrap();
            optim.step().unwrap();
        }

        assert!(last_loss < 0.01, "got loss={}", last_loss);
    }

    #[test]
    fn test_also_backward() {
        let graph = FlowBuilder::from(Linear::new(2, 2).unwrap())
            .also(Linear::new(2, 2).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    #[test]
    fn test_split_merge_backward() {
        let graph = FlowBuilder::from(Linear::new(2, 2).unwrap())
            .split(vec![
                Box::new(Linear::new(2, 2).unwrap()),
                Box::new(Linear::new(2, 2).unwrap()),
            ])
            .merge(MergeOp::Add)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    #[test]
    fn test_build_error_open_streams() {
        let result = FlowBuilder::from(Linear::new(2, 2).unwrap())
            .split(vec![Box::new(ReLU::new()), Box::new(Sigmoid::new())])
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_build_error_duplicate_tag() {
        let result = FlowBuilder::from(Linear::new(2, 2).unwrap())
            .tag("features")
            .through(ReLU::new())
            .tag("features")
            .build();
        assert!(result.is_err());
    }

    // --- Using tests ---

    #[test]
    fn test_using_backward_ref() {
        // Tag a point, then use it downstream
        // Graph: linear(x) → tag("ctx") → through_ref(AddRef).using("ctx")
        // AddRef adds ctx to stream: stream + ctx = 2 * linear(x)
        let l = Linear::new(2, 2).unwrap();
        l.weight
            .variable
            .set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        l.bias
            .as_ref()
            .unwrap()
            .variable
            .set_data(from_f32(&[0.0, 0.0], &[2]));

        let graph = FlowBuilder::from(l)
            .tag("ctx")
            .through_ref(AddRefModule)
            .using(&["ctx"])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[3.0, 5.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        // identity(x) = [3, 5], then AddRef adds ctx ([3, 5]) = [6, 10]
        assert!((data[0] - 6.0).abs() < 1e-5);
        assert!((data[1] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_using_backward_gradients() {
        let l = Linear::new(2, 2).unwrap();
        let graph = FlowBuilder::from(l)
            .tag("ctx")
            .through_ref(AddRefModule)
            .using(&["ctx"])
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        assert!(x.grad().is_some());
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }
    }

    #[test]
    fn test_using_error_plain_module() {
        // Using on a plain module (not NamedInputModule) should error
        let result = FlowBuilder::from(Linear::new(2, 2).unwrap())
            .tag("ctx")
            .through(ReLU::new())
            .using(&["ctx"])
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_using_error_unknown_tag() {
        let result = FlowBuilder::from(Linear::new(2, 2).unwrap())
            .through_ref(AddRefModule)
            .using(&["nonexistent"])
            .build();
        assert!(result.is_err());
    }

    // --- Loop tests ---

    #[test]
    fn test_loop_for() {
        // Doubler × 3 iterations: [1, 2] → [8, 16]
        let graph = FlowBuilder::from(Linear::new(2, 2).unwrap())
            .loop_body(Doubler)
            .for_n(3)
            .build()
            .unwrap();

        // Set linear to identity
        let params = graph.parameters();
        params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert!((data[0] - 8.0).abs() < 1e-5, "1*2^3=8, got {}", data[0]);
        assert!((data[1] - 16.0).abs() < 1e-5, "2*2^3=16, got {}", data[1]);
    }

    #[test]
    fn test_loop_for_backward() {
        // Loop with a learnable bias — gradient should accumulate across iterations
        let bias_step = BiasStep::new(2).unwrap();
        let graph = FlowBuilder::from(Linear::new(2, 2).unwrap())
            .loop_body(bias_step)
            .for_n(3)
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), true);
        let y = graph.forward(&x).unwrap();
        let loss = y.sum().unwrap();
        loss.backward().unwrap();

        // All parameters should have gradients
        for p in graph.parameters() {
            assert!(p.variable.grad().is_some(), "{} should have gradient", p.name);
        }

        // The bias gradient should be 3 (accumulated from 3 iterations)
        // dL/db = 1 per iteration, 3 iterations → grad = [3, 3]
        // (because sum reduces to scalar, dL/d_each_element = 1, and bias contributes at each step)
        let all_params = graph.parameters();
        // Find the loop_bias parameter (from BiasStep, not Linear's "bias")
        let bias_param = all_params.iter().find(|p| p.name == "loop_bias").unwrap();
        let grad = bias_param.variable.grad().unwrap().to_f32_vec().unwrap();
        assert!(
            (grad[0] - 3.0).abs() < 1e-5,
            "bias grad should be 3, got {}",
            grad[0]
        );
    }

    #[test]
    fn test_loop_while() {
        // While max < 10: double. Input [1, 2] → double until max >= 10
        // Iter 0: check [1,2] max=2 < 10 → double → [2,4]
        // Iter 1: check [2,4] max=4 < 10 → double → [4,8]
        // Iter 2: check [4,8] max=8 < 10 → double → [8,16]
        // Iter 3: check [8,16] max=16 >= 10 → halt
        // Result: [8, 16]
        let graph = FlowBuilder::from(Linear::new(2, 2).unwrap())
            .loop_body(Doubler)
            .while_cond(ThresholdHalt(10.0), 20)
            .build()
            .unwrap();

        let params = graph.parameters();
        params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert!((data[0] - 8.0).abs() < 1e-5, "got {}", data[0]);
        assert!((data[1] - 16.0).abs() < 1e-5, "got {}", data[1]);
    }

    #[test]
    fn test_loop_while_immediate_halt() {
        // Threshold 0.5 — input [1, 2] max=2 > 0.5, halt immediately
        // While checks before body, so body never runs
        let graph = FlowBuilder::from(Linear::new(2, 2).unwrap())
            .loop_body(Doubler)
            .while_cond(ThresholdHalt(0.5), 20)
            .build()
            .unwrap();

        let params = graph.parameters();
        params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        // Body never ran — output = input
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_loop_until() {
        // Until max > 10: double. Body runs at least once.
        // Input [1, 2]
        // Iter 0: double → [2, 4], check max=4 <= 10 → continue
        // Iter 1: double → [4, 8], check max=8 <= 10 → continue
        // Iter 2: double → [8, 16], check max=16 > 10 → halt
        // Result: [8, 16]
        let graph = FlowBuilder::from(Linear::new(2, 2).unwrap())
            .loop_body(Doubler)
            .until_cond(ThresholdHalt(10.0), 20)
            .build()
            .unwrap();

        let params = graph.parameters();
        params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        assert!((data[0] - 8.0).abs() < 1e-5, "got {}", data[0]);
        assert!((data[1] - 16.0).abs() < 1e-5, "got {}", data[1]);
    }

    #[test]
    fn test_loop_until_at_least_once() {
        // Until with threshold 0.5 — input [1, 2] would halt immediately in While,
        // but Until always runs body at least once
        let graph = FlowBuilder::from(Linear::new(2, 2).unwrap())
            .loop_body(Doubler)
            .until_cond(ThresholdHalt(0.5), 20)
            .build()
            .unwrap();

        let params = graph.parameters();
        params[0].variable.set_data(from_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        params[1].variable.set_data(from_f32(&[0.0, 0.0], &[2]));

        let x = Variable::new(from_f32(&[1.0, 2.0], &[1, 2]), false);
        let y = graph.forward(&x).unwrap();
        let data = y.data().to_f32_vec().unwrap();

        // Body ran once: [2, 4]
        assert!((data[0] - 2.0).abs() < 1e-5, "got {}", data[0]);
        assert!((data[1] - 4.0).abs() < 1e-5, "got {}", data[1]);
    }

    #[test]
    fn test_loop_parameters() {
        // Loop with learnable body — parameters should include body params
        let graph = FlowBuilder::from(Linear::new(2, 2).unwrap())
            .loop_body(Linear::new(2, 2).unwrap())
            .for_n(3)
            .build()
            .unwrap();

        let params = graph.parameters();
        // From module: weight + bias = 2, loop body Linear: weight + bias = 2
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_loop_while_parameters() {
        // While loop with body + condition — both contribute parameters
        let graph = FlowBuilder::from(Linear::new(2, 2).unwrap())
            .loop_body(Linear::new(2, 2).unwrap())
            .while_cond(Linear::new(2, 1).unwrap(), 10)
            .build()
            .unwrap();

        let params = graph.parameters();
        // From module: 2, loop body: 2, condition: 2 = 6
        assert_eq!(params.len(), 6);
    }

    #[test]
    fn test_loop_in_chain() {
        // Linear → Loop(ReLU) × 3 → Linear
        let graph = FlowBuilder::from(Linear::new(3, 4).unwrap())
            .loop_body(ReLU::new())
            .for_n(3)
            .through(Linear::new(4, 2).unwrap())
            .build()
            .unwrap();

        let x = Variable::new(from_f32(&[1.0, 2.0, 3.0], &[1, 3]), false);
        let y = graph.forward(&x).unwrap();
        assert_eq!(y.shape(), vec![1, 2]);
    }
}
