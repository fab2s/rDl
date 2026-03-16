# Quickstart

Build, train, and monitor a model with residual connections.

Uses the graph builder with a residual (`also`), Adam optimizer with gradient
clipping, graph observation, and the training monitor.

```sh
cargo run --example quickstart
```

## What it covers

- `FlowBuilder::from` / `through` / `also` / `build`
- `Adam` optimizer with `clip_grad_norm`
- `record_scalar` / `flush` for graph observation
- `Monitor::log(&model)` for training progress
