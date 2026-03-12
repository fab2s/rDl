<p align="center">
  <img src="docs/floDl.png" alt="goDl" width="640">
</p>

<h1 align="center">floDl</h1>

<p align="center">
A Rust-native deep learning framework built on libtorch.<br>
Same GPU kernels as PyTorch. No Python. No GIL. No GC. Just Rust.
</p>

<p align="center">
  <a href="#the-graph-builder">Graph Builder</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="docs/tutorials/01-tensors.md">Tutorials</a> &bull;
  <a href="#architecture">Architecture</a>
</p>

---

## The Graph Builder

floDl's fluent graph builder lets you describe complex architectures as
readable data flow — no boilerplate, no graph construction commands.

```rust
let model = FlowBuilder::from(Linear::new(2, 16)?)
    .through(GELU)                        // activation
    .through(LayerNorm::new(16)?)         // normalization
    .also(Linear::new(16, 16)?)           // residual connection
    .through(Linear::new(16, 2)?)         // output projection
    .build()?;
```

That's a trainable model. `also` adds the residual — input flows through the
Linear *and* gets added to its output. `build()` returns a `Graph` that
implements `Module` — you can nest it inside other graphs.

Things get interesting when architectures get complex:

```rust
let g = FlowBuilder::from(encoder).tag("encoded")
    .split(modules![head_a, head_b, head_c]).merge(MergeOp::Mean)
    .loop_body(refinement_block).for_n(3).tag("refined")
    .gate(router, modules![expert_a, expert_b]).using(&["encoded"])
    .switch(selector, modules![light_path, heavy_path]).using(&["refined"])
    .through(StateAdd).using(&["memory"]).tag("memory")
    .loop_body(decoder).while_cond(halt_condition, 10)
    .through(output_head)
    .build()?;
```

Every construct — `split/merge`, `also`, `loop_body`, `gate`, `switch`, `map`,
`tag/using` — composes cleanly. Sub-graphs nest like any module. Forward
references (`using` before `tag`) carry state across calls, enabling recurrent
architectures without special-casing.

See the **[Graph Builder Tutorial](docs/tutorials/05-graph-builder.md)** and
the [full showcase](flodl/examples/showcase.rs) that exercises every builder
method.

## Quick Start

Requirements: Docker (with NVIDIA Container Toolkit for GPU support).

```bash
git clone https://github.com/fab2s/floDl.git
cd floDl
make image    # build dev container (Rust + libtorch)
make test     # run all tests
make clippy   # lint
make shell    # interactive shell in container
```

### Train a model in 30 lines

```rust
use flodl::*;

// Build the model.
let model = FlowBuilder::from(Linear::new(2, 16)?)
    .through(GELU)
    .through(LayerNorm::new(16)?)
    .also(Linear::new(16, 16)?)
    .through(Linear::new(16, 2)?)
    .build()?;

// Set up training.
let params = model.parameters();
let optimizer = Adam::new(&params, 0.01);
model.set_training(true);

// Training loop.
for (input_t, target_t) in &batches {
    let input = Variable::new(input_t.clone(), true);
    let target = Variable::new(target_t.clone(), false);

    let pred = model.forward(&input)?;
    let loss = mse_loss(&pred, &target)?;

    optimizer.zero_grad();
    loss.backward()?;
    clip_grad_norm(&params, 1.0);
    optimizer.step(&params);
}
```

## Features

### Core Stack

| Layer | What it does |
|-------|-------------|
| **Tensor** | Owned RAII tensors with `Drop`, `Clone`. CPU and CUDA. |
| **Autograd** | Reverse-mode automatic differentiation. Full backward for every op. |
| **NN Modules** | `Linear`, `Conv2d`, `ConvTranspose2d`, `LayerNorm`, `BatchNorm`, `Dropout`, `Embedding`, `GRUCell`, `LSTMCell` |
| **Activations** | `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `SiLU` |
| **Losses** | `mse_loss`, `cross_entropy_loss`, `bce_with_logits_loss`, `l1_loss`, `smooth_l1_loss`, `kl_div_loss` |
| **Optimizers** | `SGD` (with momentum), `Adam`, `AdamW` |
| **LR Scheduling** | `StepDecay`, `CosineScheduler`, `WarmupScheduler` (composable), `PlateauScheduler` |
| **Mixed Precision** | `Float16`/`BFloat16` dtype casting, `GradScaler` for loss scaling |

### Graph Builder

| Method | What it does |
|--------|-------------|
| `from(m).through(m)` | Linear chain |
| `input(names)` | Auxiliary graph inputs, accessible via `using(name)` — multi-input graphs |
| `split(modules![...]).merge(op)` | Parallel branches, merged by `Add`, `Mean`, or `Cat(dim)` |
| `also(m)` | Residual connection: `input + m(input)` |
| `tag(name)` / `using(refs)` | Named references — backward (same pass) or forward (across calls) |
| `loop_body(body).for_n(n)` | Fixed iteration with BPTT |
| `loop_body(body).while_cond(cond, max)` | Condition before body (0..max iterations) |
| `loop_body(body).until_cond(cond, max)` | Condition after body (1..max iterations) |
| `gate(router, modules![...])` | Soft routing — all experts execute, weighted combination |
| `switch(selector, modules![...])` | Hard routing — only selected branch executes |
| `map(body).each()` | Apply body to each element along dim 0 |
| `map(body).over(tag)` | Iterate over a tagged tensor |
| `map(body).slices(n)` | Decompose last dim into n slices, map, recompose |
| `.batched()` | Fast path for Map — full batch in one call |
| `tag_group(name)` | Name parallel branches: `split(...).tag_group("head")` |

### Training Tools

| Tool | What it does |
|------|-------------|
| `clip_grad_norm` | L2 norm gradient clipping |
| `clip_grad_value` | Element-wise gradient clamping |
| `save_parameters` / `load_parameters` | Binary checkpoint format (file path or `Write`/`Read`) |
| `kaiming_uniform/normal`, `xavier_uniform/normal` | Weight initialization |
| LR schedulers | `StepDecay`, `CosineScheduler`, `WarmupScheduler`, `PlateauScheduler` (composable) |
| `GradScaler` | Dynamic loss scaling for mixed precision (float16) training |
| `cast_parameters` | Cast model parameters to any dtype |

### Module Traits

Beyond `Module`, modules can implement optional traits that the graph
recognizes automatically:

| Trait | Method | What happens |
|-------|--------|-------------|
| `NamedInputModule` | `forward_named(input, refs)` | Loop and node Using refs arrive as a named map |
| `Resettable` | `reset(batch_size, device)` | Graph auto-calls before each forward — per-forward state resets |
| `Detachable` | `detach()` | Breaks gradient chains on retained state |

Modules that own child modules implement `sub_modules()` on the Module trait
for recursive device placement, training mode, and parameter collection.

### Observation & Trends

Tags double as observation points — collect metrics during training, flush
to epoch history, and query trends to drive training decisions:

```rust
for epoch in 0..num_epochs {
    for (input, target) in &batches {
        let pred = graph.forward(&input)?;
        graph.collect(&["hidden"])?;                 // from graph tag

        let loss = mse_loss(&pred, &target)?;
        graph.record("loss", loss.item()?);          // external metric
    }
    graph.flush(&["hidden", "loss"]);

    if graph.trend("loss").stalled(5, 1e-4) {
        // decay learning rate
    }
}
```

| Method | What it does |
|--------|-------------|
| `g.tagged(tag)` | Access a tagged node's output after forward |
| `g.collect(tags)` / `g.flush(tags)` | Batch -> epoch metric collection |
| `g.record(tag, value)` | Inject external metrics |
| `g.trend(tag)` | Epoch-level trend: `slope`, `stalled`, `improving`, `converged` |
| `g.trends(tags)` | Group trends: `all_improving`, `any_stalled`, `mean_slope` |
| `g.end_step()` / `g.end_epoch()` | Training housekeeping |

### Visualization

```rust
println!("{}", g.dot());              // Graphviz DOT with parameter counts
let svg = g.svg("model.svg")?;       // render to SVG

// Timing-annotated: nodes colored green->yellow->red by execution time.
g.enable_profiling(true);
g.forward(&input)?;
g.svg_with_profile("profile.svg")?;

// Training curves as self-contained HTML.
g.plot_html("training.html", &["loss", "head"])?;
g.export_trends("metrics.csv", &["loss"])?;
```

### Numerical Verification

Every differentiable path is verified against finite-difference gradients:
- 37 autograd op-level checks (every op + compositions)
- Module-level checks (every NN module, input + parameter gradients)
- Exact optimizer step verifications (SGD, Adam, AdamW)
- 166+ library tests + 14 showcase tests, zero clippy warnings

## Why Rust for Deep Learning?

### The memory management problem

Python adds ~3-5 us of framework overhead to every GPU operation. For
architectures built on many small sequential operations — recurrent steps,
iterative refinement, multi-head attention — this overhead dominates.

Go solves the dispatch overhead with compiled binaries and goroutines, but
Go's garbage collector cannot manage VRAM deterministically. GPU memory lives
in libtorch's C++ allocator — invisible to Go's GC. This required goDl to
build a 5-phase memory management system: atomic refcounting, saved-tensor
lifecycle, GC callbacks, VRAM budgets, and autograd Scope. Hundreds of lines
of `runtime.KeepAlive`, `Retain()`/`Release()`, and pending-free queues.

Rust's ownership model eliminates all of this. `Tensor` owns a C++ handle.
`Drop` frees it immediately when it goes out of scope. No GC, no finalizers,
no reference counting, no VRAM budget heuristics, no KeepAlive. The entire
goDl memory management system — Phases 1 through 5 — is replaced by a single
`impl Drop for Tensor`.

### Zero-cost safety

Rust's type system catches errors at compile time that other languages defer
to runtime:

- **Ownership**: tensors are freed exactly once, exactly when no longer needed
- **Result types**: every fallible operation returns `Result<T>` — no silent
  error propagation, no nil pointer panics
- **No data races**: the borrow checker prevents concurrent mutation bugs

### Same GPU kernels

floDl binds libtorch — the same C++ library that powers PyTorch. The actual
GPU math (CUDA kernels, cuBLAS, cuDNN) is identical. floDl replaces everything
above: the dispatch path, autograd tracking, module composition, and graph
execution.

## Architecture

```
+-----------------------------------------------------------+
|  User Code / Model Definitions                            |
+-----------------------------------------------------------+
|  graph/    Fluent builder, execution, DOT/SVG             |
+-----------------------------------------------------------+
|  nn/       Modules, losses, optimizers, checkpoints       |
+-----------------------------------------------------------+
|  autograd/ Reverse-mode AD, gradient tracking             |
+-----------------------------------------------------------+
|  tensor/   Owned tensors with Drop, CPU + CUDA            |
+-----------------------------------------------------------+
|  flodl-sys   FFI bindings to libtorch C++ shim            |
+-----------------------------------------------------------+
|  libtorch / CUDA / ROCm / MPS / CPU                      |
+-----------------------------------------------------------+
```

Since floDl binds libtorch — not CUDA directly — it inherits libtorch's
backend support: NVIDIA (CUDA), AMD (ROCm), Intel (XPU), Apple Silicon (MPS),
and CPU. Switching hardware is a build flag, not a code change.

## Documentation

### Tutorials

Step-by-step guides from basics to advanced, each with code examples:

1. **[Tensors](docs/tutorials/01-tensors.md)** — creation, ops, error handling, memory
2. **[Autograd](docs/tutorials/02-autograd.md)** — variables, gradients, backward pass
3. **[Modules](docs/tutorials/03-modules.md)** — Linear, Conv2d, normalization, RNN cells
4. **[Training](docs/tutorials/04-training.md)** — losses, optimizers, full training loop
5. **[Graph Builder](docs/tutorials/05-graph-builder.md)** — the fluent API from simple to complex
6. **[Advanced Graphs](docs/tutorials/06-advanced-graphs.md)** — forward refs, loops, gates, switches
7. **[Visualization](docs/tutorials/07-visualization.md)** — DOT/SVG output, reading diagrams
8. **[Utilities](docs/tutorials/08-utilities.md)** — checkpoints, clipping, freezing, initialization

### Design

- [Roadmap](docs/design/roadmap.md) — development plan and port status
- [Trajectory Thesis](docs/design/trajectory-thesis.md) — geometric intuition behind the project

### Examples

- [`flodl/examples/showcase.rs`](flodl/examples/showcase.rs) — every graph builder method in one graph

## Lineage

floDl is a Rust port of [goDl](https://github.com/fab2s/goDl), a Go-native
DL framework. The port was motivated by Go's inability to manage VRAM
deterministically — Rust's ownership model solves this at the language level.
The graph builder API, module architecture, and design philosophy carry over
directly.

## License

floDl is open-sourced software licensed under the [MIT license](./LICENSE).
