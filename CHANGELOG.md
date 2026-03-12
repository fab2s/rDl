# Changelog

All notable changes to floDl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Project rename**: rdl -> floDl (flow Deep Learning). Crate names `flodl` + `flodl-sys`. FFI prefix `flodl_`, C type `FlodlTensor`, constants `FLODL_*`.
- **API cleanup**: 11 items completed — schedulers decoupled, default parameters, modules! macro, Result re-export, collect() returns Result, Reduce with native FFI ops, BatchNorm eval safety, build-time NamedInputModule validation.
- **NamedInputModule on routers**: SoftmaxRouter and SigmoidRouter implement NamedInputModule — sum refs into input before projection.
- **Native FFI ops**: `flodl_max` and `flodl_norm` added to shim for Reduce::Max and Reduce::Norm.

## [v0.1.0] - 2026-03-12

Initial release. Rust port of [goDl](https://github.com/fab2s/goDl).

### Core Stack
- **Tensor**: Owned RAII tensors with Drop, ~72 operations. CPU and CUDA (feature-gated).
- **Autograd**: Reverse-mode AD with 37 differentiable operations.
- **NN Modules**: Linear, Conv2d, ConvTranspose2d, LayerNorm, BatchNorm, Dropout, Embedding, GRUCell, LSTMCell.
- **Activations**: ReLU, Sigmoid, Tanh, GELU, SiLU.
- **Losses**: mse_loss, cross_entropy_loss, bce_with_logits_loss, l1_loss, smooth_l1_loss, kl_div_loss.
- **Optimizers**: SGD (with momentum), Adam, AdamW.

### Graph Builder
- Fluent API: from/through/build, split/merge, also (residual), tag/using (named refs).
- Loop constructs: for_n (fixed), while_cond (pre-condition), until_cond (post-condition).
- Routing: gate (soft, weighted), switch (hard, selected branch only).
- Map constructs: each, over, slices, with batched fast path.
- Input (auxiliary graph inputs), tag_group (auto-suffixed parallel branch names).

### Training Tools
- LR scheduling: StepDecay, CosineScheduler, WarmupScheduler (composable), PlateauScheduler.
- Mixed precision: Float16/BFloat16 dtype casting, GradScaler for loss scaling.
- Gradient clipping: clip_grad_norm, clip_grad_value.
- Checkpointing: save_parameters/load_parameters (binary format, file or io::Write).
- Weight initialization: kaiming_uniform/normal, xavier_uniform/normal.

### Observation & Visualization
- Tag-based metric collection: collect/flush/trend.
- Trend analysis: slope, stalled, improving, converged.
- Group trends with tag_group expansion.
- DOT/SVG graph visualization with parameter counts and node type shapes.
- Profiling: enable_profiling, profile, timing trends.
- Training curves: plot_html, export_trends, write_log.

### Testing
- 166 library tests + 14 showcase tests.
- Zero clippy warnings.
- Autograd numerical gradient checks.
- Module-level gradient checks.

### Key Design Decisions
- **Deterministic VRAM**: Rust's Drop trait replaces goDl's entire 5-phase memory management.
- **No GC overhead**: No runtime.KeepAlive, no pending-free queues, no VRAM budget heuristics.
- **Variable**: `Rc<RefCell<VariableInner>>` for cheap Clone with interior mutability.
- **Module trait**: single-input forward + optional NamedInputModule for multi-input.
- **Graph-as-Module**: Graph implements Module for hierarchical composition.
