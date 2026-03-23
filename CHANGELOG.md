# Changelog

All notable changes to floDl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **`MaxPool2d`** module: 2D max pooling layer with kernel size, stride, padding, dilation, and ceil mode. Full FFI chain (`shim.cpp` → `flodl-sys` → `Tensor::max_pool2d` → `autograd::max_pool2d` → `nn::MaxPool2d`). Matches PyTorch's `nn.MaxPool2d`.
- **`Rng`** struct: CPU-side random number generator wrapping SmallRng (Xoshiro256++). Methods: `seed`, `from_entropy`, `usize`, `f32`, `f64`, `shuffle`, `bernoulli`, `range`, `normal`. For data loading, shuffling, and augmentation — not for tensor ops (use `manual_seed` for those).
- **`manual_seed(u64)`**: Seed all libtorch RNGs (CPU + CUDA) for reproducible tensor operations (`rand`, `randn`, dropout masks, etc.). Equivalent to `torch.manual_seed()`.
- **`cuda_manual_seed_all(u64)`**: Seed all CUDA device RNGs independently. No-op without CUDA. Equivalent to `torch.cuda.manual_seed_all()`.
- **`cuda_active_bytes()` / `cuda_active_bytes_idx()`**: Query bytes actively used by live tensors (matches `torch.cuda.memory_allocated()` semantics). Complements `cuda_allocated_bytes()` which reports total allocator reservation.

### Fixed
- **VRAM monitoring**: `cuda_allocated_bytes()` now returns `reserved_bytes` instead of `allocated_bytes` from the CUDA caching allocator. `allocated_bytes` only counts active sub-blocks and never exceeds physical VRAM, masking unified-memory spill. `reserved_bytes` includes host-spilled pages, making spill detection actually work.
- Removed unused `ResourceSample::vram_used_bytes` field (dead since dashboard switched to allocator stats).
- Dashboard now uses `vram_alloc` as the sole VRAM metric with a physical-limit reference line.

### Improved
- **Benchmark suite**: Both Rust and Python benchmarks now report allocated (active tensors) and reserved (allocator pool) VRAM separately. Python harness resets peak stats between benchmarks with `empty_cache()` + `reset_peak_memory_stats()` for honest per-benchmark measurement.
- **Docker**: Added `.dockerignore` (excludes everything — source is bind-mounted). Libtorch downloads cached via BuildKit `--mount=type=cache`. Image targets skip rebuild when image exists.

## [0.1.2] - 2026-03-19

### Added
- **VRAM spill detection**: New FFI function `flodl_cuda_alloc_bytes` queries libtorch's CUDA caching allocator. `cuda_allocated_bytes()` / `cuda_allocated_bytes_idx()` expose it in Rust. When allocated bytes exceed physical VRAM, the monitor shows spill in terminal output, live dashboard, CSV export, and epoch log.
- `ResourceSample::vram_allocated_bytes` field for allocator-level memory tracking.
- `vram_spill` column in CSV export.

### Fixed
- README links now use absolute GitHub URLs — fixes broken links on crates.io where relative paths don't resolve.

## [0.1.0] - 2026-03-18

### Added
- **Graph identity**: `Graph::structural_hash()` — deterministic SHA-256 hash of graph topology, module names, and parameter/buffer shapes. Any architecture change produces a different hash. `Graph::short_hash()` returns the first 8 chars. `FlowBuilder::label()` sets a human-readable name (does not affect hash).
- **Checkpoint architecture validation**: Checkpoint format v1 embeds a 32-byte structural hash. `load_checkpoint` / `load_checkpoint_file` accept an optional hash and error on architecture mismatch.
- **Dashboard metadata**: `Monitor::set_metadata(serde_json::Value)` attaches hyperparameters/config to the HTML archive. `watch()` / `watch_profiled()` capture graph label and hash. Dashboard header shows `"floDl — {label} [{hash8}]"`.
- **Parameter freezing**: `Parameter::freeze()`, `unfreeze()`, `is_frozen()` — disable/enable gradient tracking per parameter. Optimizers automatically skip frozen params (no grad). `Parameter::to_device()` now preserves frozen state.
- **Named checkpoints**: `Graph::named_parameters()` and `named_buffers()` return qualified names (`"tag/weight"` or `"node_id/running_mean"`). `save_checkpoint` / `load_checkpoint` persist both parameters and buffers (e.g., BatchNorm running stats), matching by name for partial loading. `LoadReport` reports what was loaded, skipped, and missing.
- **Optimizer parameter groups**: `Adam::with_groups()`, `SGD::with_groups()`, `AdamW::with_groups()` — builder API for per-group learning rates. `Optimizer::set_group_lr()` adjusts a single group; `set_lr()` updates all groups. Groups are persisted through `Stateful` save/load.

### Core Stack
- **Tensor**: Owned RAII tensors with Drop, ~72 operations. CPU and CUDA (feature-gated).
- **Autograd**: Reverse-mode AD backed by libtorch's native autograd engine. 37 differentiable operations with numerical gradient verification.
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
- Checkpointing: save_checkpoint/load_checkpoint (named binary format with LoadReport, persists parameters + buffers, structural hash validation, file or io::Write).
- Weight initialization: kaiming_uniform/normal, xavier_uniform/normal.

### Training Monitor
- Human-readable ETA with adaptive formatting (hours/minutes/seconds/milliseconds).
- System resource tracking: CPU, RAM, GPU utilization (NVML), VRAM usage.
- Live web dashboard via embedded HTTP server with Server-Sent Events.
- Dashboard features: real-time training curves, resource usage charts, epoch log, graph SVG, label/hash header, metadata card.
- CSV and log file export.

### Observation & Visualization
- Tag-based metric collection: collect/flush/trend.
- Trend analysis: slope, stalled, improving, converged.
- Group trends with tag_group expansion.
- DOT/SVG graph visualization with parameter counts and node type shapes.
- Profiling: enable_profiling, profile, timing trends.
- Training curves: plot_html, export_trends, write_log.

### Infrastructure
- **CI**: GitHub Actions with CPU test matrix and CUDA build verification.
- **Docker**: CPU and CUDA Dockerfiles, docker-compose with GPU support.
- **Build**: Makefile with cpu/cuda targets (build, test, clippy, shell).

### Testing
- 329 library tests + showcase tests.
- Zero clippy warnings.
- Autograd numerical gradient checks.
- Module-level gradient checks.

### Key Design Decisions
- **Deterministic VRAM**: Rust's Drop trait replaces 5 phases of GC-based memory management.
- **No GC overhead**: No runtime.KeepAlive, no pending-free queues, no VRAM budget heuristics.
- **Variable**: `Rc<RefCell<VariableInner>>` for cheap Clone with interior mutability.
- **Module trait**: single-input forward + optional NamedInputModule for multi-input. `structural_hash()` for architecture identity.
- **Graph-as-Module**: Graph implements Module for hierarchical composition.
- **NamedInputModule on routers**: SoftmaxRouter and SigmoidRouter sum refs into input before projection.
- **Native FFI ops**: flodl_max, flodl_norm, flodl_cuda_mem_info, flodl_cuda_utilization.
