# The Trajectory Thesis

Neural networks are trajectory generators. An input enters a high-dimensional
space, and the network's weights define a landscape that guides it along a path
through that space. Training shapes the landscape. Inference lets the input
follow its natural trajectory. Everything else is implementation detail.

This document frames the core concepts of deep learning through the trajectory
lens and connects them to floDl's architecture decisions.

---

## Trajectories, not layers

The standard mental model of a neural network is a stack of layers: input ->
hidden -> hidden -> output. This is a useful abstraction for building software,
but it obscures what's actually happening geometrically.

Each layer transforms a point in activation space to a new point. A forward pass
is a trajectory — a sequence of positions through a high-dimensional manifold.
The weights define the vector field that determines where each point moves next.

This isn't a metaphor. Residual networks (ResNet) made it literal: the skip
connection `x + f(x)` means each layer computes a *delta* — a small step from
the current position. He et al. (2015) showed this dramatically improves
training. Chen et al. (2018) took it further with Neural ODEs, replacing
discrete residual steps with a continuous differential equation:

```
dx/dt = f(x, t, theta)
```

The forward pass becomes solving an ODE — following a continuous trajectory
through activation space. The "layers" are just discretization steps along
that trajectory.

---

## Unified vocabulary

Most DL concepts have clean trajectory interpretations:

| Standard framing | Trajectory framing |
|---|---|
| Training | Shaping the landscape so trajectories converge correctly |
| Inference | Letting an input follow its natural trajectory |
| Loss function | Measuring how far the trajectory's endpoint is from the target region |
| Gradient descent | Adjusting the landscape to pull trajectories toward targets |
| Overfitting | Trajectories that are too narrow — only work for training inputs |
| Generalization | Wide valleys — nearby inputs follow similar paths |
| Regularization | Smoothing the landscape to prevent sharp, narrow valleys |
| Attention | Dynamically choosing which dimensions matter at each trajectory step |
| Residual connections | Making trajectory steps incremental (continuous-like flow) |
| Adaptive computation | Letting the trajectory decide its own length |
| Transfer learning | A landscape from one task has valleys useful for another |
| Dropout | Randomly blocking dimensions, forcing trajectories to be robust |
| Batch normalization | Re-centering the trajectory distribution at each step |

---

## Data-dependent control flow as trajectory branching

Fixed-architecture networks force every input through the same trajectory
length and structure. A 50-layer ResNet always takes 50 steps.

Adaptive architectures let the input *choose its trajectory*:

- **Adaptive depth**: iterate until confidence is high enough — variable-length
  trajectory.
- **Conditional branches**: route different inputs through different sub-networks
  — trajectories fork based on activation position.
- **Recurrent attention**: each step chooses where to look next — the trajectory
  is a sequence of positions in the input space.
- **Early exit**: stop when a criterion is met — the trajectory terminates when
  it reaches a confident region.

These are the architectures Python penalizes most. Every branch evaluation, every
loop iteration, every early-exit check carries ~3-5us overhead plus CUDA
synchronization. The framework *discourages* trajectory branching through
performance pressure.

floDl removes that pressure. Rust's zero-cost abstractions mean branches are
pattern matches (nanoseconds). Loops are Rust `for` loops. The trajectory
structure is determined by the math, not by the framework's limitations.

---

## Gradients through trajectories

Training adaptive architectures requires backpropagation through the trajectory
structure:

- **Variable-length loops**: if input A takes 3 steps and input B takes 7, the
  backward pass unrolls 3 and 7 steps respectively.
- **Conditional branches**: only the taken branch receives gradients. Over many
  training samples, each branch's weights specialize.
- **Parallel paths**: independent trajectories get independent gradients.

This is why floDl uses a Rust-native autograd engine rather than libtorch's.
The graph must capture the actual trajectory — including branches and variable
length — not a pre-traced static computation graph.

---

## The selection bias in current research

If your framework makes certain trajectory structures expensive, researchers
avoid them. This is selection pressure:

**Well-explored** (cheap trajectories in Python):
- Fixed-depth feedforward (ResNet, ViT)
- Single-pass attention (Transformer)
- Constant-width parallel heads

**Under-explored** (expensive trajectories in Python):
- Recurrent attention with variable fixation count
- Tree search during training (MCTS-style)
- Iterative hypothesis refinement
- Adaptive computation depth
- Multi-scale processing with feedback loops

Biological cognition is in the second category. Vision involves sequential
fixations. Reasoning involves iterative refinement. Memory recall involves
variable-depth search. The architectures that model human cognition most
closely are the ones Python punishes most.

---

## Connection to floDl's architecture

floDl's layered design maps directly to the trajectory thesis:

1. **Tensor API** — the coordinate system. Points in activation space are
   tensors. Operations move points.

2. **Autograd** — trajectory analysis. Given a trajectory (forward pass),
   compute how changing the landscape (weights) would change where the
   trajectory ends up (gradients).

3. **Layers & Optimizers** — standard landscape components. A Linear layer
   defines a linear transformation. An optimizer adjusts the landscape.

4. **Graph Engine** — trajectory orchestration. Branching, looping, parallel
   paths, and adaptive depth become first-class constructs. The graph engine
   *manages trajectories* through a dynamic computation structure.

The graph engine is the key differentiator. It makes trajectory branching a
composition primitive rather than an implementation challenge.

---

## Why Rust specifically

Go (goDl) proved the graph engine concept but hit a fundamental limit:
Go's garbage collector cannot manage VRAM deterministically. GPU memory
lives in libtorch's C++ allocator — invisible to Go's GC. This required
five phases of memory management infrastructure:

1. Atomic reference counting on every tensor
2. Saved-tensor lifecycle management in autograd
3. GC callbacks for CUDA OOM recovery
4. VRAM budget heuristics with proactive GC
5. Autograd Scope for deterministic batch cleanup

All of this — hundreds of lines of `runtime.KeepAlive`, `Retain()`/`Release()`,
pending-free queues, and callback safety guards — exists because Go cannot
express "free this C++ handle when I'm done with it" as a language primitive.

Rust can. `Drop` replaces all five phases with one trait implementation.
The ownership model guarantees deterministic cleanup at the language level.
No GC, no finalizers, no VRAM budgets, no KeepAlive. This isn't a marginal
improvement — it's an entire category of bugs eliminated.

The same ownership model that manages VRAM also prevents data races, dangling
pointers, and double-frees at compile time. For a framework that interfaces
with C++ GPU kernels, these guarantees matter.

---

## Beyond single-strategy training

Current large models are trained with one strategy: predict the next token.
All knowledge must be acquired through that single lens.

### Mixture of Strategies

A deeper approach: different sub-networks trained with fundamentally different
learning strategies, composed into a single system.

- A perception module trained with supervised learning
- A reasoning module trained with reinforcement learning
- A memory module trained with contrastive learning
- A meta-controller that learns when to invoke which module

Each component has its own loss function, learning rate, and update schedule.
Gradients flow between components where strategies should reinforce each other,
and are blocked where they shouldn't interfere.

In the trajectory frame: each module defines a different kind of landscape, and
the meta-controller learns to compose trajectories across landscapes.

### Graph-as-Module composition

A trained mixture of experts is itself a Module that can be placed inside a
larger graph:

```
Level 0: Individual modules (Linear, GRU, attention heads)
Level 1: Trained sub-networks (perception, reasoning, memory)
Level 2: Strategy mixtures (MoE with learned routing)
Level 3: Meta-graph that learns to compose strategy mixtures
```

Each level is trained independently, then composed. The meta-graph doesn't
need to learn perception — it learns *when to use perception versus reasoning*,
and how to route intermediate results between them.

### Why the tools matter

The reason AI hasn't adopted modular development isn't that it's a bad idea.
It's that the tools didn't support it:

- **You can't modularize what you can't compose.** If the framework has no
  concept of sub-graphs, you can't build independent modules.
- **You can't independently retrain what you can't independently differentiate.**
  If gradients must flow through the entire system, you can't freeze one module
  while training another.
- **You can't iterate quickly on composition if composition is expensive.**

floDl's graph engine makes all of this structural: sub-graphs with independent
training contexts, selective gradient flow, and zero-overhead routing decisions.

---

## References

- He et al. (2015) — *Deep Residual Learning for Image Recognition*
- Chen et al. (2018) — *Neural Ordinary Differential Equations*
- Graves (2016) — *Adaptive Computation Time for Recurrent Neural Networks*
- Bengio et al. (2015) — *Conditional Computation in Neural Networks*
- Shazeer et al. (2017) — *Sparsely-Gated Mixture-of-Experts Layer*
- Kirkpatrick et al. (2017) — *Overcoming Catastrophic Forgetting in Neural Networks*
- Jacobs et al. (1991) — *Adaptive Mixtures of Local Experts*
