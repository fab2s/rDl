use std::cell::Cell;

thread_local! {
    static NO_GRAD_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Returns true if gradient computation is enabled.
pub fn is_grad_enabled() -> bool {
    NO_GRAD_DEPTH.with(|d| d.get() == 0)
}

/// Execute a closure with gradient computation disabled.
///
/// Operations inside `no_grad` will not build a backward graph,
/// reducing memory usage for inference and parameter updates.
///
/// Per-thread: each thread has its own depth counter. This matches
/// Rust's Rc-based autograd (single-threaded by design).
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    NO_GRAD_DEPTH.with(|d| d.set(d.get() + 1));
    let result = f();
    NO_GRAD_DEPTH.with(|d| d.set(d.get() - 1));
    result
}
