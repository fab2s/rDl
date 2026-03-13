use std::ffi::c_void;
use std::ptr;

use flodl_sys as ffi;

/// Returns true if gradient computation is enabled.
/// Queries libtorch's GradMode directly.
pub fn is_grad_enabled() -> bool {
    unsafe { ffi::flodl_is_grad_enabled() != 0 }
}

/// RAII guard that disables gradient computation while alive.
///
/// Wraps libtorch's `torch::NoGradGuard`. When the guard is dropped,
/// gradient computation is re-enabled — including on panic unwind.
///
/// ```ignore
/// {
///     let _guard = NoGradGuard::new();
///     let pred = model.forward(&x)?;
///     // gradients re-enabled when _guard drops
/// }
/// ```
pub struct NoGradGuard {
    guard: *mut c_void,
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl NoGradGuard {
    /// Disable gradient computation until this guard is dropped.
    pub fn new() -> Self {
        let guard = unsafe { ffi::flodl_no_grad_guard_new() };
        NoGradGuard { guard }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        if !self.guard.is_null() {
            unsafe { ffi::flodl_no_grad_guard_delete(self.guard) };
            self.guard = ptr::null_mut();
        }
    }
}

/// Execute a closure with gradient computation disabled.
///
/// Operations inside `no_grad` will not build a backward graph,
/// reducing memory usage for inference and parameter updates.
///
/// ```ignore
/// let pred = no_grad(|| model.forward(&x).unwrap());
/// assert!(!pred.requires_grad());
/// ```
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = NoGradGuard::new();
    f()
}
