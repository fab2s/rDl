# Contributing to floDl

Thank you for your interest in floDl. Contributions are welcome and appreciated.

## Getting Started

floDl builds against libtorch via FFI, so all development happens inside Docker:

```bash
git clone https://github.com/fab2s/floDl.git
cd floDl
make image      # build dev container (Rust + libtorch)
make shell      # interactive shell inside the container
make test       # run all tests
make clippy     # lint
```

You do **not** need Rust or libtorch installed on the host machine.

## Development Workflow

1. Fork the repository and create your branch from `main`.
2. Make your changes inside the dev container (`make shell`).
3. Run `make test` to verify all tests pass.
4. Run `make clippy` to ensure zero warnings.
5. Open a pull request.

## Code Style

- Standard Rust conventions: `rustfmt`, zero clippy warnings.
- Keep the API consistent with existing patterns.
- Every fallible operation returns `Result<T>` — use `?` for propagation.
- Every differentiable operation needs a backward function and a numerical
  gradient check in the autograd tests.
- Public types and methods should have `///` doc comments.

## What We're Looking For

**High value contributions:**
- New NN modules (with forward, backward, parameter collection, and gradient checks)
- New autograd operations (with backward and numerical verification)
- Performance improvements to the FFI dispatch path
- Bug fixes with reproducing tests

**Also welcome:**
- Documentation improvements and examples
- Doc tests for public APIs
- CI improvements

**Please discuss first:**
- Changes to public API signatures
- New dependencies
- Architecture changes

Open an issue to discuss before investing significant effort on these.

## Testing

Every PR should pass the existing test suite. If you add new functionality:

- **Tensor ops**: add tests in `tensor.rs`
- **Autograd ops**: add a numerical gradient check
- **NN modules**: add both a functional test and a gradient check
- **Graph features**: add a test in the graph module

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](./LICENSE).
