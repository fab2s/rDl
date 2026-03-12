# flodl development commands
#
# All commands run inside the Docker container via docker compose.

COMPOSE = docker compose
RUN     = $(COMPOSE) run --rm dev

.PHONY: build test test-release check clippy doc shell clean image

# Build the Docker image
image:
	$(COMPOSE) build

# Build the project (debug)
build: image
	$(RUN) cargo build

# Run all tests
test: image
	$(RUN) cargo test -- --nocapture

# Run tests in release mode
test-release: image
	$(RUN) cargo test --release -- --nocapture

# Type check without building
check: image
	$(RUN) cargo check

# Lint
clippy: image
	$(RUN) cargo clippy -- -W clippy::all

# Generate API docs → target/doc/flodl/index.html
doc: image
	$(RUN) cargo doc --no-deps --document-private-items

# Interactive shell
shell: image
	$(COMPOSE) run --rm dev bash

# Clean build artifacts
clean:
	$(COMPOSE) down -v --rmi local
