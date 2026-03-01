.PHONY: build install install-dev lint format format-check test check clean

VENV := .venv
PYTHON := $(VENV)/bin/python
UV := uv

# Build the Rust extension and install into the venv
build:
	$(UV) run maturin develop --release

# Install runtime dependencies
install:
	$(UV) sync

# Install runtime + dev dependencies, then build extension
install-dev:
	$(UV) sync --extra dev
	$(MAKE) build

lint:
	$(UV) run ruff check pkboost/ pkboost_sklearn/ tests/

format:
	$(UV) run ruff format pkboost/ pkboost_sklearn/ tests/

format-check:
	$(UV) run ruff format --check --diff pkboost/ pkboost_sklearn/ tests/

test:
	$(UV) run pytest tests/ -v --tb=short

# Run all checks (mirrors CI pr-checks workflow)
check: format-check lint test

clean:
	rm -rf $(VENV) target/wheels/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
