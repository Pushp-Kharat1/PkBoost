# Contributing to PKBoost

Thank you for your interest in contributing to PKBoost! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Be kind, constructive, and professional in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/pkboost.git
   cd pkboost
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/PKBoost-AI-Labs/pkboost.git
   ```

## How to Contribute

### Types of Contributions Welcome

- **Bug fixes** - Found a bug? We'd love a fix!
- **Performance improvements** - Make PKBoost faster
- **New features** - Extend functionality (discuss first in an issue)
- **Documentation** - Improve docs, examples, or tutorials
- **Tests** - Increase test coverage
- **Benchmarks** - Add new benchmark datasets or scenarios

### Before You Start

For significant changes, please **open an issue first** to discuss:
- What problem you're solving
- Your proposed approach
- Any breaking changes

This helps avoid duplicate work and ensures alignment with project goals.

## Development Setup

### Prerequisites

- **Rust 1.70+** (install via [rustup](https://rustup.rs/))
- **Python 3.8+** (for Python bindings testing)
- **8GB+ RAM** recommended for large dataset benchmarks

### Building

```bash
# Debug build
cargo build

# Release build (for benchmarks)
cargo build --release

# Run tests
cargo test

# Build documentation
cargo doc --no-deps --open
```

### Running Benchmarks

```bash
# Main benchmark
cargo run --release --bin benchmark

# Drift tests
cargo run --release --bin test_drift

# Multi-class benchmark
cargo run --release --bin multiclass_benchmark
```

### Python Bindings

```bash
# Install maturin
pip install maturin

# Build and install locally
maturin develop --release

# Run Python tests
python -m pytest tests/
```

## Coding Standards

### Rust Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` before committing
- Run `cargo clippy` and address warnings
- Write descriptive commit messages

### Code Quality

```bash
# Format code
cargo fmt

# Check for common mistakes
cargo clippy

# Run all checks before PR
cargo fmt && cargo clippy && cargo test
```

### Documentation

- Add doc comments (`///`) to all public items
- Include examples in doc comments where helpful
- Update README.md if adding major features

### Testing

- Add tests for new functionality
- Ensure existing tests pass
- Include both unit tests and integration tests where appropriate

## Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clean, documented code
- Add tests for new functionality
- Update documentation as needed

### 3. Commit Your Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

Use conventional commit messages:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `perf:` - Performance improvement
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance tasks

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub with:
- Clear title describing the change
- Description of what and why
- Reference to related issues (if any)
- Screenshots/benchmarks for performance changes

### 5. Review Process

- Maintainers will review your PR
- Address any requested changes
- Once approved, your PR will be merged

## Reporting Issues

### Bug Reports

Include:
- PKBoost version (`cargo pkgid pkboost`)
- Rust version (`rustc --version`)
- Operating system
- Minimal reproduction code
- Expected vs actual behavior
- Error messages/stack traces

### Feature Requests

Include:
- Problem you're trying to solve
- Proposed solution
- Use cases
- Alternatives considered

## Questions?

- Open an issue for questions
- Email: kharatpushp16@outlook.com

## License

By contributing to PKBoost, you agree that your contributions will be licensed under the GPL-3.0 License.

---

Thank you for contributing to PKBoost! 🚀
