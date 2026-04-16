# Contributing Guide

Thank you for your interest in contributing to Hetero-Paged-Infer!

## Development Setup

### Requirements

- Rust 1.70+ (2021 edition)
- Git

### Clone and Build

```bash
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer
cargo build
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_engine_creation

# Run property tests
cargo test -- --test-threads=1
```

## Code Style

### Formatting

Use `rustfmt` to maintain consistent code style:

```bash
cargo fmt --check
```

### Linting

Use `clippy` for static analysis:

```bash
cargo clippy --all-targets -- -D warnings
```

### Documentation Comments

All public APIs must have doc comments:

```rust
/// Brief description.
///
/// Detailed explanation.
///
/// # Arguments
///
/// * `param1` - Parameter description
///
/// # Returns
///
/// Return value description.
///
/// # Example
///
/// ```rust
/// use my_crate::my_function;
/// let result = my_function();
/// ```
pub fn my_function() -> i32 {
    42
}
```

## Spec-Driven Development

This project follows **Spec-Driven Development (SDD)**. All changes must start with updating specifications before code implementation.

### Workflow for Contributors

1. **Identify the relevant spec** in `/specs/`:
   - Product requirements: `/specs/product/`
   - Technical designs: `/specs/rfc/`
   - Test specifications: `/specs/testing/`

2. **Update the spec first** before writing any code:
   - If adding a feature, propose the change in the appropriate spec document
   - If fixing a bug, verify the spec accurately describes the expected behavior
   - Get review on spec changes before implementation

3. **Implement according to spec**:
   - Follow the interfaces, types, and constraints defined in specs
   - Do not add functionality not specified in the spec

4. **Test against spec**:
   - Ensure tests cover acceptance criteria from specs
   - Property tests must validate invariants defined in RFCs

## Submitting Code

### Commit Message Format

Use conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation update
- `style` - Code formatting (no functional change)
- `refactor` - Refactoring
- `test` - Test related
- `chore` - Build/tooling related

Example:

```
feat(scheduler): add decode priority scheduling

Implement decode request priority over prefill requests in scheduling
to reduce latency of in-progress requests.

Closes #123
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'feat: add feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Create a Pull Request

### PR Checklist

- [ ] Code passes `cargo fmt --check`
- [ ] Code passes `cargo clippy`
- [ ] All tests pass `cargo test`
- [ ] New features have corresponding tests
- [ ] Public APIs have doc comments
- [ ] Related specs updated (see `/specs/`)
- [ ] Documentation updated

## Testing Requirements

### Unit Tests

Each module should have unit test coverage for core functionality:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_function() {
        assert_eq!(my_function(), expected);
    }
}
```

### Property Tests

Use `proptest` for property-based testing:

```rust
#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_my_property(input in 0..100) {
            // Test property
        }
    }
}
```

### Integration Tests

End-to-end tests validating complete system behavior must be in `/tests/`:

```rust
#[test]
fn test_end_to_end_flow() {
    // Integration test
}
```

## Project Structure

```
src/
├── lib.rs           # Library entry point
├── main.rs          # CLI entry point
├── config.rs        # Configuration
├── engine.rs        # Inference engine
├── error.rs         # Error types
├── types.rs         # Core types
├── kv_cache.rs      # KV Cache manager
├── scheduler.rs     # Scheduler
├── tokenizer.rs     # Tokenizer
└── gpu_executor.rs  # GPU executor

specs/
├── product/         # Product requirements
├── rfc/             # Technical design documents
├── api/             # API specifications
├── db/              # Database schema specifications
└── testing/         # BDD test specifications

tests/
└── integration_tests.rs  # Integration tests

docs/
├── en/              # English documentation
└── zh/              # Chinese documentation
```

## Documentation

### Updating Specs

When proposing changes:

1. Create or update the relevant spec in `/specs/`
2. Use clear, testable acceptance criteria
3. Reference requirements by ID (e.g., REQ-1, REQ-2.3)
4. For RFCs, use numbered naming: `NNNN-short-description.md`

### Updating User Documentation

- User guides go in `/docs/en/` (English) and `/docs/zh/` (Chinese)
- Keep both language versions in sync
- Use MkDocs for documentation site generation

## Getting Help

If you have questions:
- Open an issue on GitHub
- Review existing code and tests as references
- Read the relevant specs in `/specs/` for design intent

## License

This project is licensed under the MIT License. By contributing code, you agree to release it under the same license.
