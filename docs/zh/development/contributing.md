# Contributing Guide

Thank you for your interest in contributing to Hetero-Paged-Infer!

## Development Setup

```bash
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer
cargo build
cargo test
```

## Code Style

```bash
# Format code
cargo fmt

# Run linter
cargo clippy --all-targets -- -D warnings

# Check formatting
cargo fmt --check
```

## Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_engine_creation

# Run property tests
cargo test -- --test-threads=1
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
