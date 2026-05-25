# Contributing Guide

Thanks for contributing to Hetero-Paged-Infer.

## Development Setup

```bash
git clone https://github.com/AICL-Lab/hetero-paged-infer.git
cd hetero-paged-infer
cargo build
```

## Core Validation Commands

Run these before opening a PR:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --verbose
cargo test --doc --verbose
cargo doc --no-deps
cargo bench --no-run
cd docs && npm run build
```

## Development Workflow

1. Make a focused change.
2. Add or update tests when behavior changes.
3. Run the relevant validation commands.
4. Update documentation and root `CHANGELOG.md` when project-facing behavior changes.

## Pull Request Checklist

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo test --verbose`
- [ ] Relevant docs updated
- [ ] Root `CHANGELOG.md` updated when needed

## Commit Message Format

Use conventional commits:

```text
<type>(<scope>): <description>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

## License

By contributing, you agree your contributions are licensed under MIT.
