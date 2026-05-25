# Contributing

This page mirrors the repository workflow in [`CONTRIBUTING.md`](https://github.com/AICL-Lab/hetero-paged-infer/blob/master/CONTRIBUTING.md).

## Quick start

```bash
git clone https://github.com/AICL-Lab/hetero-paged-infer.git
cd hetero-paged-infer
cargo build
```

## Validation

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --verbose
cargo test --doc --verbose
cargo doc --no-deps
cargo bench --no-run
cd docs && npm run build
```

## Contribution flow

1. Create a focused branch.
2. Implement the change and add tests when behavior changes.
3. Run validation commands.
4. Update documentation and root `CHANGELOG.md` when project-facing behavior changes.
5. Open a pull request.
