# AGENTS.md - AI Agent Configuration

## Project Philosophy: Spec-Driven Development (SDD)

This project strictly follows the **Spec-Driven Development (SDD)** paradigm. All code implementations must use the specification documents in the `/specs` directory as the Single Source of Truth.

## Directory Context

| Directory | Purpose |
|-----------|---------|
| `/specs/product/` | Product feature definitions and acceptance criteria |
| `/specs/rfc/` | Technical design documents and architecture decisions |
| `/specs/api/` | API interface specifications (e.g., OpenAPI.yaml) |
| `/specs/db/` | Database schema and data model specifications |
| `/specs/testing/` | BDD test case specifications |
| `/docs/` | User guides, tutorials, and deployment documentation |

## AI Agent Workflow Instructions

When you (the AI) are asked to develop a new feature, modify existing functionality, or fix a bug, **you MUST strictly follow this workflow. Do NOT skip any steps**:

### Step 1: Review Specifications

- **ALWAYS** start by reading the relevant documents in `/specs`:
  - Product requirements in `/specs/product/`
  - Technical RFCs in `/specs/rfc/`
  - API definitions in `/specs/api/`
  - Test specifications in `/specs/testing/`
- If the user's request conflicts with existing specs, **STOP coding immediately** and point out the conflict. Ask the user if they want to update the spec first.

### Step 2: Spec-First Update

- For new features or changes to interfaces/database structures, **MUST propose changes to the relevant Spec documents first** (e.g., update `/specs/product/*.md` or create a new RFC).
- Wait for user confirmation on the spec changes before entering the code implementation phase.
- When creating new RFCs, use the naming convention: `NNNN-short-description.md` (e.g., `0002-oauth2-implementation.md`).

### Step 3: Code Implementation

- When writing code, **100% adhere to the specifications** (including variable naming, API paths, data types, status codes, etc.).
- **Do NOT add features not defined in the specs** (No Gold-Plating).
- Follow Rust coding conventions: use `cargo fmt`, `cargo clippy`, and maintain comprehensive doc comments.

### Step 4: Test Against Spec

- Write unit tests and integration tests based on the acceptance criteria in `/specs`.
- Ensure test cases cover all boundary conditions described in the specs.
- Property tests must validate the correctness properties defined in RFCs.

## Code Generation Rules

1. **API Changes**: Any externally exposed API changes **MUST** be accompanied by updates to `/specs/api/` documents.
2. **Architecture Decisions**: For uncertain technical details, consult `/specs/rfc/` for architectural conventions. Do NOT invent design patterns independently.
3. **No Spec, No Code**: If no spec exists for the requested change, create one first. Never write implementation code without corresponding spec documentation.
4. **Test Coverage**: All new code must have corresponding tests as specified in `/specs/testing/`.

## Project-Specific Conventions

### Naming Conventions
- Use `snake_case` for functions, variables, and modules
- Use `PascalCase` for structs, enums, and traits
- Use `SCREAMING_SNAKE_CASE` for constants

### File Organization
- Core types in `src/types.rs`
- Error types in `src/error.rs`
- Trait definitions co-located with implementations
- Tests in same file with `#[cfg(test)]` module or in `/tests/`

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

## Quick Reference

### Key Commands
```bash
# Build
cargo build --release

# Run tests
cargo test

# Run with coverage
cargo tarpaulin --out Html

# Format check
cargo fmt --check

# Lint check
cargo clippy --all-targets -- -D warnings

# Run benchmarks
cargo bench
```

### Test Coverage Targets
| Type | Count | Coverage |
|------|-------|----------|
| Unit Tests | 78 | Core modules |
| Property Tests | 15 | Invariant verification |
| Integration Tests | 13 | End-to-end flows |
| Doc Tests | 29 | API examples |

## Why This Matters

**Preventing AI Hallucinations**: AI tends to "freely improvise" without context. Forcing it to read `/specs` first anchors its thinking scope.

**Constraining Modification Path**: By declaring "modify spec before code", documentation and code stay synchronized (Document-Code Synchronization).

**Improving PR Quality**: When AI generates Pull Requests, the implementation aligns with business logic because it's developed from acceptance criteria defined in the spec.
