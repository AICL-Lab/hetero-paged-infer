# AGENTS.md - AI Agent Configuration

> This document serves as the system prompt for AI programming assistants (Claude Code, Cursor, etc.). It defines the Spec-Driven Development workflow that must be followed for all code changes.
>
> **[English](AGENTS.md) | [中文](AGENTS.zh.md)**

## Project Philosophy: Spec-Driven Development (SDD)

This project strictly follows the **Spec-Driven Development (SDD)** paradigm. All code implementations must use the specification documents in the `/specs` directory as the **Single Source of Truth**.

## Directory Context

| Directory | Purpose |
|-----------|---------|
| `/specs/product/` | Product Requirements Documents (PRDs) - User stories, acceptance criteria, feature definitions |
| `/specs/rfc/` | Request for Comments (RFCs) - Technical design documents, architecture decisions |
| `/specs/api/` | API Interface Specifications - OpenAPI.yaml, GraphQL schemas, interface contracts |
| `/specs/db/` | Database Specifications - Schema definitions, data models |
| `/specs/testing/` | Test Specifications - BDD Gherkin feature files, property test definitions |
| `/docs/` | User Documentation - Setup guides, tutorials, deployment docs |
| `/changelog/` | Detailed Changelogs - Per-release change documentation |

## AI Agent Workflow Instructions

When you (the AI) are asked to develop a new feature, modify existing functionality, or fix a bug, **you MUST strictly follow this workflow. Do NOT skip any steps**:

### Step 1: 审查与分析 (Review & Analyze)

Before writing any code:

1. **ALWAYS** read the relevant documents in `/specs/`:
   - Product requirements in `/specs/product/`
   - Technical RFCs in `/specs/rfc/`
   - API definitions in `/specs/api/`
   - Test specifications in `/specs/testing/`

2. **If the user's request conflicts with existing specs**:
   - **STOP coding immediately**
   - Point out the conflict
   - Ask the user if they want to update the spec first

### Step 2: 规范优先 (Spec-First Update)

For new features or changes to interfaces/database structures:

1. **MUST propose changes to relevant Spec documents FIRST**:
   - New features → Create/update `/specs/product/*.md`
   - Architecture changes → Create new RFC with naming `NNNN-short-description.md`
   - API changes → Update `/specs/api/openapi.yaml`
   - Database changes → Update `/specs/db/` schemas

2. **Wait for user confirmation** on spec changes before entering code implementation phase

3. **When creating new RFCs**, use sequential naming:
   - `0001-core-architecture.md`
   - `0002-feature-name.md`
   - etc.

### Step 3: 代码实现 (Code Implementation)

When writing code:

1. **100% adhere to the specifications**:
   - Follow exact variable naming from specs
   - Use defined API paths and data types
   - Implement defined status codes and error handling
   - Match interface contracts exactly

2. **Do NOT add features not defined in the specs** (No Gold-Plating)

3. **Follow language-specific conventions**:
   - Rust: `cargo fmt`, `cargo clippy`, comprehensive doc comments
   - Use `snake_case` for functions, variables, modules
   - Use `PascalCase` for structs, enums, traits
   - Use `SCREAMING_SNAKE_CASE` for constants

### Step 4: 测试验证 (Test Against Spec)

Testing requirements:

1. **Write tests based on acceptance criteria** in `/specs/`:
   - Unit tests for individual components
   - Integration tests for end-to-end flows
   - Property tests for invariant verification

2. **Ensure test coverage** for all boundary conditions described in specs

3. **Property tests must validate correctness properties** defined in RFCs

4. **Use proper test labeling format**:
   ```
   Feature: [feature-name], Property N: [property description]
   ```

## Code Generation Rules

### Rule 1: API Changes
Any externally exposed API changes **MUST** be accompanied by updates to `/specs/api/` documents.

### Rule 2: Architecture Decisions
For uncertain technical details, consult `/specs/rfc/` for architectural conventions. **Do NOT invent design patterns independently**.

### Rule 3: No Spec, No Code
If no spec exists for the requested change, **create one first**. Never write implementation code without corresponding spec documentation.

### Rule 4: Test Coverage
All new code must have corresponding tests as specified in `/specs/testing/`.

### Rule 5: Documentation Sync
When code changes, update related documentation:
- API changes → Update `/docs/en/API.md` and `/docs/zh/API.md`
- Configuration changes → Update `/docs/en/CONFIGURATION.md` and `/docs/zh/CONFIGURATION.md`
- Architecture changes → Update `/docs/en/ARCHITECTURE.md` and `/docs/zh/ARCHITECTURE.md`

## Project-Specific Conventions

### File Organization

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

tests/
└── integration_tests.rs  # Integration tests

specs/
├── README.md        # Specs overview
├── product/         # Product requirements
├── rfc/             # Technical RFCs
├── api/             # API specifications
├── db/              # Database schemas
└── testing/         # BDD test specs
```

### Commit Message Format

Use conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation update
- `style` - Code formatting (no functional change)
- `refactor` - Refactoring
- `test` - Test related
- `chore` - Build/tooling related

**Example**:
```
feat(scheduler): add decode priority scheduling

Implement decode request priority over prefill requests in scheduling
to reduce latency of in-progress requests.

Refs: REQ-3.7
Closes #123
```

### PR Checklist

Before submitting a PR, ensure:

- [ ] Code passes `cargo fmt --check`
- [ ] Code passes `cargo clippy --all-targets -- -D warnings`
- [ ] All tests pass `cargo test`
- [ ] New features have corresponding tests
- [ ] Public APIs have doc comments
- [ ] Related specs in `/specs/` are updated
- [ ] Documentation in `/docs/` is updated

## Quick Reference

### Key Commands

```bash
# Build
cargo build --release

# Run all tests
cargo test

# Run property tests
cargo test -- --test-threads=1

# Run with coverage
cargo tarpaulin --out Html

# Format check
cargo fmt --check

# Lint check
cargo clippy --all-targets -- -D warnings

# Run benchmarks
cargo bench

# Generate documentation
cargo doc --no-deps --open
```

### Test Coverage Targets

| Type | Count | Coverage |
|------|-------|----------|
| Unit Tests | 78 | Core modules |
| Property Tests | 15 | Invariant verification |
| Integration Tests | 13 | End-to-end flows |
| Doc Tests | 29 | API examples |

### Requirements Reference Format

When implementing or testing, reference requirements by ID:
- Feature requirements: `REQ-1`, `REQ-2.3`
- Correctness properties: `PROP-5`, `PROP-12`
- RFC sections: `RFC-0001§2.3`

## Why This Matters

### Preventing AI Hallucinations
AI tends to "freely improvise" without context. Forcing it to read `/specs` first anchors its thinking scope and prevents fabricated implementations.

### Constraining Modification Path
By declaring "modify spec before code", documentation and code stay synchronized (Document-Code Synchronization). This prevents drift between documentation and implementation.

### Improving PR Quality
When AI generates Pull Requests, the implementation aligns with business logic because it's developed from acceptance criteria defined in the spec.

### Enabling Code Reviews
Specs provide objective criteria for reviewing code changes. Reviewers can verify implementation matches specification rather than subjective opinions.

---

**Remember**: Specs are the contract. Code is the implementation. The contract always comes first.
