# Specifications

This directory contains the **authoritative specifications** for the Hetero-Paged-Infer project. All implementations must adhere to these specifications as the **Single Source of Truth**.

## Directory Structure

```
specs/
├── README.md           # This file - specs overview and workflow guide
├── product/            # Product Requirements Documents (PRDs)
│   └── heterogeneous-inference-engine.md
├── rfc/                # Technical Design Documents (RFCs)
│   ├── 0001-heterogeneous-inference-architecture.md
│   └── 0001-implementation-tasks.md
├── api/                # API Interface Specifications
│   └── README.md       # API spec guidelines (OpenAPI, GraphQL, etc.)
├── db/                 # Database Schema Specifications
│   └── README.md       # Database design guidelines
└── testing/            # BDD Test Specifications
    └── inference-engine.feature
```

## Purpose of Each Directory

| Directory | Purpose | Format |
|-----------|---------|--------|
| `product/` | Product requirements and feature definitions (PRDs) | Markdown |
| `rfc/` | Technical design documents and architecture decisions (RFCs) | Markdown |
| `api/` | API interface specifications | OpenAPI YAML, GraphQL Schema, Protobuf |
| `db/` | Database schema and data model specifications | DBML, SQL, Mermaid ERD |
| `testing/` | BDD test case specifications | Gherkin `.feature` files |

## Specification Documents

### Product Requirements (PRDs)

| Document | Description |
|----------|-------------|
| [Heterogeneous Inference Engine](product/heterogeneous-inference-engine.md) | Core product requirements with user stories and acceptance criteria |

### Technical RFCs

| RFC | Title | Status |
|-----|-------|--------|
| [RFC-0001](rfc/0001-heterogeneous-inference-architecture.md) | Heterogeneous Inference System Architecture | Accepted |
| [RFC-0001](rfc/0001-implementation-tasks.md) | Implementation Tasks | Complete |

### Test Specifications

| Document | Description |
|----------|-------------|
| [Inference Engine BDD Tests](testing/inference-engine.feature) | Behavior-driven test specifications |

## Working with Specs

### Creating New Specs

#### Product Requirements

Add to `product/` with:
- Clear user stories following format: "As a [role], I want [feature] so that [benefit]"
- Acceptance criteria using Given-When-Then format
- Requirement IDs (e.g., REQ-1, REQ-2.3)

#### Technical RFCs

Add to `rfc/` using:
- Numbered naming: `NNNN-short-description.md`
- Include metadata table (ID, Status, Authors, Created)
- Follow RFC template structure

#### API Specifications

Add to `api/` in:
- OpenAPI 3.0 format (YAML preferred)
- Include examples and error documentation
- Version all APIs

#### Test Specifications

Add to `testing/` using:
- Gherkin syntax in `.feature` files
- Clear scenario descriptions
- Reference requirement IDs

### Referencing Specs

When implementing features or fixing bugs:

| Reference Type | Format | Example |
|---------------|--------|---------|
| Feature Requirements | `REQ-N` | REQ-1, REQ-2.3 |
| Correctness Properties | `PROP-N` | PROP-5, PROP-12 |
| RFC Sections | `RFC-NNNN§N` | RFC-0001§2.3 |
| Acceptance Criteria | `REQ-N.AC-N` | REQ-1.AC-3 |

### Spec Update Workflow

All spec changes must follow this process:

```
1. Propose → 2. Review → 3. Approve → 4. Implement → 5. Validate
```

1. **Propose**: Create or modify spec document
2. **Review**: Team reviews for completeness and correctness
3. **Approve**: Get sign-off before implementation
4. **Implement**: Code according to spec
5. **Validate**: Test implementation against spec

## Spec-Driven Development Workflow

See [AGENTS.md](../AGENTS.md) for the complete workflow that AI assistants and developers must follow.

### Key Principles

1. **Spec First**: Always update specs before code
2. **Single Source of Truth**: Specs are authoritative
3. **No Gold-Plating**: Don't add features not in specs
4. **Test Against Spec**: Validate implementation matches specification
5. **Document Synchronization**: Keep docs and code aligned

## Quality Checklist

### For Product Specs

- [ ] User stories are complete and clear
- [ ] Acceptance criteria are testable
- [ ] Requirements are uniquely identified
- [ ] Dependencies are documented
- [ ] Edge cases are covered

### For Technical RFCs

- [ ] Problem statement is clear
- [ ] Solution approach is well-defined
- [ ] Trade-offs are documented
- [ ] Implementation plan exists
- [ ] Testing strategy is defined

### For API Specs

- [ ] All endpoints documented
- [ ] Request/response schemas defined
- [ ] Error codes documented
- [ ] Examples provided
- [ ] Versioning strategy defined

### For Test Specs

- [ ] Scenarios cover all acceptance criteria
- [ ] Edge cases included
- [ ] Property tests defined for invariants
- [ ] Requirement references included

## Related Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| AI Agent Configuration | [AGENTS.md](../AGENTS.md) | Workflow for AI assistants |
| Contributing Guide | [CONTRIBUTING.md](../CONTRIBUTING.md) | How to contribute to the project |
| Architecture Docs | [docs/en/ARCHITECTURE.md](../docs/en/ARCHITECTURE.md) | High-level architecture overview |
| API Reference | [docs/en/API.md](../docs/en/API.md) | API usage documentation |

---

**Remember**: Specifications are the contract. Implementation is the fulfillment. Always spec first, implement second.
