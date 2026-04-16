# Specifications

This directory contains the authoritative specifications for the Hetero-Paged-Infer project. All implementations must adhere to these specifications as the Single Source of Truth.

## Structure

| Directory | Purpose |
|-----------|---------|
| `product/` | Product requirements and feature definitions (PRDs) |
| `rfc/` | Technical design documents and architecture decisions (RFCs) |
| `api/` | API interface specifications (REST, GraphQL, etc.) |
| `db/` | Database schema and data model specifications |
| `testing/` | BDD test case specifications (Gherkin feature files) |

## Specification Documents

### Product Requirements
- [Heterogeneous Inference Engine](product/heterogeneous-inference-engine.md) - Core product requirements with user stories and acceptance criteria

### Technical RFCs
- [RFC-0001: Heterogeneous Inference System Architecture](rfc/0001-heterogeneous-inference-architecture.md) - System architecture design
- [RFC-0001: Implementation Tasks](rfc/0001-implementation-tasks.md) - Task breakdown and implementation plan

### Test Specifications
- [Inference Engine BDD Tests](testing/inference-engine.feature) - Behavior-driven test specifications

## Working with Specs

### Creating New Specs

1. **Product Requirements**: Add to `product/` with clear user stories and acceptance criteria
2. **Technical RFCs**: Add to `rfc/` using numbered naming: `NNNN-short-description.md`
3. **API Specs**: Add to `api/` in machine-readable format (OpenAPI, GraphQL schema, etc.)
4. **Test Specs**: Add to `testing/` using Gherkin syntax in `.feature` files

### Referencing Specs

When implementing features or fixing bugs:
- Reference requirements by ID (e.g., REQ-1, REQ-2.3)
- Reference properties by ID (e.g., PROP-5, PROP-12)
- Always review related specs before making changes

### Updating Specs

All spec changes must:
1. Be proposed before implementation
2. Get reviewed and approved
3. Be implemented according to spec
4. Have tests validating the spec

See [AGENTS.md](../AGENTS.md) for the complete spec-driven development workflow.
