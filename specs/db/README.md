# Database Specifications

This directory contains database schema and data model specifications for the Heterogeneous Inference Engine.

## Purpose

Database specifications define:

- Data models and schemas
- Table structures and relationships
- Indexing strategies
- Data migration plans
- Query patterns

## Current Status

> **Note**: The current implementation is an in-memory inference engine without persistent storage. Database specifications are planned for future features such as:
> - Request history and audit logs
> - Model registry and versioning
> - User session management
> - Metrics and telemetry storage

## Planned Schemas

### Request History Schema

```dbml
// Request history for audit and replay
Table request_history {
  id bigint [pk, increment]
  request_id varchar(64) [unique, not null]
  input_text text [not null]
  output_text text
  generation_params jsonb
  status varchar(32) [not null]
  created_at timestamp [default: `now()`]
  completed_at timestamp
  
  indexes {
    request_id
    created_at
    status
  }
}
```

### Metrics Schema

```dbml
// Performance metrics for monitoring
Table inference_metrics {
  id bigint [pk, increment]
  request_id varchar(64) [not null]
  prefill_time_ms integer
  decode_time_ms integer
  total_tokens integer
  batch_size integer
  memory_used_blocks integer
  created_at timestamp [default: `now()`]
  
  indexes {
    request_id
    created_at
  }
}
```

### Model Registry Schema

```dbml
// Model versioning and metadata
Table models {
  id bigint [pk, increment]
  model_id varchar(128) [unique, not null]
  version varchar(32) [not null]
  config jsonb [not null]
  path text [not null]
  checksum varchar(64)
  created_at timestamp [default: `now()`]
  is_active boolean [default: true]
  
  indexes {
    model_id
    is_active
  }
}
```

## Data Model Design Principles

### 1. Immutability for Audit

Request history should be append-only:
- Never modify existing records
- Use versioning for corrections
- Enable full audit trail

### 2. Efficient Time-Series Queries

Metrics data should support:
- Time-range queries
- Aggregation by intervals
- Efficient partitioning

### 3. JSONB for Flexibility

Use PostgreSQL JSONB for:
- Configuration storage
- Parameter storage
- Extensible metadata

## Migration Strategy

When adding database support:

1. Create migration files in `migrations/`
2. Use semantic versioning for schema versions
3. Support rollback for all migrations
4. Document breaking changes

### Migration File Format

```
migrations/
├── V001__initial_schema.sql
├── V002__add_request_history.sql
├── V003__add_metrics_table.sql
└── ...
```

## Guidelines

### Creating Schema Specs

1. **Use DBML format** for visual documentation
2. **Include ER diagrams** for complex relationships
3. **Document indexing strategy** with rationale
4. **Consider query patterns** in design
5. **Plan for scalability** from the start

### Schema Review Checklist

- [ ] All tables have primary keys
- [ ] Foreign key relationships documented
- [ ] Indexes planned for common queries
- [ ] Migration path defined
- [ ] Backward compatibility considered
- [ ] Performance implications documented

## Related Documents

- [Product Requirements](../product/) - Data requirements
- [RFCs](../rfc/) - Architecture decisions affecting data
- [API Specifications](../api/) - Data exposed via APIs
