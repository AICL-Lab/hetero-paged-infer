-- Hetero-Paged-Infer Database Schema
--
-- This is a skeleton specification — the current implementation is an in-memory
-- inference engine without persistent storage. These tables are planned for
-- future features.
--
-- Related:
-- - specs/db/README.md (design principles)
-- - specs/product/heterogeneous-inference-engine.md (requirements)
-- - specs/api/openapi.yaml (API contracts)
--
-- Database: PostgreSQL 15+
-- Migration tool: sqlx / diesel (TBD)

-- ============================================================================
-- Schema Version: 001
-- Description: Initial schema for request history, metrics, and model registry
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Table: models
-- Purpose: Model registry and versioning metadata
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS models (
    id              BIGSERIAL       PRIMARY KEY,
    model_id        VARCHAR(128)    NOT NULL UNIQUE,
    version         VARCHAR(32)     NOT NULL,
    config          JSONB           NOT NULL,            -- EngineConfig serialized
    path            TEXT            NOT NULL,            -- Model file path
    checksum        VARCHAR(64),                         -- SHA-256 of model file
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    is_active       BOOLEAN         NOT NULL DEFAULT TRUE
);

CREATE INDEX idx_models_model_id ON models (model_id);
CREATE INDEX idx_models_is_active ON models (is_active);
CREATE INDEX idx_models_created_at ON models (created_at);

COMMENT ON TABLE models IS 'Model registry: tracks available inference models and their metadata';
COMMENT ON COLUMN models.config IS 'Serialized EngineConfig including tokenizer, device, and KV cache settings';

-- ----------------------------------------------------------------------------
-- Table: request_history
-- Purpose: Append-only audit log of all inference requests
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS request_history (
    id              BIGSERIAL       PRIMARY KEY,
    request_id      VARCHAR(64)     NOT NULL UNIQUE,
    model_id        VARCHAR(128)    REFERENCES models(model_id),
    input_text      TEXT            NOT NULL,
    output_text     TEXT,                              -- NULL until completed
    generation_params JSONB         NOT NULL,            -- GenerationParams serialized
    status          VARCHAR(32)     NOT NULL,            -- pending|prefill|decode|completed|failed
    error_message   TEXT,                              -- Populated on failure
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

CREATE INDEX idx_request_history_request_id ON request_history (request_id);
CREATE INDEX idx_request_history_created_at ON request_history (created_at);
CREATE INDEX idx_request_history_status ON request_history (status);
CREATE INDEX idx_request_history_model_id ON request_history (model_id);

COMMENT ON TABLE request_history IS 'Append-only audit log of inference requests. Records must never be updated or deleted.';
COMMENT ON COLUMN request_history.generation_params IS 'GenerationParams: max_tokens, temperature, top_p';

-- ----------------------------------------------------------------------------
-- Table: inference_metrics
-- Purpose: Performance telemetry for monitoring and optimization
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inference_metrics (
    id              BIGSERIAL       PRIMARY KEY,
    request_id      VARCHAR(64)     NOT NULL REFERENCES request_history(request_id),
    prefill_time_ms INTEGER,                          -- Time spent in prefill phase
    decode_time_ms  INTEGER,                          -- Time spent in decode phase
    total_tokens    INTEGER         NOT NULL,           -- Input + output tokens
    input_tokens    INTEGER         NOT NULL,
    output_tokens   INTEGER         NOT NULL,
    batch_size      INTEGER,                          -- Batch size during execution
    memory_used_blocks INTEGER,                       -- KV Cache blocks used
    gpu_memory_mb   FLOAT,                            -- GPU memory used (MB)
    device          VARCHAR(16),                       -- cpu|gpu|hybrid
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_inference_metrics_request_id ON inference_metrics (request_id);
CREATE INDEX idx_inference_metrics_created_at ON inference_metrics (created_at);
CREATE INDEX idx_inference_metrics_device ON inference_metrics (device);

COMMENT ON TABLE inference_metrics IS 'Performance telemetry: timing, throughput, and resource usage per request';

-- ----------------------------------------------------------------------------
-- Table: scheduler_events
-- Purpose: Scheduler decision log for debugging and analysis
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scheduler_events (
    id              BIGSERIAL       PRIMARY KEY,
    event_time      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    action          VARCHAR(32)     NOT NULL,            -- schedule|evict|promote|demote
    sequence_ids    JSONB           NOT NULL,            -- Array of affected seq_ids
    reason          TEXT,
    batch_tokens    INTEGER,                             -- Total tokens in scheduled batch
    prefill_count   INTEGER         NOT NULL DEFAULT 0,  -- Number of prefill sequences
    decode_count    INTEGER         NOT NULL DEFAULT 0   -- Number of decode sequences
);

CREATE INDEX idx_scheduler_events_event_time ON scheduler_events (event_time);
CREATE INDEX idx_scheduler_events_action ON scheduler_events (action);

COMMENT ON TABLE scheduler_events IS 'Scheduler decision log: records scheduling actions for debugging and analysis';

-- ----------------------------------------------------------------------------
-- Views: Common query patterns
-- ----------------------------------------------------------------------------

-- Throughput: requests per minute (last hour)
CREATE OR REPLACE VIEW v_throughput_rpm AS
SELECT
    date_trunc('minute', created_at) AS minute,
    COUNT(*)                         AS request_count,
    AVG(output_tokens)               AS avg_output_tokens,
    AVG(
        EXTRACT(EPOCH FROM (completed_at - created_at)) * 1000
    )                                AS avg_latency_ms
FROM request_history
WHERE status = 'completed'
  AND created_at > NOW() - INTERVAL '1 hour'
GROUP BY minute
ORDER BY minute DESC;

-- Resource utilization over time (5-minute intervals)
CREATE OR REPLACE VIEW v_resource_utilization AS
SELECT
    date_trunc('5 minutes', created_at) AS interval,
    device,
    AVG(memory_used_blocks)             AS avg_blocks,
    AVG(gpu_memory_mb)                  AS avg_gpu_memory_mb,
    COUNT(*)                            AS request_count
FROM inference_metrics
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY interval, device
ORDER BY interval DESC;

-- ----------------------------------------------------------------------------
-- Notes
-- ----------------------------------------------------------------------------
--
-- Migration strategy:
--   1. Create migration files in migrations/ directory
--   2. Use sequential naming: V001__initial_schema.sql, V002__add_*.sql
--   3. All migrations must be reversible (include DOWN statements)
--   4. Breaking changes require a new API version
--
-- Data retention:
--   - request_history: 90 days (configurable)
--   - inference_metrics: 30 days (configurable)
--   - scheduler_events: 7 days (configurable)
--   - models: indefinite (until manually removed)
--
-- Partitioning (future):
--   - request_history: partition by RANGE (created_at) monthly
--   - inference_metrics: partition by RANGE (created_at) weekly
--
