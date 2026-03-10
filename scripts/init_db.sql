-- ── PostgreSQL schema for ML Model Serving Platform ─────────────────────────
-- Executed automatically by the postgres container at first startup
-- (placed in /docker-entrypoint-initdb.d/)
-- ─────────────────────────────────────────────────────────────────────────────

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── predictions ───────────────────────────────────────────────────────────────
-- Every call to /predict is recorded here (async, best-effort).
CREATE TABLE IF NOT EXISTS predictions (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id      UUID        NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name      VARCHAR(50) NOT NULL,   -- "champion" | "challenger"
    model_version   VARCHAR(50) NOT NULL,
    input_features  JSONB       NOT NULL,
    prediction      SMALLINT    NOT NULL,   -- 0 | 1
    probability     FLOAT       NOT NULL,
    actual_label    SMALLINT,              -- filled in later (ground truth)
    latency_ms      FLOAT
);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp  ON predictions (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model_name ON predictions (model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_request_id ON predictions (request_id);

-- ── model_registry ────────────────────────────────────────────────────────────
-- Mirrors key info from MLflow Registry; used for audit trail.
CREATE TABLE IF NOT EXISTS model_registry (
    id              SERIAL      PRIMARY KEY,
    model_name      VARCHAR(100) NOT NULL,
    model_version   VARCHAR(50)  NOT NULL,
    stage           VARCHAR(20)  NOT NULL,  -- "Staging" | "Production" | "Archived"
    metrics         JSONB,
    registered_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    promoted_at     TIMESTAMPTZ,
    notes           TEXT,
    UNIQUE (model_name, model_version)
);

-- ── drift_reports ─────────────────────────────────────────────────────────────
-- Written by the Evidently drift monitor (Phase 4).
CREATE TABLE IF NOT EXISTS drift_reports (
    id                SERIAL      PRIMARY KEY,
    timestamp         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    drift_detected    BOOLEAN     NOT NULL,
    drift_score       FLOAT       NOT NULL,
    features_drifted  JSONB,      -- list of drifted feature names + scores
    report_path       VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS idx_drift_reports_timestamp ON drift_reports (timestamp DESC);
