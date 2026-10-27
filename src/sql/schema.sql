-- Relational schema for the analytical layer.
-- Grain and keys are documented per table. All DDL is idempotent.

-- One row per pipeline execution. Every fact row carries pipeline_run_id for lineage.
CREATE TABLE IF NOT EXISTS pipeline_runs (
    pipeline_run_id   TEXT PRIMARY KEY,
    mode              TEXT NOT NULL,            -- 'backfill' | 'incremental'
    start_time        TEXT NOT NULL,
    end_time          TEXT,
    date_from         TEXT,
    date_to           TEXT,
    rows_requested    INTEGER DEFAULT 0,
    rows_received     INTEGER DEFAULT 0,
    rows_valid        INTEGER DEFAULT 0,
    rows_quarantined  INTEGER DEFAULT 0,
    quality_score     REAL,
    status            TEXT DEFAULT 'running',   -- 'running' | 'success' | 'failed'
    errors            TEXT
);

-- Dimension: monitoring stations. PK station_id.
CREATE TABLE IF NOT EXISTS stations (
    station_id    TEXT PRIMARY KEY,
    station_name  TEXT,
    city          TEXT NOT NULL,
    country       TEXT,
    latitude      REAL,
    longitude     REAL,
    timezone      TEXT,
    source        TEXT NOT NULL,
    first_seen    TEXT,
    last_seen     TEXT
);

-- Fact: raw-but-clean pollutant observations.
-- Grain / natural key: (station_id, timestamp_utc, parameter) -> UNIQUE for idempotency.
CREATE TABLE IF NOT EXISTS air_measurements (
    station_id           TEXT NOT NULL,
    timestamp_utc        TEXT NOT NULL,
    parameter            TEXT NOT NULL,
    value                REAL,
    unit                 TEXT,
    source               TEXT,
    ingestion_timestamp  TEXT,
    pipeline_run_id      TEXT,
    PRIMARY KEY (station_id, timestamp_utc, parameter),
    FOREIGN KEY (station_id) REFERENCES stations(station_id)
);

-- Fact: hourly weather per city. Grain / natural key: (city, timestamp_utc).
CREATE TABLE IF NOT EXISTS weather_hourly (
    city             TEXT NOT NULL,
    timestamp_utc    TEXT NOT NULL,
    temperature      REAL,
    humidity         REAL,
    precipitation    REAL,
    pressure         REAL,
    wind_speed       REAL,
    wind_direction   REAL,
    source           TEXT,
    pipeline_run_id  TEXT,
    PRIMARY KEY (city, timestamp_utc)
);

-- Fact (tidy/long): data-quality metrics per station/day/metric.
CREATE TABLE IF NOT EXISTS quality_metrics (
    station_id       TEXT NOT NULL,
    date             TEXT NOT NULL,
    metric_name      TEXT NOT NULL,
    metric_value     REAL,
    threshold        REAL,
    status           TEXT,                 -- 'PASS' | 'WARN' | 'FAIL'
    rule_version     TEXT NOT NULL,
    pipeline_run_id  TEXT,                 -- lineage: last run that computed this metric
    -- run_id is intentionally NOT part of the key so that re-running the same
    -- window updates the metric in place (idempotent) rather than duplicating.
    PRIMARY KEY (station_id, date, metric_name, rule_version),
    FOREIGN KEY (station_id) REFERENCES stations(station_id)
);

-- Fact: engineered statistical features.
-- Grain / natural key: (station_id, timestamp_utc, pollutant).
CREATE TABLE IF NOT EXISTS features_hourly (
    station_id       TEXT NOT NULL,
    timestamp_utc    TEXT NOT NULL,
    pollutant        TEXT NOT NULL,
    value            REAL,
    rolling_mean     REAL,
    rolling_median   REAL,
    rolling_std      REAL,
    rolling_mad      REAL,
    rolling_iqr      REAL,
    rolling_cv       REAL,
    percentile_rank  REAL,
    lag_1            REAL,
    lag_3            REAL,
    lag_6            REAL,
    lag_12           REAL,
    lag_24           REAL,
    delta            REAL,
    pct_change       REAL,
    ewma             REAL,
    cusum_pos        REAL,
    cusum_neg        REAL,
    anomaly_flag     INTEGER DEFAULT 0,
    anomaly_method   TEXT,
    pipeline_run_id  TEXT,
    PRIMARY KEY (station_id, timestamp_utc, pollutant),
    FOREIGN KEY (station_id) REFERENCES stations(station_id)
);

-- Quarantine: rejected rows are preserved here (never deleted) with their
-- reason code + evidence, so every rejection is auditable and recoverable.
CREATE TABLE IF NOT EXISTS air_quarantine (
    station_id       TEXT,
    timestamp_utc    TEXT,
    parameter        TEXT,
    value            REAL,
    unit             TEXT,
    source           TEXT,
    reason_code      TEXT NOT NULL,
    evidence         TEXT,
    pipeline_run_id  TEXT,
    quarantined_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_quar_reason ON air_quarantine(reason_code);
CREATE INDEX IF NOT EXISTS idx_quar_run    ON air_quarantine(pipeline_run_id);
-- Unique key makes quarantine idempotent across reruns of the same window.
-- (SQLite treats NULLs as distinct, so rows with unparseable timestamps are kept.)
CREATE UNIQUE INDEX IF NOT EXISTS uq_quar_key
    ON air_quarantine(station_id, timestamp_utc, parameter, reason_code);

-- Fact: detected sustained events (not isolated spikes).
CREATE TABLE IF NOT EXISTS events (
    event_id         TEXT PRIMARY KEY,
    station_id       TEXT NOT NULL,
    pollutant        TEXT,
    start_time       TEXT NOT NULL,
    end_time         TEXT,
    event_type       TEXT,                 -- e.g. 'POSITIVE_SHIFT' | 'NEGATIVE_SHIFT'
    severity         TEXT,                 -- 'LOW' | 'MEDIUM' | 'HIGH'
    evidence         TEXT,                 -- JSON blob
    detection_method TEXT,                 -- 'CUSUM' | 'HAMPEL' | ...
    pipeline_run_id  TEXT,
    FOREIGN KEY (station_id) REFERENCES stations(station_id)
);

-- Indexes to keep the common access paths fast.
CREATE INDEX IF NOT EXISTS idx_air_station_time   ON air_measurements(station_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_air_param_time      ON air_measurements(parameter, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_weather_city_time   ON weather_hourly(city, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_quality_station_dt  ON quality_metrics(station_id, date);
CREATE INDEX IF NOT EXISTS idx_feat_station_time   ON features_hourly(station_id, timestamp_utc, pollutant);
CREATE INDEX IF NOT EXISTS idx_events_station      ON events(station_id, start_time);
