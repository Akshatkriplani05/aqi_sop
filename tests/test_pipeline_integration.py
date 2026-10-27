"""Integration tests: idempotency and SQL layer, via the synthetic pipeline."""
from __future__ import annotations

import pandas as pd

from src.common.db import read_sql
from src.pipeline.run import run_pipeline
from src.sql.queries import NAMED_QUERIES, run_named_query

DATE_FROM = "2024-01-01T00:00:00Z"
DATE_TO = "2024-01-03T23:00:00Z"


def _count(db_path, table) -> int:
    return int(read_sql(db_path, f"SELECT COUNT(*) AS n FROM {table}")["n"].iloc[0])


def test_pipeline_runs_and_is_idempotent(tmp_cfg):
    s1 = run_pipeline("backfill", DATE_FROM, DATE_TO, use_synthetic=True, cfg=tmp_cfg)
    assert s1["status"] == "success"
    assert s1["rows_valid"] > 0

    counts_after_first = {
        t: _count(tmp_cfg.db_path, t)
        for t in ["air_measurements", "weather_hourly", "features_hourly",
                  "quality_metrics", "air_quarantine", "events"]
    }

    # Rerun the SAME window: natural-key UPSERTs must not create duplicates.
    s2 = run_pipeline("backfill", DATE_FROM, DATE_TO, use_synthetic=True, cfg=tmp_cfg)
    assert s2["status"] == "success"

    counts_after_second = {t: _count(tmp_cfg.db_path, t) for t in counts_after_first}
    assert counts_after_first == counts_after_second


def test_named_views_queryable(tmp_cfg):
    run_pipeline("backfill", DATE_FROM, DATE_TO, use_synthetic=True, cfg=tmp_cfg)
    for name in NAMED_QUERIES:
        df = run_named_query(tmp_cfg.db_path, name)
        assert isinstance(df, pd.DataFrame)  # view exists and executes


def test_incremental_mode_runs(tmp_cfg):
    run_pipeline("backfill", DATE_FROM, DATE_TO, use_synthetic=True, cfg=tmp_cfg)
    s = run_pipeline("incremental", None, DATE_TO, use_synthetic=True, cfg=tmp_cfg)
    assert s["status"] == "success"
