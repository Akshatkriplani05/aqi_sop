"""Pipeline orchestrator.

Ties the layers together with production concerns: a unique pipeline_run_id for
lineage, idempotent UPSERT loads (reruns of the same date range do NOT create
duplicates), backfill vs. incremental modes, quarantine persistence, structured
logging, exception handling, and a per-run summary row.

Run modes
---------
backfill    : load an explicit [date_from, date_to] window.
incremental : load from the last loaded timestamp (per DB) to date_to/now.

Sources
-------
api         : real OpenAQ v3 + Open-Meteo ingestion (requires OPENAQ_API_KEY).
synthetic   : labeled synthetic raw data for runnable, offline demos/tests.
"""
from __future__ import annotations

import traceback
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.common.config import Config, get_config
from src.common.db import connect, init_db, read_sql, upsert_dataframe
from src.common.io_utils import write_quarantine
from src.common.logging_utils import get_logger
from src.ingestion.ingest_air import ingest_air
from src.ingestion.ingest_weather import ingest_weather
from src.quality.scoring import compute_quality_scores
from src.transform.clean import clean_air_measurements, clean_weather
from src.transform.features import compute_features, detect_events
from src.validation.validator import run_validation

logger = get_logger(__name__)


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _last_loaded_timestamp(cfg: Config) -> str | None:
    try:
        df = read_sql(cfg.db_path, "SELECT MAX(timestamp_utc) AS mx FROM air_measurements")
        return df["mx"].iloc[0] if not df.empty else None
    except Exception:
        return None


def _acquire_sources(
    cfg: Config, date_from: str, date_to: str, run_id: str, use_synthetic: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (stations_df, air_measurements_df, weather_df)."""
    if use_synthetic:
        # Imported lazily so the production path has no test-only dependency.
        from scripts.generate_synthetic import generate_sources

        return generate_sources(cfg, date_from, date_to, run_id)
    stations_df, air_df = ingest_air(cfg, date_from, date_to, run_id)
    weather_df = ingest_weather(cfg, date_from, date_to, run_id)
    return stations_df, air_df, weather_df


def run_pipeline(
    mode: str = "backfill",
    date_from: str | None = None,
    date_to: str | None = None,
    use_synthetic: bool = False,
    cfg: Config | None = None,
) -> dict:
    cfg = cfg or get_config()
    cfg.ensure_dirs()
    init_db(cfg.db_path)

    run_id = new_run_id()
    run_time = pd.Timestamp.now(tz="UTC")
    date_to = date_to or _now_iso()

    if mode == "incremental":
        last = _last_loaded_timestamp(cfg)
        date_from = last or (date_to[:10] + "T00:00:00Z")
    else:  # backfill
        date_from = date_from or (
            (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%dT00:00:00Z")
        )

    summary = {
        "pipeline_run_id": run_id,
        "mode": mode,
        "start_time": _now_iso(),
        "end_time": None,
        "date_from": date_from,
        "date_to": date_to,
        "rows_requested": 0,
        "rows_received": 0,
        "rows_valid": 0,
        "rows_quarantined": 0,
        "quality_score": None,
        "status": "running",
        "errors": None,
    }
    conn = connect(cfg.db_path)
    try:
        upsert_dataframe(conn, pd.DataFrame([summary]), "pipeline_runs", ["pipeline_run_id"])
        logger.info("=== Pipeline run %s (%s) %s -> %s ===", run_id, mode, date_from, date_to)

        stations_df, air_df, weather_df = _acquire_sources(
            cfg, date_from, date_to, run_id, use_synthetic
        )
        summary["rows_received"] = len(air_df)

        if not stations_df.empty:
            stations_df = stations_df.assign(first_seen=_now_iso(), last_seen=_now_iso())
            upsert_dataframe(
                conn, stations_df, "stations", ["station_id"],
                update_columns=["station_name", "city", "country", "latitude",
                                "longitude", "timezone", "source", "last_seen"],
            )

        known_ids = set(stations_df["station_id"]) if not stations_df.empty else set()
        vres = run_validation(air_df, known_ids, cfg, run_time)
        summary["rows_quarantined"] = len(vres.quarantined)

        if not vres.quarantined.empty:
            q = vres.quarantined.copy()
            q["quarantined_at"] = _now_iso()
            q["pipeline_run_id"] = run_id
            write_quarantine(q, cfg.quarantine_dir, run_id, "validation")
            cols = ["station_id", "timestamp_utc", "parameter", "value", "unit",
                    "source", "reason_code", "evidence", "pipeline_run_id", "quarantined_at"]
            for c in cols:
                if c not in q.columns:
                    q[c] = None
            upsert_dataframe(
                conn, q[cols], "air_quarantine",
                ["station_id", "timestamp_utc", "parameter", "reason_code"],
                update_columns=["value", "unit", "source", "evidence",
                                "pipeline_run_id", "quarantined_at"],
            )

        clean_air = clean_air_measurements(vres.valid, cfg, run_id)
        summary["rows_valid"] = len(clean_air)
        if not clean_air.empty:
            upsert_dataframe(
                conn, clean_air, "air_measurements",
                ["station_id", "timestamp_utc", "parameter"],
            )

        clean_w = clean_weather(weather_df, run_id)
        if not clean_w.empty:
            upsert_dataframe(conn, clean_w, "weather_hourly", ["city", "timestamp_utc"])

        quality = compute_quality_scores(clean_air, vres.quarantined, cfg, run_id, run_time)
        if not quality.empty:
            upsert_dataframe(
                conn, quality, "quality_metrics",
                ["station_id", "date", "metric_name", "rule_version"],
            )
            overall = quality[quality["metric_name"] == "overall"]["metric_value"]
            summary["quality_score"] = round(float(overall.mean()), 2) if len(overall) else None

        features = compute_features(clean_air, cfg, run_id)
        if not features.empty:
            upsert_dataframe(
                conn, features, "features_hourly",
                ["station_id", "timestamp_utc", "pollutant"],
            )

        events = detect_events(clean_air, cfg, run_id)
        if not events.empty:
            upsert_dataframe(conn, events, "events", ["event_id"])

        summary["status"] = "success"
        logger.info("=== Pipeline run %s SUCCESS ===", run_id)
    except Exception as exc:  # noqa: BLE001 - top-level guard records failure
        summary["status"] = "failed"
        summary["errors"] = f"{exc}\n{traceback.format_exc()}"
        logger.exception("Pipeline run %s FAILED", run_id)
    finally:
        summary["end_time"] = _now_iso()
        upsert_dataframe(
            conn, pd.DataFrame([summary]), "pipeline_runs", ["pipeline_run_id"],
            update_columns=["end_time", "date_from", "date_to", "rows_requested",
                            "rows_received", "rows_valid", "rows_quarantined",
                            "quality_score", "status", "errors"],
        )
        conn.close()
    return summary
