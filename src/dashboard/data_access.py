"""Cached data-access helpers for the dashboard.

Reads from the SQLite analytical layer (and its SQL views) so the dashboard
queries the same stable interface as any other consumer. Streamlit caching keeps
interactions responsive while controls recompute statistics in-memory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ensure the repo root is importable when launched via `streamlit run`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from src.common.config import get_config  # noqa: E402
from src.common.db import read_sql  # noqa: E402
from src.sql.queries import run_named_query  # noqa: E402

CFG = get_config()


@st.cache_data(ttl=300)
def load_measurements() -> pd.DataFrame:
    df = read_sql(
        CFG.db_path,
        """
        SELECT a.station_id, s.city, s.station_name, a.timestamp_utc,
               a.parameter, a.value
        FROM air_measurements a
        JOIN stations s ON s.station_id = a.station_id
        WHERE a.value IS NOT NULL
        """,
    )
    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


@st.cache_data(ttl=300)
def load_weather() -> pd.DataFrame:
    df = read_sql(CFG.db_path, "SELECT * FROM weather_hourly")
    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


@st.cache_data(ttl=300)
def load_quality() -> pd.DataFrame:
    return read_sql(CFG.db_path, "SELECT * FROM quality_metrics")


@st.cache_data(ttl=300)
def load_events() -> pd.DataFrame:
    return read_sql(CFG.db_path, "SELECT * FROM v_pollution_episodes")


@st.cache_data(ttl=300)
def load_stations() -> pd.DataFrame:
    return read_sql(CFG.db_path, "SELECT * FROM stations")


@st.cache_data(ttl=300)
def load_runs() -> pd.DataFrame:
    return read_sql(CFG.db_path, "SELECT * FROM pipeline_runs ORDER BY start_time DESC")


@st.cache_data(ttl=300)
def named_query(name: str) -> pd.DataFrame:
    return run_named_query(CFG.db_path, name)


@st.cache_data(ttl=300)
def weather_pollution_joined() -> pd.DataFrame:
    df = read_sql(CFG.db_path, "SELECT * FROM v_weather_pollution")
    if not df.empty:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df
