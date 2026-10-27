"""Named, reusable SQL queries exposed to the dashboard and analysis code.

Keeping query text here (rather than scattered f-strings) gives a single,
reviewable SQL surface and makes the views the stable analytical interface.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.db import read_sql

NAMED_QUERIES: dict[str, str] = {
    "monthly_trend": "SELECT * FROM v_monthly_pollutant_trend ORDER BY city, parameter, month",
    "station_completeness": "SELECT * FROM v_station_completeness ORDER BY completeness_pct ASC",
    "quality_ranking": "SELECT * FROM v_station_quality_ranking",
    "latest_observation": "SELECT * FROM v_latest_observation",
    "episodes": "SELECT * FROM v_pollution_episodes",
    "distribution": "SELECT * FROM v_pollutant_distribution",
    "weather_pollution": "SELECT * FROM v_weather_pollution",
    "freshness": "SELECT * FROM v_data_freshness ORDER BY hours_since_last DESC",
    "top_anomaly_periods": "SELECT * FROM v_top_anomaly_periods LIMIT 100",
    "city_comparison": "SELECT * FROM v_city_comparison ORDER BY city, parameter",
}


def run_named_query(db_path: Path, name: str) -> pd.DataFrame:
    if name not in NAMED_QUERIES:
        raise KeyError(f"Unknown query '{name}'. Available: {sorted(NAMED_QUERIES)}")
    return read_sql(db_path, NAMED_QUERIES[name])
