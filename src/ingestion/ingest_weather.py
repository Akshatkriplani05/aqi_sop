"""Weather ingestion orchestration (Open-Meteo archive -> raw Parquet)."""
from __future__ import annotations

import pandas as pd

from src.common.config import Config
from src.common.io_utils import write_raw
from src.common.logging_utils import get_logger
from src.ingestion.http_client import HttpClient
from src.ingestion.open_meteo_client import OpenMeteoClient

logger = get_logger(__name__)


def ingest_weather(cfg: Config, date_from: str, date_to: str, run_id: str) -> pd.DataFrame:
    """Ingest hourly weather per city and write to the raw layer.

    date_from/date_to may be full ISO timestamps; Open-Meteo wants calendar dates.
    """
    http = HttpClient(
        max_retries=cfg.ingestion["max_retries"],
        backoff_base=cfg.ingestion["backoff_base_seconds"],
        backoff_max=cfg.ingestion["backoff_max_seconds"],
        timeout=cfg.ingestion["request_timeout_seconds"],
        min_interval=cfg.ingestion["min_request_interval_seconds"],
    )
    client = OpenMeteoClient(cfg.ingestion["open_meteo_base_url"], http)
    start_date = date_from[:10]
    end_date = date_to[:10]
    variables = cfg.weather["hourly_variables"]

    frames: list[pd.DataFrame] = []
    for city in cfg.cities:
        df = client.fetch_hourly(
            city.latitude, city.longitude, start_date, end_date, variables, timezone="UTC"
        )
        if df.empty:
            continue
        df["city"] = city.name
        df["source"] = "open_meteo"
        df["pipeline_run_id"] = run_id
        frames.append(df)

    http.close()
    weather_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not weather_df.empty:
        write_raw(weather_df, cfg.raw_dir, "open_meteo", "weather_hourly", run_id)
    logger.info("Weather ingestion: %d rows across %d cities", len(weather_df), len(frames))
    return weather_df
