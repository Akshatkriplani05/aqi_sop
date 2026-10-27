"""Air-quality ingestion orchestration (OpenAQ v3 -> raw Parquet)."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.common.config import Config
from src.common.io_utils import write_raw
from src.common.logging_utils import get_logger
from src.ingestion.http_client import HttpClient
from src.ingestion.openaq_client import OpenAQClient

logger = get_logger(__name__)


def ingest_air(cfg: Config, date_from: str, date_to: str, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest air measurements and station metadata for all configured cities.

    Returns (stations_df, measurements_df) in the canonical raw schema and writes
    them to the raw layer. Only configured pollutants are kept.
    """
    http = HttpClient(
        max_retries=cfg.ingestion["max_retries"],
        backoff_base=cfg.ingestion["backoff_base_seconds"],
        backoff_max=cfg.ingestion["backoff_max_seconds"],
        timeout=cfg.ingestion["request_timeout_seconds"],
        min_interval=cfg.ingestion["min_request_interval_seconds"],
    )
    client = OpenAQClient(cfg.openaq_api_key or "", cfg.ingestion["openaq_base_url"], http)
    wanted = set(cfg.pollutants.keys())
    ingest_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    station_rows: list[dict] = []
    measurement_rows: list[dict] = []

    for city in cfg.cities:
        locations = client.find_locations(city.latitude, city.longitude, city.search_radius_m)
        for loc in locations[: cfg.max_stations_per_city]:
            station_id = f"openaq-{loc.location_id}"
            station_rows.append(
                {
                    "station_id": station_id,
                    "station_name": loc.name,
                    "city": city.name,
                    "country": city.country,
                    "latitude": loc.latitude,
                    "longitude": loc.longitude,
                    "timezone": loc.timezone or city.timezone,
                    "source": "openaq_v3",
                }
            )
            for sensor in loc.sensors:
                if sensor.parameter not in wanted:
                    continue
                for rec in client.iter_sensor_measurements(
                    sensor.sensor_id, date_from, date_to, cfg.ingestion["page_limit"]
                ):
                    period = rec.get("period") or {}
                    ts = (period.get("datetimeFrom") or {}).get("utc") or rec.get("datetime", {}).get("utc")
                    measurement_rows.append(
                        {
                            "station_id": station_id,
                            "timestamp_utc": ts,
                            "parameter": sensor.parameter,
                            "value": rec.get("value"),
                            "unit": (rec.get("parameter") or {}).get("units", sensor.unit),
                            "source": "openaq_v3",
                            "ingestion_timestamp": ingest_ts,
                            "pipeline_run_id": run_id,
                        }
                    )

    http.close()
    stations_df = pd.DataFrame(station_rows).drop_duplicates(subset=["station_id"])
    measurements_df = pd.DataFrame(measurement_rows)

    if not stations_df.empty:
        write_raw(stations_df, cfg.raw_dir, "openaq", "stations", run_id)
    if not measurements_df.empty:
        write_raw(measurements_df, cfg.raw_dir, "openaq", "air_measurements", run_id)
    logger.info("Air ingestion: %d stations, %d measurements", len(stations_df), len(measurements_df))
    return stations_df, measurements_df
