"""Open-Meteo historical weather client.

GET https://archive-api.open-meteo.com/v1/archive with latitude, longitude,
start_date, end_date, hourly=<csv>, timezone. No API key required. Returns a
columnar JSON object: hourly.time[] plus one parallel array per variable.
"""
from __future__ import annotations

import pandas as pd

from src.common.logging_utils import get_logger
from src.ingestion.http_client import HttpClient

logger = get_logger(__name__)

# Map Open-Meteo variable names -> our canonical weather column names.
_VAR_MAP = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "precipitation": "precipitation",
    "surface_pressure": "pressure",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
}


class OpenMeteoClient:
    def __init__(self, base_url: str, http: HttpClient) -> None:
        self.base_url = base_url
        self.http = http

    def fetch_hourly(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: list[str],
        timezone: str = "UTC",
    ) -> pd.DataFrame:
        """Return a tidy hourly weather DataFrame in UTC for one location."""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": timezone,
        }
        payload = self.http.get_json(self.base_url, params=params)
        hourly = payload.get("hourly") or {}
        times = hourly.get("time", [])
        if not times:
            logger.warning("No weather data returned for (%.3f, %.3f)", latitude, longitude)
            return pd.DataFrame()

        data = {"time": times}
        for var in variables:
            data[_VAR_MAP.get(var, var)] = hourly.get(var, [None] * len(times))
        df = pd.DataFrame(data)

        # Open-Meteo returns local time for the requested timezone; convert to UTC.
        ts = pd.to_datetime(df["time"])
        if timezone.upper() != "UTC":
            ts = ts.dt.tz_localize(timezone, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
        else:
            ts = ts.dt.tz_localize("UTC")
        df["timestamp_utc"] = ts.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df = df.drop(columns=["time"])
        logger.info("Fetched %d weather rows for (%.3f, %.3f)", len(df), latitude, longitude)
        return df
