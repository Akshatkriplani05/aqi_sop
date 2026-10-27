"""Synthetic data generator (CLEARLY LABELED, source='synthetic').

Purpose: make the whole pipeline runnable and testable WITHOUT API keys or
network, and give the quality/anomaly/statistics machinery realistic structure
to work on. This is NOT real OpenAQ/Open-Meteo data and must never be presented
as such; the `source` column is set to 'synthetic' end to end.

The generator intentionally bakes in:
  * diurnal (24h) + weekly + seasonal cycles and AR(1) autocorrelation,
  * genuine weather coupling (low wind / high humidity -> higher PM; rain scrubs
    PM) so correlation/regression analyses have a real signal to find,
  * injected data-quality defects (missing, duplicate, domain violations,
    stuck sensor) and a sustained pollution shift (for CUSUM event detection).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.common.config import Config, get_config
from src.common.io_utils import write_raw
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

POLLUTANTS = ["pm25", "pm10", "no2", "o3"]
STATIONS_PER_CITY = 2

# Per-city baseline multipliers: real cities have genuinely different pollution
# regimes, so between-city tests (Kruskal-Wallis) have a real effect to detect.
# Values are illustrative for the synthetic demo, not measured.
CITY_FACTOR = {
    "Delhi": 2.6, "Los Angeles": 1.3, "Madrid": 1.0,
    "Paris": 1.1, "London": 1.0, "Berlin": 0.9,
}


def _hourly_index(date_from: str, date_to: str) -> pd.DatetimeIndex:
    start = pd.to_datetime(date_from, utc=True).floor("h")
    end = pd.to_datetime(date_to, utc=True).floor("h")
    return pd.date_range(start, end, freq="h")


def _weather_for_city(idx: pd.DatetimeIndex, rng: np.random.Generator, lat: float) -> pd.DataFrame:
    n = len(idx)
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    # Seasonal + diurnal temperature; colder at high latitude.
    season = 10 * np.cos(2 * np.pi * (doy - 200) / 365.0)
    base_temp = 15 - 0.3 * (abs(lat) - 40)
    temperature = base_temp + season + 6 * np.cos(2 * np.pi * (hour - 15) / 24.0)
    temperature += rng.normal(0, 1.5, n)
    humidity = np.clip(70 - 0.8 * (temperature - base_temp) + rng.normal(0, 8, n), 10, 100)
    wind_speed = np.clip(3 + 2 * np.sin(2 * np.pi * hour / 24.0) + rng.gamma(2, 1.0, n), 0.1, None)
    wind_direction = (180 + 60 * np.sin(2 * np.pi * doy / 365.0) + rng.normal(0, 40, n)) % 360
    precipitation = np.where(rng.random(n) < 0.08, rng.gamma(1.5, 1.2, n), 0.0)
    pressure = 1013 + rng.normal(0, 6, n) + 3 * np.sin(2 * np.pi * doy / 365.0)
    return pd.DataFrame(
        {
            "timestamp_utc": idx.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "temperature": np.round(temperature, 2),
            "humidity": np.round(humidity, 1),
            "precipitation": np.round(precipitation, 2),
            "pressure": np.round(pressure, 1),
            "wind_speed": np.round(wind_speed, 2),
            "wind_direction": np.round(wind_direction, 1),
        }
    )


def _pollutant_series(
    idx: pd.DatetimeIndex,
    weather: pd.DataFrame,
    pollutant: str,
    rng: np.random.Generator,
    city_factor: float = 1.0,
) -> np.ndarray:
    n = len(idx)
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()

    base = {"pm25": 18.0, "pm10": 30.0, "no2": 25.0, "o3": 40.0}[pollutant] * city_factor
    # Diurnal: traffic peaks (NO2/PM) in morning+evening; O3 peaks midday.
    if pollutant == "o3":
        diurnal = 15 * np.clip(np.cos(2 * np.pi * (hour - 14) / 24.0), 0, None)
    else:
        diurnal = 10 * np.exp(-((hour - 8) ** 2) / 6) + 12 * np.exp(-((hour - 19) ** 2) / 6)
    weekly = np.where(dow >= 5, -4.0, 2.0)  # lower on weekends (less traffic)

    wind = weather["wind_speed"].to_numpy()
    hum = weather["humidity"].to_numpy()
    rain = weather["precipitation"].to_numpy()
    # Weather coupling (the signal correlation/regression should recover).
    if pollutant in ("pm25", "pm10"):
        weather_effect = -2.5 * (wind - wind.mean()) + 0.15 * (hum - hum.mean()) - 3.0 * rain
    elif pollutant == "no2":
        weather_effect = -2.0 * (wind - wind.mean()) + 0.05 * (hum - hum.mean())
    else:  # o3 forms in sunlight/heat, lower with high humidity
        weather_effect = 0.4 * (weather["temperature"].to_numpy() - weather["temperature"].mean())

    # AR(1) autocorrelated noise -> series is NOT i.i.d.
    noise = np.zeros(n)
    phi, sigma = 0.6, base * 0.12
    for t in range(1, n):
        noise[t] = phi * noise[t - 1] + rng.normal(0, sigma)

    series = base + diurnal + weekly + weather_effect + noise
    return np.clip(series, 0.1, None)


def _inject_defects(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject labeled-by-nature data-quality defects + a sustained shift."""
    out = df.copy().reset_index(drop=True)
    n = len(out)
    if n < 50:
        return out

    # 1.5% missing values.
    miss_idx = rng.choice(n, size=max(1, int(0.015 * n)), replace=False)
    out.loc[miss_idx, "value"] = np.nan

    # A few domain violations (impossible negatives / absurd highs).
    dv_idx = rng.choice(n, size=max(1, int(0.004 * n)), replace=False)
    out.loc[dv_idx, "value"] = rng.choice([-5.0, 99999.0], size=len(dv_idx))

    # Stuck sensor: 8 consecutive identical values somewhere mid-series.
    start = rng.integers(n // 4, n // 2)
    out.loc[start:start + 7, "value"] = float(out.loc[start, "value"] or 20.0)

    # Sustained upward shift (pollution episode) for ~12 hours -> CUSUM should fire.
    ev_start = rng.integers(n // 2, int(0.8 * n))
    out.loc[ev_start:ev_start + 12, "value"] = out.loc[ev_start:ev_start + 12, "value"] + 60.0

    # Duplicate a handful of rows (same natural key).
    dup = out.iloc[rng.choice(n, size=max(1, int(0.003 * n)), replace=False)].copy()
    out = pd.concat([out, dup], ignore_index=True)
    return out


def generate_sources(
    cfg: Config, date_from: str, date_to: str, run_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (stations_df, air_measurements_df, weather_df) in the raw schema."""
    rng = np.random.default_rng(42)
    idx = _hourly_index(date_from, date_to)
    ingest_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    station_rows, air_frames, weather_frames = [], [], []

    for city in cfg.cities:
        weather = _weather_for_city(idx, rng, city.latitude)
        weather["city"] = city.name
        weather["source"] = "synthetic"
        weather["pipeline_run_id"] = run_id
        weather_frames.append(weather)

        for s in range(STATIONS_PER_CITY):
            station_id = f"synthetic-{city.name.lower().replace(' ', '')}-{s+1}"
            station_rows.append(
                {
                    "station_id": station_id,
                    "station_name": f"{city.name} Monitor {s+1}",
                    "city": city.name,
                    "country": city.country,
                    "latitude": city.latitude + rng.normal(0, 0.02),
                    "longitude": city.longitude + rng.normal(0, 0.02),
                    "timezone": city.timezone,
                    "source": "synthetic",
                }
            )
            city_factor = CITY_FACTOR.get(city.name, 1.0)
            for pollutant in POLLUTANTS:
                values = _pollutant_series(idx, weather, pollutant, rng, city_factor)
                frame = pd.DataFrame(
                    {
                        "station_id": station_id,
                        "timestamp_utc": idx.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "parameter": pollutant,
                        "value": np.round(values, 2),
                        "unit": "ug/m3",
                        "source": "synthetic",
                        "ingestion_timestamp": ingest_ts,
                        "pipeline_run_id": run_id,
                    }
                )
                frame = _inject_defects(frame, rng)
                air_frames.append(frame)

    stations_df = pd.DataFrame(station_rows)
    air_df = pd.concat(air_frames, ignore_index=True)
    weather_df = pd.concat(weather_frames, ignore_index=True)

    # Persist to the raw layer too, so the raw storage layer is exercised.
    write_raw(stations_df, cfg.raw_dir, "synthetic", "stations", run_id)
    write_raw(air_df, cfg.raw_dir, "synthetic", "air_measurements", run_id)
    write_raw(weather_df, cfg.raw_dir, "synthetic", "weather_hourly", run_id)

    logger.info(
        "Generated SYNTHETIC data: %d stations, %d air rows, %d weather rows",
        len(stations_df), len(air_df), len(weather_df),
    )
    return stations_df, air_df, weather_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic raw data.")
    parser.add_argument("--date-from", default="2024-01-01T00:00:00Z")
    parser.add_argument("--date-to", default="2024-01-14T23:00:00Z")
    args = parser.parse_args()
    generate_sources(get_config(), args.date_from, args.date_to, "manual-synthetic")
