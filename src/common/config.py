"""Configuration loading.

Centralizes access to the YAML config files and environment variables so no
other module hard-codes paths, URLs, thresholds, or secrets. Config is loaded
once and cached; secrets come exclusively from the environment (.env).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Repository root = two levels up from this file (src/common/config.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class City:
    name: str
    country: str
    latitude: float
    longitude: float
    timezone: str
    search_radius_m: int


@dataclass(frozen=True)
class Config:
    """Immutable view over all configuration sources."""

    raw: dict[str, Any]
    cities_raw: dict[str, Any]
    quality_raw: dict[str, Any]
    openaq_api_key: str | None = field(default=None)

    # ----- paths -----
    def _path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def raw_dir(self) -> Path:
        return self._path("raw_dir")

    @property
    def processed_dir(self) -> Path:
        return self._path("processed_dir")

    @property
    def quarantine_dir(self) -> Path:
        return self._path("quarantine_dir")

    @property
    def db_path(self) -> Path:
        override = os.getenv("AQP_DB_PATH")
        if override:
            p = Path(override)
            return p if p.is_absolute() else REPO_ROOT / p
        return self._path("db_path")

    # ----- sections -----
    @property
    def ingestion(self) -> dict[str, Any]:
        return self.raw["ingestion"]

    @property
    def pollutants(self) -> dict[str, Any]:
        return self.raw["pollutants"]

    @property
    def weather(self) -> dict[str, Any]:
        return self.raw["weather"]

    @property
    def features(self) -> dict[str, Any]:
        return self.raw["features"]

    @property
    def statistics(self) -> dict[str, Any]:
        return self.raw["statistics"]

    @property
    def quality(self) -> dict[str, Any]:
        return self.quality_raw

    @property
    def log_level(self) -> str:
        return os.getenv("AQP_LOG_LEVEL", self.raw["logging"]["level"])

    @property
    def log_json(self) -> bool:
        return bool(self.raw["logging"].get("json", False))

    @property
    def cities(self) -> list[City]:
        return [
            City(
                name=c["name"],
                country=c["country"],
                latitude=float(c["latitude"]),
                longitude=float(c["longitude"]),
                timezone=c["timezone"],
                search_radius_m=int(c.get("search_radius_m", 25000)),
            )
            for c in self.cities_raw["cities"]
        ]

    @property
    def max_stations_per_city(self) -> int:
        return int(self.cities_raw.get("max_stations_per_city", 4))

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.quarantine_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load and cache configuration from disk + environment."""
    load_dotenv(REPO_ROOT / ".env")
    return Config(
        raw=_load_yaml(CONFIG_DIR / "config.yaml"),
        cities_raw=_load_yaml(CONFIG_DIR / "cities.yaml"),
        quality_raw=_load_yaml(CONFIG_DIR / "quality_rules.yaml"),
        openaq_api_key=os.getenv("OPENAQ_API_KEY"),
    )
