"""OpenAQ API v3 client.

Discovery flow (v3): locations -> sensors -> sensor measurements.
  * GET /v3/locations?coordinates=lat,lon&radius=... to find stations near a city.
  * Each location returns sensors[] with a parameter mapping.
  * GET /v3/sensors/{id}/measurements?datetime_from&datetime_to&page&limit for data.

Auth: X-API-Key header. v1/v2 are retired and intentionally not used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from src.common.logging_utils import get_logger
from src.ingestion.http_client import HttpClient

logger = get_logger(__name__)


@dataclass
class SensorRef:
    sensor_id: int
    parameter: str  # OpenAQ parameter name, e.g. 'pm25'
    unit: str


@dataclass
class LocationRef:
    location_id: int
    name: str
    latitude: float | None
    longitude: float | None
    timezone: str | None
    sensors: list[SensorRef]


class OpenAQClient:
    def __init__(self, api_key: str, base_url: str, http: HttpClient) -> None:
        if not api_key:
            raise ValueError(
                "OPENAQ_API_KEY is required for OpenAQ v3. Set it in .env "
                "(see .env.example) or run the pipeline in --synthetic mode."
            )
        self.base_url = base_url.rstrip("/")
        self.http = http
        # Attach the auth header to the shared session.
        self.http._session.headers.update({"X-API-Key": api_key})

    def find_locations(
        self, latitude: float, longitude: float, radius_m: int, limit: int = 100
    ) -> list[LocationRef]:
        """Find monitoring locations near a coordinate."""
        url = f"{self.base_url}/locations"
        params = {
            "coordinates": f"{latitude:.4f},{longitude:.4f}",
            "radius": min(radius_m, 25000),
            "limit": limit,
            "page": 1,
        }
        payload = self.http.get_json(url, params=params)
        locations: list[LocationRef] = []
        for item in payload.get("results", []):
            sensors = [
                SensorRef(
                    sensor_id=s["id"],
                    parameter=(s.get("parameter") or {}).get("name", "").lower(),
                    unit=(s.get("parameter") or {}).get("units", ""),
                )
                for s in item.get("sensors", [])
            ]
            coords = item.get("coordinates") or {}
            locations.append(
                LocationRef(
                    location_id=item["id"],
                    name=item.get("name") or f"location-{item['id']}",
                    latitude=coords.get("latitude"),
                    longitude=coords.get("longitude"),
                    timezone=(item.get("timezone") if isinstance(item.get("timezone"), str)
                              else (item.get("timezone") or {}).get("name")),
                    sensors=sensors,
                )
            )
        logger.info("Found %d OpenAQ locations near (%.3f, %.3f)", len(locations), latitude, longitude)
        return locations

    def iter_sensor_measurements(
        self,
        sensor_id: int,
        datetime_from: str,
        datetime_to: str,
        page_limit: int = 1000,
    ) -> Iterator[dict[str, Any]]:
        """Yield raw measurement records for a sensor, following pagination."""
        url = f"{self.base_url}/sensors/{sensor_id}/measurements"
        page = 1
        while True:
            params = {
                "datetime_from": datetime_from,
                "datetime_to": datetime_to,
                "limit": page_limit,
                "page": page,
            }
            payload = self.http.get_json(url, params=params)
            results = payload.get("results", [])
            if not results:
                break
            yield from results
            # Stop when the page is not full (last page) to avoid over-paging.
            if len(results) < page_limit:
                break
            page += 1
