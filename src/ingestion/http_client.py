"""Resilient HTTP client.

Implements the networking concerns a production ingestion job needs: retries
with exponential backoff + jitter, explicit HTTP 429 rate-limit handling
(honoring Retry-After), a minimum inter-request interval, request timeouts, and
structured request logging. Kept dependency-light (requests + stdlib) so the
retry logic is visible and testable rather than hidden behind a library.
"""
from __future__ import annotations

import random
import time
from typing import Any

import requests

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


class HttpError(RuntimeError):
    """Raised when a request ultimately fails after exhausting retries."""


class HttpClient:
    def __init__(
        self,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
        timeout: float = 30.0,
        min_interval: float = 0.2,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.timeout = timeout
        self.min_interval = min_interval
        self._session = requests.Session()
        if default_headers:
            self._session.headers.update(default_headers)
        self._last_request_ts = 0.0

    def _throttle(self) -> None:
        """Enforce a minimum spacing between requests (politeness)."""
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _sleep_for_retry(self, attempt: int, retry_after: str | None) -> None:
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = self.backoff_base * (2 ** attempt)
        else:
            # Exponential backoff with full jitter.
            delay = min(self.backoff_max, self.backoff_base * (2 ** attempt))
            delay = random.uniform(0, delay)
        logger.warning("Retry attempt %d after %.2fs", attempt + 1, delay)
        time.sleep(delay)

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET a URL and return parsed JSON, retrying on transient failures."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            self._last_request_ts = time.monotonic()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
                logger.info("GET %s -> %s", resp.url, resp.status_code)

                if resp.status_code == 429:
                    self._sleep_for_retry(attempt, resp.headers.get("Retry-After"))
                    continue
                if resp.status_code >= 500:
                    self._sleep_for_retry(attempt, None)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                logger.warning("Transient network error: %s", exc)
                self._sleep_for_retry(attempt, None)
            except requests.HTTPError as exc:
                # 4xx other than 429 are not retryable.
                raise HttpError(f"Non-retryable HTTP error for {url}: {exc}") from exc

        raise HttpError(f"Exhausted {self.max_retries} retries for {url}") from last_exc

    def close(self) -> None:
        self._session.close()
