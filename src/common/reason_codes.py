"""Canonical data-quality reason codes.

Every validation failure maps to exactly one of these codes, enabling auditable,
quantifiable quality reporting (vs. a vague "data is dirty").
"""
from __future__ import annotations

from enum import Enum


class ReasonCode(str, Enum):
    MISSING_VALUE = "MISSING_VALUE"
    DUPLICATE_RECORD = "DUPLICATE_RECORD"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    DOMAIN_VIOLATION = "DOMAIN_VIOLATION"
    STALE_DATA = "STALE_DATA"
    TIME_GAP = "TIME_GAP"
    PERSISTENT_CONSTANT = "PERSISTENT_CONSTANT"
    LOCAL_OUTLIER = "LOCAL_OUTLIER"
    GLOBAL_OUTLIER = "GLOBAL_OUTLIER"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    BROKEN_STATION_REF = "BROKEN_STATION_REF"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    NON_MONOTONIC = "NON_MONOTONIC"


# Codes that cause a row to be quarantined (hard failures) vs. flagged (soft).
QUARANTINE_CODES = frozenset(
    {
        ReasonCode.MISSING_VALUE,
        ReasonCode.DUPLICATE_RECORD,
        ReasonCode.INVALID_TIMESTAMP,
        ReasonCode.UNIT_MISMATCH,
        ReasonCode.DOMAIN_VIOLATION,
        ReasonCode.INVALID_PARAMETER,
        ReasonCode.BROKEN_STATION_REF,
        ReasonCode.SCHEMA_MISMATCH,
    }
)
