"""Raw/quarantine file IO helpers.

Raw data is written as partitioned Parquet so the pipeline is restartable and
the raw layer is immutable and inspectable. Partitioning by run id keeps each
ingestion reproducible and auditable.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def write_raw(df: pd.DataFrame, base_dir: Path, source: str, dataset: str, run_id: str) -> Path:
    """Write a raw DataFrame to data/raw/<source>/<dataset>/run=<run_id>/data.parquet.

    Returns the path written. Append-only: a new run id yields a new partition.
    """
    out_dir = base_dir / source / dataset / f"run={run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d raw rows to %s", len(df), out_path)
    return out_path


def read_raw(base_dir: Path, source: str, dataset: str, run_id: str | None = None) -> pd.DataFrame:
    """Read raw partitions for a dataset. If run_id is None, read all runs."""
    root = base_dir / source / dataset
    if not root.exists():
        return pd.DataFrame()
    pattern = f"run={run_id}/data.parquet" if run_id else "run=*/data.parquet"
    frames = [pd.read_parquet(p) for p in sorted(root.glob(pattern))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_quarantine(df: pd.DataFrame, quarantine_dir: Path, run_id: str, stage: str) -> Path:
    """Persist quarantined (rejected) rows with their reason codes. Never deleted."""
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    out_path = quarantine_dir / f"quarantine_{stage}_{run_id}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Quarantined %d rows (%s) -> %s", len(df), stage, out_path)
    return out_path
