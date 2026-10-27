"""SQLite access layer.

A thin wrapper around sqlite3 that: enables foreign keys, applies the schema,
and provides idempotent UPSERT helpers keyed on each table's natural key. Using
sqlite keeps the project zero-setup while still exercising real SQL, views, and
indexes. See README for the Postgres/partitioning migration path at scale.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.common.logging_utils import get_logger

logger = get_logger(__name__)

# SQLite identifiers we are willing to interpolate. Table/column names in this
# codebase are developer-controlled (never user input), but we still validate
# them against this pattern and double-quote them so no identifier can inject
# SQL. Values are always passed as bound '?' parameters, never interpolated.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(name: str) -> str:
    """Validate and quote a SQL identifier (table or column name)."""
    if not _IDENT_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return f'"{name}"'

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
VIEWS_PATH = Path(__file__).resolve().parents[1] / "sql" / "views.sql"


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    """Create tables, indexes, and views. Idempotent (uses IF NOT EXISTS)."""
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.executescript(VIEWS_PATH.read_text(encoding="utf-8"))
        conn.commit()
        logger.info("Initialized database schema + views at %s", db_path)
    finally:
        conn.close()


def upsert_dataframe(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    table: str,
    conflict_keys: Sequence[str],
    update_columns: Iterable[str] | None = None,
) -> int:
    """Idempotent bulk UPSERT via INSERT ... ON CONFLICT(keys) DO UPDATE.

    This is what makes reruns safe: inserting the same natural key again updates
    in place instead of creating duplicates. Returns number of rows submitted.
    """
    if df.empty:
        return 0
    cols = list(df.columns)
    table_q = _ident(table)
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join(_ident(c) for c in cols)
    conflict = ", ".join(_ident(c) for c in conflict_keys)

    if update_columns is None:
        update_columns = [c for c in cols if c not in conflict_keys]
    set_clause = ", ".join(f"{_ident(c)}=excluded.{_ident(c)}" for c in update_columns)

    if set_clause:
        sql = (
            f"INSERT INTO {table_q} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT({conflict}) DO UPDATE SET {set_clause}"
        )
    else:
        sql = (
            f"INSERT INTO {table_q} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT({conflict}) DO NOTHING"
        )

    # Convert NaN -> None so SQLite stores proper NULLs.
    records = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False, name=None)
    ]
    conn.executemany(sql, records)
    conn.commit()
    logger.info("Upserted %d rows into %s", len(records), table)
    return len(records)


def read_sql(db_path: Path, query: str, params: Sequence | None = None) -> pd.DataFrame:
    conn = connect(db_path)
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
