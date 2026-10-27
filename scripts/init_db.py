"""Initialize (or migrate) the SQLite schema and views. Idempotent."""
from __future__ import annotations

from src.common.config import get_config
from src.common.db import init_db


def main() -> None:
    cfg = get_config()
    cfg.ensure_dirs()
    init_db(cfg.db_path)
    print(f"Initialized database at {cfg.db_path}")


if __name__ == "__main__":
    main()
