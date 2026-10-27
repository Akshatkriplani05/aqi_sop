"""CLI entry point for the ETL pipeline.

Examples
--------
# Offline, reproducible demo on synthetic data (no API key needed):
python -m scripts.run_pipeline --synthetic --mode backfill \
    --date-from 2024-01-01T00:00:00Z --date-to 2024-01-14T23:00:00Z

# Real backfill from OpenAQ v3 + Open-Meteo (requires OPENAQ_API_KEY in .env):
python -m scripts.run_pipeline --mode backfill --date-from 2024-06-01 --date-to 2024-06-14

# Incremental load (from last loaded timestamp to now):
python -m scripts.run_pipeline --mode incremental
"""
from __future__ import annotations

import argparse
import json

from src.pipeline.run import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the air-quality ETL pipeline.")
    parser.add_argument("--mode", choices=["backfill", "incremental"], default="backfill")
    parser.add_argument("--date-from", default=None, help="ISO date/datetime (backfill start)")
    parser.add_argument("--date-to", default=None, help="ISO date/datetime (end)")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use labeled synthetic data instead of live APIs (offline demo).",
    )
    args = parser.parse_args()

    summary = run_pipeline(
        mode=args.mode,
        date_from=args.date_from,
        date_to=args.date_to,
        use_synthetic=args.synthetic,
    )
    # Print the run summary (minus the verbose traceback) for operators.
    printable = {k: v for k, v in summary.items() if k != "errors"}
    print(json.dumps(printable, indent=2))
    if summary["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
