"""Compute and print ACTUAL metrics from the current database.

Prints a JSON summary of real, computed results (row counts, DQ scores, event
counts, example statistics). Used to populate the README/resume with measured
numbers instead of fabricated ones. Run AFTER the pipeline has populated the DB.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common.config import get_config
from src.common.db import read_sql
from src.statistics import correlation, hypothesis, regression

CFG = get_config()


def _count(table: str) -> int:
    return int(read_sql(CFG.db_path, f"SELECT COUNT(*) AS n FROM {table}")["n"].iloc[0])


def main() -> None:
    db = CFG.db_path
    try:
        srcs = read_sql(db, "SELECT DISTINCT source FROM stations")["source"].tolist()
    except Exception:
        srcs = []
    report: dict = {"data_sources": srcs or ["(none loaded yet)"]}

    report["row_counts"] = {
        t: _count(t)
        for t in ["stations", "air_measurements", "weather_hourly", "features_hourly",
                  "quality_metrics", "air_quarantine", "events", "pipeline_runs"]
    }

    q = read_sql(db, "SELECT metric_name, AVG(metric_value) AS mean FROM quality_metrics GROUP BY metric_name")
    report["data_quality_score"] = {r["metric_name"]: round(r["mean"], 2) for _, r in q.iterrows()}

    quar = read_sql(db, "SELECT reason_code, COUNT(*) AS n FROM air_quarantine GROUP BY reason_code ORDER BY n DESC")
    report["quarantine_by_reason"] = {r["reason_code"]: int(r["n"]) for _, r in quar.iterrows()}

    ev = read_sql(db, "SELECT severity, COUNT(*) AS n FROM events GROUP BY severity")
    report["events_by_severity"] = {r["severity"]: int(r["n"]) for _, r in ev.iterrows()}
    report["events_total"] = _count("events")

    # Example association: PM2.5 vs wind speed (pooled over joined data).
    wp = read_sql(db, "SELECT * FROM v_weather_pollution WHERE parameter='pm25'")
    if not wp.empty:
        sp = correlation.spearman(wp["pollutant_value"], wp["wind_speed"])
        pc = correlation.partial_correlation(
            wp.assign(hour=pd.to_datetime(wp["timestamp_utc"], utc=True).dt.hour)
              .rename(columns={"pollutant_value": "pm25"}),
            "pm25", "wind_speed", ["temperature", "humidity", "hour"],
        )
        report["pm25_vs_wind"] = {
            "spearman_rho": round(sp["rho"], 3), "spearman_p": float(f"{sp['p_value']:.3g}"),
            "partial_r_controlling_temp_hum_hour": round(pc["partial_r"], 3),
            "n": sp["n"],
        }
        reg = regression.fit_ols(
            wp.rename(columns={"pollutant_value": "pm25"}),
            "pm25", ["temperature", "humidity", "wind_speed", "precipitation"],
        )
        if "r_squared" in reg:
            report["pm25_regression"] = {
                "n": reg["n"], "r_squared": round(reg["r_squared"], 3),
                "adj_r_squared": round(reg["adj_r_squared"], 3),
                "durbin_watson": round(reg["durbin_watson"], 2),
            }

    # Example hypothesis test: do cities differ in PM2.5?
    pm = read_sql(db, """
        SELECT s.city, a.value FROM air_measurements a
        JOIN stations s ON s.station_id=a.station_id
        WHERE a.parameter='pm25' AND a.value IS NOT NULL
    """)
    if not pm.empty and pm["city"].nunique() >= 2:
        groups = {c: g["value"] for c, g in pm.groupby("city")}
        kw = hypothesis.kruskal_wallis(groups)
        report["pm25_city_kruskal"] = {
            "H": round(kw["H"], 2), "p_value": float(f"{kw['p_value']:.3g}"),
            "epsilon_squared": round(kw["epsilon_squared"], 4), "k_cities": kw["k"], "n": kw["n"],
        }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
