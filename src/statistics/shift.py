"""Process-shift detection (sustained changes, not isolated spikes).

Spikes and sustained shifts are different phenomena. The Hampel filter catches
spikes; EWMA and CUSUM are designed to detect small, *persistent* shifts of the
process mean. We require a run of consecutive alarms before opening an event, so
a single outlier is never promoted to a "pollution episode".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def ewma(x: pd.Series, lam: float = 0.2) -> pd.Series:
    """Exponentially weighted moving average: S_t = lam*x_t + (1-lam)*S_{t-1}.

    Small lam -> smoother, slower to react (more weight on history).
    Large lam -> reacts faster to recent values (more weight on x_t).
    """
    return x.ewm(alpha=lam, adjust=False).mean()


def ewma_control_limits(
    x: pd.Series, lam: float = 0.2, L: float = 3.0
) -> pd.DataFrame:
    """EWMA chart with time-varying control limits.

    sigma_ewma,t = sigma * sqrt( lam/(2-lam) * (1 - (1-lam)^{2t}) ).
    Points outside [LCL, UCL] indicate the mean has likely shifted.
    """
    mu = np.nanmean(x)
    sigma = np.nanstd(x, ddof=1)
    s = ewma(x, lam)
    t = np.arange(1, len(x) + 1)
    var_factor = (lam / (2 - lam)) * (1 - (1 - lam) ** (2 * t))
    half = L * sigma * np.sqrt(var_factor)
    return pd.DataFrame(
        {"ewma": s.values, "center": mu, "ucl": mu + half, "lcl": mu - half},
        index=x.index,
    )


def cusum(
    x: pd.Series, k_sigma: float = 0.5, h_sigma: float = 5.0
) -> pd.DataFrame:
    """Tabular CUSUM on the standardized series.

    k (allowance/slack) and h (decision threshold) are expressed in std units.
      z_t         = (x_t - mu) / sigma
      S_pos_t     = max(0, S_pos_{t-1} + z_t - k)
      S_neg_t     = max(0, S_neg_{t-1} - z_t - k)
    An alarm fires when S_pos > h (upward shift) or S_neg > h (downward shift).
    """
    mu = np.nanmean(x)
    sigma = np.nanstd(x, ddof=1)
    if sigma == 0 or np.isnan(sigma):
        sigma = 1e-9
    z = (x - mu) / sigma
    s_pos = np.zeros(len(x))
    s_neg = np.zeros(len(x))
    zv = z.fillna(0).to_numpy()
    for i in range(1, len(x)):
        s_pos[i] = max(0.0, s_pos[i - 1] + zv[i] - k_sigma)
        s_neg[i] = max(0.0, s_neg[i - 1] - zv[i] - k_sigma)
    return pd.DataFrame(
        {
            "cusum_pos": s_pos,
            "cusum_neg": s_neg,
            "alarm_pos": s_pos > h_sigma,
            "alarm_neg": s_neg > h_sigma,
        },
        index=x.index,
    )


@dataclass
class Event:
    station_id: str
    pollutant: str
    start_time: str
    end_time: str
    event_type: str  # POSITIVE_SHIFT | NEGATIVE_SHIFT
    severity: str
    detection_method: str
    evidence: dict = field(default_factory=dict)


def _severity(peak_cusum: float, h: float) -> str:
    ratio = peak_cusum / h if h else 0
    if ratio >= 2.5:
        return "HIGH"
    if ratio >= 1.5:
        return "MEDIUM"
    return "LOW"


def extract_events(
    timestamps: pd.Series,
    values: pd.Series,
    cusum_df: pd.DataFrame,
    station_id: str,
    pollutant: str,
    min_persistence: int = 3,
    h_sigma: float = 5.0,
) -> list[Event]:
    """Convert CUSUM alarms into events, requiring `min_persistence` consecutive
    alarm hours before an event is opened. Produces auditable evidence.
    """
    events: list[Event] = []
    ts = pd.to_datetime(timestamps, utc=True).reset_index(drop=True)
    vals = values.reset_index(drop=True)

    for sign, alarm_col, cusum_col, etype in [
        (1, "alarm_pos", "cusum_pos", "POSITIVE_SHIFT"),
        (-1, "alarm_neg", "cusum_neg", "NEGATIVE_SHIFT"),
    ]:
        alarm = cusum_df[alarm_col].reset_index(drop=True).to_numpy()
        run_start = None
        for i in range(len(alarm)):
            if alarm[i] and run_start is None:
                run_start = i
            elif not alarm[i] and run_start is not None:
                if i - run_start >= min_persistence:
                    seg = cusum_df[cusum_col].reset_index(drop=True).iloc[run_start:i]
                    events.append(
                        Event(
                            station_id=station_id,
                            pollutant=pollutant,
                            start_time=ts.iloc[run_start].strftime("%Y-%m-%dT%H:%M:%SZ"),
                            end_time=ts.iloc[i - 1].strftime("%Y-%m-%dT%H:%M:%SZ"),
                            event_type=etype,
                            severity=_severity(float(seg.max()), h_sigma),
                            detection_method="CUSUM",
                            evidence={
                                "persistence_hours": int(i - run_start),
                                "peak_cusum": round(float(seg.max()), 3),
                                "mean_value": round(float(vals.iloc[run_start:i].mean()), 3),
                                "h_sigma": h_sigma,
                            },
                        )
                    )
                run_start = None
        # Handle an alarm run that extends to the end of the series.
        if run_start is not None and len(alarm) - run_start >= min_persistence:
            seg = cusum_df[cusum_col].reset_index(drop=True).iloc[run_start:]
            events.append(
                Event(
                    station_id=station_id,
                    pollutant=pollutant,
                    start_time=ts.iloc[run_start].strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end_time=ts.iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
                    event_type=etype,
                    severity=_severity(float(seg.max()), h_sigma),
                    detection_method="CUSUM",
                    evidence={
                        "persistence_hours": int(len(alarm) - run_start),
                        "peak_cusum": round(float(seg.max()), 3),
                        "mean_value": round(float(vals.iloc[run_start:].mean()), 3),
                        "h_sigma": h_sigma,
                    },
                )
            )
    return events


def events_to_frame(events: list[Event], run_id: str) -> pd.DataFrame:
    rows = []
    for i, e in enumerate(events):
        rows.append(
            {
                "event_id": f"{e.station_id}-{e.pollutant}-{e.start_time}-{e.event_type}",
                "station_id": e.station_id,
                "pollutant": e.pollutant,
                "start_time": e.start_time,
                "end_time": e.end_time,
                "event_type": e.event_type,
                "severity": e.severity,
                "evidence": json.dumps(e.evidence),
                "detection_method": e.detection_method,
                "pipeline_run_id": run_id,
            }
        )
    return pd.DataFrame(rows)
