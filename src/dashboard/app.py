"""Urban Air Quality Analytics - interactive dashboard.

Seven pages. Controls recompute real statistics in-memory (not relabeled charts):
changing the rolling window, EWMA lambda, CUSUM threshold, anomaly method, lag,
or confidence level re-runs the underlying math and updates flags/events/bands.

Launch:  streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard import data_access as da
from src.statistics import correlation, robust, shift, timeseries, uncertainty
from src.statistics import hypothesis, regression
from src.transform.features import add_temporal_features, add_wind_components

st.set_page_config(page_title="Urban Air Quality Analytics", layout="wide")

WEATHER_COLS = ["temperature", "humidity", "precipitation", "pressure", "wind_speed", "wind_direction"]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _series_for(meas: pd.DataFrame, station: str, pollutant: str) -> pd.DataFrame:
    sub = meas[(meas["station_id"] == station) & (meas["parameter"] == pollutant)]
    return sub.sort_values("timestamp_utc").reset_index(drop=True)


def _anomaly_flags(values: pd.Series, method: str, cfg: dict) -> pd.Series:
    """Dispatch to the chosen detector so switching methods changes the flags."""
    if method == "IQR":
        return robust.iqr_flags(values, cfg["iqr_k"])
    if method == "IQR (extreme 3x)":
        return robust.iqr_flags(values, cfg["iqr_k_extreme"])
    if method == "MAD":
        return robust.mad_flags(values, cfg["mad_threshold"])
    if method == "Hampel":
        flags, _ = robust.hampel_filter(values, cfg["hampel_window"], cfg["hampel_n_sigma"])
        return flags
    return pd.Series(False, index=values.index)


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview(meas, stations, runs, quality):
    st.header("Overview")
    if meas.empty:
        st.warning("No data loaded. Run the pipeline first (see README).")
        return
    overall = quality[quality["metric_name"] == "overall"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stations", stations["station_id"].nunique())
    c2.metric("Observations", f"{len(meas):,}")
    c3.metric("Date span", f"{(meas['timestamp_utc'].max() - meas['timestamp_utc'].min()).days} d")
    c4.metric("Mean DQ Score", f"{overall['metric_value'].mean():.1f}" if not overall.empty else "n/a")

    if runs is not None and not runs.empty:
        src = runs.iloc[0]
        st.caption(
            f"Latest run `{src['pipeline_run_id']}` ({src['mode']}): "
            f"received={src['rows_received']}, valid={src['rows_valid']}, "
            f"quarantined={src['rows_quarantined']}, status={src['status']}"
        )

    st.subheader("City comparison (mean concentration)")
    comp = da.named_query("city_comparison")
    if not comp.empty:
        fig = px.bar(comp, x="city", y="mean_value", color="parameter", barmode="group",
                     labels={"mean_value": "mean (ug/m3)"})
        st.plotly_chart(fig, use_container_width=True)


def page_trends(meas):
    st.header("Trends")
    station = st.session_state["station"]
    pollutant = st.session_state["pollutant"]
    window = st.sidebar.slider("Rolling window (hours)", 3, 72, 24, key="trend_window")
    lam = st.sidebar.slider("EWMA lambda", 0.05, 0.6, 0.2, 0.05, key="trend_lambda")
    agg = st.sidebar.selectbox("Aggregation", ["hourly", "daily"], key="trend_agg")

    sub = _series_for(meas, station, pollutant)
    if sub.empty:
        st.warning("No data for this selection.")
        return

    s = sub.set_index("timestamp_utc")["value"]
    if agg == "daily":
        s = s.resample("D").mean()

    roll = s.rolling(window, min_periods=max(2, window // 3)).mean()
    ewma = shift.ewma(s, lam)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name="observed", mode="lines",
                             line=dict(color="#9aa0a6", width=1)))
    fig.add_trace(go.Scatter(x=roll.index, y=roll.values, name=f"rolling mean ({window}h)",
                             line=dict(color="#1a73e8", width=2)))
    fig.add_trace(go.Scatter(x=ewma.index, y=ewma.values, name=f"EWMA (lambda={lam})",
                             line=dict(color="#ea4335", width=2)))
    fig.update_layout(title=f"{pollutant.upper()} at {station}", xaxis_title="time",
                      yaxis_title="ug/m3")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Smaller lambda = smoother/slower EWMA; larger lambda reacts faster to recent values.")

    with st.expander("STL decomposition (observed / trend / seasonal / residual)"):
        stl = timeseries.stl_decompose(s.interpolate(), period=24 if agg == "hourly" else 7)
        if "trend" in stl.columns:
            for col in ["observed", "trend", "seasonal", "residual"]:
                st.plotly_chart(
                    px.line(stl.reset_index(), x=stl.index.name or "index", y=col, title=col),
                    use_container_width=True,
                )
        else:
            st.info("Not enough regularly-spaced data for STL.")

    with st.expander("Autocorrelation (ACF / PACF) - why the series is not i.i.d."):
        ap = timeseries.acf_pacf(s, nlags=48)
        if not ap.empty:
            st.plotly_chart(px.bar(ap, x="lag", y="acf", title="ACF"), use_container_width=True)
            st.plotly_chart(px.bar(ap, x="lag", y="pacf", title="PACF"), use_container_width=True)


def page_distributions(meas):
    st.header("Distributions")
    pollutant = st.session_state["pollutant"]
    conf = st.sidebar.select_slider("Confidence level", [0.90, 0.95, 0.99], 0.95, key="dist_conf")
    sub = meas[meas["parameter"] == pollutant]
    if sub.empty:
        st.warning("No data for this pollutant.")
        return

    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(sub, x="value", color="city", nbins=60, marginal="box",
                                 title=f"{pollutant.upper()} distribution by city"),
                    use_container_width=True)
    c2.plotly_chart(px.box(sub, x="city", y="value", title=f"{pollutant.upper()} by city (box)"),
                    use_container_width=True)

    st.subheader(f"Bootstrap {int(conf*100)}% CI for the mean, by city")
    rows = []
    for city, g in sub.groupby("city"):
        res = uncertainty.bootstrap_statistic(g["value"], np.mean, 2000, conf)
        rows.append({"city": city, **res})
    ci = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ci["city"], y=ci["estimate"], mode="markers",
        error_y=dict(type="data", symmetric=False,
                     array=ci["ci_high"] - ci["estimate"],
                     arrayminus=ci["estimate"] - ci["ci_low"]),
        marker=dict(size=10)))
    fig.update_layout(yaxis_title="mean (ug/m3)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Interpretation: {int(conf*100)}% of such intervals, computed from repeated samples, "
        "would contain the true mean. This is NOT a probability statement about this one interval."
    )


def page_weather(meas):
    st.header("Weather relationships")
    wp = da.weather_pollution_joined()
    pollutant = st.session_state["pollutant"]
    city = st.session_state["city"]
    sub = wp[(wp["parameter"] == pollutant) & (wp["city"] == city)].dropna(subset=["pollutant_value"])
    if sub.empty:
        st.warning("No joined weather/pollution data for this city/pollutant.")
        return

    weather_var = st.sidebar.selectbox("Weather variable", WEATHER_COLS, key="w_var")
    method = st.sidebar.radio("Correlation type", ["spearman", "pearson"], key="w_method")
    lag = st.sidebar.slider("Lag (hours): pollutant_t vs weather_{t-lag}", 0, 24, 0, key="w_lag")

    sub = sub.sort_values("timestamp_utc")
    pol = sub["pollutant_value"].reset_index(drop=True)
    wx = sub[weather_var].shift(lag).reset_index(drop=True)

    if method == "spearman":
        res = correlation.spearman(pol, wx)
        label = f"Spearman rho = {res['rho']:.3f} (p={res['p_value']:.3g}, n={res['n']})"
    else:
        res = correlation.pearson(pol, wx)
        label = f"Pearson r = {res['r']:.3f} (p={res['p_value']:.3g}, n={res['n']})"

    st.plotly_chart(
        px.scatter(x=wx, y=pol, trendline="ols", opacity=0.4,
                   labels={"x": f"{weather_var} (lag {lag}h)", "y": f"{pollutant} ug/m3"},
                   title=label),
        use_container_width=True,
    )
    st.caption("Pearson = linear association; Spearman = monotonic (rank-based, robust to skew). "
               "Correlation is association, NOT causation.")

    st.subheader("Lagged correlation profile")
    lagged = correlation.lagged_correlation(pol, sub[weather_var].reset_index(drop=True),
                                            [0, 3, 6, 12, 24], method)
    st.plotly_chart(px.line(lagged, x="lag", y="corr", markers=True,
                            title=f"{pollutant.upper()} vs {weather_var} by lag"),
                    use_container_width=True)

    st.subheader("Partial correlation (controlling for confounders)")
    sub_t = add_temporal_features(sub)
    covars = st.multiselect("Control for", ["temperature", "humidity", "wind_speed", "hour", "season"],
                            default=["temperature", "humidity", "hour"], key="w_covars")
    covars = [c for c in covars if c != weather_var and c in sub_t.columns]
    pc = correlation.partial_correlation(
        sub_t.rename(columns={"pollutant_value": "pollutant"}), "pollutant", weather_var, covars
    )
    st.write(f"Partial r({pollutant}, {weather_var} | {covars}) = **{pc['partial_r']:.3f}** "
             f"(p={pc['p_value']:.3g}, n={pc['n']})")

    st.subheader("Interpretable regression (OLS, HC3 robust SE, log target)")
    reg_in = add_wind_components(sub).rename(columns={"pollutant_value": pollutant})
    res = regression.fit_ols(reg_in, pollutant, ["temperature", "humidity", "wind_speed", "precipitation"])
    if "headline" in res:
        st.write(f"n={res['n']}, R²={res['r_squared']:.3f}, adj R²={res['adj_r_squared']:.3f}, "
                 f"Durbin-Watson={res['durbin_watson']:.2f}, cov={res['cov_type']}")
        st.dataframe(res["headline"].round(4))
        st.caption("Positive wind_speed coefficient on log(PM) < 0 is expected (wind disperses PM). "
                   "Calendar (hour/dow/season) dummies are included but summarized via R².")
    else:
        st.info(res.get("error", "Regression unavailable."))


def page_quality(meas, quality, stations):
    st.header("Data Quality")
    if quality.empty:
        st.warning("No quality metrics. Run the pipeline.")
        return
    overall = quality[quality["metric_name"] == "overall"]
    st.metric("Mean overall Data Quality Score", f"{overall['metric_value'].mean():.1f}")

    st.subheader("Component scores (mean across station-days)")
    comp = (quality[quality["metric_name"] != "overall"]
            .groupby("metric_name")["metric_value"].mean().reset_index())
    st.plotly_chart(px.bar(comp, x="metric_name", y="metric_value",
                           labels={"metric_value": "score"}, range_y=[0, 100]),
                    use_container_width=True)

    st.subheader("Station quality ranking")
    rank = da.named_query("quality_ranking")
    st.dataframe(rank, use_container_width=True)

    st.subheader("Drill down: why did a station's score change?")
    sid = st.selectbox("Station", sorted(quality["station_id"].unique()), key="q_station")
    sq = quality[quality["station_id"] == sid]
    pivot = sq.pivot_table(index="date", columns="metric_name", values="metric_value")
    st.plotly_chart(px.line(pivot.reset_index(), x="date", y=pivot.columns.tolist(),
                            title=f"Quality components over time - {sid}"),
                    use_container_width=True)

    st.subheader("Completeness (missingness proxy) by station/parameter")
    completeness = da.named_query("station_completeness")
    if not completeness.empty:
        heat = completeness.pivot_table(index="station_id", columns="parameter",
                                        values="completeness_pct")
        st.plotly_chart(px.imshow(heat, color_continuous_scale="RdYlGn", zmin=0, zmax=100,
                                  aspect="auto", title="Completeness % heatmap"),
                        use_container_width=True)

    st.subheader("Data freshness")
    st.dataframe(da.named_query("freshness"), use_container_width=True)


def page_anomalies(meas):
    st.header("Anomalies & Events")
    cfg = da.CFG.statistics
    station = st.session_state["station"]
    pollutant = st.session_state["pollutant"]
    method = st.sidebar.selectbox("Outlier method",
                                  ["Hampel", "MAD", "IQR", "IQR (extreme 3x)"], key="an_method")
    k_sig = st.sidebar.slider("CUSUM threshold h (sigma)", 2.0, 10.0, 5.0, 0.5, key="an_h")
    persistence = st.sidebar.slider("Min persistence (hours)", 2, 12, 3, key="an_persist")

    sub = _series_for(meas, station, pollutant)
    if sub.empty:
        st.warning("No data for this selection.")
        return
    values = sub["value"]
    flags = _anomaly_flags(values, method, cfg)
    sub = sub.assign(flag=flags.values)

    st.metric(f"Flagged observations ({method})", int(flags.sum()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["timestamp_utc"], y=sub["value"], mode="lines",
                             name="observed", line=dict(color="#9aa0a6", width=1)))
    anom = sub[sub["flag"]]
    fig.add_trace(go.Scatter(x=anom["timestamp_utc"], y=anom["value"], mode="markers",
                             name="flagged", marker=dict(color="#ea4335", size=7)))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Switching the method changes which points are flagged: IQR/MAD are global, "
               "Hampel is local (rolling median+MAD) and adapts to trend/seasonality.")

    st.subheader("Sustained events (CUSUM, recomputed with current controls)")
    cu = shift.cusum(values.reset_index(drop=True), cfg.get("cusum_k_sigma", 0.5)
                     if isinstance(cfg, dict) else 0.5, k_sig)
    events = shift.extract_events(sub["timestamp_utc"], values.reset_index(drop=True), cu,
                                  station, pollutant, persistence, k_sig)
    if events:
        rows = []
        for e in events:
            ev = e.evidence
            rows.append({
                "pollutant": e.pollutant, "start": e.start_time, "end": e.end_time,
                "type": e.event_type, "severity": e.severity,
                "persistence_h": ev.get("persistence_hours"), "mean_value": ev.get("mean_value"),
                "peak_cusum": ev.get("peak_cusum"),
            })
        edf = pd.DataFrame(rows)
        st.dataframe(edf, use_container_width=True)

        st.markdown("##### Event context (first event)")
        e0 = events[0]
        baseline = float(np.nanmedian(values))
        st.write(
            f"**Station** {station} · **{pollutant.upper()}** · {e0.start_time} → {e0.end_time}\n\n"
            f"- Observed mean during event: **{e0.evidence.get('mean_value')}** ug/m3\n"
            f"- Series median (baseline): **{baseline:.1f}** ug/m3\n"
            f"- Method: **CUSUM**, persistence **{e0.evidence.get('persistence_hours')} h**, "
            f"peak CUSUM **{e0.evidence.get('peak_cusum')}** (h={k_sig})\n"
            f"- Classification: **potential environmental event** — not asserted as a definite "
            "pollution event. Could also be an instrument issue; cross-check Data Quality + weather."
        )
    else:
        st.info("No sustained events at the current threshold/persistence settings.")

    st.subheader("Persisted events (from pipeline)")
    st.dataframe(da.load_events(), use_container_width=True)


def page_stations(meas, stations):
    st.header("Stations")
    st.dataframe(stations, use_container_width=True)
    st.subheader("Completeness")
    st.dataframe(da.named_query("station_completeness"), use_container_width=True)
    st.subheader("Latest observation per station")
    st.dataframe(da.named_query("latest_observation"), use_container_width=True)
    if {"latitude", "longitude"}.issubset(stations.columns) and not stations.empty:
        st.map(stations.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]].dropna())


# --------------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------------- #
def main():
    st.title("Urban Air Quality Analytics & Data Quality Pipeline")
    meas = da.load_measurements()
    stations = da.load_stations()
    quality = da.load_quality()
    try:
        runs = da.load_runs()
    except Exception:
        runs = pd.DataFrame()

    st.sidebar.header("Global filters")
    if not meas.empty:
        cities = sorted(meas["city"].unique())
        city = st.sidebar.selectbox("City", cities, key="city")
        city_stations = sorted(meas[meas["city"] == city]["station_id"].unique())
        station = st.sidebar.selectbox("Station", city_stations, key="station")
        pollutants = sorted(meas[meas["station_id"] == station]["parameter"].unique())
        st.sidebar.selectbox("Pollutant", pollutants, key="pollutant")
    else:
        st.sidebar.info("No data yet. Run the pipeline.")

    page = st.sidebar.radio(
        "Page",
        ["Overview", "Trends", "Distributions", "Weather relationships",
         "Data Quality", "Anomalies & Events", "Stations"],
    )

    if page == "Overview":
        page_overview(meas, stations, runs, quality)
    elif page == "Trends":
        page_trends(meas)
    elif page == "Distributions":
        page_distributions(meas)
    elif page == "Weather relationships":
        page_weather(meas)
    elif page == "Data Quality":
        page_quality(meas, quality, stations)
    elif page == "Anomalies & Events":
        page_anomalies(meas)
    elif page == "Stations":
        page_stations(meas, stations)


if __name__ == "__main__":
    main()
