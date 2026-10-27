-- Reusable analytical views. These push aggregation into SQL instead of Pandas,
-- so the dashboard and ad-hoc analysis query a stable, documented interface.

-- 1. Monthly pollutant trend per city.
CREATE VIEW IF NOT EXISTS v_monthly_pollutant_trend AS
SELECT
    s.city,
    a.parameter,
    substr(a.timestamp_utc, 1, 7) AS month,
    COUNT(*)                       AS n_obs,
    AVG(a.value)                   AS mean_value,
    MIN(a.value)                   AS min_value,
    MAX(a.value)                   AS max_value
FROM air_measurements a
JOIN stations s ON s.station_id = a.station_id
WHERE a.value IS NOT NULL
GROUP BY s.city, a.parameter, month;

-- 2. Station completeness: observed vs. expected hourly slots over the station's span.
CREATE VIEW IF NOT EXISTS v_station_completeness AS
WITH span AS (
    SELECT
        station_id,
        parameter,
        MIN(timestamp_utc) AS first_ts,
        MAX(timestamp_utc) AS last_ts,
        COUNT(*)           AS observed
    FROM air_measurements
    GROUP BY station_id, parameter
)
SELECT
    sp.station_id,
    s.city,
    sp.parameter,
    sp.observed,
    (julianday(sp.last_ts) - julianday(sp.first_ts)) * 24 + 1 AS expected_slots,
    ROUND(
        100.0 * sp.observed /
        NULLIF((julianday(sp.last_ts) - julianday(sp.first_ts)) * 24 + 1, 0),
        2
    ) AS completeness_pct
FROM span sp
JOIN stations s ON s.station_id = sp.station_id;

-- 3. Station quality ranking: latest DQ score per station.
CREATE VIEW IF NOT EXISTS v_station_quality_ranking AS
WITH latest AS (
    SELECT station_id, MAX(date) AS max_date
    FROM quality_metrics
    WHERE metric_name = 'overall'
    GROUP BY station_id
)
SELECT
    q.station_id,
    s.city,
    s.station_name,
    q.date,
    q.metric_value AS data_quality_score,
    q.status
FROM quality_metrics q
JOIN latest l ON l.station_id = q.station_id AND l.max_date = q.date
JOIN stations s ON s.station_id = q.station_id
WHERE q.metric_name = 'overall'
ORDER BY q.metric_value DESC;

-- 4. Latest observation per station/parameter (freshness source).
CREATE VIEW IF NOT EXISTS v_latest_observation AS
SELECT
    a.station_id,
    s.city,
    a.parameter,
    MAX(a.timestamp_utc) AS latest_ts
FROM air_measurements a
JOIN stations s ON s.station_id = a.station_id
GROUP BY a.station_id, s.city, a.parameter;

-- 5. Pollution episodes (sustained events) with duration in hours.
CREATE VIEW IF NOT EXISTS v_pollution_episodes AS
SELECT
    e.event_id,
    e.station_id,
    s.city,
    e.pollutant,
    e.start_time,
    e.end_time,
    e.event_type,
    e.severity,
    e.detection_method,
    ROUND((julianday(e.end_time) - julianday(e.start_time)) * 24, 1) AS duration_hours
FROM events e
JOIN stations s ON s.station_id = e.station_id
ORDER BY e.start_time DESC;

-- 6. Pollutant distribution summary per city/parameter.
CREATE VIEW IF NOT EXISTS v_pollutant_distribution AS
SELECT
    s.city,
    a.parameter,
    COUNT(*)   AS n,
    AVG(a.value) AS mean,
    MIN(a.value) AS min,
    MAX(a.value) AS max
FROM air_measurements a
JOIN stations s ON s.station_id = a.station_id
WHERE a.value IS NOT NULL
GROUP BY s.city, a.parameter;

-- 7. Weather-pollution hourly aggregate (joined on city + hour) for correlation work.
CREATE VIEW IF NOT EXISTS v_weather_pollution AS
SELECT
    s.city,
    a.timestamp_utc,
    a.parameter,
    AVG(a.value)       AS pollutant_value,
    w.temperature,
    w.humidity,
    w.precipitation,
    w.pressure,
    w.wind_speed,
    w.wind_direction
FROM air_measurements a
JOIN stations s      ON s.station_id = a.station_id
JOIN weather_hourly w ON w.city = s.city AND w.timestamp_utc = a.timestamp_utc
WHERE a.value IS NOT NULL
GROUP BY s.city, a.timestamp_utc, a.parameter,
         w.temperature, w.humidity, w.precipitation, w.pressure, w.wind_speed, w.wind_direction;

-- 8. Data freshness: hours since last observation per station.
CREATE VIEW IF NOT EXISTS v_data_freshness AS
SELECT
    station_id,
    city,
    parameter,
    latest_ts,
    ROUND((julianday('now') - julianday(latest_ts)) * 24, 1) AS hours_since_last
FROM v_latest_observation;

-- 9. Top anomaly periods: hours with the most flagged observations.
CREATE VIEW IF NOT EXISTS v_top_anomaly_periods AS
SELECT
    f.station_id,
    s.city,
    f.pollutant,
    substr(f.timestamp_utc, 1, 13) AS hour,
    SUM(f.anomaly_flag)            AS anomalies
FROM features_hourly f
JOIN stations s ON s.station_id = f.station_id
WHERE f.anomaly_flag = 1
GROUP BY f.station_id, s.city, f.pollutant, hour
ORDER BY anomalies DESC;

-- 10. City comparison: mean pollutant level and station count per city.
CREATE VIEW IF NOT EXISTS v_city_comparison AS
SELECT
    s.city,
    a.parameter,
    COUNT(DISTINCT a.station_id) AS n_stations,
    AVG(a.value)                 AS mean_value,
    COUNT(*)                     AS n_obs
FROM air_measurements a
JOIN stations s ON s.station_id = a.station_id
WHERE a.value IS NOT NULL
GROUP BY s.city, a.parameter;
