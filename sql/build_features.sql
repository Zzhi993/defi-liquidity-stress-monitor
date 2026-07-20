DROP TABLE IF EXISTS clean_protocol_tvl;
CREATE TABLE clean_protocol_tvl AS
WITH ranked AS (
    SELECT
        date,
        CAST(source_timestamp_unix AS INTEGER) AS source_timestamp_unix,
        protocol_slug,
        protocol_name,
        category,
        CAST(tvl_usd AS REAL) AS tvl_usd,
        ROW_NUMBER() OVER (
            PARTITION BY date, protocol_slug
            ORDER BY CAST(source_timestamp_unix AS INTEGER) DESC
        ) AS observation_rank
    FROM protocol_tvl
    WHERE tvl_usd IS NOT NULL
      AND tvl_usd > 1000000
      AND date >= '2021-01-01'
)
SELECT
    date,
    source_timestamp_unix,
    protocol_slug,
    protocol_name,
    category,
    tvl_usd
FROM ranked
WHERE observation_rank = 1;

DROP TABLE IF EXISTS clean_stablecoin_supply;
CREATE TABLE clean_stablecoin_supply AS
WITH ranked AS (
    SELECT
        date,
        CAST(source_timestamp_unix AS INTEGER) AS source_timestamp_unix,
        CAST(stablecoin_supply_usd AS REAL) AS stablecoin_supply_usd,
        ROW_NUMBER() OVER (
            PARTITION BY date
            ORDER BY CAST(source_timestamp_unix AS INTEGER) DESC
        ) AS observation_rank
    FROM stablecoin_supply
    WHERE stablecoin_supply_usd IS NOT NULL
      AND stablecoin_supply_usd > 0
      AND date >= '2021-01-01'
)
SELECT date, source_timestamp_unix, stablecoin_supply_usd
FROM ranked
WHERE observation_rank = 1;

DROP TABLE IF EXISTS protocol_returns;
CREATE TABLE protocol_returns AS
WITH lagged AS (
    SELECT
        date,
        protocol_slug,
        protocol_name,
        category,
        tvl_usd,
        LAG(date) OVER (
            PARTITION BY protocol_slug
            ORDER BY date
        ) AS prev_date,
        LAG(tvl_usd) OVER (
            PARTITION BY protocol_slug
            ORDER BY date
        ) AS prev_tvl_usd,
        MAX(tvl_usd) OVER (
            PARTITION BY protocol_slug
            ORDER BY date
            ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
        ) AS trailing_7d_high_tvl_usd
    FROM clean_protocol_tvl
)
SELECT
    date,
    protocol_slug,
    protocol_name,
    category,
    tvl_usd,
    CASE
        WHEN prev_tvl_usd > 0
         AND CAST(JULIANDAY(date) - JULIANDAY(prev_date) AS INTEGER) = 1
        THEN LN(tvl_usd / prev_tvl_usd)
        ELSE NULL
    END AS log_return,
    CASE
        WHEN trailing_7d_high_tvl_usd > 0 THEN tvl_usd / trailing_7d_high_tvl_usd - 1.0
        ELSE NULL
    END AS drawdown_7d
FROM lagged;

DROP TABLE IF EXISTS system_daily;
CREATE TABLE system_daily AS
WITH stable AS (
    SELECT
        date,
        stablecoin_supply_usd,
        LAG(date) OVER (ORDER BY date) AS prev_date,
        LAG(stablecoin_supply_usd) OVER (ORDER BY date) AS prev_stablecoin_supply_usd
    FROM clean_stablecoin_supply
),
protocol_agg AS (
    SELECT
        date,
        SUM(tvl_usd) AS total_tvl_usd,
        AVG(log_return) AS equal_weight_system_return,
        AVG(CASE WHEN log_return < 0 THEN 1.0 ELSE 0.0 END) AS downside_breadth,
        COUNT(log_return) AS protocol_count,
        MIN(drawdown_7d) AS worst_protocol_drawdown_7d
    FROM protocol_returns
    WHERE log_return IS NOT NULL
    GROUP BY date
)
SELECT
    p.date,
    p.total_tvl_usd,
    p.equal_weight_system_return,
    p.downside_breadth,
    p.protocol_count,
    p.worst_protocol_drawdown_7d,
    s.stablecoin_supply_usd,
    CASE
        WHEN s.prev_stablecoin_supply_usd > 0
         AND CAST(JULIANDAY(s.date) - JULIANDAY(s.prev_date) AS INTEGER) = 1
        THEN LN(s.stablecoin_supply_usd / s.prev_stablecoin_supply_usd)
        ELSE NULL
    END AS stablecoin_supply_log_flow
FROM protocol_agg p
LEFT JOIN stable s
    ON p.date = s.date
WHERE p.protocol_count >= 6;

DROP TABLE IF EXISTS protocol_metadata;
CREATE TABLE protocol_metadata AS
SELECT DISTINCT
    protocol_slug,
    protocol_name,
    category
FROM clean_protocol_tvl;

DROP TABLE IF EXISTS protocol_coverage;
CREATE TABLE protocol_coverage AS
SELECT
    protocol_slug,
    MAX(protocol_name) AS protocol_name,
    MAX(category) AS category,
    MIN(date) AS first_observation,
    MAX(date) AS last_observation,
    COUNT(*) AS tvl_observations,
    COUNT(log_return) AS valid_daily_returns
FROM protocol_returns
GROUP BY protocol_slug;

CREATE UNIQUE INDEX IF NOT EXISTS idx_clean_protocol_day
ON clean_protocol_tvl(date, protocol_slug);
