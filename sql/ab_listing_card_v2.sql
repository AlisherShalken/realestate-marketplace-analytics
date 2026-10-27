-- Метрики эксперимента listing_card_v2 по пользователям + достаточные статистики
-- для дельта-метода, чтобы p-value по CR можно было посчитать прямо в BI без выгрузки.
--
-- user_metrics можно выгрузить целиком и считать в Python (src/abtest.py),
-- финальный SELECT — для дашборда.

DECLARE exp_from DATE DEFAULT '2025-08-04';
DECLARE exp_to   DATE DEFAULT '2025-08-24';
DECLARE pre_from DATE DEFAULT '2025-06-02';
DECLARE pre_to   DATE DEFAULT '2025-08-03';

WITH exposures AS (
  SELECT user_id, variant
  FROM `mart.exp_exposures`
  WHERE experiment = 'listing_card_v2'
),
user_metrics AS (
  SELECT
    e.user_id,
    e.variant,
    SUM(IF(d.dt BETWEEN exp_from AND exp_to, d.listing_views, 0)) AS views,
    SUM(IF(d.dt BETWEEN exp_from AND exp_to, d.contacts, 0)) AS contacts,
    SUM(IF(d.dt BETWEEN exp_from AND exp_to, d.sessions, 0)) AS sessions,
    SUM(IF(d.dt BETWEEN pre_from AND pre_to, d.listing_views, 0)) AS pre_views,
    SUM(IF(d.dt BETWEEN pre_from AND pre_to, d.contacts, 0)) AS pre_contacts
  FROM exposures e
  LEFT JOIN `mart.user_daily` d
    ON d.user_id = e.user_id
   AND d.dt BETWEEN pre_from AND exp_to
  GROUP BY e.user_id, e.variant
),
stats AS (
  SELECT
    variant,
    COUNT(*) AS n,
    SUM(contacts) AS sum_x,
    SUM(views) AS sum_y,
    VAR_SAMP(contacts) AS var_x,
    VAR_SAMP(views) AS var_y,
    COVAR_SAMP(contacts, views) AS cov_xy,
    AVG(contacts) AS mean_x,
    AVG(views) AS mean_y
  FROM user_metrics
  GROUP BY variant
),
ratio AS (
  SELECT
    variant,
    n,
    sum_x / sum_y AS cr,
    -- дельта-метод: Var(X/Y) ≈ (var_x/my² − 2·mx·cov/my³ + mx²·var_y/my⁴) / n
    (var_x / POW(mean_y, 2)
      - 2 * mean_x * cov_xy / POW(mean_y, 3)
      + POW(mean_x, 2) * var_y / POW(mean_y, 4)) / n AS cr_var
  FROM stats
)
SELECT
  c.n AS n_control,
  t.n AS n_test,
  c.cr AS cr_control,
  t.cr AS cr_test,
  t.cr / c.cr - 1 AS rel_diff,
  (t.cr - c.cr) - 1.96 * SQRT(c.cr_var + t.cr_var) AS ci_low_abs,
  (t.cr - c.cr) + 1.96 * SQRT(c.cr_var + t.cr_var) AS ci_high_abs,
  (t.cr - c.cr) / SQRT(c.cr_var + t.cr_var) AS z_score
FROM ratio c
CROSS JOIN ratio t
WHERE c.variant = 'control' AND t.variant = 'test';
