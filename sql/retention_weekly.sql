-- Недельный retention новых пользователей.
-- Когорта = неделя первой активности, только зарегистрированные в периоде.

DECLARE date_from DATE DEFAULT '2025-06-01';
DECLARE date_to   DATE DEFAULT '2025-08-31';

WITH new_users AS (
  SELECT user_id
  FROM `dwh.users`
  WHERE DATE(registered_at) BETWEEN date_from AND date_to
),
activity AS (
  SELECT DISTINCT d.user_id, d.dt
  FROM `mart.user_daily` d
  JOIN new_users USING (user_id)
  WHERE d.dt BETWEEN date_from AND date_to
),
first_seen AS (
  SELECT user_id, MIN(dt) AS first_dt
  FROM activity
  GROUP BY user_id
),
weeks AS (
  SELECT
    DATE_TRUNC(f.first_dt, WEEK(MONDAY)) AS cohort_week,
    a.user_id,
    DIV(DATE_DIFF(a.dt, f.first_dt, DAY), 7) AS week_n
  FROM activity a
  JOIN first_seen f USING (user_id)
)
SELECT
  cohort_week,
  week_n,
  COUNT(DISTINCT user_id) AS active_users,
  ROUND(COUNT(DISTINCT user_id)
        / MAX(COUNT(DISTINCT IF(week_n = 0, user_id, NULL))) OVER (PARTITION BY cohort_week), 3) AS retention
FROM weeks
-- неделя должна полностью уложиться в период, иначе retention занижен
WHERE DATE_ADD(cohort_week, INTERVAL 7 * (week_n + 2) - 1 DAY) <= date_to
GROUP BY cohort_week, week_n
ORDER BY cohort_week, week_n;
