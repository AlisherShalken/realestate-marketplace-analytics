-- Статистика объявления за день + флаг активного продвижения.
-- Результат: mart.listing_daily, партиция за {{ ds }}

DECLARE run_date DATE DEFAULT DATE('{{ ds }}');

DELETE FROM `mart.listing_daily` WHERE dt = run_date;

INSERT INTO `mart.listing_daily` (dt, listing_id, views, contacts, is_promoted)
WITH ev AS (
  SELECT
    listing_id,
    COUNTIF(event_name = 'listing_view') AS views,
    COUNT(DISTINCT IF(event_name IN ('phone_shown', 'chat_first_message'),
                      CONCAT(CAST(user_id AS STRING), ':', event_name), NULL)) AS contacts
  FROM `analytics.events`
  WHERE event_date = run_date
    AND listing_id IS NOT NULL
    AND NOT is_bot
  GROUP BY listing_id
),
active AS (
  SELECT listing_id
  FROM `dwh.listings`
  WHERE DATE(created_at) <= run_date
    AND (closed_at IS NULL OR DATE(closed_at) > run_date)
),
promo AS (
  SELECT DISTINCT listing_id
  FROM `dwh.paid_services`
  WHERE service_type = 'top_7d'
    AND run_date BETWEEN DATE(started_at) AND DATE_ADD(DATE(started_at), INTERVAL 6 DAY)
)
SELECT
  run_date AS dt,
  a.listing_id,
  IFNULL(ev.views, 0) AS views,
  IFNULL(ev.contacts, 0) AS contacts,
  IF(p.listing_id IS NOT NULL, 1, 0) AS is_promoted
FROM active a
LEFT JOIN ev USING (listing_id)
LEFT JOIN promo p USING (listing_id);
