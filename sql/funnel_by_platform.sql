-- Воронка по платформам за период: доля user-day с поиском / просмотром / контактом
-- и конверсия просмотр -> контакт.

DECLARE date_from DATE DEFAULT '2025-06-01';
DECLARE date_to   DATE DEFAULT '2025-08-31';

SELECT
  platform,
  COUNT(*) AS user_days,
  COUNT(DISTINCT user_id) AS users,
  ROUND(AVG(IF(searches > 0, 1, 0)), 3) AS share_search,
  ROUND(AVG(IF(listing_views > 0, 1, 0)), 3) AS share_view,
  ROUND(AVG(IF(contacts > 0, 1, 0)), 3) AS share_contact,
  ROUND(SAFE_DIVIDE(SUM(contacts), SUM(listing_views)), 4) AS contacts_per_view,
  ROUND(AVG(listing_views), 2) AS views_per_user_day
FROM `mart.user_daily`
WHERE dt BETWEEN date_from AND date_to
GROUP BY ROLLUP(platform)
ORDER BY platform NULLS LAST;
