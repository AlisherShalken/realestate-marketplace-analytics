-- Агрегат по пользователю за день из сырых событий.
-- Источник: analytics.events (партиционирована по event_date)
-- Результат: mart.user_daily, партиция за {{ ds }}

DECLARE run_date DATE DEFAULT DATE('{{ ds }}');

DELETE FROM `mart.user_daily` WHERE dt = run_date;

INSERT INTO `mart.user_daily` (dt, user_id, platform, sessions, searches, listing_views, favorites, contacts)
SELECT
  event_date AS dt,
  user_id,
  -- если за день был и в приложении, и на вебе — берем платформу, с которой больше событий
  APPROX_TOP_COUNT(platform, 1)[OFFSET(0)].value AS platform,
  COUNT(DISTINCT session_id) AS sessions,
  COUNTIF(event_name = 'search_results_shown') AS searches,
  COUNTIF(event_name = 'listing_view') AS listing_views,
  COUNTIF(event_name = 'favorite_add') AS favorites,
  -- контакт = показ телефона или первое сообщение в чат по объявлению
  COUNT(DISTINCT IF(event_name IN ('phone_shown', 'chat_first_message'),
                    CONCAT(CAST(listing_id AS STRING), ':', event_name), NULL)) AS contacts
FROM `analytics.events`
WHERE event_date = run_date
  AND user_id IS NOT NULL
  AND NOT is_bot
GROUP BY dt, user_id;
