"""
Генерация синтетических данных, похожих на данные площадки объявлений о недвижимости.

Реальные данные я использовать не могу, поэтому собрал генератор, который
воспроизводит основные эффекты, с которыми сталкиваешься на практике:
  - сезонность по дням недели, разная активность платформ;
  - тяжелый хвост в активности пользователей (несколько процентов юзеров
    дают большую часть просмотров);
  - "просадки" просмотров у объявлений с инерцией (AR(1) шум), из-за которых
    владельцы чаще покупают продвижение именно в плохие дни;
  - A/B тест новой карточки объявления в августе.

Запуск:  python src/generate_data.py --out data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2025-06-01")
END = pd.Timestamp("2025-08-31")

EXP_START = pd.Timestamp("2025-08-04")
EXP_END = pd.Timestamp("2025-08-24")
EXP_NAME = "listing_card_v2"

# истинные эффекты, которые потом пытаемся найти в ноутбуках
TRUE_CARD_LIFT = {"android": 0.035, "ios": 0.07, "web": 0.02}  # на конверсию просмотр -> контакт
TRUE_PROMO_VIEWS_LIFT = 0.45   # +45% к просмотрам в дни продвижения
TRUE_PROMO_CONTACT_LIFT = 0.30

CITIES = {
    "Алматы": (0.45, ["Алмалинский", "Бостандыкский", "Медеуский", "Ауэзовский",
                      "Алатауский", "Наурызбайский", "Жетысуский", "Турксибский"], 780_000),
    "Астана": (0.30, ["Есильский", "Алматинский", "Сарыаркинский", "Байконур", "Нура"], 650_000),
    "Шымкент": (0.10, ["Абайский", "Аль-Фарабийский", "Енбекшинский", "Каратауский"], 420_000),
    "Караганда": (0.06, ["Казыбек би", "Алихана Бокейханова"], 330_000),
    "Актобе": (0.05, ["Астана", "Алматы"], 310_000),
    "Атырау": (0.04, ["Атырау"], 380_000),
}


def _pick_city(rng, n):
    names = list(CITIES)
    probs = np.array([CITIES[c][0] for c in names])
    return rng.choice(names, size=n, p=probs / probs.sum())


def make_users(rng, n_users=150_000):
    days = (END - START).days + 1
    user_id = np.arange(1, n_users + 1)
    platform = rng.choice(["android", "ios", "web"], size=n_users, p=[0.52, 0.23, 0.25])
    city = _pick_city(rng, n_users)
    role = rng.choice(["buyer", "renter", "seller", "agent"], size=n_users, p=[0.38, 0.44, 0.13, 0.05])

    # часть пользователей зарегистрирована давно, часть приходит в течение периода
    reg_offset = np.where(
        rng.random(n_users) < 0.55,
        -rng.integers(1, 900, n_users),
        rng.integers(0, days, n_users),
    )
    reg_date = START + pd.to_timedelta(reg_offset, unit="D")

    # латентная активность: логнормальная, агенты сильно активнее
    activity = rng.lognormal(mean=-2.1, sigma=1.0, size=n_users)
    activity *= np.where(role == "agent", 3.0, 1.0)
    activity = np.clip(activity, 0.003, 0.9)

    # пользователь приходит "под задачу" и уходит, когда ее решил или бросил искать.
    # у старых пользователей поиск начинается в случайный момент, а не в день регистрации
    search_start = np.where(
        reg_offset < 0,
        rng.integers(-60, days, n_users),
        reg_offset,
    )
    search_len = np.select(
        [role == "agent", role == "seller"],
        [rng.exponential(400, n_users), rng.exponential(90, n_users)],
        rng.exponential(32, n_users),
    ).astype(int) + 1

    # склонность звонить/писать после просмотра
    contact_p = rng.beta(2, 30, size=n_users)
    contact_p *= np.select([role == "renter", role == "agent"], [1.35, 0.6], 1.0)
    contact_p *= np.select([platform == "ios", platform == "web"], [1.08, 0.9], 1.0)

    return pd.DataFrame({
        "user_id": user_id,
        "reg_date": reg_date.date,
        "platform": platform,
        "city": city,
        "role": role,
        "_activity": activity,
        "_contact_p": contact_p,
        "_search_start": search_start,
        "_search_len": search_len,
    })


def assign_variant(user_ids, salt=EXP_NAME):
    # как в нормальной сплит-системе: детерминированный хеш от user_id + соль
    h = pd.util.hash_array(np.array([f"{salt}:{u}" for u in user_ids], dtype=object))
    return np.where(h % 100 < 50, "control", "test")


def make_sessions(rng, users):
    dates = pd.date_range(START, END, freq="D")
    dow_mult = np.array([1.05, 1.03, 1.0, 0.98, 0.93, 0.92, 1.09])  # пн..вс
    # летом в августе небольшой рост — сезон аренды перед учебным годом
    trend = np.linspace(0.95, 1.12, len(dates))

    reg = pd.to_datetime(users["reg_date"]).values
    variant = assign_variant(users["user_id"].values)
    lift = users["platform"].map(TRUE_CARD_LIFT).values

    act = users["_activity"].values
    cp = users["_contact_p"].values
    s_start = users["_search_start"].values
    s_len = users["_search_len"].values
    platform = users["platform"].values
    views_mult = np.select([platform == "ios", platform == "web"], [1.1, 0.75], 1.0)
    rows = []
    for i, d in enumerate(dates):
        age = i - s_start
        alive = (reg <= d.to_datetime64()) & (age >= 0) & (age < s_len)
        # к концу первого месяца поиска заходят реже — кто не нашел вариант, "остывает"
        decay = np.where(age < 25, 1.0, 0.7)
        p_active = act * dow_mult[d.dayofweek] * trend[i] * decay
        is_active = alive & (rng.random(len(act)) < p_active)
        idx = np.flatnonzero(is_active)
        if len(idx) == 0:
            continue

        sessions = 1 + rng.poisson(0.6, len(idx))
        searches = rng.poisson(2.2 * sessions)
        views = rng.poisson(1.0 + 3.1 * views_mult[idx] * sessions * rng.gamma(2.0, 0.5, len(idx)))
        p = cp[idx].copy()
        in_exp = EXP_START <= d <= EXP_END
        if in_exp:
            p = np.where(variant[idx] == "test", p * (1 + lift[idx]), p)
        contacts = rng.binomial(views, np.clip(p, 0, 1))
        favorites = rng.binomial(views, 0.06)

        rows.append(pd.DataFrame({
            "dt": d.date(),
            "user_id": users["user_id"].values[idx],
            "platform": users["platform"].values[idx],
            "sessions": sessions,
            "searches": searches,
            "listing_views": views,
            "favorites": favorites,
            "contacts": contacts,
        }))
    sessions_df = pd.concat(rows, ignore_index=True)

    # лог экспозиций: пользователь попадает в эксперимент при первом визите в период теста
    exp_days = sessions_df[(pd.to_datetime(sessions_df["dt"]) >= EXP_START) &
                           (pd.to_datetime(sessions_df["dt"]) <= EXP_END)]
    first_seen = exp_days.groupby("user_id", as_index=False)["dt"].min()
    first_seen = first_seen.rename(columns={"dt": "first_exposure_dt"})
    vmap = dict(zip(users["user_id"].values, variant))
    first_seen["experiment"] = EXP_NAME
    first_seen["variant"] = first_seen["user_id"].map(vmap)
    return sessions_df, first_seen


def make_listings(rng, n_listings=22_000):
    days = (END - START).days + 1
    listing_id = np.arange(100_001, 100_001 + n_listings)
    city = _pick_city(rng, n_listings)
    district = np.array([rng.choice(CITIES[c][1]) for c in city])
    deal = rng.choice(["sale", "rent"], size=n_listings, p=[0.55, 0.45])
    owner_type = rng.choice(["owner", "agent", "developer"], size=n_listings, p=[0.58, 0.37, 0.05])
    rooms = rng.choice([1, 2, 3, 4, 5], size=n_listings, p=[0.30, 0.34, 0.24, 0.09, 0.03])
    area = np.round(rooms * rng.normal(24, 4, n_listings) + rng.normal(14, 4, n_listings), 1)
    area = np.clip(area, 18, 400)
    base_m2 = np.array([CITIES[c][2] for c in city])
    price_m2 = base_m2 * rng.lognormal(0, 0.22, n_listings)
    price = np.where(deal == "sale",
                     np.round(price_m2 * area, -5),
                     np.round(price_m2 * area * 0.0055, -4))
    photos = np.clip(rng.poisson(9, n_listings), 0, 30)
    photos = np.where(rng.random(n_listings) < 0.07, 0, photos)

    created_offset = rng.integers(-30, days - 5, n_listings)
    lifetime = np.clip(rng.exponential(38, n_listings).astype(int) + 3, 3, 180)
    lifetime = np.where(deal == "rent", (lifetime * 0.6).astype(int) + 2, lifetime)

    # "качество" объявления: фото, цена относительно района, и что-то, что не видно в данных
    quality = (0.04 * np.minimum(photos, 15)
               - 0.6 * np.log(price_m2 / base_m2)
               + rng.normal(0, 0.35, n_listings))

    return pd.DataFrame({
        "listing_id": listing_id,
        "created_dt": (START + pd.to_timedelta(created_offset, unit="D")).date,
        "closed_dt": (START + pd.to_timedelta(created_offset + lifetime, unit="D")).date,
        "city": city,
        "district": district,
        "deal_type": deal,
        "owner_type": owner_type,
        "rooms": rooms,
        "area_m2": area,
        "price_kzt": price.astype(np.int64),
        "photos_cnt": photos,
        "_quality": quality,
        "_created_offset": created_offset,
        "_lifetime": lifetime,
    })


def make_listing_daily(rng, listings):
    days = (END - START).days + 1
    n = len(listings)
    created = listings["_created_offset"].values
    life = listings["_lifetime"].values
    owner = listings["owner_type"].values
    q = listings["_quality"].values
    deal = listings["deal_type"].values

    base = np.exp(0.9 + q) * np.where(deal == "rent", 1.4, 1.0)
    dow_mult = np.array([1.05, 1.03, 1.0, 0.98, 0.93, 0.92, 1.09])

    promo_rate = np.select([owner == "agent", owner == "developer"], [0.012, 0.02], 0.004)
    promo_start = np.full(n, -10_000)
    promoted_ever = np.zeros(n, dtype=bool)

    shock = np.zeros(n)
    views_hist = []
    out = []
    for t in range(days):
        age = t - created
        active = (age >= 0) & (age < life)
        # свежие объявления смотрят больше, потом плавно затухает
        age_mult = 1.0 + 1.3 * np.exp(-np.maximum(age, 0) / 5.0)
        # AR(1) шум — объявление "проседает" на несколько дней
        shock = 0.75 * shock + rng.normal(0, 0.28, n)

        # решение о продвижении: чаще у агентов и когда последние дни были слабыми
        if t >= 3:
            recent = np.mean(views_hist[-3:], axis=0)
            dip = recent < 0.75 * base
            p = promo_rate * np.where(dip, 3.0, 1.0)
            start_now = active & (age >= 4) & (~promoted_ever) & (rng.random(n) < p)
            promo_start[start_now] = t
            promoted_ever |= start_now

        on_promo = (t >= promo_start) & (t < promo_start + 7)
        mu = base * age_mult * dow_mult[(START + pd.Timedelta(days=t)).dayofweek] * np.exp(shock)
        mu_views = mu * np.where(on_promo, 1 + TRUE_PROMO_VIEWS_LIFT, 1.0)
        views = np.where(active, rng.poisson(mu_views), 0)
        views_hist.append(views)

        # конверсия при продвижении чуть ниже: приходит менее целевой трафик
        promo_cr = (1 + TRUE_PROMO_CONTACT_LIFT) / (1 + TRUE_PROMO_VIEWS_LIFT)
        cr = 0.04 * np.exp(0.6 * q) * np.where(on_promo, promo_cr, 1.0)
        contacts = np.where(active, rng.binomial(views, np.clip(cr, 0, 1)), 0)

        idx = np.flatnonzero(active)
        out.append(pd.DataFrame({
            "dt": (START + pd.Timedelta(days=t)).date(),
            "listing_id": listings["listing_id"].values[idx],
            "views": views[idx],
            "contacts": contacts[idx],
            "is_promoted": on_promo[idx].astype(int),
        }))

    daily = pd.concat(out, ignore_index=True)
    promos = pd.DataFrame({
        "listing_id": listings["listing_id"].values[promoted_ever],
        "promo_start_dt": (START + pd.to_timedelta(promo_start[promoted_ever], unit="D")).date,
        "package": "top_7d",
    })
    return daily, promos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    users = make_users(rng)
    sessions, exposures = make_sessions(rng, users)
    listings = make_listings(rng)
    listing_daily, promos = make_listing_daily(rng, listings)

    hidden = [c for c in users.columns if c.startswith("_")]
    users.drop(columns=hidden).to_parquet(out / "users.parquet", index=False)
    sessions.to_parquet(out / "user_daily.parquet", index=False)
    exposures.to_parquet(out / "exp_exposures.parquet", index=False)

    hidden = [c for c in listings.columns if c.startswith("_")]
    listings.drop(columns=hidden).to_parquet(out / "listings.parquet", index=False)
    listing_daily.to_parquet(out / "listing_daily.parquet", index=False)
    promos.to_parquet(out / "promotions.parquet", index=False)

    print(f"users: {len(users):,}  user_daily: {len(sessions):,}  exposures: {len(exposures):,}")
    print(f"listings: {len(listings):,}  listing_daily: {len(listing_daily):,}  promotions: {len(promos):,}")


if __name__ == "__main__":
    main()
