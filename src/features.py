"""Признаки куки: только собственные события в заданном суточном окне."""

from pathlib import Path

import numpy as np
import pandas as pd


EVENT_NAMES = [
    "search_results_view", "item_view", "photo_swipe", "seller_page_view",
    "contact_phone_show", "contact_chat_open", "contact_message_sent",
    "favorite_add", "login", "captcha_shown",
]
DATE_COLUMNS = ["cookie_created_at", "window_start_ts", "window_end_ts"]


def load_data(data_dir="bot_detection_challenge/data"):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv", parse_dates=DATE_COLUMNS)
    test = pd.read_csv(data_dir / "test.csv", parse_dates=DATE_COLUMNS)
    events = pd.read_csv(data_dir / "events.csv.gz", parse_dates=["event_ts"])
    if not train.cookie_id.is_unique or not test.cookie_id.is_unique:
        raise ValueError("Ожидалась одна строка метаданных на cookie_id")
    if set(train.cookie_id) & set(test.cookie_id):
        raise ValueError("Куки train и test пересекаются")
    return train, test, events


def prepare_events(events, meta):
    """Удаляем повторы целой строки, но сохраняем разные события в одну секунду."""
    ev = events.merge(
        meta[["cookie_id", "window_start_ts", "window_end_ts"]],
        on="cookie_id", how="inner", validate="many_to_one",
    )
    inside = (ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)
    raw = ev.loc[inside].copy()
    ev = raw.drop_duplicates().copy()
    ev["platform"] = (
        ev.platform.str.strip().str.lower()
        .replace({"desktop": "web", "iphone": "ios"}).fillna("unknown")
    )
    # При совпадении времени порядок действий неизвестен. Переходы для таких
    # событий ниже исключаются; вторичный ключ нужен для стабильного результата.
    ev = ev.sort_values(["cookie_id", "event_ts", "event_name", "item_id"])
    ev = ev.reset_index(drop=True)
    audit = {
        "n_input_events": len(events),
        "n_matched_events": int(len(inside)),
        "n_outside_window": int((~inside).sum()),
        "n_duplicate_rows_in_window": len(raw) - len(ev),
        "n_clean_events": len(ev),
        "n_cookies_without_events": int((~meta.cookie_id.isin(ev.cookie_id)).sum()),
    }
    return ev, raw, audit


def category_summary(ev, column, index):
    """Разнообразие и концентрация значений без запоминания их идентификаторов."""
    counts = ev.groupby(["cookie_id", column]).size()
    totals = counts.groupby(level=0).sum()
    proportions = counts / counts.groupby(level=0).transform("sum")
    result = pd.DataFrame(index=index)
    result[f"{column}_unique"] = counts.groupby(level=0).size()
    result[f"{column}_unique_share"] = result[f"{column}_unique"] / totals
    result[f"{column}_top_share"] = proportions.groupby(level=0).max()
    result[f"{column}_entropy"] = (
        -proportions * np.log2(proportions)
    ).groupby(level=0).sum()
    result[f"{column}_unique"] = result[f"{column}_unique"].fillna(0)
    return result


def ua_family(value):
    """Выделяем семейства, не используем версии браузера и модель телефона."""
    ua = str(value).lower()
    if "headless" in ua:
        browser = "headless"
    elif any(token in ua for token in [
        "python", "curl/", "scrapy/", "go-http-client", "node-fetch",
    ]):
        browser = "http_client"
    elif ua.startswith("avito/"):
        browser = "avito_app"
    elif "yabrowser" in ua:
        browser = "yandex"
    elif "edg/" in ua:
        browser = "edge"
    elif "firefox" in ua:
        browser = "firefox"
    elif "chrome" in ua:
        browser = "chrome"
    elif "safari" in ua:
        browser = "safari"
    else:
        browser = "other"

    if "android" in ua:
        system = "android"
    elif "iphone" in ua or "ipad" in ua:
        system = "ios"
    elif "windows" in ua:
        system = "windows"
    elif "macintosh" in ua:
        system = "macos"
    elif "linux" in ua:
        system = "linux"
    else:
        system = "other"
    return browser, system


def most_common(ev, column):
    counts = ev.groupby(["cookie_id", column]).size().rename("count").reset_index()
    counts = counts.sort_values(["cookie_id", "count", column], ascending=[True, False, True])
    return counts.drop_duplicates("cookie_id").set_index("cookie_id")[column]


def build_features(events, meta):
    """Возвращает таблицу, группы признаков и счётчики очистки. Target не нужен."""
    ev, raw, audit = prepare_events(events, meta)
    index = pd.Index(meta.cookie_id, name="cookie_id")
    g = ev.groupby("cookie_id", sort=False)
    n = g.size().reindex(index, fill_value=0)
    parts = {}

    # Отдельно сохраняем два признака ровно как в официальном quickstart:
    # его baseline не удаляет дубликаты перед агрегацией.
    raw_g = raw.groupby("cookie_id")
    parts["baseline_raw"] = pd.DataFrame({
        "raw_n_events": raw_g.size(),
        "raw_item_nunique": raw_g.item_id.nunique(),
    }).reindex(index).fillna(0)
    parts["basic"] = pd.DataFrame({
        "n_events": n,
        "item_nunique": g.item_id.nunique().reindex(index, fill_value=0),
    }, index=index)
    metadata = meta.set_index("cookie_id")
    parts["metadata"] = pd.DataFrame({
        "cookie_age_days": (
            metadata.window_end_ts - metadata.cookie_created_at
        ).dt.total_seconds() / 86400,
    }, index=index)

    counts = pd.crosstab(ev.cookie_id, ev.event_name).reindex(
        index=index, columns=EVENT_NAMES, fill_value=0,
    )
    shares = counts.div(n.replace(0, np.nan), axis=0)
    behavior = pd.concat([counts.add_prefix("count_"), shares.add_prefix("share_")], axis=1)
    behavior["event_types"] = (counts > 0).sum(axis=1)
    behavior["event_entropy"] = -(shares * np.log2(shares.replace(0, np.nan))).sum(axis=1)
    contacts = counts[["contact_phone_show", "contact_chat_open", "contact_message_sent"]].sum(axis=1)
    behavior["contacts_per_item_view"] = contacts / counts.item_view.replace(0, np.nan)
    behavior["photos_per_item_view"] = counts.photo_swipe / counts.item_view.replace(0, np.nan)
    behavior["items_per_search"] = counts.item_view / counts.search_results_view.replace(0, np.nan)
    parts["events"] = behavior

    # Нулевые интервалы сохраняются: несколько действий могут иметь одну секунду.
    ev["gap_seconds"] = g.event_ts.diff().dt.total_seconds()
    ev["hour"] = ev.event_ts.dt.hour
    ev["minute"] = ev.event_ts.dt.floor("min")
    ev["session_start"] = ev.gap_seconds.isna() | ev.gap_seconds.gt(1800)
    ev["session"] = ev.groupby("cookie_id").session_start.cumsum()
    g = ev.groupby("cookie_id", sort=False)
    timing = pd.DataFrame(index=index)
    timing["active_seconds"] = (g.event_ts.max() - g.event_ts.min()).dt.total_seconds()
    timing["events_per_active_minute"] = n / (timing.active_seconds / 60 + 1)
    timing["first_event_hour"] = (g.event_ts.min() - metadata.window_start_ts).dt.total_seconds() / 3600
    timing["last_event_hour"] = (g.event_ts.max() - metadata.window_start_ts).dt.total_seconds() / 3600
    gaps = g.gap_seconds
    for stat in ["mean", "std", "min", "max", "median"]:
        timing[f"gap_{stat}"] = gaps.agg(stat)
    for quantile in [0.1, 0.25, 0.75, 0.9]:
        timing[f"gap_q{int(quantile * 100)}"] = gaps.quantile(quantile)
    timing["gap_cv"] = timing.gap_std / timing.gap_mean.replace(0, np.nan)
    timing["gap_burstiness"] = (timing.gap_std - timing.gap_mean) / (
        timing.gap_std + timing.gap_mean
    ).replace(0, np.nan)
    for seconds in [0, 1, 5, 10, 60, 300, 1800]:
        numerator = ev.gap_seconds.le(seconds).groupby(ev.cookie_id).sum()
        timing[f"gap_le_{seconds}_share"] = numerator / gaps.count().replace(0, np.nan)
    timing["timestamp_unique_share"] = g.event_ts.nunique() / n.replace(0, np.nan)
    timing["n_sessions"] = g.session.max().reindex(index, fill_value=0)
    sessions = ev.groupby(["cookie_id", "session"]).agg(
        size=("event_ts", "size"), first=("event_ts", "min"), last=("event_ts", "max"),
    )
    timing["session_size_max"] = sessions["size"].groupby(level=0).max()
    timing["session_size_mean"] = sessions["size"].groupby(level=0).mean()
    session_seconds = (sessions["last"] - sessions["first"]).dt.total_seconds()
    timing["session_seconds_max"] = session_seconds.groupby(level=0).max()
    hour_counts = ev.groupby(["cookie_id", "hour"]).size()
    hour_shares = hour_counts / hour_counts.groupby(level=0).transform("sum")
    timing["active_hours"] = g.hour.nunique()
    timing["hour_top_share"] = hour_shares.groupby(level=0).max()
    timing["hour_entropy"] = (-hour_shares * np.log2(hour_shares)).groupby(level=0).sum()
    timing["night_share"] = ev.hour.lt(6).groupby(ev.cookie_id).mean()
    minute_counts = ev.groupby(["cookie_id", "minute"]).size()
    timing["minute_count_max"] = minute_counts.groupby(level=0).max()
    parts["timing"] = timing

    # Общий разброс интервалов смешивает паузы между сессиями и ритм внутри них.
    # Кроме того, загрузка фото не должна скрывать ритм просмотра объявлений.
    rhythm = pd.DataFrame(index=index)
    within_session = ev.gap_seconds.where(ev.gap_seconds.le(1800))
    session_gaps = within_session.groupby(ev.cookie_id)
    rhythm["within_session_gap_mean"] = session_gaps.mean()
    rhythm["within_session_gap_cv"] = session_gaps.std() / session_gaps.mean().replace(0, np.nan)
    rhythm["within_session_gap_q90_q50"] = session_gaps.quantile(0.9) / session_gaps.median().replace(0, np.nan)
    gap_counts = ev.groupby(["cookie_id", "gap_seconds"]).size()
    rhythm["gap_mode_share"] = gap_counts.groupby(level=0).max() / gaps.count().replace(0, np.nan)
    rhythm["gap_unique_share"] = gaps.nunique() / gaps.count().replace(0, np.nan)
    previous_gap = g.gap_seconds.shift()
    comparable_gaps = ev.gap_seconds.notna() & previous_gap.notna()
    rhythm["same_consecutive_gap_share"] = ev.gap_seconds.eq(previous_gap).where(
        comparable_gaps
    ).groupby(ev.cookie_id).mean()
    rhythm["successive_gap_difference_median"] = (
        ev.gap_seconds - previous_gap
    ).abs().groupby(ev.cookie_id).median()
    for event_name in ["item_view", "search_results_view", "photo_swipe"]:
        selected = ev.loc[ev.event_name.eq(event_name)]
        selected_gaps = selected.groupby("cookie_id").event_ts.diff().dt.total_seconds()
        selected_group = selected_gaps.groupby(selected.cookie_id)
        rhythm[f"{event_name}_gap_median"] = selected_group.median()
        rhythm[f"{event_name}_gap_cv"] = selected_group.std() / selected_group.mean().replace(0, np.nan)
        rhythm[f"{event_name}_gap_fast_share"] = selected_gaps.le(10).where(
            selected_gaps.notna()
        ).groupby(selected.cookie_id).mean()
    parts["rhythm"] = rhythm

    diversity_parts = []
    for column in ["item_id", "item_category", "item_location", "search_query"]:
        summary = category_summary(ev, column, index)
        # item_nunique уже входит в basic.
        if column == "item_id":
            summary = summary.drop(columns="item_id_unique")
        diversity_parts.append(summary)
    diversity = pd.concat(diversity_parts, axis=1)
    diversity["seller_pro_share"] = ev.seller_type.eq("pro").where(
        ev.seller_type.notna()
    ).groupby(ev.cookie_id).mean()
    parts["diversity"] = diversity

    search = ev.loc[ev.event_name.eq("search_results_view")].copy()
    sg = search.groupby("cookie_id")
    search_features = pd.DataFrame(index=index)
    for stat in ["max", "mean", "std", "nunique"]:
        search_features[f"search_page_{stat}"] = sg.search_page.agg(stat)
    search_features["search_deep_share"] = search.search_page.ge(5).where(
        search.search_page.notna()
    ).groupby(search.cookie_id).mean()
    page_delta = sg.search_page.diff()
    # Последовательное листание имеет смысл только для того же запроса и
    # однозначного порядка поисковых событий.
    single_time = ~search.duplicated(["cookie_id", "event_ts"], keep=False)
    previous_single = single_time.groupby(search.cookie_id).shift(fill_value=False)
    search_gap = sg.event_ts.diff().dt.total_seconds()
    valid_search_pair = (
        search.search_query.eq(sg.search_query.shift())
        & search_gap.gt(0) & search_gap.le(1800) & single_time & previous_single
    )
    for name, condition in [("next_page", page_delta.eq(1)), ("same_page", page_delta.eq(0))]:
        search_features[f"search_{name}_share"] = condition.where(
            valid_search_pair
        ).groupby(search.cookie_id).mean()
    parts["search"] = search_features

    families = {ua: ua_family(ua) for ua in ev.user_agent.dropna().unique()}
    ev["browser"] = ev.user_agent.map({key: value[0] for key, value in families.items()}).fillna("other")
    ev["system"] = ev.user_agent.map({key: value[1] for key, value in families.items()}).fillna("other")
    technical = pd.DataFrame(index=index)
    for column in ["platform", "browser", "system"]:
        technical[f"{column}_main"] = most_common(ev, column).reindex(index).fillna("unknown")
        technical[f"{column}_nunique"] = ev.groupby("cookie_id")[column].nunique().reindex(index, fill_value=0)
    technical["ua_nunique"] = ev.groupby("cookie_id").user_agent.nunique().reindex(index, fill_value=0)
    technical["automation_ua_share"] = ev.browser.isin(["headless", "http_client"]).groupby(ev.cookie_id).mean()
    for platform in ["web", "android", "ios"]:
        technical[f"platform_{platform}_share"] = ev.platform.eq(platform).groupby(ev.cookie_id).mean()
    parts["technical"] = technical

    pointer = ev.loc[ev.pointer_x.notna() & ev.pointer_y.notna()].copy()
    pg = pointer.groupby("cookie_id")
    pointer_features = pd.DataFrame(index=index)
    pointer_features["pointer_share"] = pg.size().reindex(index, fill_value=0) / n.replace(0, np.nan)
    unique_points = pointer.drop_duplicates(["cookie_id", "pointer_x", "pointer_y"]).groupby("cookie_id").size()
    pointer_features["pointer_unique_share"] = unique_points / pg.size()
    for coordinate in ["pointer_x", "pointer_y"]:
        pointer_features[f"{coordinate}_std"] = pg[coordinate].std()
        pointer_features[f"{coordinate}_range"] = pg[coordinate].max() - pg[coordinate].min()
    single_pointer_time = ~pointer.duplicated(["cookie_id", "event_ts"], keep=False)
    valid_pointer_pair = single_pointer_time & single_pointer_time.groupby(
        pointer.cookie_id
    ).shift(fill_value=False) & pg.event_ts.diff().dt.total_seconds().between(1, 1800)
    distance = np.hypot(pg.pointer_x.diff(), pg.pointer_y.diff()).where(valid_pointer_pair)
    pointer_features["pointer_distance_mean"] = distance.groupby(pointer.cookie_id).mean()
    pointer_features["pointer_distance_cv"] = distance.groupby(pointer.cookie_id).std() / (
        distance.groupby(pointer.cookie_id).mean().replace(0, np.nan)
    )
    pointer_features["pointer_stationary_share"] = distance.eq(0).where(
        distance.notna()
    ).groupby(pointer.cookie_id).mean()
    parts["pointer"] = pointer_features

    # Матрица переходов проверяется отдельным экспериментом: 100 дополнительных
    # признаков оправданы только при улучшении на более поздних датах.
    single_time = ~ev.duplicated(["cookie_id", "event_ts"], keep=False)
    previous_single = single_time.groupby(ev.cookie_id).shift(fill_value=False)
    valid_pair = ev.gap_seconds.gt(0) & ev.gap_seconds.le(1800) & single_time & previous_single
    previous_event = ev.groupby("cookie_id").event_name.shift()
    pair_names = previous_event + "__" + ev.event_name
    pair_columns = [first + "__" + second for first in EVENT_NAMES for second in EVENT_NAMES]
    pair_counts = pd.crosstab(ev.loc[valid_pair, "cookie_id"], pair_names.loc[valid_pair]).reindex(
        index=index, columns=pair_columns, fill_value=0,
    )
    pair_shares = pair_counts.div(pair_counts.sum(axis=1).replace(0, np.nan), axis=0)
    parts["transitions"] = pair_shares.add_prefix("transition_")

    features = pd.concat(parts.values(), axis=1).reindex(index)
    categorical = ["platform_main", "browser_main", "system_main"]
    numeric = features.columns.difference(categorical)
    features[numeric] = features[numeric].astype(float).replace([np.inf, -np.inf], np.nan)
    groups = {name: list(frame.columns) for name, frame in parts.items()}
    return features, groups, audit
