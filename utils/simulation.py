import os
from pathlib import Path

import numpy as np


INCOME_SIMULATION = "realistic"
INCOME_TRAJECTORY_CACHE = {}
INCOME_MARKET_CURVE_CACHE = {}
FD_LAST_FULL_ITEM_CACHE = {"key": None, "value": None}
ONLINE_CLICK_BATCH_SIZE = 20


def _canonical_income_mode(mode):
    mode = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "criteo": "criteo_cva",
        "criteo_cva": "criteo_cva",
        "cva": "criteo_cva",
        "yoochoose": "yoochoose_rpv",
        "yoochoose_rpv": "yoochoose_rpv",
        "rpv": "yoochoose_rpv",
    }
    return aliases.get(mode, mode)


def set_income_simulation(mode):
    """
    Select the unit-income simulator used by f_d.
    """
    global INCOME_SIMULATION
    mode = _canonical_income_mode(mode)
    valid_modes = {"realistic", "constant", "legacy", "criteo_cva", "yoochoose_rpv"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown income simulation: {mode}")
    INCOME_SIMULATION = mode
    FD_LAST_FULL_ITEM_CACHE["key"] = None
    FD_LAST_FULL_ITEM_CACHE["value"] = None


def _full_item_cache_key(time, total_steps, peaks, query_id):
    if np.ndim(time) != 0:
        return None
    query_key = 0.0 if query_id is None else float(query_id)
    return (
        INCOME_SIMULATION,
        float(time),
        float(total_steps),
        float(peaks),
        query_key,
    )


def _is_full_item_range(item_ids):
    if item_ids is None:
        return False
    item_ids = np.asarray(item_ids)
    if item_ids.ndim != 1:
        return False
    return np.array_equal(item_ids, np.arange(item_ids.shape[0]))


def _lookup_full_item_cache(cache_key, item_ids):
    if cache_key is None or FD_LAST_FULL_ITEM_CACHE["key"] != cache_key:
        return None
    cached = FD_LAST_FULL_ITEM_CACHE["value"]
    if cached is None:
        return None
    item_ids = np.asarray(item_ids)
    if item_ids.ndim != 1 or item_ids.size == 0:
        return None
    int_item_ids = item_ids.astype(int, copy=False)
    if not np.array_equal(item_ids, int_item_ids):
        return None
    if int_item_ids.min() < 0 or int_item_ids.max() >= cached.shape[0]:
        return None
    return cached[int_item_ids]


def _store_full_item_cache(cache_key, item_ids, values):
    if cache_key is None or not _is_full_item_range(item_ids):
        return values
    FD_LAST_FULL_ITEM_CACHE["key"] = cache_key
    FD_LAST_FULL_ITEM_CACHE["value"] = np.asarray(values, dtype=float)
    return values


def sample_queryFromdata(data, query_rng, only_test=False):
    """
    This function samples a query from train/validation/test or only test.
    """
    if only_test:
        dataSplits = [data.test]
    else:
        dataSplits = [data.train, data.validation, data.test]
    splitId = sample_splitId(dataSplits, query_rng)
    Queryid = sample_Queryid(dataSplits[splitId], query_rng)
    return Queryid, dataSplits[splitId]


def sample_splitId(dataSplits, query_rng):
    """
    This function samples a split according to the number of available queries.
    """
    n_queries = [len(dataSplit.queriesList) for dataSplit in dataSplits]
    total_n_query = sum(n_queries)
    query_ratio = np.array(n_queries) / total_n_query

    splitId = query_rng.choice(len(dataSplits), size=1, p=query_ratio)
    return int(splitId)


def sample_Queryid(dataSplit, query_rng):
    """
    This function samples a query id from a data split.
    """
    Queryid = query_rng.choice(dataSplit.queriesList, size=1)[0]
    return Queryid


def getpositionBias(cutoff, positionBiasSeverity):
    """
    This function returns the position bias of each rank.
    """
    return (1 / np.log2(2 + np.arange(cutoff))) ** positionBiasSeverity


def generateClick(ranking, TrueRel, positionBias, RandomNumberGenerator):
    """
    This function generates clicks according to relevance and position bias.
    """
    RankedRel = TrueRel[ranking]
    rng = RandomNumberGenerator if RandomNumberGenerator is not None else np.random.default_rng()
    if ONLINE_CLICK_BATCH_SIZE > 1:
        click_prob = np.clip(RankedRel * positionBias[:len(RankedRel)], 0.0, 1.0)
        return rng.binomial(ONLINE_CLICK_BATCH_SIZE, click_prob) / float(ONLINE_CLICK_BATCH_SIZE)
    rand_var = rng.random(len(RankedRel))
    rand_prop = rng.random(len(positionBias))
    viewed = rand_prop < positionBias
    clicks = np.logical_and(rand_var < RankedRel, viewed)
    return clicks


def _gaussian(x, center, width):
    return np.exp(-0.5 * ((x - center) / width) ** 2)


def _hash01(values, salt=0.0):
    """
    Deterministic pseudo-random numbers in [0, 1).

    This keeps the simulation reproducible without depending on global RNG state.
    """
    values = np.asarray(values, dtype=float)
    hashed = np.sin(values * 12.9898 + salt * 78.233) * 43758.5453
    return hashed - np.floor(hashed)


def _project_root():
    return Path(__file__).resolve().parents[1]


def _income_bank_dir():
    configured = os.environ.get("DIDRF_INCOME_BANK_DIR")
    if configured:
        return Path(configured)
    return _project_root() / "data" / "income_trajectories"


def income_trajectory_bank_path(mode=None):
    mode = INCOME_SIMULATION if mode is None else _canonical_income_mode(mode)
    return _income_bank_dir() / f"{mode}.npz"


def income_simulation_source(mode=None):
    mode = INCOME_SIMULATION if mode is None else _canonical_income_mode(mode)
    if mode in {"criteo_cva", "yoochoose_rpv"}:
        bank_path = income_trajectory_bank_path(mode)
        if bank_path.exists():
            return f"trajectory_bank:{bank_path.name}"
        return f"deterministic_fallback:{mode}"
    return f"built_in:{mode}"


def _normalize_trajectory_bank(values, lower=0.05, upper=1.0):
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, np.inf)
    if values.size == 0:
        return values

    positive = values[values > 0]
    if positive.size == 0:
        return np.full_like(values, lower, dtype=np.float64)

    # If a bank already looks normalized, keep its relative scale.
    if float(np.min(values)) >= lower - 1e-12 and float(np.max(values)) <= upper + 1e-12:
        return np.clip(values, lower, upper)

    transformed = np.log1p(values)
    lo = np.percentile(transformed, 1.0)
    hi = np.percentile(transformed, 99.0)
    if hi <= lo + 1e-12:
        return np.full_like(values, (lower + upper) / 2.0, dtype=np.float64)
    transformed = np.clip(transformed, lo, hi)
    normalized = (transformed - lo) / (hi - lo)
    return lower + (upper - lower) * normalized


def _load_income_trajectory_bank(mode):
    """
    Load a real-data-calibrated trajectory bank when available.

    Expected NPZ schema:
      trajectories: array of shape [n_entities, n_time_bins]

    The optional calibration script writes this schema. When the file is absent,
    simulation falls back to deterministic semi-synthetic trajectories that
    follow the same scenario assumptions.
    """
    mode = _canonical_income_mode(mode)
    if mode in INCOME_TRAJECTORY_CACHE:
        return INCOME_TRAJECTORY_CACHE[mode]

    bank_path = _income_bank_dir() / f"{mode}.npz"
    if not bank_path.exists():
        INCOME_TRAJECTORY_CACHE[mode] = None
        return None

    with np.load(bank_path, allow_pickle=False) as data:
        if "trajectories" in data:
            trajectories = data["trajectories"]
        elif "mu" in data:
            trajectories = data["mu"]
        else:
            raise ValueError(
                f"Income trajectory bank {bank_path.name} must contain 'trajectories' or 'mu'."
            )

    trajectories = _normalize_trajectory_bank(trajectories)
    if trajectories.ndim != 2:
        raise ValueError(f"Income trajectory bank {bank_path.name} must be a 2D array.")
    if trajectories.shape[0] == 0 or trajectories.shape[1] == 0:
        raise ValueError(f"Income trajectory bank {bank_path.name} is empty.")

    INCOME_TRAJECTORY_CACHE[mode] = trajectories
    return trajectories


def _trajectory_indices(item_ids, query_id, n_entities):
    item_ids = np.asarray(item_ids, dtype=float)
    query_id = 0.0 if query_id is None else float(query_id)
    identity = item_ids + 1009.0 * query_id
    indices = np.floor(_hash01(identity, salt=9.17) * n_entities).astype(int)
    return np.clip(indices, 0, n_entities - 1)


def _income_from_trajectory_bank(mode, time, item_ids=None, query_id=0):
    bank = _load_income_trajectory_bank(mode)
    if bank is None:
        return None

    scalar_input = np.isscalar(time)
    time_array = np.asarray(time, dtype=float)
    n_bins = bank.shape[1]
    bin_ids = np.mod(np.floor(time_array).astype(int), n_bins)

    if item_ids is None:
        market_curve = INCOME_MARKET_CURVE_CACHE.get(mode)
        if market_curve is None:
            market_curve = bank.mean(axis=0)
            INCOME_MARKET_CURVE_CACHE[mode] = market_curve
        values = market_curve[bin_ids]
        if scalar_input:
            return float(values)
        return values

    item_ids = np.asarray(item_ids, dtype=float)
    entity_ids = _trajectory_indices(item_ids, query_id, bank.shape[0])
    if np.ndim(bin_ids) == 0:
        return bank[entity_ids, int(bin_ids)]

    if bin_ids.shape == item_ids.shape:
        return bank[entity_ids, bin_ids]

    return bank[entity_ids, np.expand_dims(bin_ids, axis=-1)]


def _normalize_generated_income(values, scale, lower=0.05, upper=1.0, power=1.0):
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, np.inf)
    normalized = values / (values + scale)
    if abs(power - 1.0) > 1e-12:
        normalized = np.power(np.clip(normalized, 0.0, 1.0), power)
    return lower + (upper - lower) * np.clip(normalized, 0.0, 1.0)


def _global_income_curve(time, total_steps=10_000, days=100):
    """
    Market-level unit-income intensity.

    The curve follows standard demand-forecasting components: intraday shape,
    weekly seasonality, trend, calendar/promotion pulses, and smooth regime drift.
    """
    time = np.asarray(time, dtype=float)
    days = max(float(days), 1.0)
    events_per_day = max(float(total_steps) / days, 1.0)
    day = time / events_per_day
    day_phase = np.mod(day, 1.0)
    week_phase = np.mod(day, 7.0) / 7.0

    intraday = (
        0.48
        + 0.12 * _gaussian(day_phase, 0.10, 0.09)
        + 0.22 * _gaussian(day_phase, 0.45, 0.16)
        + 0.34 * _gaussian(day_phase, 0.78, 0.13)
    )

    weekly = (
        1.00
        + 0.10 * np.sin(2.0 * np.pi * (week_phase - 0.18))
        + 0.08 * _gaussian(week_phase, 5.0 / 7.0, 0.16)
        - 0.05 * _gaussian(week_phase, 1.0 / 7.0, 0.14)
    )

    trend = (
        0.90
        + 0.16 / (1.0 + np.exp(-(day - 18.0) / 7.5))
        - 0.10 / (1.0 + np.exp(-(day - 76.0) / 9.0))
    )

    event = np.ones_like(day, dtype=float)
    for center, amplitude, width in (
        (18.0, 0.22, 2.0),
        (44.0, 0.42, 3.2),
        (71.0, 0.30, 2.6),
        (88.0, 0.55, 1.6),
    ):
        event += amplitude * _gaussian(day, center, width)
        event -= 0.08 * amplitude * _gaussian(day, center + 3.0, width * 1.35)

    regime = (
        1.00
        - 0.11 / (1.0 + np.exp(-(day - 60.0) / 3.0))
        + 0.07 / (1.0 + np.exp(-(day - 83.0) / 4.5))
    )

    low_frequency = (
        1.00
        + 0.030 * np.sin(2.0 * np.pi * day / 3.7 + 0.4)
        + 0.018 * np.sin(2.0 * np.pi * day / 11.3 + 1.7)
    )

    income = intraday * weekly * trend * event * regime * low_frequency
    return np.clip(income, 0.04, 1.60)


def _criteo_campaign_curve(time, total_steps=10_000, item_ids=None, query_id=0):
    """
    Criteo-CVA fallback simulator.

    It mimics campaign-level conversion value per impression: sparse campaign
    bursts, budget pacing, weekday effects, and delayed attribution. When an
    external Criteo-calibrated trajectory bank exists, f_d uses that bank
    instead of this deterministic fallback.
    """
    scalar_input = np.isscalar(time)
    time = np.asarray(time, dtype=float)
    days = 30.0
    events_per_day = max(float(total_steps) / days, 1.0)
    day = time / events_per_day
    day_phase = np.mod(day, 1.0)
    week_phase = np.mod(day, 7.0) / 7.0

    intraday = (
        0.58
        + 0.18 * _gaussian(day_phase, 0.36, 0.16)
        + 0.30 * _gaussian(day_phase, 0.78, 0.12)
    )
    weekday = (
        0.94
        + 0.16 * _gaussian(week_phase, 2.5 / 7.0, 0.18)
        + 0.09 * _gaussian(week_phase, 4.5 / 7.0, 0.15)
        - 0.06 * _gaussian(week_phase, 6.0 / 7.0, 0.18)
    )
    budget = (
        1.0
        + 0.28 * _gaussian(day, 6.0, 1.5)
        + 0.78 * _gaussian(day, 16.0, 2.4)
        + 0.52 * _gaussian(day, 25.0, 1.9)
    )
    market = intraday * weekday * budget

    if item_ids is None:
        values = _normalize_generated_income(market, scale=1.20, lower=0.03, upper=2.20, power=1.08)
        if scalar_input:
            return float(values)
        return values

    item_ids = np.asarray(item_ids, dtype=float)
    query_id = 0.0 if query_id is None else float(query_id)

    if np.ndim(time) == 0:
        item_day = day
        base = market
    else:
        if time.shape != item_ids.shape:
            item_day = np.expand_dims(day, axis=-1)
            base = np.expand_dims(market, axis=-1)
        else:
            item_day = day
            base = market

    identity = item_ids + 1009.0 * query_id
    h0 = _hash01(identity, salt=2.01)
    h1 = _hash01(identity, salt=2.37)
    h2 = _hash01(identity, salt=2.71)
    h3 = _hash01(identity, salt=3.13)
    h4 = _hash01(identity, salt=3.59)

    campaign_scale = 0.16 + 2.35 * (-np.log(np.clip(1.0 - h0, 1e-6, 1.0)))
    campaign_scale = np.clip(campaign_scale, 0.10, 8.50)

    launch_day = 1.0 + 27.0 * h1
    launch_width = 0.55 + 2.10 * h2
    launch_amp = 0.35 + 2.25 * h3
    burst = 1.0 + launch_amp * _gaussian(item_day, launch_day, launch_width)

    second_active = h4 > 0.38
    second_day = 3.0 + 24.0 * _hash01(identity, salt=4.11)
    second_width = 0.45 + 1.35 * _hash01(identity, salt=4.47)
    second_amp = (0.25 + 1.55 * _hash01(identity, salt=4.83)) * second_active
    burst += second_amp * _gaussian(item_day, second_day, second_width)

    # Conversions are often attributed after exposure. The shifted component
    # produces smoother delayed utility without adding state to the simulator.
    delayed = (
        0.26 * launch_amp * _gaussian(item_day, launch_day + 1.15, launch_width * 1.35)
        + 0.16 * second_amp * _gaussian(item_day, second_day + 1.30, second_width * 1.45)
    )

    fatigue_start = launch_day + 3.5 + 3.0 * h2
    fatigue = 1.0 - (0.05 + 0.18 * h4) / (
        1.0 + np.exp(-(item_day - fatigue_start) / 1.6)
    )
    retargeting = (
        1.0
        + (0.08 + 0.18 * h3)
        * np.sin(2.0 * np.pi * np.mod(item_day, 1.0) + 2.0 * np.pi * h1)
    )

    raw = base * campaign_scale * (burst + delayed) * fatigue * retargeting
    return _normalize_generated_income(raw, scale=1.55, lower=0.02, upper=2.80, power=1.12)


def _yoochoose_revenue_curve(time, total_steps=10_000, item_ids=None, query_id=0):
    """
    YOOCHOOSE-RPV fallback simulator.

    It mimics item-level revenue per view/click in e-commerce sessions:
    category-level demand, price heterogeneity, weekly/session patterns,
    product lifecycle, and sparse purchase-value smoothing.
    """
    scalar_input = np.isscalar(time)
    time = np.asarray(time, dtype=float)
    days = 180.0
    events_per_day = max(float(total_steps) / days, 1.0)
    day = time / events_per_day
    day_phase = np.mod(day, 1.0)
    week_phase = np.mod(day, 7.0) / 7.0

    session_shape = (
        0.62
        + 0.16 * _gaussian(day_phase, 0.12, 0.10)
        + 0.20 * _gaussian(day_phase, 0.50, 0.18)
        + 0.30 * _gaussian(day_phase, 0.82, 0.12)
    )
    retail_calendar = (
        1.0
        + 0.12 * _gaussian(week_phase, 5.5 / 7.0, 0.20)
        - 0.04 * _gaussian(week_phase, 1.0 / 7.0, 0.17)
    )
    promotion = (
        1.0
        + 0.36 * _gaussian(day, 28.0, 3.8)
        + 0.72 * _gaussian(day, 83.0, 5.5)
        + 0.50 * _gaussian(day, 132.0, 4.6)
        + 0.95 * _gaussian(day, 166.0, 3.2)
    )
    market = session_shape * retail_calendar * promotion

    if item_ids is None:
        values = _normalize_generated_income(market, scale=1.25, lower=0.03, upper=2.20, power=1.08)
        if scalar_input:
            return float(values)
        return values

    item_ids = np.asarray(item_ids, dtype=float)
    query_id = 0.0 if query_id is None else float(query_id)

    if np.ndim(time) == 0:
        item_day = day
        base = market
    else:
        if time.shape != item_ids.shape:
            item_day = np.expand_dims(day, axis=-1)
            base = np.expand_dims(market, axis=-1)
        else:
            item_day = day
            base = market

    identity = item_ids + 1009.0 * query_id
    h0 = _hash01(identity, salt=5.01)
    h1 = _hash01(identity, salt=5.37)
    h2 = _hash01(identity, salt=5.71)
    h3 = _hash01(identity, salt=6.13)
    h4 = _hash01(identity, salt=6.59)
    h5 = _hash01(identity, salt=7.07)

    category = np.floor(5.0 * h0).astype(int)
    price_scale = 0.18 + 2.85 * (-np.log(np.clip(1.0 - h1, 1e-6, 1.0)))
    price_scale = np.clip(price_scale, 0.12, 10.50)

    business_peak = _gaussian(week_phase, 2.5 / 7.0, 0.18)
    weekend_peak = _gaussian(week_phase, 5.8 / 7.0, 0.20)
    evening_peak = _gaussian(day_phase, 0.80, 0.13)
    noon_peak = _gaussian(day_phase, 0.48, 0.16)
    smooth_wave = np.sin(2.0 * np.pi * week_phase + 2.0 * np.pi * h2)

    category_effect = np.where(
        category == 0,
        1.00 + 0.34 * weekend_peak + 0.12 * evening_peak,
        np.where(
            category == 1,
            1.00 + 0.30 * business_peak + 0.10 * noon_peak,
            np.where(
                category == 2,
                1.00 + 0.22 * weekend_peak + 0.14 * smooth_wave,
                np.where(
                    category == 3,
                    1.00 + 0.24 * _gaussian(week_phase, 4.6 / 7.0, 0.22),
                    1.00 + 0.16 * smooth_wave,
                ),
            ),
        ),
    )

    lifecycle_center = 24.0 + 132.0 * h2
    lifecycle_width = 12.0 + 26.0 * h3
    lifecycle = 0.58 + 0.92 * _gaussian(item_day, lifecycle_center, lifecycle_width)

    restock_period = 21.0 + 24.0 * h4
    restock_phase = 2.0 * np.pi * h5
    inventory = 0.76 + 0.34 * np.maximum(
        0.0,
        np.sin(2.0 * np.pi * item_day / restock_period + restock_phase),
    )

    sparse_purchase_smoothing = 0.74 + 0.34 * h3
    raw = base * price_scale * category_effect * lifecycle * inventory * sparse_purchase_smoothing
    return _normalize_generated_income(raw, scale=1.70, lower=0.02, upper=2.80, power=1.10)


def f_d(time, total_steps=10_000, peaks=100, item_ids=None, query_id=0, use_cache=True):
    """
    Contextual unit-income signal mu_d(t).

    When item_ids is omitted, this returns a market-level scalar/vector f_d(t).
    When item_ids is provided, it returns item-level unit income with persistent
    item heterogeneity, item-specific temporal preference, event sensitivity, and
    inventory-like dips. The argument name peaks is kept for backward
    compatibility; it is interpreted as the number of days in the simulated
    horizon.
    """
    scalar_input = np.isscalar(time)
    time = np.asarray(time, dtype=float)
    item_ids_for_cache = None
    cache_key = None
    if use_cache and item_ids is not None:
        item_ids_for_cache = np.asarray(item_ids)
        cache_key = _full_item_cache_key(time, total_steps, peaks, query_id)
        cached = _lookup_full_item_cache(cache_key, item_ids_for_cache)
        if cached is not None:
            return cached

    if INCOME_SIMULATION == "constant":
        if item_ids is not None:
            item_ids = np.asarray(item_ids)
            return _store_full_item_cache(
                cache_key,
                item_ids_for_cache,
                np.ones_like(item_ids, dtype=float),
            )
        if scalar_input:
            return 1.0
        return np.ones_like(time, dtype=float)

    if INCOME_SIMULATION == "legacy":
        legacy_income = f_d_legacy(time, total_steps=total_steps, peaks=peaks)
        if item_ids is not None:
            item_ids = np.asarray(item_ids)
            return _store_full_item_cache(
                cache_key,
                item_ids_for_cache,
                np.ones_like(item_ids, dtype=float) * legacy_income,
            )
        if scalar_input:
            return float(legacy_income)
        return legacy_income

    if INCOME_SIMULATION in {"criteo_cva", "yoochoose_rpv"}:
        bank_income = _income_from_trajectory_bank(
            INCOME_SIMULATION,
            time,
            item_ids=item_ids,
            query_id=query_id,
        )
        if bank_income is not None:
            return _store_full_item_cache(cache_key, item_ids_for_cache, bank_income)
        if INCOME_SIMULATION == "criteo_cva":
            return _store_full_item_cache(
                cache_key,
                item_ids_for_cache,
                _criteo_campaign_curve(
                    time,
                    total_steps=total_steps,
                    item_ids=item_ids,
                    query_id=query_id,
                ),
            )
        return _store_full_item_cache(
            cache_key,
            item_ids_for_cache,
            _yoochoose_revenue_curve(
                time,
                total_steps=total_steps,
                item_ids=item_ids,
                query_id=query_id,
            ),
        )

    days = max(float(peaks), 1.0)
    base_income = _global_income_curve(time, total_steps=total_steps, days=days)

    if item_ids is None:
        if scalar_input:
            return float(base_income)
        return base_income

    item_ids = np.asarray(item_ids, dtype=float)
    query_id = 0.0 if query_id is None else float(query_id)

    if np.ndim(time) == 0:
        day = time / max(float(total_steps) / days, 1.0)
        base = base_income
    else:
        day = time
        if time.shape != item_ids.shape:
            day = np.expand_dims(day, axis=-1)
            base = np.expand_dims(base_income, axis=-1)
        else:
            base = base_income
        day = day / max(float(total_steps) / days, 1.0)

    identity = item_ids + 1009.0 * query_id
    h0 = _hash01(identity, salt=0.11)
    h1 = _hash01(identity, salt=0.37)
    h2 = _hash01(identity, salt=0.71)
    h3 = _hash01(identity, salt=1.13)

    item_scale = 0.72 + 0.58 * h0
    item_phase = 2.0 * np.pi * h1
    segment = np.floor(4.0 * h2).astype(int)

    week_phase = np.mod(day, 7.0) / 7.0
    business_peak = _gaussian(week_phase, 2.5 / 7.0, 0.18)
    weekend_peak = _gaussian(week_phase, 5.8 / 7.0, 0.20)
    balanced_wave = np.sin(2.0 * np.pi * week_phase + item_phase)
    event_sensitive = _gaussian(week_phase, 4.7 / 7.0, 0.22)

    item_weekly = np.where(
        segment == 0,
        1.00 + 0.13 * business_peak - 0.05 * weekend_peak,
        np.where(
            segment == 1,
            1.00 + 0.15 * weekend_peak - 0.04 * business_peak,
            np.where(
                segment == 2,
                1.00 + 0.08 * balanced_wave,
                1.00 + 0.11 * event_sensitive,
            ),
        ),
    )

    lifecycle_period = 24.0 + 18.0 * h3
    lifecycle = 1.00 + 0.075 * np.sin(2.0 * np.pi * day / lifecycle_period + item_phase)

    stockout_day = 22.0 + 62.0 * h1
    stockout_width = 1.2 + 1.8 * h3
    availability = 1.00 - (0.06 + 0.10 * h0) * _gaussian(day, stockout_day, stockout_width)

    promotion_sensitivity = 0.82 + 0.42 * h2
    item_income = base * item_scale * item_weekly * lifecycle * availability * promotion_sensitivity
    return _store_full_item_cache(
        cache_key,
        item_ids_for_cache,
        np.clip(item_income, 0.03, 1.80),
    )


def f_d_legacy(time, total_steps=10_000, peaks=100):
    """
    Original triangular-wave income simulator kept for ablation.
    """
    time = np.asarray(time, dtype=float)
    period = total_steps / peaks
    half_period = period / 2.0
    pos_in_cycle = np.mod(time, period)
    income = np.where(pos_in_cycle < half_period, pos_in_cycle / half_period, 0.0)
    return income
