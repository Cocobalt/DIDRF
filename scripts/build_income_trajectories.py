"""
Build real-data-calibrated unit-income trajectory banks for DIDRF.

Outputs NPZ files with:
  trajectories: [n_entities, n_time_bins], normalized to [0.05, 1.0]
  entity_ids: entity identifiers as strings

The runtime simulator automatically loads:
  data/income_trajectories/criteo_cva.npz
  data/income_trajectories/yoochoose_rpv.npz
when those files exist.
"""

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK_DIR = PROJECT_ROOT / "data" / "income_trajectories"


CRITEO_COLUMNS = [
    "timestamp",
    "uid",
    "campaign",
    "conversion",
    "conversion_timestamp",
    "conversion_id",
    "attribution",
    "click",
    "cost",
    "cpo",
    "cat1",
    "cat2",
    "cat3",
    "cat4",
    "cat5",
    "cat6",
    "cat7",
    "cat8",
    "cat9",
]

YOO_CLICK_COLUMNS = ["session_id", "timestamp", "item_id", "category"]
YOO_BUY_COLUMNS = ["session_id", "timestamp", "item_id", "price", "quantity"]


def normalize_bank(values, lower=0.05, upper=1.0):
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, np.inf)
    if values.size == 0:
        return values
    transformed = np.log1p(values)
    lo = np.percentile(transformed, 1.0)
    hi = np.percentile(transformed, 99.0)
    if hi <= lo + 1e-12:
        return np.full_like(values, (lower + upper) / 2.0, dtype=np.float64)
    transformed = np.clip(transformed, lo, hi)
    normalized = (transformed - lo) / (hi - lo)
    return lower + (upper - lower) * normalized


def write_bank(output, trajectories, entity_ids, metadata):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        trajectories=trajectories.astype(np.float32),
        entity_ids=np.asarray(entity_ids, dtype=str),
    )
    metadata_path = output.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"saved trajectory bank: {output.name}")
    print(f"saved metadata: {metadata_path.name}")


def iter_csv(path, sep, names, has_header, chunksize):
    import pandas as pd

    sep = sep.encode("utf-8").decode("unicode_escape")
    header = 0 if has_header else None
    read_names = None if has_header else names
    return pd.read_csv(
        path,
        sep=sep,
        names=read_names,
        header=header,
        compression="infer",
        chunksize=chunksize,
        low_memory=False,
    )


def numeric_min_timestamp(path, sep, names, has_header, chunksize):
    min_ts = None
    for chunk in iter_csv(path, sep, names, has_header, chunksize):
        ts = np.asarray(chunk["timestamp"], dtype=np.float64)
        current = float(np.nanmin(ts))
        min_ts = current if min_ts is None else min(min_ts, current)
    return min_ts


def top_entities(exposure, max_entities):
    totals = {}
    for (entity, _), value in exposure.items():
        totals[entity] = totals.get(entity, 0.0) + float(value)
    ordered = sorted(totals.items(), key=lambda item: (-item[1], str(item[0])))
    if max_entities is not None:
        ordered = ordered[:max_entities]
    return [entity for entity, _ in ordered]


def dense_from_aggregates(exposure, outcome, entities, n_bins, tau):
    entity_to_row = {entity: row for row, entity in enumerate(entities)}
    raw = np.zeros((len(entities), n_bins), dtype=np.float64)
    counts = np.zeros((len(entities), n_bins), dtype=np.float64)
    global_y = np.zeros(n_bins, dtype=np.float64)
    global_n = np.zeros(n_bins, dtype=np.float64)

    for (entity, bin_id), value in exposure.items():
        if bin_id < 0 or bin_id >= n_bins:
            continue
        global_n[bin_id] += float(value)
        row = entity_to_row.get(entity)
        if row is not None:
            counts[row, bin_id] += float(value)

    for (entity, bin_id), value in outcome.items():
        if bin_id < 0 or bin_id >= n_bins:
            continue
        global_y[bin_id] += float(value)
        row = entity_to_row.get(entity)
        if row is not None:
            raw[row, bin_id] += float(value)

    global_curve = global_y / np.clip(global_n, 1e-12, np.inf)
    raw_rate = raw / np.clip(counts, 1e-12, np.inf)
    weights = counts / np.clip(counts + tau, 1e-12, np.inf)
    smoothed = weights * raw_rate + (1.0 - weights) * global_curve[None, :]
    return normalize_bank(smoothed), counts


def build_criteo(args):
    min_ts = numeric_min_timestamp(
        args.input,
        args.sep,
        CRITEO_COLUMNS,
        args.has_header,
        args.chunksize,
    )
    bin_seconds = args.bin_hours * 3600.0
    exposure = {}
    outcome = {}
    max_bin = 0

    for chunk in iter_csv(args.input, args.sep, CRITEO_COLUMNS, args.has_header, args.chunksize):
        timestamp = np.asarray(chunk["timestamp"], dtype=np.float64)
        bin_ids = np.floor((timestamp - min_ts) / bin_seconds).astype(int)
        campaigns = chunk["campaign"].astype(str)
        if args.value == "attribution_cpo":
            values = (
                np.asarray(chunk["attribution"], dtype=np.float64)
                * np.asarray(chunk["cpo"], dtype=np.float64)
            )
        elif args.value == "conversion_cpo":
            values = (
                np.asarray(chunk["conversion"], dtype=np.float64)
                * np.asarray(chunk["cpo"], dtype=np.float64)
            )
        else:
            values = np.asarray(chunk[args.value], dtype=np.float64)

        for campaign, bin_id, value in zip(campaigns, bin_ids, values):
            if bin_id < 0:
                continue
            key = (campaign, int(bin_id))
            exposure[key] = exposure.get(key, 0.0) + 1.0
            outcome[key] = outcome.get(key, 0.0) + float(value)
            max_bin = max(max_bin, int(bin_id))

    entities = top_entities(exposure, args.max_entities)
    trajectories, counts = dense_from_aggregates(
        exposure,
        outcome,
        entities,
        max_bin + 1,
        args.tau,
    )
    metadata = {
        "scenario": "criteo_cva",
        "source": Path(args.input).name,
        "bin_hours": args.bin_hours,
        "value": args.value,
        "tau": args.tau,
        "n_entities": len(entities),
        "n_bins": int(max_bin + 1),
        "total_exposure_selected": float(counts.sum()),
    }
    write_bank(args.output, trajectories, entities, metadata)


def build_yoochoose(args):
    import pandas as pd

    min_ts = None
    for chunk in iter_csv(args.clicks, args.sep, YOO_CLICK_COLUMNS, args.has_header, args.chunksize):
        ts = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
        current = ts.min()
        if pd.notna(current):
            min_ts = current if min_ts is None else min(min_ts, current)
    if min_ts is None:
        raise ValueError("Could not parse timestamps from YOOCHOOSE clicks.")

    exposure = {}
    item_category = {}
    max_bin = 0
    bin_seconds = args.bin_hours * 3600.0

    for chunk in iter_csv(args.clicks, args.sep, YOO_CLICK_COLUMNS, args.has_header, args.chunksize):
        ts = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
        bin_ids = np.floor((ts - min_ts).dt.total_seconds() / bin_seconds).astype("Int64")
        items = chunk["item_id"].astype(str)
        categories = chunk["category"].astype(str)
        for item, category, bin_id in zip(items, categories, bin_ids):
            if pd.isna(bin_id) or int(bin_id) < 0:
                continue
            key = (item, int(bin_id))
            exposure[key] = exposure.get(key, 0.0) + 1.0
            item_category.setdefault(item, category)
            max_bin = max(max_bin, int(bin_id))

    outcome = {}
    for chunk in iter_csv(args.buys, args.sep, YOO_BUY_COLUMNS, args.has_header, args.chunksize):
        ts = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
        bin_ids = np.floor((ts - min_ts).dt.total_seconds() / bin_seconds).astype("Int64")
        items = chunk["item_id"].astype(str)
        revenue = np.asarray(chunk["price"], dtype=np.float64) * np.asarray(chunk["quantity"], dtype=np.float64)
        for item, bin_id, value in zip(items, bin_ids, revenue):
            if pd.isna(bin_id) or int(bin_id) < 0:
                continue
            key = (item, int(bin_id))
            outcome[key] = outcome.get(key, 0.0) + float(value)
            max_bin = max(max_bin, int(bin_id))

    entities = top_entities(exposure, args.max_entities)
    trajectories, counts = dense_from_aggregates(
        exposure,
        outcome,
        entities,
        max_bin + 1,
        args.tau,
    )
    metadata = {
        "scenario": "yoochoose_rpv",
        "clicks": Path(args.clicks).name,
        "buys": Path(args.buys).name,
        "bin_hours": args.bin_hours,
        "tau": args.tau,
        "n_entities": len(entities),
        "n_bins": int(max_bin + 1),
        "total_exposure_selected": float(counts.sum()),
        "note": "Uses item click/view as exposure proxy and direct item purchase revenue.",
    }
    write_bank(args.output, trajectories, entities, metadata)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="scenario", required=True)

    criteo = subparsers.add_parser("criteo_cva")
    criteo.add_argument("--input", type=Path, required=True)
    criteo.add_argument("--output", type=Path, default=DEFAULT_BANK_DIR / "criteo_cva.npz")
    criteo.add_argument("--sep", default=",")
    criteo.add_argument("--has_header", action="store_true")
    criteo.add_argument("--chunksize", type=int, default=500_000)
    criteo.add_argument("--bin_hours", type=float, default=12.0)
    criteo.add_argument("--tau", type=float, default=1000.0)
    criteo.add_argument("--max_entities", type=int, default=None)
    criteo.add_argument(
        "--value",
        choices=["attribution_cpo", "conversion_cpo", "cost", "cpo"],
        default="attribution_cpo",
    )
    criteo.set_defaults(func=build_criteo)

    yoo = subparsers.add_parser("yoochoose_rpv")
    yoo.add_argument("--clicks", type=Path, required=True)
    yoo.add_argument("--buys", type=Path, required=True)
    yoo.add_argument("--output", type=Path, default=DEFAULT_BANK_DIR / "yoochoose_rpv.npz")
    yoo.add_argument("--sep", default=",")
    yoo.add_argument("--has_header", action="store_true")
    yoo.add_argument("--chunksize", type=int, default=500_000)
    yoo.add_argument("--bin_hours", type=float, default=24.0)
    yoo.add_argument("--tau", type=float, default=50.0)
    yoo.add_argument("--max_entities", type=int, default=50_000)
    yoo.set_defaults(func=build_yoochoose)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
