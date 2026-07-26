"""Extract tsfresh and catch22 features for the S&P 500 OHLCV series."""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from pycatch22 import catch22_all
from tsfresh.feature_extraction import EfficientFCParameters, extract_features

warnings.filterwarnings("ignore")

TARGET_COLS = ["open", "high", "low", "close", "volume"]
TEST_HORIZON = 14
PROGRESS_EVERY = 25

KEY_COLS = ["Name", "Type"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract tsfresh and catch22 features for the S&P 500 OHLCV series."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join("data", "snp500"),
        help="Directory containing snp500_data.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("outputs", "snp500"),
        help="Directory shared with grid_search/: snp500_series_to_id.pkl is "
        "read from here, and the tsfresh/catch22 feature pickles are written "
        "here for baselines/ and itsop/ to consume.",
    )
    return parser.parse_args()


def extract_tsfresh_features(values):
    frame = pd.DataFrame(
        {
            "id": 0,
            "time": np.arange(len(values)),
            "value": values.astype(float),
        }
    )
    features = extract_features(
        frame,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=EfficientFCParameters(),
        disable_progressbar=True,
    )
    return features.iloc[0].to_dict()


def extract_catch22_features(values):
    result = catch22_all(values.astype(float))
    return dict(zip(result["names"], result["values"]))


def training_values(values, test_horizon):
    if len(values) <= test_horizon:
        return None

    split_index = len(values) - test_horizon
    if split_index <= 1 or split_index >= len(values):
        return None

    return values[:split_index]


def load_data(data_csv_path):
    data = pd.read_csv(data_csv_path)

    try:
        data["date"] = pd.to_datetime(
            data["date"], format="%m/%d/%Y", errors="raise"
        )
    except Exception:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

    for column in TARGET_COLS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data.dropna(subset=["date", "Name"] + TARGET_COLS).copy()


def prepare_series_map(series_map, series_map_path):
    required_columns = {"series_id", "Name", "Type"}
    missing_columns = required_columns - set(series_map.columns)
    if missing_columns:
        raise ValueError(
            f"{series_map_path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    prepared = series_map.dropna(subset=required_columns).copy()
    prepared["series_id"] = pd.to_numeric(
        prepared["series_id"], errors="raise"
    ).astype(int)
    prepared["Name"] = prepared["Name"].astype(str)
    prepared["Type"] = prepared["Type"].astype(str)

    duplicated_ids = prepared["series_id"].duplicated(keep=False)
    if duplicated_ids.any():
        examples = prepared.loc[duplicated_ids, "series_id"].tolist()[:10]
        raise ValueError(f"Duplicate series_id values found: {examples}")

    duplicated_keys = prepared.duplicated(subset=KEY_COLS, keep=False)
    if duplicated_keys.any():
        examples = (
            prepared.loc[duplicated_keys, KEY_COLS]
            .drop_duplicates()
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(f"Duplicate (Name, Type) mappings found: {examples}")

    unknown_types = sorted(set(prepared["Type"]) - set(TARGET_COLS))
    if unknown_types:
        raise ValueError(f"Unknown OHLCV series types: {unknown_types}")

    return prepared.sort_values("series_id").reset_index(drop=True)


def build_metadata_features(series_map):
    metadata = series_map[KEY_COLS].copy()
    type_one_hot = pd.get_dummies(
        metadata["Type"],
        prefix="type",
        prefix_sep="__",
        dtype=np.float32,
    )
    features = pd.concat(
        [metadata.reset_index(drop=True), type_one_hot.reset_index(drop=True)],
        axis=1,
    )
    return features.set_index(KEY_COLS, drop=False)


def print_progress(processed, total, successful, skipped, start_time):
    if processed % PROGRESS_EVERY != 0 and processed != total:
        return

    elapsed = time.time() - start_time
    print(
        f"[PROGRESS] {processed}/{total} processed | "
        f"successful={successful} | skipped={skipped} | elapsed={elapsed:.1f}s"
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data_csv_path = os.path.join(args.data_dir, "snp500_data.csv")
    series_map_path = os.path.join(args.output_dir, "snp500_series_to_id.pkl")
    output_tsfresh_pkl = os.path.join(args.output_dir, "snp500_feature_extraction_tsfresh.pkl")
    output_catch22_pkl = os.path.join(args.output_dir, "snp500_feature_extraction_catch22.pkl")

    start_time = time.time()

    data = load_data(data_csv_path)
    series_map = prepare_series_map(pd.read_pickle(series_map_path), series_map_path)
    metadata_lookup = build_metadata_features(series_map)
    data_by_name = {
        str(name): group.sort_values("date").copy()
        for name, group in data.groupby("Name", sort=False)
    }

    tsfresh_rows = []
    catch22_rows = []
    successful = 0
    skipped = 0
    total_series = len(series_map)

    print(f"[INFO] Found {total_series} mapped S&P 500 series.")
    print(f"[INFO] Series types: {sorted(series_map['Type'].unique().tolist())}")

    for index, mapping in enumerate(series_map.itertuples(index=False), start=1):
        name = str(mapping.Name)
        series_type = str(mapping.Type)
        key = (name, series_type)

        if name not in data_by_name:
            skipped += 1
            print(
                f"[SKIP] series_id={mapping.series_id} | Name={name} | "
                f"Type={series_type} | reason=stock not found"
            )
            print_progress(index, total_series, successful, skipped, start_time)
            continue

        values = data_by_name[name][series_type].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            skipped += 1
            invalid_count = int((~np.isfinite(values)).sum())
            print(
                f"[SKIP] series_id={mapping.series_id} | Name={name} | "
                f"Type={series_type} | reason={invalid_count} non-finite values"
            )
            print_progress(index, total_series, successful, skipped, start_time)
            continue

        train_values = training_values(values, TEST_HORIZON)
        if train_values is None:
            skipped += 1
            print_progress(index, total_series, successful, skipped, start_time)
            continue

        metadata_row = metadata_lookup.loc[key]
        base_values = {"Name": name, "Type": series_type}
        metadata_values = {
            column: metadata_row[column]
            for column in metadata_lookup.columns
            if column not in KEY_COLS
        }

        tsfresh_rows.append(
            {
                **base_values,
                **metadata_values,
                **extract_tsfresh_features(train_values),
            }
        )
        catch22_rows.append(
            {
                **base_values,
                **metadata_values,
                **extract_catch22_features(train_values),
            }
        )

        successful += 1
        print_progress(index, total_series, successful, skipped, start_time)

    tsfresh_features = pd.DataFrame(tsfresh_rows)
    catch22_features = pd.DataFrame(catch22_rows)

    if tsfresh_features.empty:
        raise RuntimeError("No tsfresh feature rows were created.")
    if catch22_features.empty:
        raise RuntimeError("No catch22 feature rows were created.")

    tsfresh_features.to_pickle(output_tsfresh_pkl)
    catch22_features.to_pickle(output_catch22_pkl)

    print(f"Saved {output_tsfresh_pkl} | shape={tsfresh_features.shape}")
    print(f"Saved {output_catch22_pkl} | shape={catch22_features.shape}")


if __name__ == "__main__":
    main()
