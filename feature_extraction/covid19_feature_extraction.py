"""Extract tsfresh and catch22 features for the COVID-19 time series."""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from pycatch22 import catch22_all
from tsfresh.feature_extraction import EfficientFCParameters, extract_features

warnings.filterwarnings("ignore")

TEST_HORIZON = 14
PROGRESS_EVERY = 25

KEY_COLS = ["country", "type"]
NON_DATE_COLS = {"country", "type", "Lat", "Long"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract tsfresh and catch22 features for the COVID-19 time series."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join("data", "covid19"),
        help="Directory containing covid19_dataset.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("outputs", "covid19"),
        help="Directory shared with grid_search/: covid19_series_to_id.pkl is "
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


def get_sorted_date_columns(data):
    date_mapping = {}

    for column in data.columns:
        if column in NON_DATE_COLS:
            continue

        parsed_date = pd.to_datetime(str(column), format="%m/%d/%Y", errors="coerce")
        if pd.isna(parsed_date):
            parsed_date = pd.to_datetime(str(column), format="%m/%d/%y", errors="coerce")
        if pd.isna(parsed_date):
            parsed_date = pd.to_datetime(str(column), errors="coerce")

        if not pd.isna(parsed_date):
            date_mapping[column] = parsed_date

    date_columns = sorted(date_mapping, key=date_mapping.get)
    if not date_columns:
        raise RuntimeError("No date columns found in covid19_dataset.pkl")

    return date_columns


def prepare_series_map(series_map, series_map_path):
    required_columns = {"series_id", "country", "type"}
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
    prepared["country"] = prepared["country"].astype(str)
    prepared["type"] = prepared["type"].astype(str)

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
        raise ValueError(f"Duplicate (country, type) mappings found: {examples}")

    return prepared.sort_values("series_id").reset_index(drop=True)


def prepare_data(data, date_columns, data_pkl_path):
    missing_columns = set(KEY_COLS) - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"{data_pkl_path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    prepared = data.dropna(subset=KEY_COLS).copy()
    prepared["country"] = prepared["country"].astype(str)
    prepared["type"] = prepared["type"].astype(str)

    for column in date_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    duplicated_keys = prepared.duplicated(subset=KEY_COLS, keep=False)
    if duplicated_keys.any():
        examples = (
            prepared.loc[duplicated_keys, KEY_COLS]
            .drop_duplicates()
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{data_pkl_path} contains multiple rows for the same "
            f"(country, type): {examples}"
        )

    return prepared.set_index(KEY_COLS, drop=False)


def build_metadata_features(series_map):
    metadata = series_map[KEY_COLS].copy()
    country_one_hot = pd.get_dummies(
        metadata["country"],
        prefix="country",
        prefix_sep="__",
        dtype=np.float32,
    )
    type_one_hot = pd.get_dummies(
        metadata["type"],
        prefix="type",
        prefix_sep="__",
        dtype=np.float32,
    )

    features = pd.concat(
        [
            metadata.reset_index(drop=True),
            country_one_hot.reset_index(drop=True),
            type_one_hot.reset_index(drop=True),
        ],
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

    data_pkl_path = os.path.join(args.data_dir, "covid19_dataset.pkl")
    series_map_path = os.path.join(args.output_dir, "covid19_series_to_id.pkl")
    output_tsfresh_pkl = os.path.join(args.output_dir, "covid19_feature_extraction_tsfresh.pkl")
    output_catch22_pkl = os.path.join(args.output_dir, "covid19_feature_extraction_catch22.pkl")

    start_time = time.time()

    data = pd.read_pickle(data_pkl_path).copy()
    series_map = pd.read_pickle(series_map_path).copy()

    date_columns = get_sorted_date_columns(data)
    data_lookup = prepare_data(data, date_columns, data_pkl_path)
    series_map = prepare_series_map(series_map, series_map_path)
    metadata_lookup = build_metadata_features(series_map)

    tsfresh_rows = []
    catch22_rows = []
    successful = 0
    skipped = 0
    total_series = len(series_map)

    print(f"[INFO] Found {total_series} mapped COVID-19 series.")
    print(f"[INFO] Found {len(date_columns)} chronological date columns.")

    for index, mapping in enumerate(series_map.itertuples(index=False), start=1):
        country = str(mapping.country)
        series_type = str(mapping.type)
        key = (country, series_type)

        if key not in data_lookup.index:
            skipped += 1
            print(
                f"[SKIP] series_id={mapping.series_id} | country={country} | "
                f"type={series_type} | reason=series not found"
            )
            print_progress(index, total_series, successful, skipped, start_time)
            continue

        values = data_lookup.loc[key, date_columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            skipped += 1
            invalid_count = int((~np.isfinite(values)).sum())
            print(
                f"[SKIP] series_id={mapping.series_id} | country={country} | "
                f"type={series_type} | reason={invalid_count} non-finite values"
            )
            print_progress(index, total_series, successful, skipped, start_time)
            continue

        train_values = training_values(values, TEST_HORIZON)
        if train_values is None:
            skipped += 1
            print_progress(index, total_series, successful, skipped, start_time)
            continue

        metadata_row = metadata_lookup.loc[key]
        base_values = {"country": country, "type": series_type}
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
