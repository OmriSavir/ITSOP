"""Extract tsfresh and catch22 features for the electricity-consumption series."""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from pycatch22 import catch22_all
from tsfresh.feature_extraction import EfficientFCParameters, extract_features

warnings.filterwarnings("ignore")

TARGET_COL = "consumption"
TIMESTAMP_COL = "ts"
DATA_ID_COL = "id"
MATRIX_ID_COL = "building_id"

CATEGORICAL_FEATURES = ["neighborhood", "socio-economic_cluster"]
NUMERIC_FEATURES = ["number_of_meters", "area", "floors"]

TEST_HORIZON = 48
PROGRESS_EVERY = 25


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract tsfresh and catch22 features for the electricity-consumption series."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join("data", "electricity"),
        help="Directory containing consumption_dataset.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("outputs", "electricity"),
        help="Directory shared with grid_search/: electricity_smape_matrix.pkl "
        "is read from here to obtain the building IDs in scope, and the "
        "tsfresh/catch22 feature pickles are written here for baselines/ and "
        "itsop/ to consume. NOTE: this is a different, train-only feature set "
        "than the whole-portfolio catch22 file (with a city_Electricity "
        "indicator column) that grid_search/electricity_grid_search.py reads "
        "as an input via --features-dir — keep that file in a separate "
        "location so this script does not overwrite it.",
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


def load_series_ids(series_matrix_path):
    matrix = pd.read_pickle(series_matrix_path)
    if MATRIX_ID_COL not in matrix.columns:
        raise ValueError(
            f"{series_matrix_path} is missing required column: {MATRIX_ID_COL}"
        )

    series_ids = matrix[MATRIX_ID_COL].dropna().astype(str)
    if series_ids.duplicated().any():
        examples = series_ids[series_ids.duplicated(keep=False)].tolist()[:10]
        raise ValueError(f"Duplicate building IDs found in {series_matrix_path}: {examples}")

    return sorted(
        series_ids.tolist(),
        key=lambda value: int(value) if value.isdigit() else value,
    )


def load_data(data_pkl_path):
    data = pd.read_pickle(data_pkl_path).copy()
    required_columns = {
        DATA_ID_COL,
        TIMESTAMP_COL,
        TARGET_COL,
        *CATEGORICAL_FEATURES,
        *NUMERIC_FEATURES,
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"{data_pkl_path} is missing required columns: {sorted(missing_columns)}"
        )

    data[TIMESTAMP_COL] = pd.to_datetime(data[TIMESTAMP_COL], errors="coerce")
    data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
    data[DATA_ID_COL] = data[DATA_ID_COL].astype(str)

    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=[DATA_ID_COL, TIMESTAMP_COL, TARGET_COL]).copy()
    return data


def get_static_value(group, column, series_id):
    values = group[column].dropna().drop_duplicates()
    if len(values) != 1:
        raise ValueError(
            f"Series id={series_id} must have exactly one value for '{column}', "
            f"found {len(values)}"
        )
    return values.iloc[0]


def build_metadata_features(data_by_id, series_ids):
    rows = []

    for series_id in series_ids:
        if series_id not in data_by_id:
            continue

        group = data_by_id[series_id]
        row = {"id": series_id}

        for column in CATEGORICAL_FEATURES + NUMERIC_FEATURES:
            row[column] = get_static_value(group, column, series_id)

        rows.append(row)

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise RuntimeError("No electricity metadata rows were created.")

    for column in CATEGORICAL_FEATURES:
        metadata[column] = metadata[column].astype(str)

    categorical_parts = [
        pd.get_dummies(
            metadata[column],
            prefix=column,
            prefix_sep="__",
            dtype=np.float32,
        )
        for column in CATEGORICAL_FEATURES
    ]

    feature_table = pd.concat(
        [
            metadata[["id"] + NUMERIC_FEATURES].reset_index(drop=True),
            *[part.reset_index(drop=True) for part in categorical_parts],
        ],
        axis=1,
    )
    return feature_table.set_index("id", drop=False)


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

    data_pkl_path = os.path.join(args.data_dir, "consumption_dataset.pkl")
    series_matrix_path = os.path.join(args.output_dir, "electricity_smape_matrix.pkl")
    output_tsfresh_pkl = os.path.join(args.output_dir, "electricity_feature_extraction_tsfresh.pkl")
    output_catch22_pkl = os.path.join(args.output_dir, "electricity_feature_extraction_catch22.pkl")

    start_time = time.time()

    series_ids = load_series_ids(series_matrix_path)
    data = load_data(data_pkl_path)
    data_by_id = {
        str(series_id): group.sort_values(TIMESTAMP_COL).copy()
        for series_id, group in data.groupby(DATA_ID_COL, sort=False)
    }
    metadata_lookup = build_metadata_features(data_by_id, series_ids)

    tsfresh_rows = []
    catch22_rows = []
    successful = 0
    skipped = 0
    total_series = len(series_ids)

    print(f"[INFO] Found {total_series} electricity series in {series_matrix_path}.")

    for index, series_id in enumerate(series_ids, start=1):
        if series_id not in data_by_id:
            skipped += 1
            print(f"[SKIP] id={series_id} | reason=series not found")
            print_progress(index, total_series, successful, skipped, start_time)
            continue

        values = data_by_id[series_id][TARGET_COL].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            skipped += 1
            invalid_count = int((~np.isfinite(values)).sum())
            print(
                f"[SKIP] id={series_id} | "
                f"reason={invalid_count} non-finite values"
            )
            print_progress(index, total_series, successful, skipped, start_time)
            continue

        train_values = training_values(values, TEST_HORIZON)
        if train_values is None:
            skipped += 1
            print_progress(index, total_series, successful, skipped, start_time)
            continue

        metadata_row = metadata_lookup.loc[series_id]
        metadata_values = {
            column: metadata_row[column]
            for column in metadata_lookup.columns
            if column != "id"
        }

        tsfresh_rows.append(
            {
                "id": series_id,
                **metadata_values,
                **extract_tsfresh_features(train_values),
            }
        )
        catch22_rows.append(
            {
                "id": series_id,
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
