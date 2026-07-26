"""AutoForecast baseline for forecasting-model selection."""

import argparse
import math
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg

warnings.filterwarnings("ignore")

DATASET = "covid19"  # options: "electricity", "snp500", "covid19"
MODEL_TYPE = "all"  # options: "prophet", "sarima", "lstm", "all"

FEATURE_NAMES = ["tsfresh_landmarker"]
METHODS = ["regression"]
ALGO_NAMES = ["linear_regression"]

DELTA = 0.05
ORDER_MULTIPLIERS = [0.2, 0.5, 1.0, 2.0, 5.0]
N_REPEATS = 10
RANDOM_SEED = 42

AR_K = 10
LANDMARKER_LOOKBACK = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoForecast baseline for forecasting-model selection."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing the dataset's raw file "
        "(consumption_dataset.pkl / snp500_data.csv / covid19_dataset.pkl). "
        "Defaults to ./data/<dataset>.",
    )
    parser.add_argument(
        "--features-dir",
        default=None,
        help="Directory containing the catch22/tsfresh feature-extraction "
        "pickle files produced by feature_extraction/. Defaults to "
        "--output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory shared with grid_search/ and test_forecasting/: "
        "validation matrices, combination maps, and test forecasts are read "
        "from here, and this script's own landmarker-feature cache is also "
        "written here. Defaults to ./outputs/<dataset>.",
    )
    return parser.parse_args()


def get_dataset_config(dataset, data_dir=None, output_dir=None, features_dir=None):
    """Return paths and evaluation settings for a dataset."""
    data_dir = data_dir or os.path.join("data", dataset)
    output_dir = output_dir or os.path.join("outputs", dataset)
    features_dir = features_dir or output_dir

    if dataset == "electricity":
        return {
            "dataset": dataset,
            "id_col": "building_id",
            "raw_data_path": os.path.join(data_dir, "consumption_dataset.pkl"),
            "series_map_path": None,
            "target_col": "consumption",
            "time_col": "ts",
            "test_horizon": 48,
            "seasonal_period": 168,
            "cache_dir": output_dir,
            "feature_files": {
                "catch22": os.path.join(features_dir, "electricity_feature_extraction_catch22.pkl"),
                "tsfresh": os.path.join(features_dir, "electricity_feature_extraction_tsfresh.pkl"),
            },
            "model_files": {
                "prophet": {
                    "smape": os.path.join(output_dir, "electricity_smape_matrix.pkl"),
                    "forecast": os.path.join(output_dir, "electricity_forecasting_test.pkl"),
                },
                "sarima": {
                    "smape": os.path.join(output_dir, "electricity_smape_matrix_sarima.pkl"),
                    "forecast": os.path.join(output_dir, "electricity_forecasting_test_sarima.pkl"),
                },
                "lstm": {
                    "smape": os.path.join(output_dir, "electricity_smape_matrix_lstm.pkl"),
                    "forecast": os.path.join(output_dir, "electricity_forecasting_test_lstm.pkl"),
                },
            },
        }

    if dataset == "snp500":
        return {
            "dataset": dataset,
            "id_col": "series_id",
            "raw_data_path": os.path.join(data_dir, "snp500_data.csv"),
            "series_map_path": os.path.join(output_dir, "snp500_series_to_id.pkl"),
            "target_cols": ["open", "high", "low", "close", "volume"],
            "test_horizon": 14,
            "seasonal_period": 5,
            "cache_dir": output_dir,
            "feature_files": {
                "catch22": os.path.join(features_dir, "snp500_feature_extraction_catch22.pkl"),
                "tsfresh": os.path.join(features_dir, "snp500_feature_extraction_tsfresh.pkl"),
            },
            "model_files": {
                "prophet": {
                    "smape": os.path.join(output_dir, "snp500_smape_matrix.pkl"),
                    "forecast": os.path.join(output_dir, "snp500_forecasting_test.pkl"),
                },
                "sarima": {
                    "smape": os.path.join(output_dir, "snp500_smape_matrix_sarima.pkl"),
                    "forecast": os.path.join(output_dir, "snp500_forecasting_test_sarima.pkl"),
                },
                "lstm": {
                    "smape": os.path.join(output_dir, "snp500_smape_matrix_lstm.pkl"),
                    "forecast": os.path.join(output_dir, "snp500_forecasting_test_lstm.pkl"),
                },
            },
        }

    if dataset == "covid19":
        return {
            "dataset": dataset,
            "id_col": "series_id",
            "raw_data_path": os.path.join(data_dir, "covid19_dataset.pkl"),
            "series_map_path": os.path.join(output_dir, "covid19_series_to_id.pkl"),
            "test_horizon": 14,
            "seasonal_period": 7,
            "cache_dir": output_dir,
            "feature_files": {
                "catch22": os.path.join(features_dir, "covid19_feature_extraction_catch22.pkl"),
                "tsfresh": os.path.join(features_dir, "covid19_feature_extraction_tsfresh.pkl"),
            },
            "model_files": {
                "prophet": {
                    "smape": os.path.join(output_dir, "covid19_smape_matrix.pkl"),
                    "forecast": os.path.join(output_dir, "covid19_forecasting_test.pkl"),
                },
                "sarima": {
                    "smape": os.path.join(output_dir, "covid19_smape_matrix_sarima.pkl"),
                    "forecast": os.path.join(output_dir, "covid19_forecasting_test_sarima.pkl"),
                },
                "lstm": {
                    "smape": os.path.join(output_dir, "covid19_smape_matrix_lstm.pkl"),
                    "forecast": os.path.join(output_dir, "covid19_forecasting_test_lstm.pkl"),
                },
            },
        }

    raise ValueError(f"Unknown DATASET: {dataset}")


_ARGS = parse_args()
CONFIG = get_dataset_config(DATASET, _ARGS.data_dir, _ARGS.output_dir, _ARGS.features_dir)
ID_COL = CONFIG["id_col"]
TEST_HORIZON = CONFIG["test_horizon"]
SEASONAL_PERIOD = CONFIG["seasonal_period"]


def seasonal_scale_q(y_train, m=SEASONAL_PERIOD, eps=1e-12):
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= m:
        return np.nan

    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if not np.isfinite(scale) or scale <= eps:
        return np.nan
    return float(scale)


def global_smape(y_true_all, y_pred_all):
    y_true_all = np.asarray(y_true_all, dtype=float)
    y_pred_all = np.asarray(y_pred_all, dtype=float)
    return 100.0 * np.mean(
        2.0
        * np.abs(y_true_all - y_pred_all)
        / (np.abs(y_true_all) + np.abs(y_pred_all) + 1e-9)
    )


def compute_order_sample_settings(n_series, n_combos):
    total_cells = int(n_series * n_combos)
    base_order = (n_series + n_combos) * (
        math.log(n_series + n_combos) + math.log(1.0 / DELTA)
    )

    settings = []
    for multiplier in ORDER_MULTIPLIERS:
        sample_cells = int(math.ceil(multiplier * base_order))
        matrix_percentage = (
            100.0 * sample_cells / total_cells if total_cells > 0 else np.nan
        )

        if sample_cells > total_cells:
            settings.append(
                {
                    "multiplier": float(multiplier),
                    "sample_cells": sample_cells,
                    "matrix_percentage": float(matrix_percentage),
                    "train_size": None,
                    "effective_cells": None,
                    "effective_percentage": None,
                    "skip": True,
                }
            )
            continue

        train_size = max(1, int(math.ceil(sample_cells / n_combos)))
        train_size = min(train_size, n_series - 1) if n_series > 1 else 1
        effective_cells = int(train_size * n_combos)
        effective_percentage = (
            100.0 * effective_cells / total_cells if total_cells > 0 else np.nan
        )

        settings.append(
            {
                "multiplier": float(multiplier),
                "sample_cells": sample_cells,
                "matrix_percentage": float(matrix_percentage),
                "train_size": int(train_size),
                "effective_cells": effective_cells,
                "effective_percentage": float(effective_percentage),
                "skip": False,
            }
        )

    return float(base_order), settings


def make_split(n, train_size, sample_cells, repeat_idx):
    train_size = max(1, min(int(train_size), n - 1))
    seed = (
        RANDOM_SEED
        + repeat_idx * 1_000_003
        + train_size * 10_007
        + int(sample_cells) * 101
    )
    rng = np.random.default_rng(seed)

    train_idx = rng.choice(np.arange(n), size=train_size, replace=False)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    test_idx = np.where(~train_mask)[0]
    return train_idx, test_idx


def get_covid_date_columns(data):
    non_date_cols = {"country", "type", "Lat", "Long"}
    date_lookup = {}

    for column in data.columns:
        if column in non_date_cols:
            continue

        parsed = pd.to_datetime(
            str(column), format="%m/%d/%Y", errors="coerce"
        )
        if pd.isna(parsed):
            parsed = pd.to_datetime(
                str(column), format="%m/%d/%y", errors="coerce"
            )
        if pd.isna(parsed):
            parsed = pd.to_datetime(str(column), errors="coerce")
        if not pd.isna(parsed):
            date_lookup[column] = parsed

    date_cols = sorted(date_lookup, key=date_lookup.get)
    if not date_cols:
        raise RuntimeError("No date columns found in covid19_dataset.pkl")
    return date_cols


def load_raw_data(config):
    dataset = config["dataset"]

    if dataset == "electricity":
        data = pd.read_pickle(config["raw_data_path"])
        if not np.issubdtype(data["ts"].dtype, np.datetime64):
            data["ts"] = pd.to_datetime(data["ts"])
        data[config["target_col"]] = pd.to_numeric(
            data[config["target_col"]], errors="coerce"
        )
        data = data.dropna(
            subset=["ts", config["target_col"], "id"]
        ).copy()
        data["id"] = data["id"].astype(str)
        return data

    if dataset == "snp500":
        data = pd.read_csv(config["raw_data_path"])
        try:
            data["date"] = pd.to_datetime(
                data["date"], format="%m/%d/%Y", errors="raise"
            )
        except Exception:
            data["date"] = pd.to_datetime(data["date"], errors="coerce")

        for column in config["target_cols"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")

        return data.dropna(
            subset=["date", "Name"] + config["target_cols"]
        ).copy()

    if dataset == "covid19":
        data = pd.read_pickle(config["raw_data_path"]).copy()
        date_cols = get_covid_date_columns(data)
        for column in date_cols:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(subset=["country", "type"]).copy()
        data["country"] = data["country"].astype(str)
        data["type"] = data["type"].astype(str)
        return data

    raise ValueError(dataset)


def load_series_map(config):
    if config["series_map_path"] is None:
        return None

    series_map = pd.read_pickle(config["series_map_path"]).copy()
    series_map[ID_COL] = series_map[ID_COL].astype(int).astype(str)

    if config["dataset"] == "covid19":
        required_cols = {"country", "type"}
        missing_cols = required_cols - set(series_map.columns)
        if missing_cols:
            raise ValueError(
                "covid19_series_to_id.pkl is missing required columns: "
                f"{sorted(missing_cols)}"
            )
        series_map["country"] = series_map["country"].astype(str)
        series_map["type"] = series_map["type"].astype(str)

    return series_map


def load_features(path, config, series_map):
    dataset = config["dataset"]
    features = pd.read_pickle(path).copy()

    if dataset == "electricity":
        feature_id_col = "id"
        if feature_id_col not in features.columns:
            raise ValueError(f"{path} must contain column {feature_id_col}")
        output = features.rename(columns={feature_id_col: ID_COL})
        output[ID_COL] = output[ID_COL].astype(str)
        drop_cols = [ID_COL]

    elif dataset == "snp500":
        if "Name" not in features.columns or "Type" not in features.columns:
            raise ValueError(f"{path} must contain columns Name and Type")
        output = series_map[[ID_COL, "Name", "Type"]].merge(
            features, on=["Name", "Type"], how="inner"
        )
        output[ID_COL] = output[ID_COL].astype(int).astype(str)
        drop_cols = [ID_COL, "Name", "Type"]

    elif dataset == "covid19":
        required_cols = {"country", "type"}
        missing_cols = required_cols - set(features.columns)
        if missing_cols:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing_cols)}"
            )
        output = series_map[[ID_COL, "country", "type"]].merge(
            features, on=["country", "type"], how="inner"
        )
        output[ID_COL] = output[ID_COL].astype(int).astype(str)
        drop_cols = [ID_COL, "country", "type"]

    else:
        raise ValueError(dataset)

    feature_cols = [column for column in output if column not in drop_cols]
    non_numeric = [
        column
        for column in feature_cols
        if not pd.api.types.is_numeric_dtype(output[column])
    ]
    if non_numeric:
        raise ValueError(
            f"{path} contains non-numeric feature columns: {non_numeric[:10]}"
        )

    output[feature_cols] = output[feature_cols].replace(
        [np.inf, -np.inf], np.nan
    )
    return output[[ID_COL] + feature_cols]


def make_combo_cols_int(data, id_col=ID_COL):
    output = data.copy()
    output.columns = [
        column if column == id_col else int(column)
        for column in output.columns
    ]
    output[id_col] = output[id_col].astype(str)
    return output


def load_single_model_matrices(model_type, config):
    files = config["model_files"][model_type]
    smape_matrix = make_combo_cols_int(pd.read_pickle(files["smape"]))
    forecasting_matrix = make_combo_cols_int(pd.read_pickle(files["forecast"]))
    return smape_matrix, forecasting_matrix


def load_all_model_matrices(config):
    smape_parts = []
    forecast_parts = []

    for model_name in ["prophet", "sarima", "lstm"]:
        smape_matrix, forecasting_matrix = load_single_model_matrices(
            model_name, config
        )
        smape_cols = [column for column in smape_matrix if column != ID_COL]
        forecast_cols = [
            column for column in forecasting_matrix if column != ID_COL
        ]
        smape_parts.append(
            smape_matrix.rename(
                columns={
                    column: f"{model_name}__{int(column)}"
                    for column in smape_cols
                }
            )
        )
        forecast_parts.append(
            forecasting_matrix.rename(
                columns={
                    column: f"{model_name}__{int(column)}"
                    for column in forecast_cols
                }
            )
        )

    common_ids = set(smape_parts[0][ID_COL])
    for part in smape_parts[1:]:
        common_ids &= set(part[ID_COL])
    for part in forecast_parts:
        common_ids &= set(part[ID_COL])

    common_ids = sorted(
        common_ids,
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )
    smape_output = pd.DataFrame({ID_COL: common_ids})
    forecast_output = pd.DataFrame({ID_COL: common_ids})

    for part in smape_parts:
        aligned = part.set_index(ID_COL).loc[common_ids].reset_index()
        smape_output = smape_output.merge(aligned, on=ID_COL, how="inner")
    for part in forecast_parts:
        aligned = part.set_index(ID_COL).loc[common_ids].reset_index()
        forecast_output = forecast_output.merge(aligned, on=ID_COL, how="inner")

    return smape_output, forecast_output


def prepare_y_matrix(smape_matrix, combo_cols):
    y_raw = smape_matrix[combo_cols].astype(float).to_numpy()
    y_raw[~np.isfinite(y_raw)] = np.nan

    valid_rows = np.isfinite(y_raw).any(axis=1)
    if not np.any(valid_rows):
        raise RuntimeError("No rows with at least one finite sMAPE value.")

    y_valid = y_raw[valid_rows]
    combo_array = np.asarray(combo_cols, dtype=object)
    y_class = np.asarray(
        [combo_array[int(np.nanargmin(row))] for row in y_valid],
        dtype=object,
    )

    y_frame = pd.DataFrame(y_valid, columns=combo_cols)
    global_median = np.nanmedian(y_valid)
    global_median = float(global_median) if np.isfinite(global_median) else 1e6
    medians = y_frame.median(axis=0, numeric_only=True).fillna(global_median)
    y_filled = (
        y_frame.fillna(medians)
        .fillna(global_median)
        .to_numpy(dtype=float)
    )
    return valid_rows, y_class, y_filled


def align_inputs(smape_matrix, forecasting_matrix, features):
    smape_matrix = smape_matrix.copy()
    forecasting_matrix = forecasting_matrix.copy()
    features = features.copy()

    for frame in [smape_matrix, forecasting_matrix, features]:
        frame[ID_COL] = frame[ID_COL].astype(str)

    common_ids = (
        set(smape_matrix[ID_COL])
        & set(forecasting_matrix[ID_COL])
        & set(features[ID_COL])
    )
    common_ids = sorted(
        common_ids,
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )

    smape_matrix = smape_matrix.set_index(ID_COL).loc[common_ids].reset_index()
    forecasting_matrix = (
        forecasting_matrix.set_index(ID_COL).loc[common_ids].reset_index()
    )
    features = features.set_index(ID_COL).loc[common_ids].reset_index()
    combo_cols = [column for column in smape_matrix if column != ID_COL]

    valid_rows, y_class, y_reg = prepare_y_matrix(smape_matrix, combo_cols)
    forecasting_matrix = forecasting_matrix.loc[valid_rows].reset_index(drop=True)
    features = features.loc[valid_rows].reset_index(drop=True)

    series_ids = features[ID_COL].astype(str).to_numpy()
    x_values = features.drop(columns=[ID_COL]).to_numpy(dtype=float)
    return (
        series_ids,
        x_values,
        y_class,
        y_reg,
        combo_cols,
        forecasting_matrix,
    )


def apply_knn_pca_if_needed(algo_name, x_train, x_test):
    if algo_name != "knn":
        return x_train, x_test

    n_components = int(round(np.sqrt(x_train.shape[1])))
    n_components = max(
        1,
        min(n_components, x_train.shape[0], x_train.shape[1]),
    )
    pca = PCA(n_components=n_components)
    return pca.fit_transform(x_train), pca.transform(x_test)


def impute_train_test_by_train_median(x_train, x_test):
    train_frame = pd.DataFrame(x_train)
    test_frame = pd.DataFrame(x_test)
    medians = train_frame.median(axis=0)

    all_nan_cols = medians[medians.isna()].index.tolist()
    if all_nan_cols:
        train_frame = train_frame.drop(columns=all_nan_cols)
        test_frame = test_frame.drop(columns=all_nan_cols)
        medians = medians.drop(index=all_nan_cols)

    train_frame = train_frame.fillna(medians)
    test_frame = test_frame.fillna(medians)
    return train_frame.to_numpy(dtype=float), test_frame.to_numpy(dtype=float)


def scale_train_test_by_train(x_train, x_test):
    scaler = StandardScaler()
    return scaler.fit_transform(x_train), scaler.transform(x_test)


def sanitize_train_test_for_float32(x_train, x_test):
    limit = np.finfo(np.float32).max / 10.0
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=limit, neginf=-limit)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=limit, neginf=-limit)
    return np.clip(x_train, -limit, limit), np.clip(x_test, -limit, limit)


def build_test_cache(data, series_map, series_ids, config):
    dataset = config["dataset"]
    cache = {}

    if dataset == "electricity":
        data_by_id = {
            str(series_id): frame.sort_values("ts").copy()
            for series_id, frame in data.groupby("id", sort=False)
        }
        for series_id in series_ids:
            series_id = str(series_id)
            if series_id not in data_by_id:
                continue
            series = data_by_id[series_id]
            if len(series) <= TEST_HORIZON:
                continue
            y_all = series[config["target_col"]].astype(float).to_numpy()
            split_index = len(y_all) - TEST_HORIZON
            y_train, y_test = y_all[:split_index], y_all[split_index:]
            scale = seasonal_scale_q(y_train)
            if np.isfinite(scale):
                cache[series_id] = (y_test, scale)
        return cache

    if dataset == "snp500":
        data_by_name = {
            str(name): frame.sort_values("date").copy()
            for name, frame in data.groupby("Name", sort=False)
        }
        map_lookup = series_map.copy()
        map_lookup[ID_COL] = map_lookup[ID_COL].astype(str)
        map_lookup = map_lookup.set_index(ID_COL)

        for series_id in series_ids:
            series_id = str(series_id)
            if series_id not in map_lookup.index:
                continue
            stock_name = str(map_lookup.loc[series_id, "Name"])
            target_type = map_lookup.loc[series_id, "Type"]
            if stock_name not in data_by_name:
                continue
            series = data_by_name[stock_name]
            if len(series) <= TEST_HORIZON:
                continue
            y_all = series[target_type].astype(float).to_numpy()
            split_index = len(y_all) - TEST_HORIZON
            y_train, y_test = y_all[:split_index], y_all[split_index:]
            scale = seasonal_scale_q(y_train)
            if np.isfinite(scale):
                cache[series_id] = (y_test, scale)
        return cache

    if dataset == "covid19":
        date_cols = get_covid_date_columns(data)
        data_lookup = (
            data.drop_duplicates(subset=["country", "type"], keep="last")
            .set_index(["country", "type"])
        )
        map_lookup = series_map.copy()
        map_lookup[ID_COL] = map_lookup[ID_COL].astype(str)
        map_lookup = map_lookup.set_index(ID_COL)

        for series_id in series_ids:
            series_id = str(series_id)
            if series_id not in map_lookup.index:
                continue
            key = (
                str(map_lookup.loc[series_id, "country"]),
                str(map_lookup.loc[series_id, "type"]),
            )
            if key not in data_lookup.index:
                continue
            data_row = data_lookup.loc[key]
            if isinstance(data_row, pd.DataFrame):
                data_row = data_row.iloc[0]
            y_all = pd.to_numeric(
                data_row[date_cols], errors="coerce"
            ).to_numpy(dtype=float)
            y_all = y_all[np.isfinite(y_all)]
            if len(y_all) <= TEST_HORIZON:
                continue
            split_index = len(y_all) - TEST_HORIZON
            y_train, y_test = y_all[:split_index], y_all[split_index:]
            scale = seasonal_scale_q(y_train)
            if np.isfinite(scale):
                cache[series_id] = (y_test, scale)
        return cache

    raise ValueError(dataset)


def get_forecast_value(forecasting_lookup, series_id, combo_label):
    possible_keys = [combo_label]
    try:
        possible_keys.append(int(combo_label))
    except Exception:
        pass
    possible_keys.append(str(combo_label))

    for key in possible_keys:
        if key in forecasting_lookup.columns:
            return forecasting_lookup.loc[series_id, key]
    raise KeyError(f"Forecast column not found for combo_label={combo_label}")


def evaluate_selected_combos(
    test_cache,
    series_ids_test,
    selected_combo_ids,
    forecasting_lookup,
):
    y_true_all = []
    y_pred_all = []
    absolute_scaled = []
    squared_scaled = []

    for series_id, combo_label in zip(series_ids_test, selected_combo_ids):
        series_id = str(series_id)
        if series_id not in test_cache or series_id not in forecasting_lookup.index:
            continue

        y_test, scale = test_cache[series_id]
        try:
            y_pred = get_forecast_value(
                forecasting_lookup, series_id, combo_label
            )
        except Exception:
            continue

        y_test = np.asarray(y_test, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_pred) != len(y_test):
            continue

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        absolute_scaled.append(np.abs(y_test - y_pred) / scale)
        squared_scaled.append(((y_test - y_pred) / scale) ** 2)

    if not y_true_all:
        return np.nan, np.nan, np.nan

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    absolute_scaled = np.concatenate(absolute_scaled)
    squared_scaled = np.concatenate(squared_scaled)
    return (
        global_smape(y_true_all, y_pred_all),
        float(np.mean(absolute_scaled)),
        float(np.sqrt(np.mean(squared_scaled))),
    )


def build_all_train_series(data, series_map, candidate_ids, config):
    dataset = config["dataset"]
    result = {}

    if dataset == "electricity":
        data_by_id = {
            str(series_id): frame.sort_values("ts").copy()
            for series_id, frame in data.groupby("id", sort=False)
        }
        for series_id in candidate_ids:
            series_id = str(series_id)
            if series_id not in data_by_id:
                continue
            series = data_by_id[series_id]
            if len(series) <= TEST_HORIZON:
                continue
            y_all = series[config["target_col"]].astype(float).to_numpy()
            result[series_id] = y_all[: len(y_all) - TEST_HORIZON]
        return result

    if dataset == "snp500":
        data_by_name = {
            str(name): frame.sort_values("date").copy()
            for name, frame in data.groupby("Name", sort=False)
        }
        map_lookup = series_map.copy()
        map_lookup[ID_COL] = map_lookup[ID_COL].astype(str)
        map_lookup = map_lookup.set_index(ID_COL)

        for series_id in candidate_ids:
            series_id = str(series_id)
            if series_id not in map_lookup.index:
                continue
            stock_name = str(map_lookup.loc[series_id, "Name"])
            target_type = map_lookup.loc[series_id, "Type"]
            if stock_name not in data_by_name:
                continue
            series = data_by_name[stock_name]
            if len(series) <= TEST_HORIZON:
                continue
            y_all = series[target_type].astype(float).to_numpy()
            result[series_id] = y_all[: len(y_all) - TEST_HORIZON]
        return result

    if dataset == "covid19":
        date_cols = get_covid_date_columns(data)
        data_lookup = (
            data.drop_duplicates(subset=["country", "type"], keep="last")
            .set_index(["country", "type"])
        )
        map_lookup = series_map.copy()
        map_lookup[ID_COL] = map_lookup[ID_COL].astype(str)
        map_lookup["country"] = map_lookup["country"].astype(str)
        map_lookup["type"] = map_lookup["type"].astype(str)
        map_lookup = map_lookup.set_index(ID_COL)

        for series_id in candidate_ids:
            series_id = str(series_id)
            if series_id not in map_lookup.index:
                continue
            row = map_lookup.loc[series_id]
            key = (str(row["country"]), str(row["type"]))
            if key not in data_lookup.index:
                continue
            data_row = data_lookup.loc[key]
            if isinstance(data_row, pd.DataFrame):
                data_row = data_row.iloc[0]
            y_all = pd.to_numeric(
                data_row[date_cols], errors="coerce"
            ).to_numpy(dtype=float)
            y_all = y_all[np.isfinite(y_all)]
            if len(y_all) <= TEST_HORIZON:
                continue
            result[series_id] = y_all[: len(y_all) - TEST_HORIZON]
        return result

    raise ValueError(dataset)


def build_lag_window_dataset(y_values, lookback):
    y_values = np.asarray(y_values, dtype=float)
    if len(y_values) <= lookback:
        return None, None

    x_values = np.array(
        [
            y_values[index : index + lookback]
            for index in range(len(y_values) - lookback)
        ]
    )
    targets = np.array(
        [
            y_values[index + lookback]
            for index in range(len(y_values) - lookback)
        ]
    )
    return x_values, targets


def compute_ar_features(y_values, k=AR_K):
    names = [f"ar_coef_{index}" for index in range(k)]
    features = {name: np.nan for name in names}
    try:
        model = AutoReg(
            np.asarray(y_values, dtype=float), lags=k, old_names=False
        ).fit()
        coefficients = np.asarray(model.params)[1 : 1 + k]
        for index in range(min(k, len(coefficients))):
            features[names[index]] = float(coefficients[index])
    except Exception:
        pass
    return features


def compute_rf_landmarker_features(
    y_values,
    lookback=LANDMARKER_LOOKBACK,
):
    stat_names = ["min", "max", "mean", "std", "skew", "kurtosis"]
    metric_names = ["depth", "n_leaves", "mean_feat_imp", "max_feat_imp"]
    features = {
        f"rf_{metric}_{stat}": np.nan
        for metric in metric_names
        for stat in stat_names
    }
    features["rf_oob_score"] = np.nan

    try:
        x_values, targets = build_lag_window_dataset(y_values, lookback)
        if x_values is None or len(x_values) < 5:
            return features

        model = RandomForestRegressor(
            oob_score=True,
            bootstrap=True,
            random_state=0,
            n_jobs=-1,
        )
        model.fit(x_values, targets)

        depths = np.array(
            [estimator.get_depth() for estimator in model.estimators_],
            dtype=float,
        )
        leaves = np.array(
            [estimator.get_n_leaves() for estimator in model.estimators_],
            dtype=float,
        )
        mean_importance = np.array(
            [
                np.mean(estimator.feature_importances_)
                for estimator in model.estimators_
            ],
            dtype=float,
        )
        max_importance = np.array(
            [
                np.max(estimator.feature_importances_)
                for estimator in model.estimators_
            ],
            dtype=float,
        )

        def summarize(values):
            return {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "skew": float(skew(values)),
                "kurtosis": float(kurtosis(values)),
            }

        metric_values = [depths, leaves, mean_importance, max_importance]
        for metric_name, values in zip(metric_names, metric_values):
            for stat_name, value in summarize(values).items():
                features[f"rf_{metric_name}_{stat_name}"] = value
        features["rf_oob_score"] = float(model.oob_score_)
    except Exception:
        pass

    return features


def compute_bayesian_ridge_landmarker_features(
    y_values,
    lookback=LANDMARKER_LOOKBACK,
):
    features = {
        "br_mean_of_distribution": np.nan,
        "br_log_marginal_likelihood": np.nan,
        "br_alpha": np.nan,
        "br_lambda": np.nan,
        "br_n_iter": np.nan,
    }

    try:
        x_values, targets = build_lag_window_dataset(y_values, lookback)
        if x_values is None or len(x_values) < 5:
            return features

        model = BayesianRidge(compute_score=True)
        model.fit(x_values, targets)
        predictions = model.predict(x_values)
        features["br_mean_of_distribution"] = float(np.mean(predictions))
        features["br_log_marginal_likelihood"] = (
            float(model.scores_[-1]) if len(model.scores_) > 0 else np.nan
        )
        features["br_alpha"] = float(model.alpha_)
        features["br_lambda"] = float(model.lambda_)
        features["br_n_iter"] = float(model.n_iter_)
    except Exception:
        pass

    return features


def compute_landmarker_features_for_series(y_values):
    features = {}
    features.update(compute_ar_features(y_values))
    features.update(compute_rf_landmarker_features(y_values))
    features.update(compute_bayesian_ridge_landmarker_features(y_values))
    return features


def compute_or_load_landmarker_features(
    data,
    series_map,
    candidate_ids,
    config,
):
    cache_path = os.path.join(
        config["cache_dir"], f"{config['dataset']}_landmarker_features.pkl"
    )
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)

    train_series = build_all_train_series(
        data, series_map, candidate_ids, config
    )
    rows = []
    for series_id, y_train in train_series.items():
        features = compute_landmarker_features_for_series(y_train)
        features[ID_COL] = series_id
        rows.append(features)

    output = pd.DataFrame(rows)
    output = output[[ID_COL] + [column for column in output if column != ID_COL]]
    os.makedirs(config["cache_dir"], exist_ok=True)
    output.to_pickle(cache_path)
    return output


def run_one_setting(
    test_cache,
    sample_setting,
    series_ids,
    x_values,
    y_reg,
    combo_cols,
    forecasting_lookup,
):
    smape_values = []
    mase_values = []
    rmsse_values = []
    n_series = len(series_ids)
    train_size = sample_setting["train_size"]
    sample_cells = sample_setting["sample_cells"]

    for repeat_idx in range(N_REPEATS):
        train_idx, test_idx = make_split(
            n_series, train_size, sample_cells, repeat_idx
        )
        x_train = x_values[train_idx]
        x_test = x_values[test_idx]
        x_train, x_test = impute_train_test_by_train_median(x_train, x_test)
        x_train, x_test = scale_train_test_by_train(x_train, x_test)
        x_train, x_test = sanitize_train_test_for_float32(x_train, x_test)
        x_train, x_test = apply_knn_pca_if_needed(
            "linear_regression", x_train, x_test
        )

        model = LinearRegression()
        model.fit(x_train, y_reg[train_idx])
        predicted_smape = np.asarray(model.predict(x_test), dtype=float)
        if predicted_smape.ndim == 1:
            predicted_smape = predicted_smape.reshape(-1, 1)

        best_positions = np.argmin(predicted_smape, axis=1)
        selected_combo_ids = np.asarray(combo_cols, dtype=object)[
            best_positions
        ]
        smape_value, mase_value, rmsse_value = evaluate_selected_combos(
            test_cache=test_cache,
            series_ids_test=series_ids[test_idx],
            selected_combo_ids=selected_combo_ids,
            forecasting_lookup=forecasting_lookup,
        )
        smape_values.append(smape_value)
        mase_values.append(mase_value)
        rmsse_values.append(rmsse_value)

    return (
        float(np.nanmean(smape_values)),
        float(np.nanmean(mase_values)),
        float(np.nanmean(rmsse_values)),
    )


def main():
    start_time = time.time()

    if MODEL_TYPE not in ["prophet", "sarima", "lstm", "all"]:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    data = load_raw_data(CONFIG)
    series_map = load_series_map(CONFIG)
    if MODEL_TYPE == "all":
        smape_matrix, forecasting_matrix = load_all_model_matrices(CONFIG)
    else:
        smape_matrix, forecasting_matrix = load_single_model_matrices(
            MODEL_TYPE, CONFIG
        )

    print("=== AUTOFORECAST META-LEARNING ===", flush=True)
    print(f"DATASET    = {DATASET}", flush=True)
    print(f"MODEL_TYPE = {MODEL_TYPE}", flush=True)
    print(f"FEATURES   = {FEATURE_NAMES}", flush=True)
    print(f"METHODS    = {METHODS}", flush=True)
    print(f"ALGORITHMS = {ALGO_NAMES}", flush=True)
    print(f"N_REPEATS  = {N_REPEATS}", flush=True)
    print("NOTE: only the general meta-learner Phi is implemented; the", flush=True)
    print(
        "      time-series meta-learner Theta (LSTM over rolling windows)",
        flush=True,
    )
    print("      is out of scope for this data pipeline.", flush=True)
    print("==============================\n", flush=True)

    summary_rows = []
    for feature_name in FEATURE_NAMES:
        tsfresh_features = load_features(
            CONFIG["feature_files"]["tsfresh"], CONFIG, series_map
        )
        landmarker_features = compute_or_load_landmarker_features(
            data,
            series_map,
            tsfresh_features[ID_COL].tolist(),
            CONFIG,
        )
        features = tsfresh_features.merge(
            landmarker_features, on=ID_COL, how="inner"
        )
        (
            series_ids,
            x_values,
            y_class,
            y_reg,
            combo_cols,
            forecasting_aligned,
        ) = align_inputs(smape_matrix, forecasting_matrix, features)

        n_series = len(series_ids)
        n_combos = len(combo_cols)
        total_cells = n_series * n_combos
        base_order, sample_settings = compute_order_sample_settings(
            n_series, n_combos
        )

        print("\n====================", flush=True)
        print(f"FEATURE SOURCE: {feature_name}", flush=True)
        print("====================", flush=True)
        print(
            f"Aligned matrix shape: n={n_series}, m={n_combos}, "
            f"cells={total_cells}",
            flush=True,
        )
        print(f"Base order: {base_order:.6f}", flush=True)
        print("Sample settings:", flush=True)
        for setting in sample_settings:
            if setting["skip"]:
                print(
                    f"  multiplier={setting['multiplier']} | "
                    f"sample_cells={setting['sample_cells']} | "
                    f"matrix_percentage={setting['matrix_percentage']:.6f}% | "
                    "SKIP: sample_cells > total_cells",
                    flush=True,
                )
            else:
                print(
                    f"  multiplier={setting['multiplier']} | "
                    f"sample_cells={setting['sample_cells']} | "
                    f"matrix_percentage={setting['matrix_percentage']:.6f}% | "
                    f"train_series={setting['train_size']} | "
                    f"effective_cells={setting['effective_cells']} | "
                    f"effective_percentage="
                    f"{setting['effective_percentage']:.6f}%",
                    flush=True,
                )

        test_cache = build_test_cache(
            data, series_map, series_ids, CONFIG
        )
        forecasting_lookup = forecasting_aligned.set_index(ID_COL)

        for method in METHODS:
            for algo_name in ALGO_NAMES:
                for setting in sample_settings:
                    if setting["skip"]:
                        continue

                    print(
                        f"[START] dataset={DATASET} | model={MODEL_TYPE} | "
                        f"method={method} | features={feature_name} | "
                        f"algorithm={algo_name} | "
                        f"order_multiplier={setting['multiplier']} | "
                        f"sample_cells={setting['sample_cells']} | "
                        f"matrix_percentage="
                        f"{setting['matrix_percentage']:.6f}% | "
                        f"train_series={setting['train_size']}",
                        flush=True,
                    )

                    smape_value, mase_value, rmsse_value = run_one_setting(
                        test_cache=test_cache,
                        sample_setting=setting,
                        series_ids=series_ids,
                        x_values=x_values,
                        y_reg=y_reg,
                        combo_cols=combo_cols,
                        forecasting_lookup=forecasting_lookup,
                    )
                    line = (
                        f"dataset={DATASET} | model={MODEL_TYPE} | "
                        f"method={method} | features={feature_name} | "
                        f"algorithm={algo_name} | "
                        f"order_multiplier={setting['multiplier']} | "
                        f"sample_cells={setting['sample_cells']} | "
                        f"matrix_percentage="
                        f"{setting['matrix_percentage']:.6f}% | "
                        f"train_series={setting['train_size']} | "
                        f"effective_cells={setting['effective_cells']} | "
                        f"effective_percentage="
                        f"{setting['effective_percentage']:.6f}% | "
                        f"sMAPE={smape_value:.6f} | "
                        f"MASE={mase_value:.6f} | RMSSE={rmsse_value:.6f}"
                    )
                    print(line, flush=True)
                    summary_rows.append(line)

    print("\n=== SUMMARY ===", flush=True)
    for line in summary_rows:
        print(line, flush=True)
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
