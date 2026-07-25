"""Evaluate random model-selection baselines using saved validation and test results."""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# Configuration
DATASET = "covid19"  # options: "electricity", "snp500", "covid19"
MODEL_NAME = "all_models"

N_REPEATS = 10
RANDOM_SEED = 42

BASELINE_MODES = ("random@1", "random@5")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate random model-selection baselines using saved "
        "validation and test results."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing the dataset's raw file "
        "(consumption_dataset.pkl / snp500_data.csv / covid19_dataset.pkl). "
        "Defaults to ./data/<dataset>.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory shared with grid_search/ and test_forecasting/: "
        "validation matrices, combination maps, and test forecasts are read "
        "from here. Defaults to ./outputs/<dataset>.",
    )
    return parser.parse_args()


def build_dataset_configs(data_dir=None, output_dir=None):
    data_dir = data_dir or os.path.join("data", DATASET)
    output_dir = output_dir or os.path.join("outputs", DATASET)

    return {
        "electricity": {
            "id_col": "building_id",
            "data_kind": "electricity",
            "data_path": os.path.join(data_dir, "consumption_dataset.pkl"),
            "test_horizon": 48,
            "seasonal_period": 168,
        },
        "snp500": {
            "id_col": "series_id",
            "data_kind": "snp500",
            "data_path": os.path.join(data_dir, "snp500_data.csv"),
            "series_map_path": os.path.join(output_dir, "snp500_series_to_id.pkl"),
            "target_cols": ["open", "high", "low", "close", "volume"],
            "test_horizon": 14,
            "seasonal_period": 5,
        },
        "covid19": {
            "id_col": "series_id",
            "data_kind": "covid19",
            "data_path": os.path.join(data_dir, "covid19_dataset.pkl"),
            "series_map_path": os.path.join(output_dir, "covid19_series_to_id.pkl"),
            "test_horizon": 14,
            "seasonal_period": 7,
        },
    }


def build_model_file_configs(output_dir=None):
    output_dir = output_dir or os.path.join("outputs", DATASET)

    return {
        "prophet": {
            "electricity": {
                "smape": os.path.join(output_dir, "electricity_smape_matrix.pkl"),
                "forecast": os.path.join(output_dir, "electricity_forecasting_test.pkl"),
                "comb_map": os.path.join(output_dir, "electricity_comb_to_id.pkl"),
            },
            "snp500": {
                "smape": os.path.join(output_dir, "snp500_smape_matrix.pkl"),
                "forecast": os.path.join(output_dir, "snp500_forecasting_test.pkl"),
                "comb_map": os.path.join(output_dir, "snp500_comb_to_id.pkl"),
            },
            "covid19": {
                "smape": os.path.join(output_dir, "covid19_smape_matrix.pkl"),
                "forecast": os.path.join(output_dir, "covid19_forecasting_test.pkl"),
                "comb_map": os.path.join(output_dir, "covid19_comb_to_id.pkl"),
            },
        },
        "sarima": {
            "electricity": {
                "smape": os.path.join(output_dir, "electricity_smape_matrix_sarima.pkl"),
                "forecast": os.path.join(output_dir, "electricity_forecasting_test_sarima.pkl"),
                "comb_map": os.path.join(output_dir, "electricity_comb_to_id_sarima.pkl"),
            },
            "snp500": {
                "smape": os.path.join(output_dir, "snp500_smape_matrix_sarima.pkl"),
                "forecast": os.path.join(output_dir, "snp500_forecasting_test_sarima.pkl"),
                "comb_map": os.path.join(output_dir, "snp500_comb_to_id_sarima.pkl"),
            },
            "covid19": {
                "smape": os.path.join(output_dir, "covid19_smape_matrix_sarima.pkl"),
                "forecast": os.path.join(output_dir, "covid19_forecasting_test_sarima.pkl"),
                "comb_map": os.path.join(output_dir, "covid19_comb_to_id_sarima.pkl"),
            },
        },
        "lstm": {
            "electricity": {
                "smape": os.path.join(output_dir, "electricity_smape_matrix_lstm.pkl"),
                "forecast": os.path.join(output_dir, "electricity_forecasting_test_lstm.pkl"),
                "comb_map": os.path.join(output_dir, "electricity_comb_to_id_lstm.pkl"),
            },
            "snp500": {
                "smape": os.path.join(output_dir, "snp500_smape_matrix_lstm.pkl"),
                "forecast": os.path.join(output_dir, "snp500_forecasting_test_lstm.pkl"),
                "comb_map": os.path.join(output_dir, "snp500_comb_to_id_lstm.pkl"),
            },
            "covid19": {
                "smape": os.path.join(output_dir, "covid19_smape_matrix_lstm.pkl"),
                "forecast": os.path.join(output_dir, "covid19_forecasting_test_lstm.pkl"),
                "comb_map": os.path.join(output_dir, "covid19_comb_to_id_lstm.pkl"),
            },
        },
    }


_ARGS = parse_args()
DATASET_CONFIGS = build_dataset_configs(_ARGS.data_dir, _ARGS.output_dir)
MODEL_FILE_CONFIGS = build_model_file_configs(_ARGS.output_dir)


# Utilities
def _maybe_int_col(c):
    try:
        return int(c)
    except Exception:
        return c


def normalize_matrix_columns(df, id_col):
    out = df.copy()
    new_cols = []
    for c in out.columns:
        if c == id_col:
            new_cols.append(c)
        else:
            new_cols.append(_maybe_int_col(c))
    out.columns = new_cols
    out[id_col] = out[id_col].astype(str)
    return out


def get_forecast_value(forecast_row, combo_id):
    candidates = [combo_id]
    try:
        candidates.append(int(combo_id))
    except Exception:
        pass
    candidates.append(str(combo_id))

    for c in candidates:
        if c in forecast_row.index:
            return forecast_row[c]

    raise KeyError(f"combo_id={combo_id} not found in forecasting matrix")


def seasonal_scale_q(y_train, m, eps=1e-12):
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= m:
        return np.nan
    q = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if (not np.isfinite(q)) or (q <= eps):
        return np.nan
    return float(q)


def global_smape(y_true_all, y_pred_all):
    y_true_all = np.asarray(y_true_all, dtype=float)
    y_pred_all = np.asarray(y_pred_all, dtype=float)
    return 100.0 * np.mean(
        2.0 * np.abs(y_true_all - y_pred_all)
        / (np.abs(y_true_all) + np.abs(y_pred_all) + 1e-9)
    )

# Data loading
def load_dataset_data(dataset):
    cfg = DATASET_CONFIGS[dataset]

    if cfg["data_kind"] == "electricity":
        data = pd.read_pickle(cfg["data_path"])
        if not np.issubdtype(data["ts"].dtype, np.datetime64):
            data["ts"] = pd.to_datetime(data["ts"], errors="coerce")
        data[cfg["target_col"]] = pd.to_numeric(data[cfg["target_col"]], errors="coerce")
        data = data.dropna(subset=["id", "ts", cfg["target_col"]]).copy()
        data["id"] = data["id"].astype(str)
        return data, None

    if cfg["data_kind"] == "snp500":
        data = pd.read_csv(cfg["data_path"])
        try:
            data["date"] = pd.to_datetime(data["date"], format="%m/%d/%Y", errors="raise")
        except Exception:
            data["date"] = pd.to_datetime(data["date"], errors="coerce")

        for c in cfg["target_cols"]:
            data[c] = pd.to_numeric(data[c], errors="coerce")

        data = data.dropna(subset=["date", "Name"] + cfg["target_cols"]).copy()
        series_map = pd.read_pickle(cfg["series_map_path"]).copy()
        series_map[cfg["id_col"]] = series_map[cfg["id_col"]].astype(str)
        return data, series_map

    if cfg["data_kind"] == "covid19":
        data = pd.read_pickle(cfg["data_path"])

        non_date_cols = ["country", "type", "Lat", "Long"]
        date_col_to_datetime = {}

        for col in data.columns:
            if col not in non_date_cols:
                parsed_date = pd.to_datetime(col, format="%m/%d/%Y", errors="coerce")
                if pd.isna(parsed_date):
                    parsed_date = pd.to_datetime(col, format="%m/%d/%y", errors="coerce")
                if not pd.isna(parsed_date):
                    date_col_to_datetime[col] = parsed_date

        date_cols = sorted(date_col_to_datetime, key=lambda col: date_col_to_datetime[col])

        for col in date_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna(subset=["country", "type"] + date_cols).copy()

        series_map = pd.read_pickle(cfg["series_map_path"]).copy()
        series_map[cfg["id_col"]] = series_map[cfg["id_col"]].astype(str)
        return data, series_map

    raise ValueError(dataset)


def build_test_cache(dataset, data, series_map, series_ids):
    cfg = DATASET_CONFIGS[dataset]
    test_horizon = cfg["test_horizon"]
    seasonal_period = cfg["seasonal_period"]
    cache = {}

    if cfg["data_kind"] == "electricity":
        data_by_id = {
            str(series_id): df_s.sort_values("ts").copy()
            for series_id, df_s in data.groupby("id", sort=False)
        }

        for series_id in series_ids:
            series_id = str(series_id)
            if series_id not in data_by_id:
                continue

            df_s = data_by_id[series_id]
            if len(df_s) <= test_horizon:
                continue

            y_all = df_s[cfg["target_col"]].astype(float).to_numpy()
            split_idx = len(y_all) - test_horizon
            y_train = y_all[:split_idx]
            y_test = y_all[split_idx:]

            q_i = seasonal_scale_q(y_train, m=seasonal_period)
            if not np.isfinite(q_i):
                continue

            cache[series_id] = (y_test, q_i)

        return cache

    if cfg["data_kind"] == "snp500":
        data_by_name = {
            str(name): df_s.sort_values("date").copy()
            for name, df_s in data.groupby("Name", sort=False)
        }
        series_lookup = series_map.set_index(cfg["id_col"])

        for series_id in series_ids:
            series_id = str(series_id)
            if series_id not in series_lookup.index:
                continue

            row = series_lookup.loc[series_id]
            stock_name = str(row["Name"])
            target_type = row["Type"]

            if stock_name not in data_by_name:
                continue

            df_s = data_by_name[stock_name]
            if len(df_s) <= test_horizon:
                continue

            y_all = df_s[target_type].astype(float).to_numpy()
            split_idx = len(y_all) - test_horizon
            y_train = y_all[:split_idx]
            y_test = y_all[split_idx:]

            q_i = seasonal_scale_q(y_train, m=seasonal_period)
            if not np.isfinite(q_i):
                continue

            cache[series_id] = (y_test, q_i)

        return cache

    if cfg["data_kind"] == "covid19":
        non_date_cols = ["country", "type", "Lat", "Long"]
        date_col_to_datetime = {}

        for col in data.columns:
            if col not in non_date_cols:
                parsed_date = pd.to_datetime(col, format="%m/%d/%Y", errors="coerce")
                if pd.isna(parsed_date):
                    parsed_date = pd.to_datetime(col, format="%m/%d/%y", errors="coerce")
                if not pd.isna(parsed_date):
                    date_col_to_datetime[col] = parsed_date

        date_cols = sorted(date_col_to_datetime, key=lambda col: date_col_to_datetime[col])
        series_lookup = series_map.set_index(cfg["id_col"])

        for series_id in series_ids:
            series_id = str(series_id)
            if series_id not in series_lookup.index:
                continue

            row = series_lookup.loc[series_id]
            country = row["country"]
            target_type = row["type"]

            df_s = data[(data["country"] == country) & (data["type"] == target_type)]
            if len(df_s) == 0:
                continue

            y_all = df_s.iloc[0][date_cols].astype(float).to_numpy()
            if len(y_all) <= test_horizon:
                continue

            split_idx = len(y_all) - test_horizon
            y_train = y_all[:split_idx]
            y_test = y_all[split_idx:]

            q_i = seasonal_scale_q(y_train, m=seasonal_period)
            if not np.isfinite(q_i):
                continue

            cache[series_id] = (y_test, q_i)

        return cache

    raise ValueError(dataset)

# Model table loading
def load_comb_map(dataset, model_name):
    path = MODEL_FILE_CONFIGS[model_name][dataset].get("comb_map")
    if path is None or not os.path.exists(path):
        return None
    comb_map = pd.read_pickle(path).copy()
    if "id" in comb_map.columns:
        comb_map["id"] = comb_map["id"].astype(int)
    return comb_map


def get_combo_cols(smape_matrix, forecasting_matrix, comb_map, id_col):
    if comb_map is not None and "id" in comb_map.columns:
        candidate_cols = comb_map["id"].astype(int).tolist()
    else:
        candidate_cols = [c for c in smape_matrix.columns if c != id_col]

    combo_cols = []
    for c in candidate_cols:
        if c in smape_matrix.columns and c in forecasting_matrix.columns:
            combo_cols.append(c)
        elif str(c) in smape_matrix.columns and str(c) in forecasting_matrix.columns:
            combo_cols.append(str(c))

    combo_cols = [_maybe_int_col(c) for c in combo_cols]
    return combo_cols


def load_single_model_tables(dataset, model_name):
    ds_cfg = DATASET_CONFIGS[dataset]
    file_cfg = MODEL_FILE_CONFIGS[model_name][dataset]
    id_col = ds_cfg["id_col"]

    smape_matrix = pd.read_pickle(file_cfg["smape"])
    forecasting_matrix = pd.read_pickle(file_cfg["forecast"])

    smape_matrix = normalize_matrix_columns(smape_matrix, id_col)
    forecasting_matrix = normalize_matrix_columns(forecasting_matrix, id_col)

    comb_map = load_comb_map(dataset, model_name)
    combo_cols = get_combo_cols(smape_matrix, forecasting_matrix, comb_map, id_col)

    if len(combo_cols) == 0:
        raise RuntimeError("No common combo columns found in validation and forecasting matrices")

    common_ids = sorted(
        set(smape_matrix[id_col]) & set(forecasting_matrix[id_col]),
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )

    smape_matrix = smape_matrix.set_index(id_col).loc[common_ids].reset_index()
    forecasting_matrix = forecasting_matrix.set_index(id_col).loc[common_ids].reset_index()

    smape_matrix = smape_matrix[[id_col] + combo_cols]
    forecasting_matrix = forecasting_matrix[[id_col] + combo_cols]

    return {
        "id_col": id_col,
        "series_ids": np.array(common_ids, dtype=str),
        "smape_matrix": smape_matrix,
        "forecasting_matrix": forecasting_matrix,
        "forecasting_lookup": forecasting_matrix.set_index(id_col),
        "comb_map": comb_map,
        "combo_cols": combo_cols,
    }

# Combined model tables
def load_all_models_tables(dataset):
    id_col = DATASET_CONFIGS[dataset]["id_col"]
    model_names = ["prophet", "sarima", "lstm"]

    loaded = {
        model_name: load_single_model_tables(dataset, model_name)
        for model_name in model_names
    }

    common_ids = None
    for model_name in model_names:
        ids = set(loaded[model_name]["series_ids"].tolist())
        common_ids = ids if common_ids is None else common_ids & ids

    common_ids = sorted(common_ids, key=lambda x: int(x) if str(x).isdigit() else str(x))

    smape_parts = []
    option_info = {}
    forecasting_lookups = {}

    for model_name in model_names:
        tab = loaded[model_name]
        smape_aligned = tab["smape_matrix"].set_index(id_col).loc[common_ids]
        forecast_aligned = tab["forecasting_matrix"].set_index(id_col).loc[common_ids]

        rename_map = {}
        for combo_id in tab["combo_cols"]:
            label = f"{model_name}__{combo_id}"
            rename_map[combo_id] = label
            option_info[label] = (model_name, combo_id)

        smape_parts.append(smape_aligned[tab["combo_cols"]].rename(columns=rename_map))
        forecasting_lookups[model_name] = forecast_aligned

    smape_all = pd.concat(smape_parts, axis=1)
    smape_all.insert(0, id_col, common_ids)
    option_cols = [c for c in smape_all.columns if c != id_col]

    return {
        "id_col": id_col,
        "series_ids": np.array(common_ids, dtype=str),
        "smape_matrix": smape_all,
        "combo_cols": option_cols,
        "forecasting_lookups": forecasting_lookups,
        "option_info": option_info,
    }

# Baseline selection
def choose_options_for_baseline(mode, smape_matrix, option_cols, rng):
    n_series = len(smape_matrix)

    if mode == "random@1":
        return rng.choice(option_cols, size=n_series, replace=True).tolist()

    if mode == "random@5":
        sample_size = min(5, len(option_cols))
        values = smape_matrix[option_cols].astype(float)
        selected = []

        for row_idx in range(n_series):
            sampled = rng.choice(option_cols, size=sample_size, replace=False).tolist()
            row_scores = values.iloc[row_idx][sampled]
            if row_scores.notna().any():
                selected.append(row_scores.idxmin())
            else:
                selected.append(rng.choice(sampled))

        return selected

    raise ValueError(f"Unknown baseline mode: {mode}")

# Evaluation
def evaluate_single_model_selection(test_cache, series_ids, selected_combo_ids, forecasting_lookup):
    y_true_all = []
    y_pred_all = []
    q_all = []
    failures = 0

    for series_id, combo_id in zip(series_ids, selected_combo_ids):
        series_id = str(series_id)
        if series_id not in test_cache:
            failures += 1
            continue

        try:
            forecast_row = forecasting_lookup.loc[series_id]
            y_pred = get_forecast_value(forecast_row, combo_id)
            y_pred = np.asarray(y_pred, dtype=float)
        except Exception:
            failures += 1
            continue

        y_test, q_i = test_cache[series_id]
        y_test = np.asarray(y_test, dtype=float)

        if len(y_pred) != len(y_test):
            failures += 1
            continue

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        q_all.append(np.repeat(q_i, len(y_test)))

    return compute_metrics(y_true_all, y_pred_all, q_all, failures)


def evaluate_all_models_selection(
    test_cache, series_ids, selected_options, forecasting_lookups, option_info
):
    y_true_all = []
    y_pred_all = []
    q_all = []
    failures = 0

    for series_id, option in zip(series_ids, selected_options):
        series_id = str(series_id)
        if series_id not in test_cache:
            failures += 1
            continue

        try:
            model_name, combo_id = option_info[option]
            forecast_row = forecasting_lookups[model_name].loc[series_id]
            y_pred = get_forecast_value(forecast_row, combo_id)
            y_pred = np.asarray(y_pred, dtype=float)
        except Exception:
            failures += 1
            continue

        y_test, q_i = test_cache[series_id]
        y_test = np.asarray(y_test, dtype=float)

        if len(y_pred) != len(y_test):
            failures += 1
            continue

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        q_all.append(np.repeat(q_i, len(y_test)))

    return compute_metrics(y_true_all, y_pred_all, q_all, failures)


def compute_metrics(y_true_list, y_pred_list, q_list, failures):
    if len(y_true_list) == 0:
        return {
            "successful_series": 0,
            "failed_series": int(failures),
            "global_smape": np.nan,
            "global_mase": np.nan,
            "global_rmsse": np.nan,
        }

    y_true_all = np.concatenate(y_true_list)
    y_pred_all = np.concatenate(y_pred_list)
    q_all = np.concatenate(q_list)

    return {
        "successful_series": int(len(y_true_list)),
        "failed_series": int(failures),
        "global_smape": float(global_smape(y_true_all, y_pred_all)),
        "global_mase": float(np.mean(np.abs(y_true_all - y_pred_all) / (q_all + 1e-9))),
        "global_rmsse": float(np.sqrt(np.mean(((y_true_all - y_pred_all) / (q_all + 1e-9)) ** 2))),
    }

# Repeated evaluation
def run_single_model_baselines(dataset, model_name):
    data, series_map = load_dataset_data(dataset)
    tables = load_single_model_tables(dataset, model_name)
    test_cache = build_test_cache(
        dataset=dataset,
        data=data,
        series_map=series_map,
        series_ids=tables["series_ids"],
    )

    result_lines = []
    for mode in BASELINE_MODES:
        repeat_results = []

        for repeat_idx in range(N_REPEATS):
            rng = np.random.default_rng(RANDOM_SEED + repeat_idx)
            selected = choose_options_for_baseline(
                mode=mode,
                smape_matrix=tables["smape_matrix"],
                option_cols=tables["combo_cols"],
                rng=rng,
            )
            repeat_results.append(
                evaluate_single_model_selection(
                    test_cache=test_cache,
                    series_ids=tables["series_ids"],
                    selected_combo_ids=selected,
                    forecasting_lookup=tables["forecasting_lookup"],
                )
            )

        result_lines.append(
            format_mean_result(dataset, model_name, mode, repeat_results)
        )

    return result_lines


def run_all_models_baselines(dataset):
    data, series_map = load_dataset_data(dataset)
    tables = load_all_models_tables(dataset)
    test_cache = build_test_cache(
        dataset=dataset,
        data=data,
        series_map=series_map,
        series_ids=tables["series_ids"],
    )

    result_lines = []
    for mode in BASELINE_MODES:
        repeat_results = []

        for repeat_idx in range(N_REPEATS):
            rng = np.random.default_rng(RANDOM_SEED + repeat_idx)
            selected = choose_options_for_baseline(
                mode=mode,
                smape_matrix=tables["smape_matrix"],
                option_cols=tables["combo_cols"],
                rng=rng,
            )
            repeat_results.append(
                evaluate_all_models_selection(
                    test_cache=test_cache,
                    series_ids=tables["series_ids"],
                    selected_options=selected,
                    forecasting_lookups=tables["forecasting_lookups"],
                    option_info=tables["option_info"],
                )
            )

        result_lines.append(
            format_mean_result(dataset, "all_models", mode, repeat_results)
        )

    return result_lines


def format_mean_result(dataset, model_name, mode, repeat_results):
    successful = np.array(
        [result["successful_series"] for result in repeat_results], dtype=float
    )
    failed = np.array(
        [result["failed_series"] for result in repeat_results], dtype=float
    )
    smape = np.array(
        [result["global_smape"] for result in repeat_results], dtype=float
    )
    mase = np.array(
        [result["global_mase"] for result in repeat_results], dtype=float
    )
    rmsse = np.array(
        [result["global_rmsse"] for result in repeat_results], dtype=float
    )

    return (
        f"dataset={dataset} | model={model_name} | baseline={mode} | "
        f"n_repeats={N_REPEATS} | "
        f"mean_successful_series={np.nanmean(successful):.2f} | "
        f"mean_failed_series={np.nanmean(failed):.2f} | "
        f"sMAPE={np.nanmean(smape):.6f} | "
        f"MASE={np.nanmean(mase):.6f} | "
        f"RMSSE={np.nanmean(rmsse):.6f}"
    )


def main():
    if DATASET not in DATASET_CONFIGS:
        raise ValueError(f"Unknown DATASET={DATASET}")

    if MODEL_NAME == "all_models":
        lines = run_all_models_baselines(DATASET)
    else:
        lines = run_single_model_baselines(DATASET, MODEL_NAME)

    for line in lines:
        print(line, flush=True)


if __name__ == "__main__":
    main()
