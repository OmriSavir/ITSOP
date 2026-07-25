"""Evaluate the validation oracle across Prophet, SARIMA, and LSTM."""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATASET = "covid19"  # options: "covid19", "electricity", "snp500"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the validation oracle across Prophet, SARIMA, and LSTM."
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


def _maybe_int_col(column):
    try:
        return int(column)
    except Exception:
        return column


def normalize_matrix_columns(dataframe, id_col):
    """Normalize combination columns and cast series identifiers to strings."""
    normalized = dataframe.copy()
    normalized.columns = [
        column if column == id_col else _maybe_int_col(column)
        for column in normalized.columns
    ]
    normalized[id_col] = normalized[id_col].astype(str)
    return normalized


def get_forecast_value(forecast_row, combo_id):
    """Read a forecast regardless of whether its combination ID is typed as int or str."""
    candidates = [combo_id]
    try:
        candidates.append(int(combo_id))
    except Exception:
        pass
    candidates.append(str(combo_id))

    for candidate in candidates:
        if candidate in forecast_row.index:
            return forecast_row[candidate]

    raise KeyError(f"combo_id={combo_id} not found in forecasting matrix")


def seasonal_scale_q(y_train, seasonal_period, eps=1e-12):
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= seasonal_period:
        return np.nan

    scale = np.mean(
        np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    )
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


def load_electricity_data(path, target_col):
    data = pd.read_pickle(path)
    if not np.issubdtype(data["ts"].dtype, np.datetime64):
        data["ts"] = pd.to_datetime(data["ts"], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna(subset=["ts", target_col, "id"]).copy()
    data["id"] = data["id"].astype(str)
    return data


def load_snp500_data(path, target_cols):
    data = pd.read_csv(path)
    try:
        data["date"] = pd.to_datetime(
            data["date"], format="%m/%d/%Y", errors="raise"
        )
    except Exception:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

    for target_col in target_cols:
        data[target_col] = pd.to_numeric(data[target_col], errors="coerce")

    return data.dropna(subset=["date", "Name"] + target_cols).copy()


def load_covid19_data(path):
    data = pd.read_pickle(path)
    non_date_cols = ["country", "type", "Lat", "Long"]
    date_col_to_datetime = {}

    for column in data.columns:
        if column in non_date_cols:
            continue

        parsed_date = pd.to_datetime(
            column, format="%m/%d/%Y", errors="coerce"
        )
        if pd.isna(parsed_date):
            parsed_date = pd.to_datetime(
                column, format="%m/%d/%y", errors="coerce"
            )
        if not pd.isna(parsed_date):
            date_col_to_datetime[column] = parsed_date

    date_cols = sorted(
        date_col_to_datetime,
        key=lambda column: date_col_to_datetime[column],
    )
    for column in date_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["country", "type"] + date_cols).copy()
    return data, date_cols


def truth_cache_electricity(
    data,
    series_ids,
    target_col,
    test_horizon,
    seasonal_period,
):
    data_by_id = {
        str(series_id): group.sort_values("ts").copy()
        for series_id, group in data.groupby("id", sort=False)
    }
    cache = {}

    for series_id in series_ids:
        series_id = str(series_id)
        if series_id not in data_by_id:
            continue

        series_data = data_by_id[series_id]
        if len(series_data) <= test_horizon:
            continue

        y_all = series_data[target_col].astype(float).to_numpy()
        split_idx = len(y_all) - test_horizon
        scale = seasonal_scale_q(y_all[:split_idx], seasonal_period)
        if np.isfinite(scale):
            cache[series_id] = (y_all[split_idx:], scale)

    return cache


def truth_cache_snp500(
    data,
    series_map,
    series_ids,
    test_horizon,
    seasonal_period,
):
    data_by_name = {
        str(name): group.sort_values("date").copy()
        for name, group in data.groupby("Name", sort=False)
    }
    indexed_series_map = series_map.copy()
    indexed_series_map["series_id"] = indexed_series_map["series_id"].astype(str)
    indexed_series_map = indexed_series_map.set_index("series_id")
    cache = {}

    for series_id in series_ids:
        series_id = str(series_id)
        if series_id not in indexed_series_map.index:
            continue

        name = str(indexed_series_map.loc[series_id, "Name"])
        target_col = indexed_series_map.loc[series_id, "Type"]
        if name not in data_by_name:
            continue

        series_data = data_by_name[name]
        if len(series_data) <= test_horizon:
            continue

        y_all = series_data[target_col].astype(float).to_numpy()
        split_idx = len(y_all) - test_horizon
        scale = seasonal_scale_q(y_all[:split_idx], seasonal_period)
        if np.isfinite(scale):
            cache[series_id] = (y_all[split_idx:], scale)

    return cache


def truth_cache_covid19(
    data,
    series_map,
    series_ids,
    date_cols,
    test_horizon,
    seasonal_period,
):
    indexed_series_map = series_map.copy()
    indexed_series_map["series_id"] = indexed_series_map["series_id"].astype(str)
    indexed_series_map = indexed_series_map.set_index("series_id")
    cache = {}

    for series_id in series_ids:
        series_id = str(series_id)
        if series_id not in indexed_series_map.index:
            continue

        country = indexed_series_map.loc[series_id, "country"]
        series_type = indexed_series_map.loc[series_id, "type"]
        series_data = data[
            (data["country"] == country) & (data["type"] == series_type)
        ]
        if len(series_data) == 0:
            continue

        y_all = series_data.iloc[0][date_cols].astype(float).to_numpy()
        y_all = y_all[np.isfinite(y_all)]
        if len(y_all) <= test_horizon:
            continue

        split_idx = len(y_all) - test_horizon
        scale = seasonal_scale_q(y_all[:split_idx], seasonal_period)
        if np.isfinite(scale):
            cache[series_id] = (y_all[split_idx:], scale)

    return cache


def compute_metrics(y_true_list, y_pred_list, q_list):
    if len(y_true_list) == 0:
        raise RuntimeError("No successful series were evaluated.")

    y_true_all = np.concatenate(y_true_list)
    y_pred_all = np.concatenate(y_pred_list)
    q_all = np.concatenate(q_list)

    smape = global_smape(y_true_all, y_pred_all)
    mase = float(
        np.mean(np.abs(y_true_all - y_pred_all) / (q_all + 1e-9))
    )
    rmsse = float(
        np.sqrt(
            np.mean(
                ((y_true_all - y_pred_all) / (q_all + 1e-9)) ** 2
            )
        )
    )
    return smape, mase, rmsse


def build_dataset_configs(data_dir=None, output_dir=None):
    data_dir = data_dir or os.path.join("data", DATASET)
    output_dir = output_dir or os.path.join("outputs", DATASET)

    return {
        "electricity": {
            "id_col": "building_id",
            "data_path": os.path.join(data_dir, "consumption_dataset.pkl"),
            "target_col": "consumption",
            "test_horizon": 48,
            "seasonal_period": 168,
            "prophet_smape_matrix_pkl": os.path.join(output_dir, "electricity_smape_matrix.pkl"),
            "prophet_forecasting_test_pkl": os.path.join(output_dir, "electricity_forecasting_test.pkl"),
            "prophet_comb_map_pkl": os.path.join(output_dir, "electricity_comb_to_id.pkl"),
            "sarima_smape_matrix_pkl": os.path.join(output_dir, "electricity_smape_matrix_sarima.pkl"),
            "sarima_forecasting_test_pkl": os.path.join(output_dir, "electricity_forecasting_test_sarima.pkl"),
            "sarima_comb_map_pkl": os.path.join(output_dir, "electricity_comb_to_id_sarima.pkl"),
            "lstm_smape_matrix_pkl": os.path.join(output_dir, "electricity_smape_matrix_lstm.pkl"),
            "lstm_forecasting_test_pkl": os.path.join(output_dir, "electricity_forecasting_test_lstm.pkl"),
            "lstm_comb_map_pkl": os.path.join(output_dir, "electricity_comb_to_id_lstm.pkl"),
        },
        "snp500": {
            "id_col": "series_id",
            "data_path": os.path.join(data_dir, "snp500_data.csv"),
            "series_map_pkl": os.path.join(output_dir, "snp500_series_to_id.pkl"),
            "target_cols": ["open", "high", "low", "close", "volume"],
            "test_horizon": 14,
            "seasonal_period": 5,
            "prophet_smape_matrix_pkl": os.path.join(output_dir, "snp500_smape_matrix.pkl"),
            "prophet_forecasting_test_pkl": os.path.join(output_dir, "snp500_forecasting_test.pkl"),
            "prophet_comb_map_pkl": os.path.join(output_dir, "snp500_comb_to_id.pkl"),
            "sarima_smape_matrix_pkl": os.path.join(output_dir, "snp500_smape_matrix_sarima.pkl"),
            "sarima_forecasting_test_pkl": os.path.join(output_dir, "snp500_forecasting_test_sarima.pkl"),
            "sarima_comb_map_pkl": os.path.join(output_dir, "snp500_comb_to_id_sarima.pkl"),
            "lstm_smape_matrix_pkl": os.path.join(output_dir, "snp500_smape_matrix_lstm.pkl"),
            "lstm_forecasting_test_pkl": os.path.join(output_dir, "snp500_forecasting_test_lstm.pkl"),
            "lstm_comb_map_pkl": os.path.join(output_dir, "snp500_comb_to_id_lstm.pkl"),
        },
        "covid19": {
            "id_col": "series_id",
            "data_path": os.path.join(data_dir, "covid19_dataset.pkl"),
            "series_map_pkl": os.path.join(output_dir, "covid19_series_to_id.pkl"),
            "test_horizon": 14,
            "seasonal_period": 7,
            "prophet_smape_matrix_pkl": os.path.join(output_dir, "covid19_smape_matrix.pkl"),
            "prophet_forecasting_test_pkl": os.path.join(output_dir, "covid19_forecasting_test.pkl"),
            "prophet_comb_map_pkl": os.path.join(output_dir, "covid19_comb_to_id.pkl"),
            "sarima_smape_matrix_pkl": os.path.join(output_dir, "covid19_smape_matrix_sarima.pkl"),
            "sarima_forecasting_test_pkl": os.path.join(output_dir, "covid19_forecasting_test_sarima.pkl"),
            "sarima_comb_map_pkl": os.path.join(output_dir, "covid19_comb_to_id_sarima.pkl"),
            "lstm_smape_matrix_pkl": os.path.join(output_dir, "covid19_smape_matrix_lstm.pkl"),
            "lstm_forecasting_test_pkl": os.path.join(output_dir, "covid19_forecasting_test_lstm.pkl"),
            "lstm_comb_map_pkl": os.path.join(output_dir, "covid19_comb_to_id_lstm.pkl"),
        },
    }


def combo_cols_for_part(
    smape_matrix,
    forecasting_matrix,
    comb_map_pkl,
    id_col,
):
    if comb_map_pkl is not None and os.path.exists(comb_map_pkl):
        comb_map = pd.read_pickle(comb_map_pkl).copy()
        comb_map["id"] = comb_map["id"].astype(int)
        candidates = comb_map["id"].astype(int).tolist()
    else:
        candidates = [
            column for column in smape_matrix.columns if column != id_col
        ]

    combo_cols = [
        column
        for column in candidates
        if column in smape_matrix.columns
        and column in forecasting_matrix.columns
    ]
    if len(combo_cols) == 0:
        raise RuntimeError(
            f"No common combo columns found for {comb_map_pkl}"
        )
    return combo_cols


def load_part(config, prefix):
    id_col = config["id_col"]
    smape_matrix = normalize_matrix_columns(
        pd.read_pickle(config[f"{prefix}_smape_matrix_pkl"]),
        id_col,
    )
    forecasting_matrix = normalize_matrix_columns(
        pd.read_pickle(config[f"{prefix}_forecasting_test_pkl"]),
        id_col,
    )
    combo_cols = combo_cols_for_part(
        smape_matrix,
        forecasting_matrix,
        config[f"{prefix}_comb_map_pkl"],
        id_col,
    )
    series_ids = set(smape_matrix[id_col].astype(str)) & set(
        forecasting_matrix[id_col].astype(str)
    )
    return {
        "smape": smape_matrix,
        "forecast": forecasting_matrix,
        "combo_cols": combo_cols,
        "ids": series_ids,
    }


def build_truth_cache(config, series_ids):
    if DATASET == "electricity":
        data = load_electricity_data(
            config["data_path"], config["target_col"]
        )
        return truth_cache_electricity(
            data,
            series_ids,
            config["target_col"],
            config["test_horizon"],
            config["seasonal_period"],
        )

    if DATASET == "snp500":
        data = load_snp500_data(
            config["data_path"], config["target_cols"]
        )
        series_map = pd.read_pickle(config["series_map_pkl"]).copy()
        return truth_cache_snp500(
            data,
            series_map,
            series_ids,
            config["test_horizon"],
            config["seasonal_period"],
        )

    if DATASET == "covid19":
        data, date_cols = load_covid19_data(config["data_path"])
        series_map = pd.read_pickle(config["series_map_pkl"]).copy()
        return truth_cache_covid19(
            data,
            series_map,
            series_ids,
            date_cols,
            config["test_horizon"],
            config["seasonal_period"],
        )

    raise ValueError(DATASET)


def main():
    args = parse_args()
    dataset_configs = build_dataset_configs(args.data_dir, args.output_dir)

    if DATASET not in dataset_configs:
        raise ValueError(f"Unknown DATASET={DATASET}")

    config = dataset_configs[DATASET]
    id_col = config["id_col"]

    prophet = load_part(config, "prophet")
    sarima = load_part(config, "sarima")
    lstm = load_part(config, "lstm")

    common_ids = sorted(
        prophet["ids"] & sarima["ids"] & lstm["ids"],
        key=lambda value: (
            int(value) if str(value).isdigit() else str(value)
        ),
    )
    truth_cache = build_truth_cache(config, common_ids)

    prophet_smape = prophet["smape"].set_index(id_col).loc[common_ids]
    sarima_smape = sarima["smape"].set_index(id_col).loc[common_ids]
    lstm_smape = lstm["smape"].set_index(id_col).loc[common_ids]

    prophet_forecast = prophet["forecast"].set_index(id_col).loc[common_ids]
    sarima_forecast = sarima["forecast"].set_index(id_col).loc[common_ids]
    lstm_forecast = lstm["forecast"].set_index(id_col).loc[common_ids]

    y_true_list = []
    y_pred_list = []
    q_list = []
    failures = []
    selected_models = []
    selected_options = []

    for series_id in common_ids:
        if series_id not in truth_cache:
            failures.append((series_id, "missing truth/q_i cache"))
            continue

        prophet_scores = pd.to_numeric(
            prophet_smape.loc[series_id, prophet["combo_cols"]],
            errors="coerce",
        )
        sarima_scores = pd.to_numeric(
            sarima_smape.loc[series_id, sarima["combo_cols"]],
            errors="coerce",
        )
        lstm_scores = pd.to_numeric(
            lstm_smape.loc[series_id, lstm["combo_cols"]],
            errors="coerce",
        )

        candidates = []
        if prophet_scores.notna().sum() > 0:
            combo_id = prophet_scores.idxmin()
            candidates.append(
                ("prophet", combo_id, float(prophet_scores.loc[combo_id]))
            )
        if sarima_scores.notna().sum() > 0:
            combo_id = sarima_scores.idxmin()
            candidates.append(
                ("sarima", combo_id, float(sarima_scores.loc[combo_id]))
            )
        if lstm_scores.notna().sum() > 0:
            combo_id = lstm_scores.idxmin()
            candidates.append(
                ("lstm", combo_id, float(lstm_scores.loc[combo_id]))
            )

        if len(candidates) == 0:
            failures.append(
                (series_id, "all model/combo scores are non-finite")
            )
            continue

        chosen_model, chosen_combo, _ = min(
            candidates, key=lambda item: item[2]
        )
        selected_models.append(chosen_model)
        selected_options.append(f"{chosen_model}__{chosen_combo}")

        try:
            if chosen_model == "prophet":
                y_pred = get_forecast_value(
                    prophet_forecast.loc[series_id], chosen_combo
                )
            elif chosen_model == "sarima":
                y_pred = get_forecast_value(
                    sarima_forecast.loc[series_id], chosen_combo
                )
            elif chosen_model == "lstm":
                y_pred = get_forecast_value(
                    lstm_forecast.loc[series_id], chosen_combo
                )
            else:
                raise ValueError(chosen_model)
        except Exception as error:
            failures.append((series_id, str(error)))
            continue

        y_test, scale = truth_cache[series_id]
        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_pred) != len(y_test):
            failures.append(
                (
                    series_id,
                    "forecast length mismatch: "
                    f"pred={len(y_pred)}, true={len(y_test)}",
                )
            )
            continue

        y_true_list.append(np.asarray(y_test, dtype=float))
        y_pred_list.append(y_pred)
        q_list.append(
            np.full(len(y_test), float(scale), dtype=float)
        )

    smape, mase, rmsse = compute_metrics(
        y_true_list,
        y_pred_list,
        q_list,
    )
    model_counts = pd.Series(selected_models).value_counts().to_dict()

    print("=== FULL GRID SUBSTITUTE RESULTS: All Models ===")
    print(f"Dataset                 : {DATASET}")
    print("Selection matrices      : Prophet + SARIMA + LSTM")
    print("Forecast matrices       : Prophet + SARIMA + LSTM")
    print(f"Forecast horizon        : {config['test_horizon']}")
    print(f"Successful series       : {len(y_true_list)}")
    print(f"Failed series           : {len(failures)}")
    print(f"Selected model counts   : {model_counts}")
    print(f"Unique selected options : {len(set(selected_options))}")
    print(f"Global sMAPE            : {smape:.6f}")
    print(f"Global MASE             : {mase:.6f}")
    print(f"Global RMSSE            : {rmsse:.6f}")

    if len(failures) > 0:
        print("\nFirst 10 failures:")
        for failure in failures[:10]:
            print(failure)


if __name__ == "__main__":
    main()
