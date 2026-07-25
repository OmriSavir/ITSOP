"""Complete the validation-loss matrix with the nuclear-norm M-estimator."""

import argparse
import os
import time
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


DATASET = "covid19"  # options: "electricity", "snp500", "covid19"

RNG_SEED = 42
N_REPEATS = 10

DELTA = 0.05
ORDER_MULTIPLIERS = [0.2, 0.5, 1.0, 2.0, 5.0]

SIGMA = 0.1
C_LAMBDA = 0.5
ALPHA = 5.0

SCS_MAX_ITERS = 10000
SCS_EPS = 1e-7


def parse_args():
    parser = argparse.ArgumentParser(
        description="Complete the validation-loss matrix with the nuclear-norm M-estimator."
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
        "from here, and this script's own summary pickle is also written "
        "here. Defaults to ./outputs/<dataset>.",
    )
    return parser.parse_args()


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


_ARGS = parse_args()
_OUTPUT_DIR = _ARGS.output_dir or os.path.join("outputs", DATASET)
DATASET_CONFIGS = build_dataset_configs(_ARGS.data_dir, _ARGS.output_dir)


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


def smape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return 100.0 * np.mean((2.0 * np.abs(y_true - y_pred)) / denom)


def global_metrics(y_true_all, y_pred_all, mase_scale_all, rmsse_scale_all, eps=1e-9):
    y_true_all = np.asarray(y_true_all, dtype=float)
    y_pred_all = np.asarray(y_pred_all, dtype=float)
    mase_scale_all = np.asarray(mase_scale_all, dtype=float)
    rmsse_scale_all = np.asarray(rmsse_scale_all, dtype=float)

    s = smape(y_true_all, y_pred_all, eps=eps)
    mase = float(np.mean(mase_scale_all))
    rmsse = float(np.sqrt(np.mean(rmsse_scale_all)))
    return s, mase, rmsse


def seasonal_scale_q(y_train, m, eps=1e-12):
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= m:
        return np.nan
    q = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if (not np.isfinite(q)) or (q <= eps):
        return np.nan
    return float(q)


def frobenius_on_known(X_true, X_hat, valid_mask):
    return float(np.linalg.norm((X_true - X_hat)[valid_mask]))


def training_error(X_true, X_hat, obs_mask):
    return float(np.linalg.norm((X_true - X_hat)[obs_mask]))


def get_forecast_value(forecast_row, combo_id):
    if combo_id in forecast_row.index:
        return forecast_row[combo_id]
    try:
        combo_id_int = int(combo_id)
        if combo_id_int in forecast_row.index:
            return forecast_row[combo_id_int]
    except Exception:
        pass
    combo_id_str = str(combo_id)
    if combo_id_str in forecast_row.index:
        return forecast_row[combo_id_str]
    raise KeyError(f"combo_id={combo_id} not found in forecasting matrix")


def scale_by_observed_frobenius(X, obs_mask):
    obs_vals = X[obs_mask].astype(float)

    n, m = X.shape
    p = obs_mask.sum() / (n * m)

    scale = float(np.sqrt(np.sum(obs_vals ** 2) / p))

    if (not np.isfinite(scale)) or scale <= 0:
        scale = 1.0

    X_scaled = X / scale
    return X_scaled, scale


def complete_m_estimator_cvxpy(Y_scaled, obs_mask, scale_factor):
    start = time.time()

    n, m = Y_scaled.shape
    omega_size = int(obs_mask.sum())
    sigma_scaled = SIGMA / scale_factor
    lambda_n = (
        C_LAMBDA
        * sigma_scaled
        * np.sqrt(((n + m) * np.log(n + m)) / omega_size)
    )
    spikiness_limit = ALPHA / np.sqrt(n * m)

    print(
        f"[MEstimator] Start | n={n}, m={m}, |Omega|={omega_size}, "
        f"lambda_n={lambda_n:.6f}, spikiness_limit={spikiness_limit:.8f}",
        flush=True,
    )

    mask_float = obs_mask.astype(float)
    Y_obs = np.nan_to_num(Y_scaled, nan=0.0)

    Z = cp.Variable((n, m))
    residual = cp.multiply(mask_float, Y_obs - Z)

    loss = (1.0 / (2.0 * omega_size)) * cp.sum_squares(residual)
    penalty = lambda_n * cp.normNuc(Z)

    problem = cp.Problem(
        cp.Minimize(loss + penalty),
        [cp.max(cp.abs(Z)) <= spikiness_limit],
    )

    problem.solve(
        solver=cp.SCS,
        verbose=True,
        max_iters=SCS_MAX_ITERS,
        eps=SCS_EPS,
    )

    elapsed = time.time() - start

    if Z.value is None:
        raise RuntimeError("M-estimator CVXPY failed: Z.value is None")

    Z_hat = np.asarray(Z.value, dtype=float)
    train_err = training_error(Y_scaled, Z_hat, obs_mask)

    print(
        f"[MEstimator] Done | status={problem.status} | "
        f"objective={problem.value:.6f} | train_error={train_err:.6f} | "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )

    return Z_hat


def run_m_estimator(X, valid_mask, obs_mask):
    X_scaled, scale = scale_by_observed_frobenius(X, obs_mask)
    Z_scaled = complete_m_estimator_cvxpy(X_scaled, obs_mask, scale)
    Z_hat = Z_scaled * scale
    Z_hat[~valid_mask] = np.nan
    return Z_hat


def make_theoretical_sample_sizes(n, m, valid_count):
    base = (n + m) * (np.log(n + m) + np.log(1.0 / DELTA))

    out = []
    seen = set()

    for multiplier in ORDER_MULTIPLIERS:
        sample_size = int(np.ceil(multiplier * base))
        if sample_size < 1:
            sample_size = 1

        if sample_size in seen:
            continue
        seen.add(sample_size)

        test_percentage = 100.0 * sample_size / (n * m)
        feasible = sample_size <= valid_count

        out.append({
            "multiplier": float(multiplier),
            "sample_size": int(sample_size),
            "test_percentage": float(test_percentage),
            "feasible": bool(feasible),
        })

    return out


def sample_cumulative_masks(valid_mask, sample_sizes, seed):
    rng = np.random.default_rng(seed)
    valid_indices = np.flatnonzero(valid_mask.ravel())

    feasible_sizes = [s["sample_size"] for s in sample_sizes if s["feasible"]]
    if not feasible_sizes:
        return {}

    max_sample_size = max(feasible_sizes)
    if max_sample_size > len(valid_indices):
        raise ValueError("max_sample_size exceeds number of valid cells")

    shuffled_indices = rng.choice(valid_indices, size=max_sample_size, replace=False)

    masks = {}
    for s in sample_sizes:
        if not s["feasible"]:
            continue

        current_indices = shuffled_indices[:s["sample_size"]]

        obs_mask_flat = np.zeros(valid_mask.size, dtype=bool)
        obs_mask_flat[current_indices] = True
        masks[s["sample_size"]] = obs_mask_flat.reshape(valid_mask.shape)

    return masks


def load_electricity_data(data_path, target_col):
    data = pd.read_pickle(data_path)
    if not np.issubdtype(data["ts"].dtype, np.datetime64):
        data["ts"] = pd.to_datetime(data["ts"], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna(subset=["ts", target_col, "id"]).copy()
    data["id"] = data["id"].astype(str)
    return data


def load_snp500_data(csv_path, target_cols):
    data = pd.read_csv(csv_path)
    try:
        data["date"] = pd.to_datetime(data["date"], format="%m/%d/%Y", errors="raise")
    except Exception:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

    for c in target_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    data = data.dropna(subset=["date", "Name"] + target_cols).copy()
    return data


def load_covid19_data(pkl_path):
    data = pd.read_pickle(pkl_path)

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

    for c in date_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    data = data.dropna(subset=["country", "type"] + date_cols).copy()
    return data, date_cols


def evaluate_selected_combos_electricity(
    data,
    forecasting_matrix,
    id_col,
    series_ids,
    selected_combo_ids,
    target_col,
    test_horizon,
    seasonal_period,
):
    y_true_all = []
    y_pred_all = []
    mase_scale_all = []
    rmsse_scale_all = []
    failures = []

    data_by_id = {
        str(series_id): df_s.sort_values("ts").copy()
        for series_id, df_s in data.groupby("id", sort=False)
    }
    forecasting_lookup = forecasting_matrix.set_index(id_col)

    for series_id, combo_id in zip(series_ids, selected_combo_ids):
        series_id_str = str(series_id)

        if series_id_str not in data_by_id:
            failures.append((series_id_str, "empty series"))
            continue

        if series_id_str not in forecasting_lookup.index:
            failures.append((series_id_str, "missing forecasting row"))
            continue

        df_s = data_by_id[series_id_str]
        if len(df_s) <= test_horizon:
            failures.append((series_id_str, "series too short"))
            continue

        y_all = df_s[target_col].astype(float).to_numpy()
        split_idx = len(y_all) - test_horizon
        y_train = y_all[:split_idx]
        y_test = y_all[split_idx:]

        q_i = seasonal_scale_q(y_train, m=seasonal_period)
        if not np.isfinite(q_i):
            failures.append((series_id_str, "invalid Q_i for MASE/RMSSE"))
            continue

        try:
            y_pred = get_forecast_value(forecasting_lookup.loc[series_id_str], combo_id)
        except Exception as e:
            failures.append((series_id_str, str(e)))
            continue

        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_pred) != len(y_test):
            failures.append(
                (
                    series_id_str,
                    f"forecast length mismatch: pred={len(y_pred)}, "
                    f"true={len(y_test)}",
                )
            )
            continue

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        mase_scale_all.extend(np.abs(y_test - y_pred) / q_i)
        rmsse_scale_all.extend(((y_test - y_pred) / q_i) ** 2)

    if len(y_true_all) == 0:
        return None

    s, m, r = global_metrics(y_true_all, y_pred_all, mase_scale_all, rmsse_scale_all)

    return {
        "n_series": len(y_true_all) // test_horizon,
        "n_failures": len(failures),
        "failures": failures,
        "sMAPE": s,
        "MASE": m,
        "RMSSE": r,
    }


def evaluate_selected_combos_snp500(
    data,
    series_map,
    forecasting_matrix,
    id_col,
    series_ids,
    selected_combo_ids,
    test_horizon,
    seasonal_period,
):
    y_true_all = []
    y_pred_all = []
    mase_scale_all = []
    rmsse_scale_all = []
    failures = []

    data_by_name = {
        str(name): df_s.sort_values("date").copy()
        for name, df_s in data.groupby("Name", sort=False)
    }

    series_map_lookup = series_map.copy()
    series_map_lookup["series_id"] = series_map_lookup["series_id"].astype(str)
    series_map_lookup = series_map_lookup.set_index("series_id")

    forecasting_lookup = forecasting_matrix.set_index(id_col)

    for series_id, combo_id in zip(series_ids, selected_combo_ids):
        series_id_str = str(series_id)

        if series_id_str not in series_map_lookup.index:
            failures.append((series_id_str, "missing series_map row"))
            continue

        if series_id_str not in forecasting_lookup.index:
            failures.append((series_id_str, "missing forecasting row"))
            continue

        info = series_map_lookup.loc[series_id_str]
        name = str(info["Name"])
        typ = info["Type"]

        if name not in data_by_name:
            failures.append((series_id_str, "empty series"))
            continue

        df_s = data_by_name[name]
        if len(df_s) <= test_horizon:
            failures.append((series_id_str, "series too short"))
            continue

        y_all = df_s[typ].astype(float).to_numpy()
        split_idx = len(y_all) - test_horizon
        y_train = y_all[:split_idx]
        y_test = y_all[split_idx:]

        q_i = seasonal_scale_q(y_train, m=seasonal_period)
        if not np.isfinite(q_i):
            failures.append((series_id_str, "invalid Q_i for MASE/RMSSE"))
            continue

        try:
            y_pred = get_forecast_value(forecasting_lookup.loc[series_id_str], combo_id)
        except Exception as e:
            failures.append((series_id_str, str(e)))
            continue

        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_pred) != len(y_test):
            failures.append(
                (
                    series_id_str,
                    f"forecast length mismatch: pred={len(y_pred)}, "
                    f"true={len(y_test)}",
                )
            )
            continue

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        mase_scale_all.extend(np.abs(y_test - y_pred) / q_i)
        rmsse_scale_all.extend(((y_test - y_pred) / q_i) ** 2)

    if len(y_true_all) == 0:
        return None

    s, m, r = global_metrics(y_true_all, y_pred_all, mase_scale_all, rmsse_scale_all)

    return {
        "n_series": len(y_true_all) // test_horizon,
        "n_failures": len(failures),
        "failures": failures,
        "sMAPE": s,
        "MASE": m,
        "RMSSE": r,
    }


def evaluate_selected_combos_covid19(
    data,
    date_cols,
    series_map,
    forecasting_matrix,
    id_col,
    series_ids,
    selected_combo_ids,
    test_horizon,
    seasonal_period,
):
    y_true_all = []
    y_pred_all = []
    mase_scale_all = []
    rmsse_scale_all = []
    failures = []

    series_map_lookup = series_map.copy()
    series_map_lookup["series_id"] = series_map_lookup["series_id"].astype(str)
    series_map_lookup = series_map_lookup.set_index("series_id")

    forecasting_lookup = forecasting_matrix.set_index(id_col)

    for series_id, combo_id in zip(series_ids, selected_combo_ids):
        series_id_str = str(series_id)

        if series_id_str not in series_map_lookup.index:
            failures.append((series_id_str, "missing series_map row"))
            continue

        if series_id_str not in forecasting_lookup.index:
            failures.append((series_id_str, "missing forecasting row"))
            continue

        info = series_map_lookup.loc[series_id_str]
        country = info["country"]
        typ = info["type"]

        df_s = data[(data["country"] == country) & (data["type"] == typ)]
        if len(df_s) == 0:
            failures.append((series_id_str, "empty series"))
            continue

        y_all = df_s.iloc[0][date_cols].astype(float).to_numpy()
        if len(y_all) <= test_horizon:
            failures.append((series_id_str, "series too short"))
            continue

        split_idx = len(y_all) - test_horizon
        y_train = y_all[:split_idx]
        y_test = y_all[split_idx:]

        q_i = seasonal_scale_q(y_train, m=seasonal_period)
        if not np.isfinite(q_i):
            failures.append((series_id_str, "invalid Q_i for MASE/RMSSE"))
            continue

        try:
            y_pred = get_forecast_value(forecasting_lookup.loc[series_id_str], combo_id)
        except Exception as e:
            failures.append((series_id_str, str(e)))
            continue

        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_pred) != len(y_test):
            failures.append(
                (
                    series_id_str,
                    f"forecast length mismatch: pred={len(y_pred)}, "
                    f"true={len(y_test)}",
                )
            )
            continue

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        mase_scale_all.extend(np.abs(y_test - y_pred) / q_i)
        rmsse_scale_all.extend(((y_test - y_pred) / q_i) ** 2)

    if len(y_true_all) == 0:
        return None

    s, m, r = global_metrics(y_true_all, y_pred_all, mase_scale_all, rmsse_scale_all)

    return {
        "n_series": len(y_true_all) // test_horizon,
        "n_failures": len(failures),
        "failures": failures,
        "sMAPE": s,
        "MASE": m,
        "RMSSE": r,
    }


def summarize_repeat_results(rows):
    if len(rows) == 0:
        return {
            "mean_frob_all": np.nan,
            "mean_train_error": np.nan,
            "mean_sMAPE": np.nan,
            "mean_MASE": np.nan,
            "mean_RMSSE": np.nan,
            "mean_n_series": 0.0,
            "mean_n_failures": np.nan,
            "mean_elapsed_sec": np.nan,
        }

    df = pd.DataFrame(rows)
    return {
        "mean_frob_all": float(df["frob_all"].mean()),
        "mean_train_error": float(df["train_error"].mean()),
        "mean_sMAPE": float(df["sMAPE"].mean()),
        "mean_MASE": float(df["MASE"].mean()),
        "mean_RMSSE": float(df["RMSSE"].mean()),
        "mean_n_series": float(df["n_series"].mean()),
        "mean_n_failures": float(df["n_failures"].mean()),
        "mean_elapsed_sec": float(df["elapsed_sec"].mean()),
    }


def get_combo_columns_for_part(smape_matrix, forecasting_matrix, comb_map_pkl, id_col):
    if comb_map_pkl is not None and os.path.exists(comb_map_pkl):
        comb_map = pd.read_pickle(comb_map_pkl).copy()
        comb_map["id"] = comb_map["id"].astype(int)
        candidate_cols = comb_map["id"].astype(int).tolist()
    else:
        candidate_cols = [c for c in smape_matrix.columns if c != id_col]

    combo_cols = [
        c for c in candidate_cols
        if c in smape_matrix.columns and c in forecasting_matrix.columns
    ]

    if len(combo_cols) == 0:
        raise RuntimeError(f"No common combo columns found for comb_map={comb_map_pkl}")

    return combo_cols


def load_part(cfg, prefix):
    id_col = cfg["id_col"]

    smape_matrix = pd.read_pickle(cfg[f"{prefix}_smape_matrix_pkl"])
    forecasting_matrix = pd.read_pickle(cfg[f"{prefix}_forecasting_test_pkl"])

    smape_matrix = normalize_matrix_columns(smape_matrix, id_col)
    forecasting_matrix = normalize_matrix_columns(forecasting_matrix, id_col)

    combo_cols = get_combo_columns_for_part(
        smape_matrix=smape_matrix,
        forecasting_matrix=forecasting_matrix,
        comb_map_pkl=cfg[f"{prefix}_comb_map_pkl"],
        id_col=id_col,
    )

    common_ids = set(smape_matrix[id_col].astype(str)) & set(
        forecasting_matrix[id_col].astype(str)
    )

    return {
        "smape_matrix": smape_matrix,
        "forecasting_matrix": forecasting_matrix,
        "combo_cols": combo_cols,
        "common_ids": common_ids,
    }


def load_all_models_matrix():
    cfg = DATASET_CONFIGS[DATASET]
    id_col = cfg["id_col"]

    prophet_part = load_part(cfg, "prophet")
    sarima_part = load_part(cfg, "sarima")
    lstm_part = load_part(cfg, "lstm")

    common_ids = sorted(
        prophet_part["common_ids"] & sarima_part["common_ids"] & lstm_part["common_ids"],
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )

    if len(common_ids) == 0:
        raise RuntimeError("No common series across Prophet, SARIMA, and LSTM matrices.")

    parts = []
    option_lookup = {}

    for prefix, part in [
        ("prophet", prophet_part),
        ("sarima", sarima_part),
        ("lstm", lstm_part),
    ]:
        smape_aligned = part["smape_matrix"].set_index(id_col).loc[common_ids]

        for combo_id in part["combo_cols"]:
            label = f"{prefix}__{combo_id}"
            parts.append(smape_aligned[[combo_id]].rename(columns={combo_id: label}))
            option_lookup[label] = (prefix, combo_id)

    X_df = pd.concat(parts, axis=1)
    X_raw = X_df.to_numpy(dtype=float)

    row_has_finite_value = np.isfinite(X_raw).any(axis=1)
    n_removed = int((~row_has_finite_value).sum())
    if n_removed > 0:
        print(
            f"[INFO] Removed {n_removed} series with no finite values "
            "in the all-models matrix.",
            flush=True,
        )

    common_ids = [sid for sid, keep in zip(common_ids, row_has_finite_value) if keep]
    X = X_raw[row_has_finite_value, :]
    X[~np.isfinite(X)] = np.nan

    valid_mask = np.isfinite(X)
    option_ids = list(X_df.columns)

    prophet_forecast = prophet_part["forecasting_matrix"].set_index(id_col).loc[common_ids]
    sarima_forecast = sarima_part["forecasting_matrix"].set_index(id_col).loc[common_ids]
    lstm_forecast = lstm_part["forecasting_matrix"].set_index(id_col).loc[common_ids]

    forecast_lookup_by_prefix = {
        "prophet": prophet_forecast,
        "sarima": sarima_forecast,
        "lstm": lstm_forecast,
    }

    return X, valid_mask, common_ids, option_ids, option_lookup, forecast_lookup_by_prefix


def evaluate_all_models_forecasts(
    series_ids,
    selected_option_ids,
    option_lookup,
    forecast_lookup_by_prefix,
):
    cfg = DATASET_CONFIGS[DATASET]

    if DATASET == "electricity":
        data = load_electricity_data(cfg["data_path"], cfg["target_col"])
    elif DATASET == "snp500":
        data = load_snp500_data(cfg["data_path"], cfg["target_cols"])
        series_map = pd.read_pickle(cfg["series_map_pkl"]).copy()
    elif DATASET == "covid19":
        data, date_cols = load_covid19_data(cfg["data_path"])
        series_map = pd.read_pickle(cfg["series_map_pkl"]).copy()
    else:
        raise ValueError(f"Unknown DATASET={DATASET}")

    y_true_all = []
    y_pred_all = []
    mase_scale_all = []
    rmsse_scale_all = []
    failures = []

    if DATASET == "electricity":
        data_by_id = {
            str(series_id): df_s.sort_values("ts").copy()
            for series_id, df_s in data.groupby("id", sort=False)
        }

    elif DATASET == "snp500":
        data_by_name = {
            str(name): df_s.sort_values("date").copy()
            for name, df_s in data.groupby("Name", sort=False)
        }
        series_map_lookup = series_map.copy()
        series_map_lookup["series_id"] = series_map_lookup["series_id"].astype(str)
        series_map_lookup = series_map_lookup.set_index("series_id")

    elif DATASET == "covid19":
        series_map_lookup = series_map.copy()
        series_map_lookup["series_id"] = series_map_lookup["series_id"].astype(str)
        series_map_lookup = series_map_lookup.set_index("series_id")

    for series_id, option_id in zip(series_ids, selected_option_ids):
        series_id_str = str(series_id)

        if option_id not in option_lookup:
            failures.append((series_id_str, f"unknown option_id={option_id}"))
            continue

        prefix, combo_id = option_lookup[option_id]

        if series_id_str not in forecast_lookup_by_prefix[prefix].index:
            failures.append((series_id_str, f"missing forecasting row for {prefix}"))
            continue

        try:
            forecast_row = forecast_lookup_by_prefix[prefix].loc[series_id_str]
            y_pred = get_forecast_value(forecast_row, combo_id)
        except Exception as e:
            failures.append((series_id_str, str(e)))
            continue

        if DATASET == "electricity":
            if series_id_str not in data_by_id:
                failures.append((series_id_str, "empty series"))
                continue

            df_s = data_by_id[series_id_str]
            if len(df_s) <= cfg["test_horizon"]:
                failures.append((series_id_str, "series too short"))
                continue

            y_all = df_s[cfg["target_col"]].astype(float).to_numpy()

        elif DATASET == "snp500":
            if series_id_str not in series_map_lookup.index:
                failures.append((series_id_str, "missing series_map row"))
                continue

            info = series_map_lookup.loc[series_id_str]
            name = str(info["Name"])
            typ = info["Type"]

            if name not in data_by_name:
                failures.append((series_id_str, "empty series"))
                continue

            df_s = data_by_name[name]
            if len(df_s) <= cfg["test_horizon"]:
                failures.append((series_id_str, "series too short"))
                continue

            y_all = df_s[typ].astype(float).to_numpy()

        elif DATASET == "covid19":
            if series_id_str not in series_map_lookup.index:
                failures.append((series_id_str, "missing series_map row"))
                continue

            info = series_map_lookup.loc[series_id_str]
            country = info["country"]
            typ = info["type"]

            df_s = data[(data["country"] == country) & (data["type"] == typ)]
            if len(df_s) == 0:
                failures.append((series_id_str, "empty series"))
                continue

            y_all = df_s.iloc[0][date_cols].astype(float).to_numpy()
            if len(y_all) <= cfg["test_horizon"]:
                failures.append((series_id_str, "series too short"))
                continue

        split_idx = len(y_all) - cfg["test_horizon"]
        y_train = y_all[:split_idx]
        y_test = y_all[split_idx:]

        q_i = seasonal_scale_q(y_train, m=cfg["seasonal_period"])
        if not np.isfinite(q_i):
            failures.append((series_id_str, "invalid Q_i for MASE/RMSSE"))
            continue

        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_pred) != len(y_test):
            failures.append(
                (
                    series_id_str,
                    f"forecast length mismatch: pred={len(y_pred)}, "
                    f"true={len(y_test)}",
                )
            )
            continue

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        mase_scale_all.extend(np.abs(y_test - y_pred) / q_i)
        rmsse_scale_all.extend(((y_test - y_pred) / q_i) ** 2)

    if len(y_true_all) == 0:
        return None

    s, m, r = global_metrics(y_true_all, y_pred_all, mase_scale_all, rmsse_scale_all)

    return {
        "n_series": len(y_true_all) // DATASET_CONFIGS[DATASET]["test_horizon"],
        "n_failures": len(failures),
        "failures": failures,
        "sMAPE": s,
        "MASE": m,
        "RMSSE": r,
    }



def main():
    if DATASET not in DATASET_CONFIGS:
        raise ValueError(f"Unknown DATASET={DATASET}")

    (
        X,
        valid_mask,
        series_ids,
        option_ids,
        option_lookup,
        forecast_lookup_by_prefix,
    ) = load_all_models_matrix()

    n, m = X.shape
    valid_count = int(valid_mask.sum())
    sample_sizes = make_theoretical_sample_sizes(n, m, valid_count)

    print("=== M-ESTIMATOR MATRIX COMPLETION: all_models ===", flush=True)
    print(f"Dataset       : {DATASET}", flush=True)
    print(f"Matrix shape  : {X.shape}", flush=True)
    print(f"Valid cells   : {valid_count}", flush=True)
    print(f"Sample sizes  : {[s for s in sample_sizes]}", flush=True)
    print("", flush=True)

    results_by_sample_size = {
        s["sample_size"]: []
        for s in sample_sizes
        if s["feasible"]
    }

    for repeat_idx in range(N_REPEATS):
        seed = RNG_SEED + 1000 * repeat_idx
        masks = sample_cumulative_masks(valid_mask, sample_sizes, seed)

        for s in sample_sizes:
            sample_size = s["sample_size"]

            if not s["feasible"]:
                continue

            obs_mask = masks[sample_size]

            print(
                f"[START] repeat={repeat_idx + 1}/{N_REPEATS} | "
                f"sample_size={sample_size} | "
                f"multiplier={s['multiplier']} | "
                f"test_percentage={s['test_percentage']:.6f}%",
                flush=True
            )

            start = time.time()

            try:
                Z_hat = run_m_estimator(X, valid_mask, obs_mask)

                frob_all = frobenius_on_known(X, Z_hat, valid_mask)
                train_err = training_error(X, Z_hat, obs_mask)

                selected_col_idx = np.nanargmin(Z_hat, axis=1)
                selected_option_ids = np.array(option_ids, dtype=object)[selected_col_idx]

                forecast_res = evaluate_all_models_forecasts(
                    series_ids=series_ids,
                    selected_option_ids=selected_option_ids,
                    option_lookup=option_lookup,
                    forecast_lookup_by_prefix=forecast_lookup_by_prefix,
                )

                elapsed = time.time() - start

                if forecast_res is None:
                    row = {
                        "repeat": repeat_idx + 1,
                        "sample_size": sample_size,
                        "test_percentage": s["test_percentage"],
                        "frob_all": frob_all,
                        "train_error": train_err,
                        "n_series": 0,
                        "n_failures": len(series_ids),
                        "sMAPE": np.nan,
                        "MASE": np.nan,
                        "RMSSE": np.nan,
                        "elapsed_sec": elapsed,
                    }
                else:
                    row = {
                        "repeat": repeat_idx + 1,
                        "sample_size": sample_size,
                        "test_percentage": s["test_percentage"],
                        "frob_all": frob_all,
                        "train_error": train_err,
                        "n_series": forecast_res["n_series"],
                        "n_failures": forecast_res["n_failures"],
                        "sMAPE": forecast_res["sMAPE"],
                        "MASE": forecast_res["MASE"],
                        "RMSSE": forecast_res["RMSSE"],
                        "elapsed_sec": elapsed,
                    }

            except Exception as e:
                elapsed = time.time() - start
                row = {
                    "repeat": repeat_idx + 1,
                    "sample_size": sample_size,
                    "test_percentage": s["test_percentage"],
                    "frob_all": np.nan,
                    "train_error": np.nan,
                    "n_series": 0,
                    "n_failures": len(series_ids),
                    "sMAPE": np.nan,
                    "MASE": np.nan,
                    "RMSSE": np.nan,
                    "elapsed_sec": elapsed,
                    "error": repr(e),
                }

            results_by_sample_size[sample_size].append(row)

            print(
                f"[RESULT] repeat={repeat_idx + 1}/{N_REPEATS} | "
                f"sample_size={sample_size} | "
                f"sMAPE={row['sMAPE']:.6f} | "
                f"MASE={row['MASE']:.6f} | "
                f"RMSSE={row['RMSSE']:.6f} | "
                f"n_series={row['n_series']} | "
                f"n_failures={row['n_failures']} | "
                f"elapsed={row['elapsed_sec']:.2f}s",
                flush=True
            )

    summary_rows = []

    for s in sample_sizes:
        if not s["feasible"]:
            print(
                f"[SKIPPED] sample_size={s['sample_size']} | "
                f"test_percentage={s['test_percentage']:.6f}% | "
                f"reason=sample_size exceeds valid cells",
                flush=True
            )
            continue

        sample_size = s["sample_size"]
        summary = summarize_repeat_results(results_by_sample_size[sample_size])

        out_row = {
            "dataset": DATASET,
            "model": "all_models",
            "algorithm": "m_estimator",
            "multiplier": s["multiplier"],
            "sample_size": sample_size,
            "test_percentage": s["test_percentage"],
            "n_repeats": N_REPEATS,
            **summary,
        }
        summary_rows.append(out_row)

        print(
            f"[MEAN RESULT] dataset={DATASET} | model=all_models | "
            f"algorithm=m_estimator | multiplier={s['multiplier']} | "
            f"sample_size={sample_size} | "
            f"test_percentage={s['test_percentage']:.6f}% | "
            f"mean_frob_all={summary['mean_frob_all']:.6f} | "
            f"mean_train_error={summary['mean_train_error']:.6f} | "
            f"mean_n_series={summary['mean_n_series']:.2f} | "
            f"mean_n_failures={summary['mean_n_failures']:.2f} | "
            f"mean_sMAPE={summary['mean_sMAPE']:.6f} | "
            f"mean_MASE={summary['mean_MASE']:.6f} | "
            f"mean_RMSSE={summary['mean_RMSSE']:.6f} | "
            f"mean_elapsed_sec={summary['mean_elapsed_sec']:.2f}",
            flush=True
        )

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    output_pkl = os.path.join(_OUTPUT_DIR, f"{DATASET}_m_estimator_all_models_summary.pkl")
    summary_df.to_pickle(output_pkl)
    print(f"Saved {output_pkl}", flush=True)


if __name__ == "__main__":
    main()
