"""Evaluate one-hot MLP matrix completion across all forecasting models."""

import argparse
import logging
import os
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)

# Configuration
DATASET = "covid19"  # options: "electricity", "snp500", "covid19"

HIDDEN_DIM = 128
N_HIDDEN_LAYERS = 2
DROPOUT = 0.25
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 4096
VALIDATION_FRACTION = 0.1

RNG_SEED = 42
N_REPEATS = 10

DELTA = 0.05
ORDER_MULTIPLIERS = [0.2, 0.5, 1.0, 2.0, 5.0]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one-hot MLP matrix completion across all forecasting models."
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
            "prophet_matrix_pkl": os.path.join(output_dir, "electricity_smape_matrix.pkl"),
            "prophet_forecasting_test_pkl": os.path.join(output_dir, "electricity_forecasting_test.pkl"),
            "prophet_comb_map_pkl": os.path.join(output_dir, "electricity_comb_to_id.pkl"),
            "sarima_matrix_pkl": os.path.join(output_dir, "electricity_smape_matrix_sarima.pkl"),
            "sarima_forecasting_test_pkl": os.path.join(output_dir, "electricity_forecasting_test_sarima.pkl"),
            "sarima_comb_map_pkl": os.path.join(output_dir, "electricity_comb_to_id_sarima.pkl"),
            "lstm_matrix_pkl": os.path.join(output_dir, "electricity_smape_matrix_lstm.pkl"),
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
            "prophet_matrix_pkl": os.path.join(output_dir, "snp500_smape_matrix.pkl"),
            "prophet_forecasting_test_pkl": os.path.join(output_dir, "snp500_forecasting_test.pkl"),
            "prophet_comb_map_pkl": os.path.join(output_dir, "snp500_comb_to_id.pkl"),
            "sarima_matrix_pkl": os.path.join(output_dir, "snp500_smape_matrix_sarima.pkl"),
            "sarima_forecasting_test_pkl": os.path.join(output_dir, "snp500_forecasting_test_sarima.pkl"),
            "sarima_comb_map_pkl": os.path.join(output_dir, "snp500_comb_to_id_sarima.pkl"),
            "lstm_matrix_pkl": os.path.join(output_dir, "snp500_smape_matrix_lstm.pkl"),
            "lstm_forecasting_test_pkl": os.path.join(output_dir, "snp500_forecasting_test_lstm.pkl"),
            "lstm_comb_map_pkl": os.path.join(output_dir, "snp500_comb_to_id_lstm.pkl"),
        },
        "covid19": {
            "id_col": "series_id",
            "data_path": os.path.join(data_dir, "covid19_dataset.pkl"),
            "series_map_pkl": os.path.join(output_dir, "covid19_series_to_id.pkl"),
            "test_horizon": 14,
            "seasonal_period": 7,
            "prophet_matrix_pkl": os.path.join(output_dir, "covid19_smape_matrix.pkl"),
            "prophet_forecasting_test_pkl": os.path.join(output_dir, "covid19_forecasting_test.pkl"),
            "prophet_comb_map_pkl": os.path.join(output_dir, "covid19_comb_to_id.pkl"),
            "sarima_matrix_pkl": os.path.join(output_dir, "covid19_smape_matrix_sarima.pkl"),
            "sarima_forecasting_test_pkl": os.path.join(output_dir, "covid19_forecasting_test_sarima.pkl"),
            "sarima_comb_map_pkl": os.path.join(output_dir, "covid19_comb_to_id_sarima.pkl"),
            "lstm_matrix_pkl": os.path.join(output_dir, "covid19_smape_matrix_lstm.pkl"),
            "lstm_forecasting_test_pkl": os.path.join(output_dir, "covid19_forecasting_test_lstm.pkl"),
            "lstm_comb_map_pkl": os.path.join(output_dir, "covid19_comb_to_id_lstm.pkl"),
        },
    }


_ARGS = parse_args()
_OUTPUT_DIR = _ARGS.output_dir or os.path.join("outputs", DATASET)
DATASET_CONFIGS = build_dataset_configs(_ARGS.data_dir, _ARGS.output_dir)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return np.nan
    return 1.0 - (ss_res / ss_tot)


def frob_norm_masked(A_true, A_pred, mask):
    diff = (A_true - A_pred)[mask]
    return float(np.sqrt(np.sum(diff * diff)))


def seasonal_scale_q(y_train, m, eps=1e-12):
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= m:
        return np.nan
    q = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if (not np.isfinite(q)) or (q <= eps):
        return np.nan
    return float(q)


def global_smape(y_true_all, y_pred_all, eps=1e-9):
    y_true_all = np.asarray(y_true_all, dtype=float)
    y_pred_all = np.asarray(y_pred_all, dtype=float)
    return 100.0 * np.mean(
        2.0 * np.abs(y_true_all - y_pred_all)
        / (np.abs(y_true_all) + np.abs(y_pred_all) + eps)
    )


def get_forecast_value(forecast_row, combo_id):
    candidates = [combo_id, str(combo_id)]
    try:
        candidates.append(int(combo_id))
    except Exception:
        pass

    for c in candidates:
        if c in forecast_row.index:
            return forecast_row[c]

    raise KeyError(f"combo_id={combo_id} not found in forecasting matrix")


def make_theoretical_sample_sizes(n, m, valid_count):
    base = (n + m) * (np.log(n + m) + np.log(1.0 / DELTA))
    out = []
    seen = set()

    for multiplier in ORDER_MULTIPLIERS:
        sample_size = int(np.ceil(multiplier * base))
        sample_size = max(1, sample_size)

        if sample_size in seen:
            continue
        seen.add(sample_size)

        out.append({
            "multiplier": float(multiplier),
            "sample_size": int(sample_size),
            "test_percentage": float(100.0 * sample_size / (n * m)),
            "feasible": bool(sample_size <= valid_count),
        })

    return out


def sample_cumulative_masks(valid_mask, sample_sizes, seed):
    feasible_sizes = [s["sample_size"] for s in sample_sizes if s["feasible"]]
    if len(feasible_sizes) == 0:
        return {}

    rng = np.random.default_rng(seed)
    valid_indices = np.flatnonzero(valid_mask.ravel())
    max_sample_size = max(feasible_sizes)

    if max_sample_size > len(valid_indices):
        raise ValueError("max_sample_size exceeds number of valid cells")

    shuffled_indices = rng.choice(valid_indices, size=max_sample_size, replace=False)

    masks = {}
    for sample_size in feasible_sizes:
        current_indices = shuffled_indices[:sample_size]
        obs_mask_flat = np.zeros(valid_mask.size, dtype=bool)
        obs_mask_flat[current_indices] = True
        masks[sample_size] = obs_mask_flat.reshape(valid_mask.shape)

    return masks

# MLP one-hot model


class MLP(nn.Module):
    """MLP used to predict an entry from series and model-option one-hot vectors."""

    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, inputs):
        return self.network(inputs).squeeze(1)


def build_onehot_features(series_indices, option_indices, n_series, n_options):
    series_onehot = np.zeros((len(series_indices), n_series), dtype=np.float32)
    option_onehot = np.zeros((len(option_indices), n_options), dtype=np.float32)
    rows = np.arange(len(series_indices))
    series_onehot[rows, series_indices] = 1.0
    option_onehot[rows, option_indices] = 1.0
    return np.concatenate([series_onehot, option_onehot], axis=1)


def split_observed_entries(features, targets, seed):
    if len(targets) < 2:
        raise RuntimeError("Need at least two observed cells for train/validation split")

    set_seed(seed)
    permutation = np.random.permutation(len(targets))
    n_validation = max(1, int(VALIDATION_FRACTION * len(targets)))
    n_validation = min(n_validation, len(targets) - 1)

    validation_indices = permutation[:n_validation]
    train_indices = permutation[n_validation:]

    return (
        features[train_indices],
        targets[train_indices],
        features[validation_indices],
        targets[validation_indices],
    )


def train_mlp(train_features, train_targets, validation_features, validation_targets):
    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_targets = torch.tensor(train_targets, dtype=torch.float32)
    validation_features = torch.tensor(validation_features, dtype=torch.float32, device=DEVICE)
    validation_targets = torch.tensor(validation_targets, dtype=torch.float32, device=DEVICE)

    train_loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=min(BATCH_SIZE, len(train_features)),
        shuffle=True,
    )

    model = MLP(input_dim=train_features.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()

    best_state = None
    best_validation_loss = np.inf
    epochs_without_improvement = 0

    for _ in range(MAX_EPOCHS):
        model.train()
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = loss_function(predictions, batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_predictions = model(validation_features)
            validation_loss = loss_function(
                validation_predictions,
                validation_targets,
            ).item()

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("MLP training did not produce a valid model state")

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_full_matrix(model, matrix_shape):
    n_series, n_options = matrix_shape
    matrix_prediction = np.full(matrix_shape, np.nan, dtype=float)
    series_block_size = 128

    for start in range(0, n_series, series_block_size):
        end = min(start + series_block_size, n_series)
        series_indices = np.repeat(np.arange(start, end), n_options)
        option_indices = np.tile(np.arange(n_options), end - start)
        block_features = build_onehot_features(
            series_indices,
            option_indices,
            n_series,
            n_options,
        )

        with torch.no_grad():
            block_prediction = model(
                torch.tensor(block_features, dtype=torch.float32, device=DEVICE)
            ).cpu().numpy()

        matrix_prediction[start:end] = block_prediction.reshape(end - start, n_options)

    return matrix_prediction


def complete_matrix_with_mlp(X, valid_mask, mask_obs, repeat_idx, sample_size):
    seed = RNG_SEED + 100000 * repeat_idx + sample_size
    observed_series, observed_options = np.where(mask_obs)
    observed_targets = X[mask_obs].astype(np.float32)
    observed_features = build_onehot_features(
        observed_series,
        observed_options,
        X.shape[0],
        X.shape[1],
    )

    split = split_observed_entries(observed_features, observed_targets, seed)
    model = train_mlp(*split)
    matrix_prediction = predict_full_matrix(model, X.shape)
    matrix_prediction[~valid_mask] = np.nan

    prediction_mask = valid_mask & (~mask_obs)
    metrics = {
        "hidden_dim": HIDDEN_DIM,
        "n_hidden_layers": N_HIDDEN_LAYERS,
        "dropout": DROPOUT,
        "lr": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "activation": "relu",
        "r2_pred": (
            r2_score(X[prediction_mask], matrix_prediction[prediction_mask])
            if prediction_mask.any()
            else np.nan
        ),
        "r2_all": r2_score(X[valid_mask], matrix_prediction[valid_mask]),
        "frob": frob_norm_masked(X, matrix_prediction, valid_mask),
    }

    completed = np.full_like(X, np.nan, dtype=float)
    completed[mask_obs] = X[mask_obs]
    completed[prediction_mask] = matrix_prediction[prediction_mask]
    return completed, metrics

# Evaluation data loaders


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

    for col in date_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["country", "type"] + date_cols).copy()
    return data, date_cols


def summarize_repeat_rows(rows):
    if len(rows) == 0:
        return {
            "mean_r2_pred": np.nan,
            "mean_r2_all": np.nan,
            "mean_frob": np.nan,
            "mean_sMAPE": np.nan,
            "mean_MASE": np.nan,
            "mean_RMSSE": np.nan,
            "mean_n_series": 0.0,
            "mean_n_failures": np.nan,
            "mean_elapsed_sec": np.nan,
        }

    df = pd.DataFrame(rows)
    return {
        "mean_r2_pred": float(df["r2_pred"].mean()),
        "mean_r2_all": float(df["r2_all"].mean()),
        "mean_frob": float(df["frob"].mean()),
        "mean_sMAPE": float(df["sMAPE"].mean()),
        "mean_MASE": float(df["MASE"].mean()),
        "mean_RMSSE": float(df["RMSSE"].mean()),
        "mean_n_series": float(df["n_series"].mean()),
        "mean_n_failures": float(df["n_failures"].mean()),
        "mean_elapsed_sec": float(df["elapsed_sec"].mean()),
    }


def load_part(cfg, prefix):
    id_col = cfg["id_col"]

    matrix_df = pd.read_pickle(cfg[f"{prefix}_matrix_pkl"])
    matrix_df = normalize_matrix_columns(matrix_df, id_col)

    forecasting_matrix = pd.read_pickle(cfg[f"{prefix}_forecasting_test_pkl"])
    forecasting_matrix = normalize_matrix_columns(forecasting_matrix, id_col)

    comb_map_pkl = cfg[f"{prefix}_comb_map_pkl"]
    if comb_map_pkl is not None and os.path.exists(comb_map_pkl):
        comb_df = pd.read_pickle(comb_map_pkl).sort_values("id")
        combo_cols = comb_df["id"].astype(int).tolist()
    else:
        combo_cols = [c for c in matrix_df.columns if c != id_col]

    combo_cols = [
        combo_id
        for combo_id in combo_cols
        if combo_id in matrix_df.columns and combo_id in forecasting_matrix.columns
    ]
    if len(combo_cols) == 0:
        raise RuntimeError(f"No common combo columns found for {prefix}.")

    common_ids = set(matrix_df[id_col].astype(str)) & set(forecasting_matrix[id_col].astype(str))

    return {
        "matrix_df": matrix_df,
        "forecasting_matrix": forecasting_matrix,
        "combo_cols": combo_cols,
        "common_ids": common_ids,
    }


def load_all_models_matrix_and_forecasts():
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
        raise RuntimeError("No common series across Prophet, SARIMA, and LSTM.")

    pieces = []
    option_lookup = {}

    for prefix, part in [
        ("prophet", prophet_part),
        ("sarima", sarima_part),
        ("lstm", lstm_part),
    ]:
        aligned = part["matrix_df"].set_index(id_col).loc[common_ids]
        for combo_id in part["combo_cols"]:
            option_id = f"{prefix}__{combo_id}"
            pieces.append(aligned[[combo_id]].rename(columns={combo_id: option_id}))
            option_lookup[option_id] = (prefix, combo_id)

    X_df = pd.concat(pieces, axis=1)
    X_raw = X_df.to_numpy(dtype=float)

    row_has_finite_value = np.isfinite(X_raw).any(axis=1)
    if int((~row_has_finite_value).sum()) > 0:
        removed_rows = int((~row_has_finite_value).sum())
        print(
            f"[INFO] Removed {removed_rows} rows with no finite values.",
            flush=True,
        )

    common_ids = [sid for sid, keep in zip(common_ids, row_has_finite_value) if keep]
    X = X_raw[row_has_finite_value, :]
    X[~np.isfinite(X)] = np.nan
    valid_mask = np.isfinite(X)
    option_ids = list(X_df.columns)

    forecast_lookup_by_prefix = {
        "prophet": prophet_part["forecasting_matrix"].set_index(id_col).loc[common_ids],
        "sarima": sarima_part["forecasting_matrix"].set_index(id_col).loc[common_ids],
        "lstm": lstm_part["forecasting_matrix"].set_index(id_col).loc[common_ids],
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
        data_by_id = {
            str(series_id): df_s.sort_values("ts").copy()
            for series_id, df_s in data.groupby("id", sort=False)
        }
    elif DATASET == "snp500":
        data = load_snp500_data(cfg["data_path"], cfg["target_cols"])
        data_by_name = {
            str(name): df_s.sort_values("date").copy()
            for name, df_s in data.groupby("Name", sort=False)
        }
        series_map = pd.read_pickle(cfg["series_map_pkl"]).copy()
        series_map["series_id"] = series_map["series_id"].astype(str)
        series_map = series_map.set_index("series_id")
    elif DATASET == "covid19":
        data, date_cols = load_covid19_data(cfg["data_path"])
        series_map = pd.read_pickle(cfg["series_map_pkl"]).copy()
        series_map["series_id"] = series_map["series_id"].astype(str)
        series_map = series_map.set_index("series_id")
    else:
        raise ValueError(f"Unknown DATASET={DATASET}")

    y_true_all = []
    y_pred_all = []
    abs_scaled_all = []
    sq_scaled_all = []
    failures = []

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
            if series_id_str not in series_map.index:
                failures.append((series_id_str, "series_id not found"))
                continue
            row = series_map.loc[series_id_str]
            name = str(row["Name"])
            typ = row["Type"]
            if name not in data_by_name:
                failures.append((series_id_str, "empty series"))
                continue
            df_s = data_by_name[name]
            if len(df_s) <= cfg["test_horizon"]:
                failures.append((series_id_str, "series too short"))
                continue
            y_all = df_s[typ].astype(float).to_numpy()

        elif DATASET == "covid19":
            if series_id_str not in series_map.index:
                failures.append((series_id_str, "series_id not found"))
                continue
            row = series_map.loc[series_id_str]
            country = row["country"]
            typ = row["type"]
            df_s = data[(data["country"] == country) & (data["type"] == typ)]
            if df_s.empty:
                failures.append((series_id_str, "empty series"))
                continue
            y_all = df_s.iloc[0][date_cols].astype(float).to_numpy()
            if len(y_all) <= cfg["test_horizon"]:
                failures.append((series_id_str, "series too short"))
                continue

        split_idx = len(y_all) - cfg["test_horizon"]
        y_train = y_all[:split_idx]
        y_test = y_all[split_idx:]

        q_i = seasonal_scale_q(y_train, cfg["seasonal_period"])
        if not np.isfinite(q_i):
            failures.append((series_id_str, "invalid Q_i"))
            continue

        y_pred = np.asarray(y_pred, dtype=float)
        if len(y_pred) != len(y_test):
            failures.append(
                (
                    series_id_str,
                    f"forecast length mismatch: pred={len(y_pred)}, true={len(y_test)}",
                )
            )
            continue

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        abs_scaled_all.append(np.abs(y_test - y_pred) / q_i)
        sq_scaled_all.append(((y_test - y_pred) / q_i) ** 2)

    if len(y_true_all) == 0:
        return None

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    abs_scaled_all = np.concatenate(abs_scaled_all)
    sq_scaled_all = np.concatenate(sq_scaled_all)

    return {
        "n_series": int(len(y_pred_all) // cfg["test_horizon"]),
        "n_failures": int(len(failures)),
        "global_smape": float(global_smape(y_true_all, y_pred_all)),
        "global_mase": float(np.mean(abs_scaled_all)),
        "global_rmsse": float(np.sqrt(np.mean(sq_scaled_all))),
        "failures": failures,
    }


def run_one_mask(
    X,
    valid_mask,
    mask_obs,
    option_ids,
    series_ids,
    option_lookup,
    forecast_lookup_by_prefix,
    repeat_idx,
    sample_size,
):
    start = pd.Timestamp.now()

    completed, model_metrics = complete_matrix_with_mlp(
        X=X,
        valid_mask=valid_mask,
        mask_obs=mask_obs,
        repeat_idx=repeat_idx,
        sample_size=sample_size,
    )

    selected_col_idx = np.nanargmin(completed, axis=1)
    selected_option_ids = np.array(option_ids, dtype=object)[selected_col_idx]

    forecast_summary = evaluate_all_models_forecasts(
        series_ids=series_ids,
        selected_option_ids=selected_option_ids,
        option_lookup=option_lookup,
        forecast_lookup_by_prefix=forecast_lookup_by_prefix,
    )

    elapsed = (pd.Timestamp.now() - start).total_seconds()

    if forecast_summary is None:
        return {
            **model_metrics,
            "n_series": 0,
            "n_failures": len(series_ids),
            "sMAPE": np.nan,
            "MASE": np.nan,
            "RMSSE": np.nan,
            "elapsed_sec": elapsed,
        }

    return {
        **model_metrics,
        "n_series": forecast_summary["n_series"],
        "n_failures": forecast_summary["n_failures"],
        "sMAPE": forecast_summary["global_smape"],
        "MASE": forecast_summary["global_mase"],
        "RMSSE": forecast_summary["global_rmsse"],
        "elapsed_sec": elapsed,
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
    ) = load_all_models_matrix_and_forecasts()
    n, m = X.shape
    valid_count = int(valid_mask.sum())
    sample_sizes = make_theoretical_sample_sizes(n, m, valid_count)

    print("=== MLP ONE-HOT MATRIX COMPLETION: all_models ===", flush=True)
    print(f"Dataset      : {DATASET}", flush=True)
    print(f"Matrix shape : {X.shape}", flush=True)
    print(f"Valid cells  : {valid_count}", flush=True)
    print(f"Sample sizes : {sample_sizes}", flush=True)
    print("", flush=True)

    results_by_sample_size = {s["sample_size"]: [] for s in sample_sizes if s["feasible"]}

    for repeat_idx in range(N_REPEATS):
        seed = RNG_SEED + 1000 * repeat_idx
        masks = sample_cumulative_masks(valid_mask, sample_sizes, seed)

        for s in sample_sizes:
            sample_size = s["sample_size"]
            if not s["feasible"]:
                continue

            print(
                f"[START] repeat={repeat_idx + 1}/{N_REPEATS} | "
                f"sample_size={sample_size} | multiplier={s['multiplier']} | "
                f"test_percentage={s['test_percentage']:.6f}%",
                flush=True,
            )

            try:
                row = run_one_mask(
                    X=X,
                    valid_mask=valid_mask,
                    mask_obs=masks[sample_size],
                    option_ids=option_ids,
                    series_ids=series_ids,
                    option_lookup=option_lookup,
                    forecast_lookup_by_prefix=forecast_lookup_by_prefix,
                    repeat_idx=repeat_idx,
                    sample_size=sample_size,
                )
            except Exception as e:
                row = {
                    "hidden_dim": HIDDEN_DIM,
                    "n_hidden_layers": N_HIDDEN_LAYERS,
                    "dropout": DROPOUT,
                    "lr": LEARNING_RATE,
                    "max_epochs": MAX_EPOCHS,
                    "activation": "relu",
                    "r2_pred": np.nan,
                    "r2_all": np.nan,
                    "frob": np.nan,
                    "n_series": 0,
                    "n_failures": len(series_ids),
                    "sMAPE": np.nan,
                    "MASE": np.nan,
                    "RMSSE": np.nan,
                    "elapsed_sec": np.nan,
                    "error": repr(e),
                }

            row["repeat"] = repeat_idx + 1
            row["sample_size"] = sample_size
            row["test_percentage"] = s["test_percentage"]
            row["multiplier"] = s["multiplier"]
            results_by_sample_size[sample_size].append(row)

            print(
                f"[RESULT] repeat={repeat_idx + 1}/{N_REPEATS} | "
                f"sample_size={sample_size} | hidden_dim={row['hidden_dim']} | "
                f"layers={row['n_hidden_layers']} | dropout={row['dropout']} | "
                f"lr={row['lr']} | epochs={row['max_epochs']} | act={row['activation']} | "
                f"r2_pred={row['r2_pred']:.6f} | r2_all={row['r2_all']:.6f} | "
                f"frob={row['frob']:.6f} | sMAPE={row['sMAPE']:.6f} | "
                f"MASE={row['MASE']:.6f} | RMSSE={row['RMSSE']:.6f} | "
                f"n_series={row['n_series']} | n_failures={row['n_failures']}",
                flush=True,
            )

    summary_rows = []

    for s in sample_sizes:
        if not s["feasible"]:
            print(
                f"[SKIPPED] sample_size={s['sample_size']} | "
                f"test_percentage={s['test_percentage']:.6f}% | "
                f"reason=sample_size exceeds valid cells",
                flush=True,
            )
            continue

        sample_size = s["sample_size"]
        summary = summarize_repeat_rows(results_by_sample_size[sample_size])
        out_row = {
            "dataset": DATASET,
            "model": "all_models",
            "algorithm": "mlp_onehot",
            "multiplier": s["multiplier"],
            "sample_size": sample_size,
            "test_percentage": s["test_percentage"],
            "n_repeats": N_REPEATS,
            **summary,
        }
        summary_rows.append(out_row)

        print(
            f"[MEAN RESULT] dataset={DATASET} | model=all_models | "
            f"algorithm=mlp_onehot | multiplier={s['multiplier']} | "
            f"sample_size={sample_size} | test_percentage={s['test_percentage']:.6f}% | "
            f"mean_r2_pred={summary['mean_r2_pred']:.6f} | "
            f"mean_r2_all={summary['mean_r2_all']:.6f} | "
            f"mean_frob={summary['mean_frob']:.6f} | "
            f"mean_n_series={summary['mean_n_series']:.2f} | "
            f"mean_n_failures={summary['mean_n_failures']:.2f} | "
            f"mean_sMAPE={summary['mean_sMAPE']:.6f} | "
            f"mean_MASE={summary['mean_MASE']:.6f} | "
            f"mean_RMSSE={summary['mean_RMSSE']:.6f} | "
            f"mean_elapsed_sec={summary['mean_elapsed_sec']:.2f}",
            flush=True,
        )

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    output_pkl = os.path.join(_OUTPUT_DIR, f"{DATASET}_mlp_onehot_all_models_summary.pkl")
    summary_df.to_pickle(output_pkl)
    print(f"Saved {output_pkl}", flush=True)

if __name__ == "__main__":
    main()
