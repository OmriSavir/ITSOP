"""Evaluate feature-based MLP matrix completion across forecasting models."""

import argparse
import logging
import os
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

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
        description="Evaluate feature-based MLP matrix completion across forecasting models."
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
        "pickle files. Defaults to --data-dir.",
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
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return np.nan
    return 1.0 - (ss_res / ss_tot)


def frob_norm_masked(A_true, A_pred, mask):
    diff = (A_true - A_pred)[mask]
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return np.nan
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

# MLP with series features and model-option one-hot vectors


class MLP(nn.Module):
    """MLP used to predict an entry from series features and an option one-hot vector."""

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


def build_model_inputs(series_features, option_features):
    return np.concatenate([series_features, option_features], axis=1)


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


def train_and_predict_matrix(X, valid_mask, mask_obs, X_user, X_option, seed):
    observed_series, observed_options = np.where(mask_obs)
    observed_targets = X[mask_obs].astype(np.float32)
    observed_features = build_model_inputs(
        X_user[observed_series].astype(np.float32),
        X_option[observed_options].astype(np.float32),
    )

    split = split_observed_entries(observed_features, observed_targets, seed)
    model = train_mlp(*split)

    matrix_prediction = np.full_like(X, np.nan, dtype=float)
    n_options = X.shape[1]
    series_block_size = 128

    for start in range(0, X.shape[0], series_block_size):
        end = min(start + series_block_size, X.shape[0])
        local_series = np.repeat(np.arange(end - start), n_options)
        option_indices = np.tile(np.arange(n_options), end - start)
        block_features = build_model_inputs(
            X_user[start:end][local_series],
            X_option[option_indices],
        )

        with torch.no_grad():
            block_prediction = model(
                torch.tensor(block_features, dtype=torch.float32, device=DEVICE)
            ).cpu().numpy()

        matrix_prediction[start:end] = block_prediction.reshape(end - start, n_options)

    matrix_prediction[~valid_mask] = np.nan
    return matrix_prediction

# Data loading and feature loading


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


def get_covid19_date_cols(data):
    non_date_cols = {"country", "type", "Lat", "Long"}
    date_col_to_datetime = {}

    for col in data.columns:
        if col in non_date_cols:
            continue

        parsed = pd.to_datetime(str(col), format="%m/%d/%Y", errors="coerce")
        if pd.isna(parsed):
            parsed = pd.to_datetime(str(col), format="%m/%d/%y", errors="coerce")
        if pd.isna(parsed):
            parsed = pd.to_datetime(str(col), errors="coerce")

        if not pd.isna(parsed):
            date_col_to_datetime[col] = parsed

    return sorted(date_col_to_datetime, key=date_col_to_datetime.get)


def load_covid19_data(pkl_path):
    data = pd.read_pickle(pkl_path).copy()
    date_cols = get_covid19_date_cols(data)

    if len(date_cols) == 0:
        raise RuntimeError("No date columns found in covid19 dataset")

    for c in date_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    data = data.dropna(subset=["country", "type"]).copy()
    data["country"] = data["country"].astype(str)
    data["type"] = data["type"].astype(str)
    data.attrs["date_cols"] = date_cols
    return data


def load_user_features(series_ids, series_map, feature_file):
    feats = pd.read_pickle(feature_file).copy()

    if DATASET == "electricity":
        feats["id"] = feats["id"].astype(str)
        feats = feats.drop_duplicates(subset=["id"], keep="last")
        feats = feats.set_index("id").reindex([str(sid) for sid in series_ids])
        feats = feats.drop(columns=["id"], errors="ignore")

    elif DATASET == "snp500":
        series_map_local = series_map.copy()
        series_map_local["series_id"] = series_map_local["series_id"].astype(str)
        ordered_keys = (
            series_map_local.set_index("series_id")
            .loc[[str(sid) for sid in series_ids], ["Name", "Type"]]
            .reset_index(drop=True)
        )
        feats = feats.drop_duplicates(subset=["Name", "Type"], keep="last")
        feats = ordered_keys.merge(feats, on=["Name", "Type"], how="left")
        feats = feats.drop(columns=["Name", "Type"], errors="ignore")

    elif DATASET == "covid19":
        series_map_local = series_map.copy()
        series_map_local["series_id"] = series_map_local["series_id"].astype(str)
        ordered_keys = (
            series_map_local.set_index("series_id")
            .loc[[str(sid) for sid in series_ids], ["country", "type"]]
            .reset_index(drop=True)
        )
        feats = feats.drop_duplicates(subset=["country", "type"], keep="last")
        feats = ordered_keys.merge(feats, on=["country", "type"], how="left")
        feats = feats.drop(columns=["country", "type"], errors="ignore")

    else:
        raise ValueError(f"Unknown DATASET={DATASET}")

    numeric_cols = feats.select_dtypes(include=[np.number, bool]).columns.tolist()
    other_cols = [c for c in feats.columns if c not in numeric_cols]

    feats_num = (
        feats[numeric_cols].apply(pd.to_numeric, errors="coerce")
        if numeric_cols
        else pd.DataFrame(index=feats.index)
    )
    feats_cat = (
        pd.get_dummies(feats[other_cols].astype("string"), dummy_na=True)
        if other_cols
        else pd.DataFrame(index=feats.index)
    )

    X_user_df = pd.concat([feats_num, feats_cat], axis=1)
    X_user_df = X_user_df.replace([np.inf, -np.inf], np.nan)

    med = X_user_df.median(numeric_only=True)
    X_user_df = X_user_df.fillna(med).fillna(0.0)

    scaler = StandardScaler()
    return scaler.fit_transform(X_user_df.to_numpy(dtype=np.float32))

# Forecast evaluation


def summarize_repeat_results(rows):
    if len(rows) == 0:
        return {
            "hidden_dim": HIDDEN_DIM,
            "n_hidden_layers": N_HIDDEN_LAYERS,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "activation": "relu",
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
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

    results = pd.DataFrame(rows)
    return {
        "hidden_dim": HIDDEN_DIM,
        "n_hidden_layers": N_HIDDEN_LAYERS,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "activation": "relu",
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "mean_r2_pred": float(results["r2_pred"].mean()),
        "mean_r2_all": float(results["r2_all"].mean()),
        "mean_frob": float(results["frob"].mean()),
        "mean_sMAPE": float(results["sMAPE"].mean()),
        "mean_MASE": float(results["MASE"].mean()),
        "mean_RMSSE": float(results["RMSSE"].mean()),
        "mean_n_series": float(results["n_series"].mean()),
        "mean_n_failures": float(results["n_failures"].mean()),
        "mean_elapsed_sec": float(results["elapsed_sec"].mean()),
    }


def train_select_and_evaluate(
    X,
    valid_mask,
    obs_mask,
    combo_ids,
    series_ids,
    X_user,
    X_option,
    evaluate_forecasts_func,
    repeat_idx,
    sample_size,
):
    prediction_mask = valid_mask & (~obs_mask)
    seed = RNG_SEED + 100000 * repeat_idx + sample_size
    matrix_prediction = train_and_predict_matrix(
        X=X,
        valid_mask=valid_mask,
        mask_obs=obs_mask,
        X_user=X_user,
        X_option=X_option,
        seed=seed,
    )

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
    completed[obs_mask] = X[obs_mask]
    completed[prediction_mask] = matrix_prediction[prediction_mask]

    selected_columns = np.nanargmin(completed, axis=1)
    selected_combo_ids = np.asarray(combo_ids, dtype=object)[selected_columns]
    forecast_summary = evaluate_forecasts_func(
        series_ids=series_ids,
        selected_combo_ids=selected_combo_ids,
    )

    if forecast_summary is None:
        return {
            **metrics,
            "n_series": 0,
            "n_failures": len(series_ids),
            "sMAPE": np.nan,
            "MASE": np.nan,
            "RMSSE": np.nan,
        }

    return {
        **metrics,
        "n_series": forecast_summary["n_series"],
        "n_failures": forecast_summary["n_failures"],
        "sMAPE": forecast_summary["global_smape"],
        "MASE": forecast_summary["global_mase"],
        "RMSSE": forecast_summary["global_rmsse"],
    }


def build_dataset_configs(data_dir=None, output_dir=None, features_dir=None):
    data_dir = data_dir or os.path.join("data", DATASET)
    output_dir = output_dir or os.path.join("outputs", DATASET)
    features_dir = features_dir or data_dir

    return {
        "electricity": {
            "id_col": "building_id",
            "data_path": os.path.join(data_dir, "consumption_dataset.pkl"),
            "target_col": "consumption",
            "test_horizon": 48,
            "seasonal_period": 168,
            "feature_files": {
                "tsfresh": os.path.join(features_dir, "electricity_feature_extraction_tsfresh.pkl"),
                "catch22": os.path.join(features_dir, "electricity_feature_extraction_catch22.pkl"),
            },
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
            "feature_files": {
                "tsfresh": os.path.join(features_dir, "snp500_feature_extraction_tsfresh.pkl"),
                "catch22": os.path.join(features_dir, "snp500_feature_extraction_catch22.pkl"),
            },
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
            "feature_files": {
                "tsfresh": os.path.join(features_dir, "covid19_feature_extraction_tsfresh.pkl"),
                "catch22": os.path.join(features_dir, "covid19_feature_extraction_catch22.pkl"),
            },
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
DATASET_CONFIGS = build_dataset_configs(_ARGS.data_dir, _ARGS.output_dir, _ARGS.features_dir)


def get_combo_cols_for_part(smape_matrix, forecasting_matrix, comb_map_pkl, id_col):
    if comb_map_pkl is not None and os.path.exists(comb_map_pkl):
        comb_map = pd.read_pickle(comb_map_pkl).copy()
        comb_map["id"] = comb_map["id"].astype(int)
        candidate_cols = comb_map["id"].astype(int).tolist()
    else:
        candidate_cols = [c for c in smape_matrix.columns if c != id_col]

    combo_cols = [
        combo_id
        for combo_id in candidate_cols
        if combo_id in smape_matrix.columns and combo_id in forecasting_matrix.columns
    ]
    if len(combo_cols) == 0:
        raise RuntimeError(f"No common combo columns found for {comb_map_pkl}")
    return combo_cols


def load_part(cfg, prefix):
    id_col = cfg["id_col"]
    smape_matrix = pd.read_pickle(cfg[f"{prefix}_matrix_pkl"])
    forecasting_matrix = pd.read_pickle(cfg[f"{prefix}_forecasting_test_pkl"])

    smape_matrix = normalize_matrix_columns(smape_matrix, id_col)
    forecasting_matrix = normalize_matrix_columns(forecasting_matrix, id_col)

    combo_cols = get_combo_cols_for_part(
        smape_matrix=smape_matrix,
        forecasting_matrix=forecasting_matrix,
        comb_map_pkl=cfg[f"{prefix}_comb_map_pkl"],
        id_col=id_col,
    )

    common_ids = set(smape_matrix[id_col].astype(str)) & set(forecasting_matrix[id_col].astype(str))

    return {
        "smape_matrix": smape_matrix,
        "forecasting_matrix": forecasting_matrix,
        "combo_cols": combo_cols,
        "comb_map_pkl": cfg[f"{prefix}_comb_map_pkl"],
        "common_ids": common_ids,
    }


def load_all_models_matrix_and_option_features():
    cfg = DATASET_CONFIGS[DATASET]
    id_col = cfg["id_col"]

    parts = {
        "prophet": load_part(cfg, "prophet"),
        "sarima": load_part(cfg, "sarima"),
        "lstm": load_part(cfg, "lstm"),
    }

    common_ids = sorted(
        set.intersection(*(part["common_ids"] for part in parts.values())),
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )
    if len(common_ids) == 0:
        raise RuntimeError("No common series across Prophet, SARIMA, and LSTM matrices")

    matrix_parts = []
    option_ids = []
    option_lookup = {}

    for model_name, part in parts.items():
        aligned = part["smape_matrix"].set_index(id_col).loc[common_ids]
        for combo_id in part["combo_cols"]:
            option_id = f"{model_name}__{combo_id}"
            matrix_parts.append(aligned[[combo_id]].rename(columns={combo_id: option_id}))
            option_ids.append(option_id)
            option_lookup[option_id] = (model_name, combo_id)

    matrix_frame = pd.concat(matrix_parts, axis=1)
    raw_matrix = matrix_frame.to_numpy(dtype=float)
    row_mask = np.isfinite(raw_matrix).any(axis=1)

    if (~row_mask).any():
        print(
            f"[INFO] Removed {int((~row_mask).sum())} series with no finite values.",
            flush=True,
        )

    common_ids = [series_id for series_id, keep in zip(common_ids, row_mask) if keep]
    matrix = raw_matrix[row_mask]
    matrix[~np.isfinite(matrix)] = np.nan
    valid_mask = np.isfinite(matrix)

    option_features = np.eye(len(option_ids), dtype=np.float32)
    forecast_lookup_by_model = {
        model_name: part["forecasting_matrix"].set_index(id_col).loc[common_ids]
        for model_name, part in parts.items()
    }

    return (
        matrix,
        valid_mask,
        common_ids,
        option_ids,
        option_features,
        option_lookup,
        forecast_lookup_by_model,
    )


def load_eval_data_and_series_map():
    cfg = DATASET_CONFIGS[DATASET]

    if DATASET == "electricity":
        data = load_electricity_data(cfg["data_path"], cfg["target_col"])
        return data, None

    if DATASET == "snp500":
        data = load_snp500_data(cfg["data_path"], cfg["target_cols"])
        series_map = pd.read_pickle(cfg["series_map_pkl"]).copy()
        series_map["series_id"] = series_map["series_id"].astype(str)
        return data, series_map

    if DATASET == "covid19":
        data = load_covid19_data(cfg["data_path"])
        series_map = pd.read_pickle(cfg["series_map_pkl"]).copy()
        series_map["series_id"] = series_map["series_id"].astype(str)
        series_map["country"] = series_map["country"].astype(str)
        series_map["type"] = series_map["type"].astype(str)
        return data, series_map

    raise ValueError(f"Unknown DATASET={DATASET}")


def make_evaluate_all_models_forecasts_func(
    data,
    series_map,
    option_lookup,
    forecast_lookup_by_model,
):
    cfg = DATASET_CONFIGS[DATASET]

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
        date_cols = data.attrs.get("date_cols", get_covid19_date_cols(data))

    else:
        raise ValueError(f"Unknown DATASET={DATASET}")

    def _evaluate(series_ids, selected_combo_ids):
        y_true_all = []
        y_pred_all = []
        all_abs_scaled = []
        all_sq_scaled = []
        failures = []

        for sid, option_id in zip(series_ids, selected_combo_ids):
            sid = str(sid)

            if option_id not in option_lookup:
                failures.append((sid, f"unknown option_id={option_id}"))
                continue

            model_name, combo_id = option_lookup[option_id]

            if sid not in forecast_lookup_by_model[model_name].index:
                failures.append((sid, f"missing forecasting row for {model_name}"))
                continue

            try:
                y_pred = get_forecast_value(forecast_lookup_by_model[model_name].loc[sid], combo_id)
            except Exception as e:
                failures.append((sid, str(e)))
                continue

            if DATASET == "electricity":
                if sid not in data_by_id:
                    failures.append((sid, "empty series"))
                    continue
                df_s = data_by_id[sid]
                if len(df_s) <= cfg["test_horizon"]:
                    failures.append((sid, "series too short"))
                    continue
                y_all = df_s[cfg["target_col"]].astype(float).to_numpy()

            elif DATASET == "snp500":
                if sid not in series_map_lookup.index:
                    failures.append((sid, "series_id not found"))
                    continue
                row = series_map_lookup.loc[sid]
                name = str(row["Name"])
                typ = row["Type"]
                if name not in data_by_name:
                    failures.append((sid, "empty series"))
                    continue
                df_s = data_by_name[name]
                if len(df_s) <= cfg["test_horizon"]:
                    failures.append((sid, "series too short"))
                    continue
                y_all = df_s[typ].astype(float).to_numpy()

            elif DATASET == "covid19":
                if sid not in series_map_lookup.index:
                    failures.append((sid, "series_id not found"))
                    continue
                row = series_map_lookup.loc[sid]
                country = str(row["country"])
                typ = str(row["type"])
                df_s = data[(data["country"] == country) & (data["type"] == typ)]
                if df_s.empty:
                    failures.append((sid, "empty series"))
                    continue
                y_all = df_s.iloc[0][date_cols].astype(float).to_numpy()

            split_idx = len(y_all) - cfg["test_horizon"]
            y_train = y_all[:split_idx]
            y_test = y_all[split_idx:]

            q_i = seasonal_scale_q(y_train, cfg["seasonal_period"])
            if not np.isfinite(q_i):
                failures.append((sid, "invalid Q_i"))
                continue

            y_pred = np.asarray(y_pred, dtype=float)
            if len(y_pred) != len(y_test):
                failures.append(
                    (
                        sid,
                        f"forecast length mismatch: pred={len(y_pred)}, true={len(y_test)}",
                    )
                )
                continue

            y_true_all.append(y_test)
            y_pred_all.append(y_pred)
            all_abs_scaled.append(np.abs(y_test - y_pred) / q_i)
            all_sq_scaled.append(((y_test - y_pred) / q_i) ** 2)

        if len(y_true_all) == 0:
            return None

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        all_abs_scaled = np.concatenate(all_abs_scaled)
        all_sq_scaled = np.concatenate(all_sq_scaled)

        return {
            "n_series": int(len(y_true_all) // cfg["test_horizon"]),
            "n_failures": int(len(failures)),
            "failures": failures,
            "global_smape": global_smape(y_true_all, y_pred_all),
            "global_mase": float(np.mean(all_abs_scaled)),
            "global_rmsse": float(np.sqrt(np.mean(all_sq_scaled))),
        }

    return _evaluate


def main():
    if DATASET not in DATASET_CONFIGS:
        raise ValueError(f"Unknown DATASET={DATASET}")

    cfg = DATASET_CONFIGS[DATASET]

    (
        X,
        valid_mask,
        series_ids,
        option_ids,
        X_option,
        option_lookup,
        forecast_lookup_by_model,
    ) = load_all_models_matrix_and_option_features()
    data, series_map = load_eval_data_and_series_map()
    evaluate_forecasts_func = make_evaluate_all_models_forecasts_func(
        data=data,
        series_map=series_map,
        option_lookup=option_lookup,
        forecast_lookup_by_model=forecast_lookup_by_model,
    )

    n, m = X.shape
    valid_count = int(valid_mask.sum())
    sample_sizes = make_theoretical_sample_sizes(n, m, valid_count)

    print("=== MLP WITH SERIES FEATURES: all_models ===", flush=True)
    print(f"Dataset      : {DATASET}", flush=True)
    print(f"Matrix shape : {X.shape}", flush=True)
    print(f"Valid cells  : {valid_count}", flush=True)
    print(f"Feature files: {cfg['feature_files']}", flush=True)
    print(f"Sample sizes : {sample_sizes}", flush=True)
    print("", flush=True)

    final_summary_rows = []

    for feature_name, feature_file in cfg["feature_files"].items():
        print("\n" + "=" * 80, flush=True)
        print(f"[FEATURES] {feature_name} | file={feature_file}", flush=True)
        print("=" * 80, flush=True)

        X_user = load_user_features(series_ids, series_map, feature_file)

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
                    f"[START] feature={feature_name} | repeat={repeat_idx + 1}/{N_REPEATS} | "
                    f"sample_size={sample_size} | multiplier={s['multiplier']} | "
                    f"test_percentage={s['test_percentage']:.6f}%",
                    flush=True,
                )

                start = pd.Timestamp.now()

                try:
                    row = train_select_and_evaluate(
                        X=X,
                        valid_mask=valid_mask,
                        obs_mask=obs_mask,
                        combo_ids=option_ids,
                        series_ids=series_ids,
                        X_user=X_user,
                        X_option=X_option,
                        evaluate_forecasts_func=evaluate_forecasts_func,
                        repeat_idx=repeat_idx,
                        sample_size=sample_size,
                    )

                    elapsed = (pd.Timestamp.now() - start).total_seconds()

                    if row is None:
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
                        }

                    row.update({
                        "feature_name": feature_name,
                        "repeat": repeat_idx + 1,
                        "sample_size": sample_size,
                        "test_percentage": s["test_percentage"],
                        "elapsed_sec": elapsed,
                    })

                except Exception as e:
                    elapsed = (pd.Timestamp.now() - start).total_seconds()
                    row = {
                        "feature_name": feature_name,
                        "repeat": repeat_idx + 1,
                        "sample_size": sample_size,
                        "test_percentage": s["test_percentage"],
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
                        "elapsed_sec": elapsed,
                        "error": repr(e),
                    }

                results_by_sample_size[sample_size].append(row)

                print(
                    f"[RESULT] feature={feature_name} | repeat={repeat_idx + 1}/{N_REPEATS} | "
                    f"sample_size={sample_size} | r2_pred={row['r2_pred']:.6f} | "
                    f"r2_all={row['r2_all']:.6f} | frob={row['frob']:.6f} | "
                    f"sMAPE={row['sMAPE']:.6f} | MASE={row['MASE']:.6f} | "
                    f"RMSSE={row['RMSSE']:.6f} | n_series={row['n_series']} | "
                    f"n_failures={row['n_failures']} | elapsed={row['elapsed_sec']:.2f}s",
                    flush=True,
                )

        for s in sample_sizes:
            if not s["feasible"]:
                print(
                    f"[SKIPPED] feature={feature_name} | sample_size={s['sample_size']} | "
                    f"test_percentage={s['test_percentage']:.6f}% | "
                    f"reason=sample_size exceeds valid cells",
                    flush=True,
                )
                continue

            sample_size = s["sample_size"]
            summary = summarize_repeat_results(results_by_sample_size[sample_size])

            out_row = {
                "dataset": DATASET,
                "model": "all_models",
                "feature_name": feature_name,
                "multiplier": s["multiplier"],
                "sample_size": sample_size,
                "test_percentage": s["test_percentage"],
                "n_repeats": N_REPEATS,
                **summary,
            }
            final_summary_rows.append(out_row)

            print(
                f"[MEAN RESULT] dataset={DATASET} | model=all_models | feature={feature_name} | "
                f"multiplier={s['multiplier']} | sample_size={sample_size} | "
                f"test_percentage={s['test_percentage']:.6f}% | "
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

    summary_df = pd.DataFrame(final_summary_rows)
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    output_pkl = os.path.join(_OUTPUT_DIR, f"{DATASET}_mlp_features_all_models_summary.pkl")
    summary_df.to_pickle(output_pkl)
    print(f"Saved {output_pkl}", flush=True)

if __name__ == "__main__":
    main()
