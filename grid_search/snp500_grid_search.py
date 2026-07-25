"""Run Prophet, SARIMA, and LSTM grid searches for the S&P 500 dataset."""

import argparse
import multiprocessing as mp
import os
import time
import warnings
from queue import Empty as QueueEmpty

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

CSV_PATH = os.path.join("data", "snp500", "snp500_data.csv")
TARGET_COLS = ["open", "high", "low", "close", "volume"]
SERIES_MAP_PKL = "snp500_series_to_id.pkl"

TEST_HORIZON = 14
CV_N_WINDOWS = 5
CV_VAL_LEN = 14
TRAIN_FRAC_TOTAL = 0.3

PROPHET_TIMEOUT_SEC = 180
PROPHET_CHANGEPOINT_PRIOR_SCALES = [0.005, 0.05, 0.5]
PROPHET_SEASONALITY_PRIOR_SCALES = [5, 10, 20]
PROPHET_N_CHANGEPOINTS = [15, 25, 50]
PROPHET_CHANGEPOINT_RANGES = [0.8, 0.9]
PROPHET_GRID = [
    {
        "changepoint_prior_scale": float(changepoint_prior_scale),
        "seasonality_prior_scale": float(seasonality_prior_scale),
        "n_changepoints": int(n_changepoints),
        "changepoint_range": float(changepoint_range),
    }
    for changepoint_prior_scale in PROPHET_CHANGEPOINT_PRIOR_SCALES
    for seasonality_prior_scale in PROPHET_SEASONALITY_PRIOR_SCALES
    for n_changepoints in PROPHET_N_CHANGEPOINTS
    for changepoint_range in PROPHET_CHANGEPOINT_RANGES
]
PROPHET_COMB_MAP_PKL = "snp500_comb_to_id.pkl"
PROPHET_SMAPE_MATRIX_PKL = "snp500_smape_matrix.pkl"

SARIMA_TIMEOUT_SEC = 180
SARIMA_SEASONAL_PERIOD = 5
SARIMA_VALUES = [0, 1]
SARIMA_GRID = [
    {
        "p": int(p),
        "d": int(d),
        "q": int(q),
        "P": int(seasonal_p),
        "D": int(seasonal_d),
        "Q": int(seasonal_q),
    }
    for p in SARIMA_VALUES
    for d in SARIMA_VALUES
    for q in SARIMA_VALUES
    for seasonal_p in SARIMA_VALUES
    for seasonal_d in SARIMA_VALUES
    for seasonal_q in SARIMA_VALUES
]
SARIMA_COMB_MAP_PKL = "snp500_comb_to_id_sarima.pkl"
SARIMA_SMAPE_MATRIX_PKL = "snp500_smape_matrix_sarima.pkl"

RANDOM_SEED = 42
LSTM_LOOKBACKS = [5, 10, 20]
LSTM_HIDDEN_SIZES = [32, 64]
LSTM_NUM_LAYERS = [1, 2]
LSTM_DROPOUTS = [0.0, 0.2]
LSTM_LEARNING_RATES = [1e-3, 3e-4]
LSTM_BATCH_SIZE = 64
LSTM_MAX_EPOCHS = 50
LSTM_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LSTM_GRID = [
    {
        "lookback": int(lookback),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "batch_size": LSTM_BATCH_SIZE,
        "max_epochs": LSTM_MAX_EPOCHS,
    }
    for lookback in LSTM_LOOKBACKS
    for hidden_size in LSTM_HIDDEN_SIZES
    for num_layers in LSTM_NUM_LAYERS
    for dropout in LSTM_DROPOUTS
    for learning_rate in LSTM_LEARNING_RATES
]
LSTM_COMB_MAP_PKL = "snp500_comb_to_id_lstm.pkl"
LSTM_SMAPE_MATRIX_PKL = "snp500_smape_matrix_lstm.pkl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Prophet, SARIMA, and LSTM grid searches for the S&P 500 dataset."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join("data", "snp500"),
        help="Directory containing snp500_data.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("outputs", "snp500"),
        help="Directory to write validation grid-search matrices and "
        "combination maps to. test_forecasting/snp500_test_forecasting.py "
        "reads its combination maps from this same directory.",
    )
    return parser.parse_args()


def smape(y_true, y_pred, eps=1e-9):
    """Compute symmetric mean absolute percentage error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred) + eps
    return 200.0 * np.mean(np.abs(y_true - y_pred) / denominator)


def rolling_cv_slices(
    n_train_full,
    n_total,
    n_windows=CV_N_WINDOWS,
    val_len=CV_VAL_LEN,
    train_frac_total=TRAIN_FRAC_TOTAL,
):
    """Construct fixed-length rolling training and validation slices."""
    train_window = int(np.floor(train_frac_total * n_total))
    if train_window <= 0 or train_window + val_len > n_train_full:
        return []

    first_val_start = train_window
    last_val_start = n_train_full - val_len
    if last_val_start < first_val_start:
        return []

    if n_windows == 1:
        validation_starts = [last_val_start]
    else:
        validation_starts = np.linspace(first_val_start, last_val_start, n_windows)
        validation_starts = np.rint(validation_starts).astype(int).tolist()
        validation_starts[-1] = last_val_start

    validation_starts = sorted(set(validation_starts))

    if len(validation_starts) < n_windows:
        needed = n_windows - len(validation_starts)
        candidate = last_val_start
        while needed > 0 and candidate >= first_val_start:
            if candidate not in validation_starts:
                validation_starts.append(candidate)
                needed -= 1
            candidate -= 1
        validation_starts = sorted(validation_starts)

    if len(validation_starts) < n_windows:
        return []

    if len(validation_starts) > n_windows:
        indices = np.linspace(0, len(validation_starts) - 1, n_windows)
        indices = np.rint(indices).astype(int).tolist()
        validation_starts = [validation_starts[index] for index in indices]
        validation_starts[-1] = last_val_start

    splits = []
    for val_start in validation_starts:
        train_start = val_start - train_window
        val_end = val_start + val_len
        if train_start < 0 or val_end > n_train_full:
            continue
        splits.append((slice(train_start, val_start), slice(val_start, val_end)))

    if not splits or splits[-1][1].stop != n_train_full:
        train_start = last_val_start - train_window
        if train_start >= 0:
            splits.append(
                (
                    slice(train_start, last_val_start),
                    slice(last_val_start, last_val_start + val_len),
                )
            )
            splits = splits[-n_windows:]

    return splits


def _process_target(queue, function, args, kwargs):
    try:
        queue.put(("ok", function(*args, **(kwargs or {}))))
    except Exception as exc:
        queue.put(("error", exc))


def run_with_timeout(function, args=(), kwargs=None, timeout=180):
    """Run a function in a subprocess and enforce a time limit."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_process_target,
        args=(queue, function, args, kwargs),
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Operation timed out after {timeout} seconds")

    try:
        status, payload = queue.get_nowait()
    except QueueEmpty as exc:
        raise RuntimeError("Subprocess ended without returning a result") from exc

    if status == "ok":
        return payload
    raise payload


def save_combo_map(grid, output_path, extra_fields=None):
    """Save a one-based mapping from combination IDs to parameters."""
    records = []
    for combo_id, params in enumerate(grid, start=1):
        record = {"id": combo_id, **params}
        if extra_fields:
            record.update(extra_fields)
        records.append(record)

    combo_map = pd.DataFrame(records)
    combo_map.to_pickle(output_path)
    print(f"[INFO] Saved {output_path} with {len(combo_map)} combinations.", flush=True)
    return combo_map


def load_or_initialize_matrix(output_path, id_column, combo_ids, total_series):
    """Load an incremental result matrix or create an empty one."""
    columns = [id_column] + list(combo_ids)

    if os.path.exists(output_path):
        matrix = pd.read_pickle(output_path)
        for column in columns:
            if column not in matrix.columns:
                matrix[column] = np.nan
        matrix = matrix[columns]
        completed_ids = set(matrix[id_column].tolist())
        print(
            f"[INFO] Loaded existing {output_path}: "
            f"{len(completed_ids)}/{total_series} already done.",
            flush=True,
        )
        return matrix, completed_ids

    matrix = pd.DataFrame(columns=columns)
    matrix.to_pickle(output_path)
    print(f"[INFO] Created empty {output_path}.", flush=True)
    return matrix, set()


def append_result(matrix, output_path, id_column, series_id, scores):
    row = {id_column: series_id, **scores}
    matrix = pd.concat([matrix, pd.DataFrame([row])], ignore_index=True)
    matrix.to_pickle(output_path)
    return matrix


def load_data():
    """Load and clean the S&P 500 observations."""
    data = pd.read_csv(CSV_PATH)

    try:
        data["date"] = pd.to_datetime(
            data["date"],
            format="%m/%d/%Y",
            errors="raise",
        )
    except Exception:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")

    for column in TARGET_COLS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data.dropna(subset=["date"] + TARGET_COLS)


def build_series_map(data):
    """Assign a series ID to each ticker and target variable."""
    records = []
    series_id = 1

    for name in sorted(data["Name"].dropna().unique().tolist()):
        for target in TARGET_COLS:
            records.append({"series_id": series_id, "Name": name, "Type": target})
            series_id += 1

    series_map = pd.DataFrame(records)
    series_map.to_pickle(SERIES_MAP_PKL)
    print(f"[INFO] Saved {SERIES_MAP_PKL} with {len(series_map)} series.", flush=True)
    return series_map


def get_series_values(data, name, target):
    series_data = data[data["Name"] == name].sort_values("date")
    values = series_data[target].astype(float).values
    dates = series_data["date"].values
    return values, dates


def prophet_fit_predict(train_data, predict_data, params):
    model = Prophet(
        n_changepoints=params["n_changepoints"],
        changepoint_range=params["changepoint_range"],
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
    )
    model.fit(train_data[["ds", "y"]], algorithm="LBFGS", iter=1000)
    forecast = model.predict(predict_data[["ds"]].copy())
    return forecast["yhat"].values


def evaluate_prophet_grid(train_data, splits):
    scores_by_combo = {}

    for combo_id, params in enumerate(PROPHET_GRID, start=1):
        scores = []
        timed_out = False

        for train_slice, val_slice in splits:
            train_window = train_data.iloc[train_slice]
            validation_window = train_data.iloc[val_slice]

            try:
                predictions = run_with_timeout(
                    prophet_fit_predict,
                    args=(train_window, validation_window, params),
                    timeout=PROPHET_TIMEOUT_SEC,
                )
            except TimeoutError:
                timed_out = True
                break

            scores.append(smape(validation_window["y"].values, predictions))

        scores_by_combo[combo_id] = (
            np.inf if timed_out or not scores else float(np.mean(scores))
        )

    return scores_by_combo


def run_prophet_grid_search(data, series_map):
    combo_map = save_combo_map(PROPHET_GRID, PROPHET_COMB_MAP_PKL)
    combo_ids = combo_map["id"].astype(int).tolist()
    total_series = len(series_map)
    matrix, completed_ids = load_or_initialize_matrix(
        PROPHET_SMAPE_MATRIX_PKL,
        "series_id",
        combo_ids,
        total_series,
    )
    completed_count = len(completed_ids)

    print(
        f"[INFO] Starting S&P 500 Prophet grid search: "
        f"{len(combo_ids)} combinations.",
        flush=True,
    )

    for row in series_map.itertuples(index=False):
        series_id = int(row.series_id)
        if series_id in completed_ids:
            continue

        values, dates = get_series_values(data, row.Name, row.Type)
        n_total = len(values)

        if n_total <= TEST_HORIZON:
            completed_count += 1
            print(
                f"[INFO] Prophet {completed_count}/{total_series} "
                f"(skipped: too short, series_id={series_id}, Name={row.Name}, Type={row.Type})",
                flush=True,
            )
            continue

        split_index = n_total - TEST_HORIZON
        if split_index <= 0 or split_index >= n_total:
            completed_count += 1
            print(
                f"[INFO] Prophet {completed_count}/{total_series} "
                f"(skipped: bad split, series_id={series_id}, Name={row.Name}, Type={row.Type})",
                flush=True,
            )
            continue

        train_data = pd.DataFrame(
            {"ds": dates[:split_index], "y": values[:split_index]}
        )
        splits = rolling_cv_slices(len(train_data), n_total)

        if not splits:
            completed_count += 1
            print(
                f"[INFO] Prophet {completed_count}/{total_series} "
                f"(skipped: no CV splits, series_id={series_id}, Name={row.Name}, Type={row.Type})",
                flush=True,
            )
            continue

        combo_scores = evaluate_prophet_grid(train_data, splits)
        matrix = append_result(
            matrix,
            PROPHET_SMAPE_MATRIX_PKL,
            "series_id",
            series_id,
            combo_scores,
        )
        completed_ids.add(series_id)
        completed_count += 1
        print(
            f"[INFO] Prophet {completed_count}/{total_series} series "
            f"(last series_id={series_id}, Name={row.Name}, Type={row.Type})",
            flush=True,
        )


def sarima_fit_predict(y_train, validation_length, params):
    model = SARIMAX(
        endog=y_train,
        order=(params["p"], params["d"], params["q"]),
        seasonal_order=(
            params["P"],
            params["D"],
            params["Q"],
            SARIMA_SEASONAL_PERIOD,
        ),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted_model = model.fit(disp=False)
    return np.asarray(fitted_model.forecast(steps=validation_length), dtype=float)


def evaluate_sarima_grid(y_train_full, splits):
    scores_by_combo = {}

    for combo_id, params in enumerate(SARIMA_GRID, start=1):
        scores = []
        failed = False

        for train_slice, val_slice in splits:
            y_train = y_train_full[train_slice]
            y_validation = y_train_full[val_slice]

            try:
                predictions = run_with_timeout(
                    sarima_fit_predict,
                    args=(y_train, len(y_validation), params),
                    timeout=SARIMA_TIMEOUT_SEC,
                )
            except Exception:
                failed = True
                break

            scores.append(smape(y_validation, predictions))

        scores_by_combo[combo_id] = (
            np.inf if failed or not scores else float(np.mean(scores))
        )

    return scores_by_combo


def run_sarima_grid_search(data, series_map):
    combo_map = save_combo_map(
        SARIMA_GRID,
        SARIMA_COMB_MAP_PKL,
        extra_fields={"s": SARIMA_SEASONAL_PERIOD},
    )
    combo_ids = combo_map["id"].astype(int).tolist()
    total_series = len(series_map)
    matrix, completed_ids = load_or_initialize_matrix(
        SARIMA_SMAPE_MATRIX_PKL,
        "series_id",
        combo_ids,
        total_series,
    )
    completed_count = len(completed_ids)

    print(f"[INFO] Starting S&P 500 SARIMA grid search: {len(combo_ids)} combinations.", flush=True)

    for row in series_map.itertuples(index=False):
        series_id = int(row.series_id)
        if series_id in completed_ids:
            continue

        values, _ = get_series_values(data, row.Name, row.Type)
        values = values[np.isfinite(values)]
        n_total = len(values)

        if n_total <= TEST_HORIZON:
            completed_count += 1
            print(
                f"[INFO] SARIMA {completed_count}/{total_series} "
                f"(skipped: too short, series_id={series_id}, Name={row.Name}, Type={row.Type})",
                flush=True,
            )
            continue

        split_index = n_total - TEST_HORIZON
        if split_index <= 0 or split_index >= n_total:
            completed_count += 1
            print(
                f"[INFO] SARIMA {completed_count}/{total_series} "
                f"(skipped: bad split, series_id={series_id}, Name={row.Name}, Type={row.Type})",
                flush=True,
            )
            continue

        y_train_full = values[:split_index]
        splits = rolling_cv_slices(len(y_train_full), n_total)

        if not splits:
            completed_count += 1
            print(
                f"[INFO] SARIMA {completed_count}/{total_series} "
                f"(skipped: no CV splits, series_id={series_id}, Name={row.Name}, Type={row.Type})",
                flush=True,
            )
            continue

        combo_scores = evaluate_sarima_grid(y_train_full, splits)
        matrix = append_result(
            matrix,
            SARIMA_SMAPE_MATRIX_PKL,
            "series_id",
            series_id,
            combo_scores,
        )
        completed_ids.add(series_id)
        completed_count += 1
        print(
            f"[INFO] SARIMA {completed_count}/{total_series} series "
            f"(last series_id={series_id}, Name={row.Name}, Type={row.Type})",
            flush=True,
        )


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cyclical_feature(values, period):
    values = np.asarray(values, dtype=float)
    angle = 2.0 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def build_time_features(timestamps):
    timestamps = pd.to_datetime(pd.Series(timestamps))
    day_of_week = timestamps.dt.dayofweek.to_numpy()
    day_sin, day_cos = cyclical_feature(day_of_week, 7)
    return np.column_stack([day_sin, day_cos]).astype(np.float32)


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        return self.output_layer(outputs[:, -1, :]).squeeze(1)


def make_lstm_sequences(y_scaled, x_scaled, lookback):
    sequences = []
    targets = []

    for index in range(lookback, len(y_scaled)):
        y_window = y_scaled[index - lookback:index].reshape(-1, 1)
        x_window = x_scaled[index - lookback:index]
        sequences.append(np.concatenate([y_window, x_window], axis=1))
        targets.append(y_scaled[index])

    if not sequences:
        return None, None

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def scale_lstm_data(y_train, x_train, x_future):
    y_train = np.asarray(y_train, dtype=float)
    x_train = np.asarray(x_train, dtype=float)
    x_future = np.asarray(x_future, dtype=float)

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if not np.isfinite(y_std) or y_std <= 1e-12:
        y_std = 1.0

    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)
    x_std[~np.isfinite(x_std)] = 1.0
    x_std[x_std <= 1e-12] = 1.0

    return (
        (y_train - y_mean) / y_std,
        (x_train - x_mean) / x_std,
        (x_future - x_mean) / x_std,
        y_mean,
        y_std,
    )


def train_lstm(y_train_scaled, x_train_scaled, params, seed):
    set_seed(seed)
    sequences, targets = make_lstm_sequences(
        y_train_scaled,
        x_train_scaled,
        params["lookback"],
    )
    if sequences is None:
        raise RuntimeError("Not enough observations to create LSTM sequences.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(sequences, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        ),
        batch_size=params["batch_size"],
        shuffle=True,
        generator=generator,
    )

    model = LSTMForecaster(
        input_dim=sequences.shape[2],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(LSTM_DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_function = nn.MSELoss()

    model.train()
    for _ in range(params["max_epochs"]):
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(LSTM_DEVICE)
            batch_targets = batch_targets.to(LSTM_DEVICE)
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = loss_function(predictions, batch_targets)
            loss.backward()
            optimizer.step()

    return model


def recursive_lstm_forecast(
    model,
    y_train_scaled,
    x_train_scaled,
    x_future_scaled,
    lookback,
    y_mean,
    y_std,
):
    history_y = list(y_train_scaled[-lookback:])
    history_x = list(x_train_scaled[-lookback:])
    predictions = []

    model.eval()
    with torch.no_grad():
        for step in range(len(x_future_scaled)):
            y_window = np.asarray(history_y[-lookback:], dtype=np.float32).reshape(-1, 1)
            x_window = np.asarray(history_x[-lookback:], dtype=np.float32)
            sequence = np.concatenate([y_window, x_window], axis=1)
            sequence_tensor = torch.tensor(
                sequence[None, :, :],
                dtype=torch.float32,
                device=LSTM_DEVICE,
            )
            prediction = float(model(sequence_tensor).cpu().numpy()[0])
            predictions.append(prediction)
            history_y.append(prediction)
            history_x.append(x_future_scaled[step])

    return np.asarray(predictions, dtype=float) * y_std + y_mean


def lstm_fit_predict(y_train, x_train, x_future, params, seed):
    y_scaled, x_scaled, x_future_scaled, y_mean, y_std = scale_lstm_data(
        y_train,
        x_train,
        x_future,
    )
    model = train_lstm(y_scaled, x_scaled, params, seed)
    return recursive_lstm_forecast(
        model,
        y_scaled,
        x_scaled,
        x_future_scaled,
        params["lookback"],
        y_mean,
        y_std,
    )


def evaluate_lstm_grid(y_train_full, x_train_full, splits):
    scores_by_combo = {}

    for combo_id, params in enumerate(LSTM_GRID, start=1):
        scores = []
        failed = False

        for window_index, (train_slice, val_slice) in enumerate(splits, start=1):
            try:
                predictions = lstm_fit_predict(
                    y_train_full[train_slice],
                    x_train_full[train_slice],
                    x_train_full[val_slice],
                    params,
                    RANDOM_SEED + combo_id * 100 + window_index,
                )
            except Exception:
                failed = True
                break

            scores.append(smape(y_train_full[val_slice], predictions))

        scores_by_combo[combo_id] = (
            np.inf if failed or not scores else float(np.mean(scores))
        )

    return scores_by_combo


def run_lstm_grid_search(data, series_map):
    set_seed(RANDOM_SEED)
    combo_map = save_combo_map(LSTM_GRID, LSTM_COMB_MAP_PKL)
    combo_ids = combo_map["id"].astype(int).tolist()
    total_series = len(series_map)
    matrix, completed_ids = load_or_initialize_matrix(
        LSTM_SMAPE_MATRIX_PKL,
        "series_id",
        combo_ids,
        total_series,
    )
    completed_count = len(completed_ids)

    print(f"[INFO] Starting S&P 500 LSTM grid search: {len(combo_ids)} combinations.", flush=True)
    print(f"[INFO] LSTM device: {LSTM_DEVICE}.", flush=True)

    for row in series_map.itertuples(index=False):
        series_id = int(row.series_id)
        if series_id in completed_ids:
            continue

        start_time = time.time()
        values, dates = get_series_values(data, row.Name, row.Type)
        features = build_time_features(dates)
        finite_mask = np.isfinite(values)
        values = values[finite_mask]
        features = features[finite_mask]
        n_total = len(values)

        if n_total <= TEST_HORIZON:
            completed_count += 1
            print(
                f"[INFO] LSTM {completed_count}/{total_series} "
                f"(skipped: too short, series_id={series_id}, Name={row.Name}, Type={row.Type})",
                flush=True,
            )
            continue

        split_index = n_total - TEST_HORIZON
        if split_index <= 0 or split_index >= n_total:
            completed_count += 1
            print(
                f"[INFO] LSTM {completed_count}/{total_series} "
                f"(skipped: bad split, series_id={series_id}, Name={row.Name}, Type={row.Type})",
                flush=True,
            )
            continue

        y_train_full = values[:split_index]
        x_train_full = features[:split_index]
        splits = rolling_cv_slices(len(y_train_full), n_total)

        if not splits:
            completed_count += 1
            print(
                f"[INFO] LSTM {completed_count}/{total_series} "
                f"(skipped: no CV splits, series_id={series_id}, Name={row.Name}, Type={row.Type})",
                flush=True,
            )
            continue

        combo_scores = evaluate_lstm_grid(y_train_full, x_train_full, splits)
        matrix = append_result(
            matrix,
            LSTM_SMAPE_MATRIX_PKL,
            "series_id",
            series_id,
            combo_scores,
        )
        completed_ids.add(series_id)
        completed_count += 1
        elapsed_minutes = (time.time() - start_time) / 60.0
        print(
            f"[INFO] LSTM {completed_count}/{total_series} series "
            f"(last series_id={series_id}, Name={row.Name}, Type={row.Type}, "
            f"time={elapsed_minutes:.2f} min)",
            flush=True,
        )


def main():
    global CSV_PATH, SERIES_MAP_PKL, PROPHET_COMB_MAP_PKL, PROPHET_SMAPE_MATRIX_PKL
    global SARIMA_COMB_MAP_PKL, SARIMA_SMAPE_MATRIX_PKL, LSTM_COMB_MAP_PKL, LSTM_SMAPE_MATRIX_PKL

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    CSV_PATH = os.path.join(args.data_dir, "snp500_data.csv")
    SERIES_MAP_PKL = os.path.join(args.output_dir, "snp500_series_to_id.pkl")
    PROPHET_COMB_MAP_PKL = os.path.join(args.output_dir, "snp500_comb_to_id.pkl")
    PROPHET_SMAPE_MATRIX_PKL = os.path.join(args.output_dir, "snp500_smape_matrix.pkl")
    SARIMA_COMB_MAP_PKL = os.path.join(args.output_dir, "snp500_comb_to_id_sarima.pkl")
    SARIMA_SMAPE_MATRIX_PKL = os.path.join(args.output_dir, "snp500_smape_matrix_sarima.pkl")
    LSTM_COMB_MAP_PKL = os.path.join(args.output_dir, "snp500_comb_to_id_lstm.pkl")
    LSTM_SMAPE_MATRIX_PKL = os.path.join(args.output_dir, "snp500_smape_matrix_lstm.pkl")

    data = load_data()
    series_map = build_series_map(data)

    run_prophet_grid_search(data, series_map)
    run_sarima_grid_search(data, series_map)
    run_lstm_grid_search(data, series_map)


if __name__ == "__main__":
    main()
