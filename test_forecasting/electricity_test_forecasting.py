"""Evaluate all Prophet, SARIMA, and LSTM configurations on the electricity test set."""

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

DATA_PKL_PATH = os.path.join("data", "electricity", "consumption_dataset.pkl")
FEATURES_PKL_PATH = os.path.join("data", "electricity", "electricity_feature_extraction_catch22.pkl")
TARGET_COL = "consumption"
EXOG_COL = "heat_index"
FEATURE_ID_COL = "id"
ELECTRICITY_INDICATOR_COL = "city_Electricity"
BUILDING_ID_COL = "building_id"
SERIES_TO_PART_PKL = "series_to_part.pkl"
TEST_HORIZON = 48

PROPHET_TIMEOUT_SEC = 180
PROPHET_COMB_MAP_PKL = "electricity_comb_to_id.pkl"
PROPHET_SMAPE_TEST_PKL = "electricity_smape_matrix_test.pkl"
PROPHET_FORECAST_TEST_PKL = "electricity_forecasting_test.pkl"
PROPHET_METRIC_PERIOD = 1

SARIMA_TIMEOUT_SEC = 60
SARIMA_SEASONAL_PERIOD = 24
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
SARIMA_SMAPE_TEST_PKL = "electricity_smape_matrix_test_sarima.pkl"
SARIMA_FORECAST_TEST_PKL = "electricity_forecasting_test_sarima.pkl"
SARIMA_METRIC_PERIOD = 24

RANDOM_SEED = 42
LSTM_LOOKBACKS = [24, 48, 168]
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
LSTM_COMB_MAP_PKL = "electricity_comb_to_id_lstm.pkl"
LSTM_SMAPE_TEST_PKL = "electricity_smape_matrix_test_lstm.pkl"
LSTM_FORECAST_TEST_PKL = "electricity_forecasting_test_lstm.pkl"
LSTM_METRIC_PERIOD = 24


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all Prophet, SARIMA, and LSTM configurations on the electricity test set."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join("data", "electricity"),
        help="Directory containing consumption_dataset.pkl.",
    )
    parser.add_argument(
        "--features-dir",
        default=None,
        help="Directory containing electricity_feature_extraction_catch22.pkl. "
        "Defaults to --data-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("outputs", "electricity"),
        help="Directory shared with grid_search/electricity_grid_search.py: the "
        "Prophet combination map and the optional series_to_part.pkl ordering "
        "are read from here, and test matrices/forecasts are written here.",
    )
    return parser.parse_args()


def smape(y_true, y_pred, eps=1e-9):
    """Compute symmetric mean absolute percentage error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred) + eps
    return 200.0 * np.mean(np.abs(y_true - y_pred) / denominator)


def global_metrics(y_true, y_pred, mase_scales, rmsse_scales, eps=1e-9):
    """Compute global sMAPE, MASE, and RMSSE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mase_scales = np.asarray(mase_scales, dtype=float)
    rmsse_scales = np.asarray(rmsse_scales, dtype=float)

    global_smape = smape(y_true, y_pred, eps=eps)
    mase = np.mean(np.abs(y_true - y_pred) / (mase_scales + eps))
    rmsse = np.sqrt(np.mean((y_true - y_pred) ** 2 / (rmsse_scales + eps)))
    return global_smape, mase, rmsse


def seasonal_scales(y_train, period):
    """Return the absolute and squared seasonal-naive scaling terms."""
    y_train = np.asarray(y_train, dtype=float)
    differences = y_train[period:] - y_train[:-period]
    return np.mean(np.abs(differences)), np.mean(differences**2)


def _process_target(queue, function, args, kwargs):
    try:
        queue.put(("ok", function(*args, **(kwargs or {}))))
    except Exception as exc:
        queue.put(("error", exc))


def run_with_timeout(function, args=(), kwargs=None, timeout=180):
    """Run a function in a subprocess with a fixed time limit."""
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


def load_result_matrices(smape_path, forecast_path, id_column, combo_ids):
    """Load or initialize the test-error and forecast matrices."""
    columns = [id_column] + combo_ids

    if os.path.exists(smape_path):
        smape_matrix = pd.read_pickle(smape_path)
        for column in columns:
            if column not in smape_matrix.columns:
                smape_matrix[column] = np.nan
        smape_matrix = smape_matrix[columns]
    else:
        smape_matrix = pd.DataFrame(columns=columns)
        smape_matrix.to_pickle(smape_path)

    if os.path.exists(forecast_path):
        forecast_matrix = pd.read_pickle(forecast_path)
        for column in columns:
            if column not in forecast_matrix.columns:
                forecast_matrix[column] = np.nan
        forecast_matrix = forecast_matrix[columns]
    else:
        forecast_matrix = pd.DataFrame(columns=columns)
        forecast_matrix.to_pickle(forecast_path)

    return smape_matrix, forecast_matrix


def append_test_results(
    smape_matrix,
    forecast_matrix,
    smape_path,
    forecast_path,
    smape_row,
    forecast_row,
):
    smape_matrix = pd.concat(
        [smape_matrix, pd.DataFrame([smape_row])], ignore_index=True
    )
    forecast_matrix = pd.concat(
        [forecast_matrix, pd.DataFrame([forecast_row])], ignore_index=True
    )
    smape_matrix.to_pickle(smape_path)
    forecast_matrix.to_pickle(forecast_path)
    return smape_matrix, forecast_matrix


def load_data():
    """Load and clean the electricity-consumption observations."""
    data = pd.read_pickle(DATA_PKL_PATH)
    if not np.issubdtype(data["ts"].dtype, np.datetime64):
        data["ts"] = pd.to_datetime(data["ts"])

    data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
    data[EXOG_COL] = pd.to_numeric(data[EXOG_COL], errors="coerce")
    return data.dropna(subset=["ts", TARGET_COL, EXOG_COL, "id"]).copy()


def load_electricity_ids():
    """Load the building identifiers included in the electricity subset."""
    features = pd.read_pickle(FEATURES_PKL_PATH)

    if FEATURE_ID_COL not in features.columns:
        raise ValueError(f"Missing column in features file: {FEATURE_ID_COL}")
    if ELECTRICITY_INDICATOR_COL not in features.columns:
        raise ValueError(
            f"Missing column in features file: {ELECTRICITY_INDICATOR_COL}"
        )

    building_ids = (
        features.loc[features[ELECTRICITY_INDICATOR_COL] == 1, FEATURE_ID_COL]
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return sorted(building_ids)


def load_prophet_building_ids(data):
    """Load the building order used by the original Prophet test script."""
    if os.path.exists(SERIES_TO_PART_PKL):
        series_to_part = pd.read_pickle(SERIES_TO_PART_PKL)
        return series_to_part[BUILDING_ID_COL].tolist()
    return sorted(data["id"].dropna().unique().tolist())


def get_building_data(data, building_id):
    return data[data["id"] == building_id].sort_values("ts").copy()


def prophet_fit_predict(train_data, predict_data, params):
    model = Prophet(
        n_changepoints=int(params["n_changepoints"]),
        changepoint_range=float(params["changepoint_range"]),
        changepoint_prior_scale=float(params["changepoint_prior_scale"]),
        seasonality_prior_scale=float(params["seasonality_prior_scale"]),
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=True,
    )
    model.add_regressor(EXOG_COL)
    model.fit(train_data[["ds", "y", EXOG_COL]], algorithm="LBFGS", iter=1000)
    forecast = model.predict(predict_data[["ds", EXOG_COL]].copy())
    return forecast["yhat"].values


def sarima_fit_predict(y_train, x_train, forecast_length, x_test, params):
    model = SARIMAX(
        endog=y_train,
        exog=x_train,
        order=(int(params["p"]), int(params["d"]), int(params["q"])),
        seasonal_order=(
            int(params["P"]),
            int(params["D"]),
            int(params["Q"]),
            SARIMA_SEASONAL_PERIOD,
        ),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted_model = model.fit(disp=False)
    return np.asarray(
        fitted_model.forecast(steps=forecast_length, exog=x_test), dtype=float
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
    hour = timestamps.dt.hour.to_numpy()
    day_of_week = timestamps.dt.dayofweek.to_numpy()
    month = timestamps.dt.month.to_numpy() - 1

    hour_sin, hour_cos = cyclical_feature(hour, 24)
    day_sin, day_cos = cyclical_feature(day_of_week, 7)
    month_sin, month_cos = cyclical_feature(month, 12)
    return np.column_stack(
        [hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos]
    ).astype(np.float32)


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
    lookback = int(params["lookback"])
    sequences, targets = make_lstm_sequences(
        y_train_scaled, x_train_scaled, lookback
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
        batch_size=int(params["batch_size"]),
        shuffle=True,
        generator=generator,
    )

    model = LSTMForecaster(
        input_dim=sequences.shape[2],
        hidden_size=int(params["hidden_size"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params["dropout"]),
    ).to(LSTM_DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(params["learning_rate"])
    )
    loss_function = nn.MSELoss()

    model.train()
    for _ in range(int(params["max_epochs"])):
        for inputs, targets_batch in loader:
            inputs = inputs.to(LSTM_DEVICE)
            targets_batch = targets_batch.to(LSTM_DEVICE)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_function(predictions, targets_batch)
            loss.backward()
            optimizer.step()

    return model


def recursive_lstm_forecast(
    model,
    y_train_scaled,
    x_train_scaled,
    x_future_scaled,
    params,
    y_mean,
    y_std,
):
    lookback = int(params["lookback"])
    y_history = list(y_train_scaled[-lookback:])
    x_history = list(x_train_scaled[-lookback:])
    predictions = []

    model.eval()
    with torch.no_grad():
        for step in range(len(x_future_scaled)):
            y_window = np.asarray(
                y_history[-lookback:], dtype=np.float32
            ).reshape(-1, 1)
            x_window = np.asarray(x_history[-lookback:], dtype=np.float32)
            sequence = np.concatenate([y_window, x_window], axis=1)
            sequence_tensor = torch.tensor(
                sequence[None, :, :],
                dtype=torch.float32,
                device=LSTM_DEVICE,
            )
            prediction = float(model(sequence_tensor).cpu().numpy()[0])
            predictions.append(prediction)
            y_history.append(prediction)
            x_history.append(x_future_scaled[step])

    return np.asarray(predictions, dtype=float) * y_std + y_mean


def lstm_fit_predict(y_train, x_train, x_future, params, seed):
    scaled = scale_lstm_data(y_train, x_train, x_future)
    y_train_scaled, x_train_scaled, x_future_scaled, y_mean, y_std = scaled
    model = train_lstm(y_train_scaled, x_train_scaled, params, seed)
    return recursive_lstm_forecast(
        model,
        y_train_scaled,
        x_train_scaled,
        x_future_scaled,
        params,
        y_mean,
        y_std,
    )


def build_lstm_combo_map():
    records = [{"id": combo_id, **params} for combo_id, params in enumerate(LSTM_GRID, 1)]
    combo_map = pd.DataFrame(records)
    combo_map.to_pickle(LSTM_COMB_MAP_PKL)
    print(
        f"[INFO] Saved {LSTM_COMB_MAP_PKL} with {len(combo_map)} combinations.",
        flush=True,
    )
    return combo_map


def load_required_table(path, sort_column):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_pickle(path).sort_values(sort_column)


def print_oracle_metrics(
    label,
    smape_path,
    forecast_path,
    combo_ids,
    data,
    metric_period,
    use_exogenous_filter,
    skip_invalid_scales,
):
    smape_matrix = pd.read_pickle(smape_path)
    forecast_matrix = pd.read_pickle(forecast_path)
    y_true_all = []
    y_pred_all = []
    mase_scales = []
    rmsse_scales = []

    for _, row in smape_matrix.iterrows():
        building_id = row[BUILDING_ID_COL]
        best_combo_id = int(row[combo_ids].astype(float).idxmin())
        building_data = get_building_data(data, building_id)
        y_all = building_data[TARGET_COL].astype(float).values

        if use_exogenous_filter:
            x_all = building_data[EXOG_COL].astype(float).values
            finite = np.isfinite(y_all) & np.isfinite(x_all)
            y_all = y_all[finite]

        if len(y_all) <= TEST_HORIZON:
            continue

        split_index = len(y_all) - TEST_HORIZON
        y_train = y_all[:split_index]
        y_test = y_all[split_index:]
        y_pred = forecast_matrix.loc[
            forecast_matrix[BUILDING_ID_COL] == building_id, best_combo_id
        ].iloc[0]
        mase_scale, rmsse_scale = seasonal_scales(y_train, metric_period)
        if skip_invalid_scales and (
            not np.isfinite(mase_scale)
            or not np.isfinite(rmsse_scale)
            or mase_scale <= 1e-9
            or rmsse_scale <= 1e-9
        ):
            continue

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        mase_scales.extend(np.repeat(mase_scale, TEST_HORIZON))
        rmsse_scales.extend(np.repeat(rmsse_scale, TEST_HORIZON))

    if not y_true_all:
        print(f"{label} oracle test metrics could not be computed.")
        return

    global_smape, mase, rmsse = global_metrics(
        y_true_all, y_pred_all, mase_scales, rmsse_scales
    )
    print(f"{label} oracle test sMAPE: {global_smape:.4f}")
    print(f"{label} oracle test MASE: {mase:.4f}")
    print(f"{label} oracle test RMSSE: {rmsse:.4f}")


def run_prophet_test(data):
    combo_map = load_required_table(PROPHET_COMB_MAP_PKL, "id")
    combo_ids = combo_map["id"].astype(int).tolist()
    building_ids = load_prophet_building_ids(data)
    smape_matrix, forecast_matrix = load_result_matrices(
        PROPHET_SMAPE_TEST_PKL,
        PROPHET_FORECAST_TEST_PKL,
        BUILDING_ID_COL,
        combo_ids,
    )
    completed = set(smape_matrix[BUILDING_ID_COL].tolist())
    completed_count = len(completed)
    total_series = len(building_ids)

    print(
        f"[INFO] Starting Electricity Prophet test run. Already done: "
        f"{completed_count}/{total_series}",
        flush=True,
    )

    for building_id in building_ids:
        if building_id in completed:
            continue

        building_data = get_building_data(data, building_id)
        y_all = building_data[TARGET_COL].astype(float).values
        dates = building_data["ts"].values
        exogenous = building_data[EXOG_COL].astype(float).values

        if len(y_all) <= TEST_HORIZON:
            print(f"[INFO] Skipped building_id={building_id}: too short", flush=True)
            continue

        split_index = len(y_all) - TEST_HORIZON
        y_train = y_all[:split_index]
        y_test = y_all[split_index:]
        train_data = pd.DataFrame(
            {
                "ds": dates[:split_index],
                "y": y_train,
                EXOG_COL: exogenous[:split_index],
            }
        )
        test_data = pd.DataFrame(
            {
                "ds": dates[split_index:],
                "y": y_test,
                EXOG_COL: exogenous[split_index:],
            }
        )
        fallback = np.repeat(np.mean(y_train), TEST_HORIZON)
        smape_row = {BUILDING_ID_COL: building_id}
        forecast_row = {BUILDING_ID_COL: building_id}

        for _, combination in combo_map.iterrows():
            combo_id = int(combination["id"])
            try:
                predictions = run_with_timeout(
                    prophet_fit_predict,
                    args=(train_data, test_data, combination.to_dict()),
                    timeout=PROPHET_TIMEOUT_SEC,
                )
            except Exception:
                predictions = fallback.copy()

            smape_row[combo_id] = float(smape(y_test, predictions))
            forecast_row[combo_id] = np.asarray(predictions, dtype=float)

        smape_matrix, forecast_matrix = append_test_results(
            smape_matrix,
            forecast_matrix,
            PROPHET_SMAPE_TEST_PKL,
            PROPHET_FORECAST_TEST_PKL,
            smape_row,
            forecast_row,
        )
        completed.add(building_id)
        completed_count += 1
        print(
            f"[INFO] Finished {completed_count}/{total_series} series "
            f"(building_id={building_id})",
            flush=True,
        )

    print_oracle_metrics(
        "Prophet",
        PROPHET_SMAPE_TEST_PKL,
        PROPHET_FORECAST_TEST_PKL,
        combo_ids,
        data,
        PROPHET_METRIC_PERIOD,
        use_exogenous_filter=False,
        skip_invalid_scales=False,
    )


def run_sarima_test(data, building_ids):
    combo_map = pd.DataFrame(
        [{"id": combo_id, **params} for combo_id, params in enumerate(SARIMA_GRID, 1)]
    )
    combo_ids = combo_map["id"].astype(int).tolist()
    smape_matrix, forecast_matrix = load_result_matrices(
        SARIMA_SMAPE_TEST_PKL,
        SARIMA_FORECAST_TEST_PKL,
        BUILDING_ID_COL,
        combo_ids,
    )
    completed = set(smape_matrix[BUILDING_ID_COL].tolist())
    completed_count = len(completed)
    total_series = len(building_ids)

    print(
        f"[INFO] Starting Electricity SARIMA test run. Already done: "
        f"{completed_count}/{total_series}",
        flush=True,
    )
    print(f"[INFO] Total SARIMA combinations: {len(combo_ids)}", flush=True)

    for building_id in building_ids:
        if building_id in completed:
            continue

        building_data = get_building_data(data, building_id)
        y_all = building_data[TARGET_COL].astype(float).values
        x_all = building_data[EXOG_COL].astype(float).values.reshape(-1, 1)
        finite = np.isfinite(y_all) & np.isfinite(x_all[:, 0])
        y_all = y_all[finite]
        x_all = x_all[finite]

        if len(y_all) <= TEST_HORIZON:
            print(f"[INFO] Skipped building_id={building_id}: too short", flush=True)
            continue

        split_index = len(y_all) - TEST_HORIZON
        y_train = y_all[:split_index]
        x_train = x_all[:split_index]
        y_test = y_all[split_index:]
        x_test = x_all[split_index:]
        fallback = np.repeat(np.mean(y_train), TEST_HORIZON)
        smape_row = {BUILDING_ID_COL: building_id}
        forecast_row = {BUILDING_ID_COL: building_id}

        for _, combination in combo_map.iterrows():
            combo_id = int(combination["id"])
            try:
                predictions = run_with_timeout(
                    sarima_fit_predict,
                    args=(
                        y_train,
                        x_train,
                        TEST_HORIZON,
                        x_test,
                        combination.to_dict(),
                    ),
                    timeout=SARIMA_TIMEOUT_SEC,
                )
            except Exception:
                predictions = fallback.copy()

            smape_row[combo_id] = float(smape(y_test, predictions))
            forecast_row[combo_id] = np.asarray(predictions, dtype=float)

        smape_matrix, forecast_matrix = append_test_results(
            smape_matrix,
            forecast_matrix,
            SARIMA_SMAPE_TEST_PKL,
            SARIMA_FORECAST_TEST_PKL,
            smape_row,
            forecast_row,
        )
        completed.add(building_id)
        completed_count += 1
        print(
            f"[INFO] Finished {completed_count}/{total_series} series "
            f"(building_id={building_id})",
            flush=True,
        )

    print_oracle_metrics(
        "SARIMA",
        SARIMA_SMAPE_TEST_PKL,
        SARIMA_FORECAST_TEST_PKL,
        combo_ids,
        data,
        SARIMA_METRIC_PERIOD,
        use_exogenous_filter=True,
        skip_invalid_scales=True,
    )


def run_lstm_test(data, building_ids):
    set_seed(RANDOM_SEED)
    if os.path.exists(LSTM_COMB_MAP_PKL):
        combo_map = pd.read_pickle(LSTM_COMB_MAP_PKL).sort_values("id")
    else:
        combo_map = build_lstm_combo_map()

    combo_ids = combo_map["id"].astype(int).tolist()
    smape_matrix, forecast_matrix = load_result_matrices(
        LSTM_SMAPE_TEST_PKL,
        LSTM_FORECAST_TEST_PKL,
        BUILDING_ID_COL,
        combo_ids,
    )
    completed = set(smape_matrix[BUILDING_ID_COL].tolist())
    completed_count = len(completed)
    total_series = len(building_ids)

    print(
        f"[INFO] Starting Electricity LSTM test run. Already done: "
        f"{completed_count}/{total_series}",
        flush=True,
    )
    print(f"[INFO] Total LSTM combinations: {len(combo_ids)}", flush=True)
    print(f"[INFO] Device: {LSTM_DEVICE}", flush=True)

    for building_id in building_ids:
        if building_id in completed:
            continue

        start_time = time.time()
        building_data = get_building_data(data, building_id)
        y_all = building_data[TARGET_COL].astype(float).values
        heat_index = building_data[EXOG_COL].astype(float).values.reshape(-1, 1)
        time_features = build_time_features(building_data["ts"].values)
        x_all = np.concatenate([time_features, heat_index], axis=1)
        finite = np.isfinite(y_all) & np.all(np.isfinite(x_all), axis=1)
        y_all = y_all[finite]
        x_all = x_all[finite]

        if len(y_all) <= TEST_HORIZON:
            print(f"[INFO] Skipped building_id={building_id}: too short", flush=True)
            continue

        split_index = len(y_all) - TEST_HORIZON
        y_train = y_all[:split_index]
        x_train = x_all[:split_index]
        y_test = y_all[split_index:]
        x_test = x_all[split_index:]
        fallback = np.repeat(np.mean(y_train), TEST_HORIZON)
        smape_row = {BUILDING_ID_COL: building_id}
        forecast_row = {BUILDING_ID_COL: building_id}

        for _, combination in combo_map.iterrows():
            combo_id = int(combination["id"])
            try:
                predictions = lstm_fit_predict(
                    y_train,
                    x_train,
                    x_test,
                    combination.to_dict(),
                    RANDOM_SEED + combo_id,
                )
            except Exception:
                predictions = fallback.copy()

            smape_row[combo_id] = float(smape(y_test, predictions))
            forecast_row[combo_id] = np.asarray(predictions, dtype=float)

        smape_matrix, forecast_matrix = append_test_results(
            smape_matrix,
            forecast_matrix,
            LSTM_SMAPE_TEST_PKL,
            LSTM_FORECAST_TEST_PKL,
            smape_row,
            forecast_row,
        )
        completed.add(building_id)
        completed_count += 1
        elapsed_minutes = (time.time() - start_time) / 60.0
        print(
            f"[INFO] Finished {completed_count}/{total_series} series "
            f"(building_id={building_id}, time={elapsed_minutes:.2f} min)",
            flush=True,
        )

    print_oracle_metrics(
        "LSTM",
        LSTM_SMAPE_TEST_PKL,
        LSTM_FORECAST_TEST_PKL,
        combo_ids,
        data,
        LSTM_METRIC_PERIOD,
        use_exogenous_filter=True,
        skip_invalid_scales=True,
    )


def main():
    global DATA_PKL_PATH, FEATURES_PKL_PATH, SERIES_TO_PART_PKL
    global PROPHET_COMB_MAP_PKL, PROPHET_SMAPE_TEST_PKL, PROPHET_FORECAST_TEST_PKL
    global SARIMA_SMAPE_TEST_PKL, SARIMA_FORECAST_TEST_PKL
    global LSTM_COMB_MAP_PKL, LSTM_SMAPE_TEST_PKL, LSTM_FORECAST_TEST_PKL

    args = parse_args()
    features_dir = args.features_dir or args.data_dir
    os.makedirs(args.output_dir, exist_ok=True)

    DATA_PKL_PATH = os.path.join(args.data_dir, "consumption_dataset.pkl")
    FEATURES_PKL_PATH = os.path.join(features_dir, "electricity_feature_extraction_catch22.pkl")
    SERIES_TO_PART_PKL = os.path.join(args.output_dir, "series_to_part.pkl")

    PROPHET_COMB_MAP_PKL = os.path.join(args.output_dir, "electricity_comb_to_id.pkl")
    PROPHET_SMAPE_TEST_PKL = os.path.join(args.output_dir, "electricity_smape_matrix_test.pkl")
    PROPHET_FORECAST_TEST_PKL = os.path.join(args.output_dir, "electricity_forecasting_test.pkl")

    SARIMA_SMAPE_TEST_PKL = os.path.join(args.output_dir, "electricity_smape_matrix_test_sarima.pkl")
    SARIMA_FORECAST_TEST_PKL = os.path.join(args.output_dir, "electricity_forecasting_test_sarima.pkl")

    LSTM_COMB_MAP_PKL = os.path.join(args.output_dir, "electricity_comb_to_id_lstm.pkl")
    LSTM_SMAPE_TEST_PKL = os.path.join(args.output_dir, "electricity_smape_matrix_test_lstm.pkl")
    LSTM_FORECAST_TEST_PKL = os.path.join(args.output_dir, "electricity_forecasting_test_lstm.pkl")

    data = load_data()
    electricity_ids = load_electricity_ids()
    run_prophet_test(data)
    run_sarima_test(data, electricity_ids)
    run_lstm_test(data, electricity_ids)


if __name__ == "__main__":
    mp.freeze_support()
    main()
