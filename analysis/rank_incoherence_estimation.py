"""Estimate effective rank and incoherence for concatenated model-loss matrices."""

import argparse
import os
import pickle

import numpy as np
import pandas as pd


ENERGY_THRESHOLDS = [0.90, 0.95, 0.99]
CENTER_BEFORE_SVD = False

# Supported values: "error" and "drop_rows_cols".
NONFINITE_POLICY = "drop_rows_cols"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate effective rank and incoherence for concatenated "
        "model-loss matrices."
    )
    parser.add_argument(
        "--electricity-dir",
        default=os.path.join("outputs", "electricity"),
        help="Directory containing electricity_smape_matrix*.pkl "
        "(grid_search/electricity_grid_search.py output).",
    )
    parser.add_argument(
        "--snp500-dir",
        default=os.path.join("outputs", "snp500"),
        help="Directory containing snp500_smape_matrix*.pkl "
        "(grid_search/snp500_grid_search.py output).",
    )
    parser.add_argument(
        "--covid19-dir",
        default=os.path.join("outputs", "covid19"),
        help="Directory containing covid19_smape_matrix*.pkl "
        "(grid_search/covid19_grid_search.py output).",
    )
    parser.add_argument(
        "--output-path",
        default=os.path.join("outputs", "rank_mu_all_models_matrices.csv"),
        help="Path to write the combined rank/incoherence results CSV to.",
    )
    return parser.parse_args()


def build_dataset_matrix_groups(electricity_dir, snp500_dir, covid19_dir):
    return [
        {
            "matrix_name": "Electricity All Models",
            "id_col": "building_id",
            "model_files": [
                ("prophet", os.path.join(electricity_dir, "electricity_smape_matrix.pkl")),
                ("sarima", os.path.join(electricity_dir, "electricity_smape_matrix_sarima.pkl")),
                ("lstm", os.path.join(electricity_dir, "electricity_smape_matrix_lstm.pkl")),
            ],
        },
        {
            "matrix_name": "SNP500 All Models",
            "id_col": "series_id",
            "model_files": [
                ("prophet", os.path.join(snp500_dir, "snp500_smape_matrix.pkl")),
                ("sarima", os.path.join(snp500_dir, "snp500_smape_matrix_sarima.pkl")),
                ("lstm", os.path.join(snp500_dir, "snp500_smape_matrix_lstm.pkl")),
            ],
        },
        {
            "matrix_name": "COVID19 All Models",
            "id_col": "series_id",
            "model_files": [
                ("prophet", os.path.join(covid19_dir, "covid19_smape_matrix.pkl")),
                ("sarima", os.path.join(covid19_dir, "covid19_smape_matrix_sarima.pkl")),
                ("lstm", os.path.join(covid19_dir, "covid19_smape_matrix_lstm.pkl")),
            ],
        },
    ]


def load_pkl_object(file_path):
    """Load a serialized Python object from a pickle file."""
    with open(file_path, "rb") as file:
        return pickle.load(file)


def find_id_column(dataframe, preferred_id_col, file_path):
    """Return the preferred ID column or an available fallback."""
    if preferred_id_col in dataframe.columns:
        return preferred_id_col

    fallback_cols = ["series_id", "building_id", "id"]
    for column in fallback_cols:
        if column in dataframe.columns:
            print(
                f"[WARNING] {file_path}: expected id_col='{preferred_id_col}', "
                f"using existing column '{column}' instead."
            )
            return column

    raise ValueError(
        f"{file_path}: could not find ID column. "
        f"Expected '{preferred_id_col}' or one of {fallback_cols}."
    )


def load_one_model_matrix_for_concat(model_name, file_path, id_col):
    """Load one loss matrix and prefix its value columns with the model name."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    loaded_object = load_pkl_object(file_path)
    if not isinstance(loaded_object, pd.DataFrame):
        raise TypeError(
            f"{file_path}: expected pandas DataFrame for concatenation by series ID. "
            f"Got: {type(loaded_object)}"
        )

    dataframe = loaded_object.copy()
    actual_id_col = find_id_column(dataframe, id_col, file_path)
    dataframe["__series_key__"] = dataframe[actual_id_col].astype(str)

    value_cols = [
        column
        for column in dataframe.columns
        if column not in [actual_id_col, "__series_key__"]
    ]
    values = dataframe[value_cols].apply(pd.to_numeric, errors="coerce")
    values = values.rename(
        columns={column: f"{model_name}__{column}" for column in value_cols}
    )

    output = pd.concat([dataframe[["__series_key__"]], values], axis=1)

    if output["__series_key__"].duplicated().any():
        duplicated = (
            output.loc[output["__series_key__"].duplicated(), "__series_key__"]
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"{file_path}: duplicate series IDs found, for example: {duplicated}"
        )

    return output


def load_concatenated_all_models_matrix(group):
    """Concatenate Prophet, SARIMA, and LSTM matrices by common series IDs."""
    id_col = group["id_col"]
    parts = []

    for model_name, file_path in group["model_files"]:
        part = load_one_model_matrix_for_concat(
            model_name=model_name,
            file_path=file_path,
            id_col=id_col,
        )
        print(f"[INFO] Loaded {model_name}: {file_path} | shape={part.shape}")
        parts.append(part)

    common_ids = set(parts[0]["__series_key__"])
    for part in parts[1:]:
        common_ids &= set(part["__series_key__"])

    common_ids = sorted(
        common_ids,
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )

    if len(common_ids) == 0:
        raise RuntimeError(
            f"{group['matrix_name']}: no common series IDs across "
            "Prophet, SARIMA and LSTM."
        )

    combined = pd.DataFrame({"__series_key__": common_ids})
    for part in parts:
        aligned_part = part.set_index("__series_key__").loc[common_ids].reset_index()
        combined = combined.merge(aligned_part, on="__series_key__", how="inner")

    value_cols = [column for column in combined.columns if column != "__series_key__"]
    matrix = combined[value_cols].to_numpy(dtype=float)

    return matrix, {
        "n_common_series": len(common_ids),
        "n_total_columns": len(value_cols),
        "combined_columns": value_cols,
    }


def handle_nonfinite_values(matrix, matrix_name, policy="drop_rows_cols"):
    """Apply the configured NaN and infinity policy before SVD."""
    matrix = np.asarray(matrix, dtype=float)
    raw_shape = matrix.shape
    n_nonfinite = np.size(matrix) - np.isfinite(matrix).sum()

    if n_nonfinite == 0:
        return matrix, {
            "raw_n": raw_shape[0],
            "raw_m": raw_shape[1],
            "used_n": raw_shape[0],
            "used_m": raw_shape[1],
            "n_nonfinite": 0,
            "dropped_rows": 0,
            "dropped_cols": 0,
        }

    if policy == "error":
        raise ValueError(
            f"{matrix_name}: matrix contains {n_nonfinite} NaN/inf values. "
            "SVD cannot be computed directly."
        )

    if policy != "drop_rows_cols":
        raise ValueError(f"Unknown NONFINITE_POLICY: {policy}")

    finite_mask = np.isfinite(matrix)
    keep_rows = finite_mask.all(axis=1)
    keep_cols = finite_mask.all(axis=0)
    clean_matrix = matrix[keep_rows, :][:, keep_cols]

    if clean_matrix.shape[0] == 0 or clean_matrix.shape[1] == 0:
        raise ValueError(
            f"{matrix_name}: after removing rows/columns with NaN/inf, "
            f"matrix is empty. Original shape: {raw_shape}"
        )

    return clean_matrix, {
        "raw_n": raw_shape[0],
        "raw_m": raw_shape[1],
        "used_n": clean_matrix.shape[0],
        "used_m": clean_matrix.shape[1],
        "n_nonfinite": int(n_nonfinite),
        "dropped_rows": int((~keep_rows).sum()),
        "dropped_cols": int((~keep_cols).sum()),
    }


def estimate_r_and_mu(matrix, energy_threshold=0.95, center=False):
    """Estimate effective rank and incoherence from a truncated SVD."""
    values = np.asarray(matrix, dtype=float).copy()

    if center:
        values = values - np.mean(values)

    n, m = values.shape
    left_vectors, singular_values, right_vectors_transposed = np.linalg.svd(
        values,
        full_matrices=False,
    )

    singular_energy = singular_values**2
    total_energy = np.sum(singular_energy)

    if total_energy <= 0:
        return {
            "r": 0,
            "mu_U": np.nan,
            "mu_V": np.nan,
            "mu": np.nan,
            "mu_times_r": np.nan,
            "tail_ratio": np.nan,
            "explained_energy": np.nan,
            "largest_singular_value": np.nan,
            "smallest_used_singular_value": np.nan,
        }

    cumulative_energy = np.cumsum(singular_energy) / total_energy
    rank = int(np.searchsorted(cumulative_energy, energy_threshold) + 1)

    truncated_left = left_vectors[:, :rank]
    truncated_right = right_vectors_transposed.T[:, :rank]

    left_row_norms = np.sum(truncated_left**2, axis=1)
    right_row_norms = np.sum(truncated_right**2, axis=1)

    mu_u = (n / rank) * np.max(left_row_norms)
    mu_v = (m / rank) * np.max(right_row_norms)
    mu = max(mu_u, mu_v)

    tail_energy = np.sum(singular_energy[rank:])
    tail_ratio = np.sqrt(tail_energy / total_energy)

    return {
        "r": rank,
        "mu_U": float(mu_u),
        "mu_V": float(mu_v),
        "mu": float(mu),
        "mu_times_r": float(mu * rank),
        "tail_ratio": float(tail_ratio),
        "explained_energy": float(cumulative_energy[rank - 1]),
        "largest_singular_value": float(singular_values[0]),
        "smallest_used_singular_value": float(singular_values[rank - 1]),
    }


def analyze_one_dataset_group(group):
    """Estimate rank and incoherence for each configured energy threshold."""
    matrix_name = group["matrix_name"]

    print("=" * 100)
    print(f"Analyzing: {matrix_name}")
    print("Model matrices:")
    for model_name, file_path in group["model_files"]:
        print(f"  {model_name}: {file_path}")

    raw_matrix, concat_info = load_concatenated_all_models_matrix(group)
    matrix, cleaning_info = handle_nonfinite_values(
        raw_matrix,
        matrix_name=matrix_name,
        policy=NONFINITE_POLICY,
    )

    rows = []
    for threshold in ENERGY_THRESHOLDS:
        result = estimate_r_and_mu(
            matrix,
            energy_threshold=threshold,
            center=CENTER_BEFORE_SVD,
        )

        rows.append(
            {
                "matrix": matrix_name,
                "energy_threshold": threshold,
                "n_common_series_before_cleaning": concat_info["n_common_series"],
                "n_columns_before_cleaning": concat_info["n_total_columns"],
                "raw_n": cleaning_info["raw_n"],
                "raw_m": cleaning_info["raw_m"],
                "used_n": cleaning_info["used_n"],
                "used_m": cleaning_info["used_m"],
                "n_nonfinite": cleaning_info["n_nonfinite"],
                "dropped_rows": cleaning_info["dropped_rows"],
                "dropped_cols": cleaning_info["dropped_cols"],
                "r": result["r"],
                "mu_U": result["mu_U"],
                "mu_V": result["mu_V"],
                "mu": result["mu"],
                "mu_times_r": result["mu_times_r"],
                "tail_ratio": result["tail_ratio"],
                "explained_energy": result["explained_energy"],
                "largest_singular_value": result["largest_singular_value"],
                "smallest_used_singular_value": result[
                    "smallest_used_singular_value"
                ],
            }
        )

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    dataset_matrix_groups = build_dataset_matrix_groups(
        args.electricity_dir, args.snp500_dir, args.covid19_dir
    )

    all_results = []

    for group in dataset_matrix_groups:
        try:
            result_df = analyze_one_dataset_group(group)
            all_results.append(result_df)

            display_cols = [
                "matrix",
                "energy_threshold",
                "raw_n",
                "raw_m",
                "used_n",
                "used_m",
                "n_nonfinite",
                "dropped_rows",
                "dropped_cols",
                "r",
                "mu",
                "mu_times_r",
                "tail_ratio",
                "explained_energy",
            ]
            print(result_df[display_cols].to_string(index=False))

        except Exception as error:
            print(f"Failed to analyze {group['matrix_name']}: {error}")

    if len(all_results) == 0:
        print("\nNo matrices were analyzed.")
        return None

    final_results = pd.concat(all_results, ignore_index=True)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    final_results.to_csv(args.output_path, index=False)

    print("=" * 100)
    print("Combined results:")
    print(final_results.to_string(index=False))
    print(f"\nSaved results to: {args.output_path}")

    return final_results


if __name__ == "__main__":
    main()
