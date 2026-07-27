"""Run statistical significance tests on repeated matrix-completion results.

The script reads per-run sMAPE values from log files, validates that all
expected repetitions and sampling budgets are present, and applies the
Friedman test followed by pairwise Wilcoxon signed-rank tests with Holm
correction.
"""

import argparse
import os
import re
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


LOG_FILES = {
    ("Electricity", "MLP"): "electricity_mlp_vanilla_all_models.log",
    ("Electricity", "M-estimator"): "electricity_m_estimator_all_models.log",
    ("S&P 500", "MLP"): "snp500_mlp_vanilla_all_models.log",
    ("S&P 500", "M-estimator"): "snp500_m_estimator_all_models.log",
    ("COVID-19", "MLP"): "covid19_mlp_vanilla_all_models.log",
    ("COVID-19", "M-estimator"): "covid19_m_estimator_all_models.log",
}

EXPECTED_MULTIPLIERS = [0.2, 0.5, 1.0, 2.0, 5.0]
EXPECTED_REPEATS = list(range(1, 11))
ALPHA = 0.05


START_PATTERN = re.compile(
    r"^\s*\[START\]\s+"
    r"repeat=(\d+)/10\s*\|\s*"
    r"sample_size=(\d+)\s*\|\s*"
    r"multiplier=([0-9.]+)",
    flags=re.IGNORECASE,
)

RESULT_PATTERN = re.compile(
    r"^\s*\[RESULT\]\s+"
    r"repeat=(\d+)/10\s*\|\s*"
    r"sample_size=(\d+).*?"
    r"\|\s*sMAPE=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
    flags=re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run statistical significance tests on repeated "
        "matrix-completion results."
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory containing the six stdout log files listed in "
        "LOG_FILES. These logs are captured manually by redirecting stdout "
        "when running itsop/mlp_onehot.py and itsop/m_estimator.py for each "
        "dataset (e.g. `python itsop/mlp_onehot.py > logs/"
        "electricity_mlp_vanilla_all_models.log`) -- they are not produced "
        "automatically by any other pipeline script.",
    )
    return parser.parse_args()


def parse_log_file(file_name, dataset, method):
    path = Path(file_name)

    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {file_name}")

    records = {}

    current_repeat = None
    current_sample_size = None
    current_multiplier = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            start_match = START_PATTERN.search(line)

            if start_match:
                current_repeat = int(start_match.group(1))
                current_sample_size = int(start_match.group(2))
                current_multiplier = float(start_match.group(3))
                continue

            result_match = RESULT_PATTERN.search(line)

            if result_match is None:
                continue

            result_repeat = int(result_match.group(1))
            result_sample_size = int(result_match.group(2))
            smape = float(result_match.group(3))

            if current_repeat is None or current_multiplier is None:
                raise ValueError(
                    "{} line {}: RESULT without preceding START".format(
                        file_name,
                        line_number,
                    )
                )

            if result_repeat != current_repeat:
                raise ValueError(
                    "{} line {}: repeat mismatch, START={}, RESULT={}".format(
                        file_name,
                        line_number,
                        current_repeat,
                        result_repeat,
                    )
                )

            if result_sample_size != current_sample_size:
                raise ValueError(
                    "{} line {}: sample_size mismatch, START={}, RESULT={}".format(
                        file_name,
                        line_number,
                        current_sample_size,
                        result_sample_size,
                    )
                )

            key = (current_repeat, current_multiplier)

            if key in records:
                old_value = records[key]["smape"]

                if np.isclose(old_value, smape):
                    continue

                raise ValueError(
                    "{}: two different RESULT values for repeat={} "
                    "and multiplier={}: {} and {}".format(
                        file_name,
                        current_repeat,
                        current_multiplier,
                        old_value,
                        smape,
                    )
                )

            records[key] = {
                "dataset": dataset,
                "method": method,
                "repeat": current_repeat,
                "sample_size": current_sample_size,
                "multiplier": current_multiplier,
                "smape": smape,
            }

    if not records:
        raise ValueError(
            f"No valid per-run RESULT lines were found in {file_name}"
        )

    return list(records.values())


def validate_records(records):
    lookup = {}

    for row in records:
        group_key = (row["dataset"], row["method"])
        multiplier = row["multiplier"]
        repeat = row["repeat"]

        lookup.setdefault(group_key, {})
        lookup[group_key].setdefault(multiplier, {})
        lookup[group_key][multiplier][repeat] = row["smape"]

    errors = []

    for dataset_method in LOG_FILES:
        dataset, method = dataset_method

        if dataset_method not in lookup:
            errors.append(
                "No records for {} | {}".format(dataset, method)
            )
            continue

        for multiplier in EXPECTED_MULTIPLIERS:
            matched_multiplier = None

            for observed_multiplier in lookup[dataset_method]:
                if np.isclose(observed_multiplier, multiplier):
                    matched_multiplier = observed_multiplier
                    break

            if matched_multiplier is None:
                errors.append(
                    "{} | {}: missing multiplier {}".format(
                        dataset,
                        method,
                        multiplier,
                    )
                )
                continue

            repeats = sorted(
                lookup[dataset_method][matched_multiplier].keys()
            )

            if repeats != EXPECTED_REPEATS:
                errors.append(
                    "{} | {} | multiplier={}: expected repeats {}, found {}".format(
                        dataset,
                        method,
                        multiplier,
                        EXPECTED_REPEATS,
                        repeats,
                    )
                )

    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(errors))

    return lookup


def get_values(group_data, multiplier):
    matched_multiplier = None

    for observed_multiplier in group_data:
        if np.isclose(observed_multiplier, multiplier):
            matched_multiplier = observed_multiplier
            break

    if matched_multiplier is None:
        raise ValueError(
            "Multiplier {} was not found".format(multiplier)
        )

    return np.array(
        [
            group_data[matched_multiplier][repeat]
            for repeat in EXPECTED_REPEATS
        ],
        dtype=float,
    )


def holm_correction(p_values):
    p_values = np.asarray(p_values, dtype=float)
    number_of_tests = len(p_values)

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(number_of_tests, dtype=float)

    running_max = 0.0

    for index, p_value in enumerate(sorted_p):
        multiplier = number_of_tests - index
        adjusted_value = min(1.0, multiplier * p_value)
        running_max = max(running_max, adjusted_value)
        adjusted_sorted[index] = running_max

    adjusted = np.empty(number_of_tests, dtype=float)
    adjusted[order] = adjusted_sorted

    rejected = adjusted < ALPHA

    return adjusted, rejected


def format_p_value(value):
    if value < 0.000001:
        return "{:.3e}".format(value)

    return "{:.6f}".format(value)


def print_summary(dataset, method, group_data):
    print("")
    print("=" * 100)
    print("{} | {}".format(dataset, method))
    print("=" * 100)

    print("")
    print("sMAPE summary:")
    print(
        "{:<12} {:>4} {:>12} {:>12} {:>12} {:>12}".format(
            "Budget",
            "n",
            "Mean",
            "Std",
            "Min",
            "Max",
        )
    )

    for multiplier in EXPECTED_MULTIPLIERS:
        values = get_values(group_data, multiplier)

        print(
            "{:<12} {:>4d} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f}".format(
                "{}B0".format(multiplier),
                len(values),
                np.mean(values),
                np.std(values, ddof=1),
                np.min(values),
                np.max(values),
            )
        )


def run_friedman_test(dataset, method, group_data):
    samples = [
        get_values(group_data, multiplier)
        for multiplier in EXPECTED_MULTIPLIERS
    ]

    statistic, p_value = friedmanchisquare(*samples)

    print("")
    print("Friedman omnibus test:")
    print("Statistic: {:.6f}".format(statistic))
    print("p-value:   {}".format(format_p_value(p_value)))
    print(
        "Significant at alpha=0.05: {}".format(
            "YES" if p_value < ALPHA else "NO"
        )
    )

    return p_value


def run_pairwise_tests(dataset, method, group_data):
    results = []

    for multiplier_a, multiplier_b in combinations(
        EXPECTED_MULTIPLIERS,
        2,
    ):
        values_a = get_values(group_data, multiplier_a)
        values_b = get_values(group_data, multiplier_b)

        differences = values_a - values_b

        if np.allclose(differences, 0.0):
            statistic = 0.0
            p_value = 1.0
        else:
            statistic, p_value = wilcoxon(
                values_a,
                values_b,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )

        results.append(
            {
                "multiplier_a": multiplier_a,
                "multiplier_b": multiplier_b,
                "mean_a": np.mean(values_a),
                "mean_b": np.mean(values_b),
                "mean_difference": np.mean(differences),
                "statistic": statistic,
                "p_raw": p_value,
            }
        )

    raw_p_values = [row["p_raw"] for row in results]
    adjusted_p_values, rejected = holm_correction(raw_p_values)

    for index, row in enumerate(results):
        row["p_holm"] = adjusted_p_values[index]
        row["significant"] = bool(rejected[index])

    print("")
    print("Pairwise Wilcoxon signed-rank tests:")
    print("Holm correction is applied within this dataset-method group.")
    print("")
    print(
        "{:<18} {:>11} {:>11} {:>12} {:>12} {:>12} {:>12}".format(
            "Comparison",
            "Mean A",
            "Mean B",
            "A minus B",
            "Raw p",
            "Holm p",
            "Significant",
        )
    )

    for row in results:
        comparison = "{}B0 vs {}B0".format(
            row["multiplier_a"],
            row["multiplier_b"],
        )

        print(
            "{:<18} {:>11.6f} {:>11.6f} {:>12.6f} {:>12} {:>12} {:>12}".format(
                comparison,
                row["mean_a"],
                row["mean_b"],
                row["mean_difference"],
                format_p_value(row["p_raw"]),
                format_p_value(row["p_holm"]),
                "YES" if row["significant"] else "NO",
            )
        )

    print("")
    print("Significant comparisons after Holm correction:")

    significant_rows = [
        row for row in results if row["significant"]
    ]

    if not significant_rows:
        print("None")
    else:
        for row in significant_rows:
            better_multiplier = (
                row["multiplier_a"]
                if row["mean_a"] < row["mean_b"]
                else row["multiplier_b"]
            )

            print(
                "{}B0 vs {}B0: Holm p={}, lower mean sMAPE at {}B0".format(
                    row["multiplier_a"],
                    row["multiplier_b"],
                    format_p_value(row["p_holm"]),
                    better_multiplier,
                )
            )


def main():
    args = parse_args()
    all_records = []

    print("Reading six log files...")

    for (dataset, method), file_name in LOG_FILES.items():
        log_path = os.path.join(args.logs_dir, file_name)
        print("Reading: {}".format(log_path))

        file_records = parse_log_file(
            file_name=log_path,
            dataset=dataset,
            method=method,
        )

        print(
            "Found {} unique per-run sMAPE results".format(
                len(file_records)
            )
        )

        all_records.extend(file_records)

    lookup = validate_records(all_records)

    print("")
    print("Validation passed.")
    print("Each dataset-method group contains 10 runs for each budget.")

    for dataset, method in LOG_FILES:
        group_data = lookup[(dataset, method)]

        print_summary(
            dataset=dataset,
            method=method,
            group_data=group_data,
        )

        run_friedman_test(
            dataset=dataset,
            method=method,
            group_data=group_data,
        )

        run_pairwise_tests(
            dataset=dataset,
            method=method,
            group_data=group_data,
        )

    print("")
    print("=" * 100)
    print("Analysis completed successfully.")
    print("=" * 100)


if __name__ == "__main__":
    main()
