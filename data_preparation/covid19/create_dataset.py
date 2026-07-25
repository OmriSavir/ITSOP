"""Create the COVID-19 dataset used in the experiments."""

import argparse
import os

import pandas as pd


INPUT_FILES = {
    "cases": "time_series_covid19_cases.csv",
    "deaths": "time_series_covid19_deaths.csv",
    "recovered": "time_series_covid19_recovered.csv",
}

START_DATE = pd.Timestamp("2020-02-20")
END_DATE = pd.Timestamp("2021-05-25")
NON_DATE_COLUMNS = ["Lat", "Long", "type", "country"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the COVID-19 dataset used in the experiments."
    )
    parser.add_argument(
        "--raw-data-dir",
        default=os.path.join("data", "covid19", "raw"),
        help="Directory containing the raw JHU CSSE COVID-19 time series CSV files.",
    )
    parser.add_argument(
        "--output-path",
        default=os.path.join("data", "covid19", "covid19_dataset.pkl"),
        help="Path to write the combined dataset pickle to.",
    )
    return parser.parse_args()


def load_series_file(series_type, file_name, raw_data_dir):
    """Load one raw file and add the series type and country identifier."""
    file_path = os.path.join(raw_data_dir, file_name)
    data = pd.read_csv(file_path)
    data["type"] = series_type

    province = data["Province/State"]
    country_region = data["Country/Region"]
    has_no_province = province.isna() | (province.astype(str).str.strip() == "")

    data["country"] = country_region.where(
        has_no_province,
        country_region.astype(str) + "_" + province.astype(str),
    )

    return data.drop(columns=["Province/State", "Country/Region"])


def get_selected_date_columns(data):
    """Return date columns within the requested period in chronological order."""
    date_columns = {}

    for column in data.columns:
        if column in NON_DATE_COLUMNS:
            continue

        parsed_date = pd.to_datetime(column, format="%m/%d/%Y", errors="coerce")
        if pd.isna(parsed_date):
            parsed_date = pd.to_datetime(column, format="%m/%d/%y", errors="coerce")

        if not pd.isna(parsed_date) and START_DATE <= parsed_date <= END_DATE:
            date_columns[column] = parsed_date

    return sorted(date_columns, key=date_columns.get)


def build_dataset(raw_data_dir):
    """Build the combined dataset and identify series containing missing values."""
    datasets = [
        load_series_file(series_type, file_name, raw_data_dir)
        for series_type, file_name in INPUT_FILES.items()
    ]
    dataset = pd.concat(datasets, ignore_index=True)

    selected_date_columns = get_selected_date_columns(dataset)
    output_columns = ["country", "type", "Lat", "Long", *selected_date_columns]
    dataset = dataset[output_columns]

    missing_mask = dataset[selected_date_columns].isna().any(axis=1)
    removed_series = dataset.loc[missing_mask, ["country", "type"]].copy()
    dataset = dataset.loc[~missing_mask].reset_index(drop=True)

    return dataset, removed_series


def main():
    args = parse_args()
    dataset, removed_series = build_dataset(args.raw_data_dir)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    dataset.to_pickle(args.output_path)

    print(f"Number of series in saved dataset: {len(dataset)}")

    if not removed_series.empty:
        print("Removed series:")
        for row in removed_series.itertuples(index=False):
            print(f"country: {row.country}, type: {row.type}")


if __name__ == "__main__":
    main()
