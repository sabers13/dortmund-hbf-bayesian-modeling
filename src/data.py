"""Data loading and preprocessing utilities."""
import pandas as pd


from .config import REQUIRED_COLS


def load_dataset(path):
    if not path:
        raise ValueError("Dataset path is required.")
    df = pd.read_csv(path)
    return df


def validate_and_prepare(df):
    if "weekday" not in df.columns:
        if "time" not in df.columns:
            raise ValueError("Missing both 'weekday' and 'time' columns; cannot derive weekday.")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        if df["time"].isna().all():
            raise ValueError("'time' column could not be parsed to datetime for weekday derivation.")
        df["weekday"] = df["time"].dt.dayofweek

    missing = sorted(set(REQUIRED_COLS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def clean_data(df):
    clean_df = df.copy()

    num_cols = ["delay_in_min", "hour_dec", "weekday", "day_of_month", "days_until_xmas"]
    for col in num_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    clean_df["is_canceled"] = clean_df["is_canceled"].replace({True: 1, False: 0, "True": 1, "False": 0})
    clean_df["is_canceled"] = pd.to_numeric(clean_df["is_canceled"], errors="coerce").fillna(0).astype(int)

    clean_df = clean_df.dropna(subset=["hour_dec", "weekday", "day_of_month", "days_until_xmas"])
    clean_df = clean_df[(clean_df["delay_in_min"].isna()) | (clean_df["delay_in_min"] >= 0)]

    clean_df["is_delayed"] = (clean_df["delay_in_min"] > 0).astype(int)
    clean_df["positive_delay"] = clean_df["delay_in_min"].where(clean_df["delay_in_min"] > 0, pd.NA)

    for col in ["train_cat", "train_type", "final_destination_station"]:
        clean_df[col] = clean_df[col].fillna("Unknown").astype(str)

    clean_df["weekday"] = clean_df["weekday"].astype(int)

    return clean_df
