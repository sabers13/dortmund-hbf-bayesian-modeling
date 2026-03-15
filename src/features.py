"""Feature engineering helpers."""
import numpy as np
import pandas as pd


def standardize_series(series):
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        std = 1.0
    return (series - mean) / std, float(mean), float(std)


def add_standardized_columns(df, cont_cols):
    scalers = {}
    for col in cont_cols:
        _, mean, std = standardize_series(df[col].astype(float))
        scalers[col] = {"mean": mean, "std": std}
    return apply_standardized_columns(df, scalers), scalers


def add_hour_3h_bin(df, hour_col="hour_dec"):
    df = df.copy()
    hour_int = np.floor(df[hour_col].astype(float)).astype(int).clip(0, 23)
    bin_start = (hour_int // 3) * 3
    df["hour_3h_bin"] = (
        bin_start.astype(str).str.zfill(2) + "-" +
        (bin_start + 2).astype(str).str.zfill(2)
    )
    return df


def build_grouped_binomial_data(
    df,
    outcome_col,
    group_cols,
    cont_cols=None,
):
    df = df.copy()

    agg_dict = {
        outcome_col: ["sum", "count"],
    }

    if cont_cols is not None:
        for col in cont_cols:
            agg_dict[col] = "mean"

    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(agg_dict)
        .reset_index()
    )

    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    grouped = grouped.rename(
        columns={
            f"{outcome_col}_sum": "y",
            f"{outcome_col}_count": "n",
        }
    )

    return grouped


def apply_top_k(series, categories, other_label="Other"):
    series = series.fillna("Unknown").astype(str)
    keep = [c for c in categories if c != other_label]
    return series.where(series.isin(keep), other_label)


def apply_standardized_columns(df, scalers):
    for col, stats in scalers.items():
        mean = stats["mean"]
        std = stats["std"]
        if std == 0 or np.isnan(std):
            std = 1.0
        df[f"{col}_z"] = (df[col].astype(float) - mean) / std
    return df


def map_top_k(series, k=10, other_label="Other"):
    series = series.fillna("Unknown").astype(str)
    top = series.value_counts().nlargest(k).index.tolist()
    mapped = apply_top_k(series, top, other_label=other_label)
    categories = top + ([other_label] if other_label not in top else [])
    return mapped, categories


def choose_grouping(df, min_count=50):
    candidates = ["train_cat", "train_type"]
    stats = {}
    for col in candidates:
        counts = df[col].fillna("Unknown").value_counts()
        coverage = (counts >= min_count).mean()
        stats[col] = {
            "coverage": float(coverage),
            "min_count": int(counts.min()),
            "n_groups": int(counts.shape[0]),
        }
    best = sorted(
        stats.items(),
        key=lambda x: (-x[1]["coverage"], -x[1]["min_count"], x[1]["n_groups"]),
    )[0][0]
    return best, stats


def encode_categorical(series, categories=None):
    if categories is None:
        cat = pd.Categorical(series)
    else:
        cat = pd.Categorical(series, categories=categories)
    if cat.isna().any():
        raise ValueError("Missing category levels after encoding.")
    return cat.codes, list(cat.categories)


def build_model_data(df, cont_cols, cat_cols, cat_levels=None):
    if cat_levels is None:
        cat_levels = {}
    coords = {
        "obs_id": np.arange(len(df)),
        "cont": cont_cols,
    }
    cat_idx = {}
    for col in cat_cols:
        levels = cat_levels.get(col)
        codes, levels = encode_categorical(df[col], levels)
        cat_idx[col] = codes
        coords[col] = levels
        cat_levels[col] = levels
    x_cols = []
    for col in cont_cols:
        z_col = f"{col}_z"
        if z_col in df.columns:
            x_cols.append(z_col)
        elif col in df.columns:
            x_cols.append(col)
        else:
            raise KeyError(f"Missing continuous feature '{col}' or '{z_col}'")
    X_cont = df[x_cols].to_numpy()
    return {
        "X_cont": X_cont,
        "X_cat": cat_idx,
        "cat_idx": cat_idx,
        "coords": coords,
        "cat_levels": cat_levels,
        "cont_cols": cont_cols,
    }


def prepare_event_train_test(
    train_df,
    test_df,
    cont_cols,
    topk_source_col="final_destination_station",
    topk_k=10,
):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # destination bucketing learned on train only
    _, dest_levels = map_top_k(train_df[topk_source_col], k=topk_k)
    train_df["dest_top"] = apply_top_k(train_df[topk_source_col], dest_levels)
    test_df["dest_top"] = apply_top_k(test_df[topk_source_col], dest_levels)

    # standardization learned on train only
    train_df, scalers = add_standardized_columns(train_df, cont_cols)
    test_df = apply_standardized_columns(test_df, scalers)

    train_df["weekday_cat"] = train_df["weekday"].astype(int).astype(str)
    test_df["weekday_cat"] = test_df["weekday"].astype(int).astype(str)

    return train_df, test_df, {"dest_top_levels": dest_levels, "scalers": scalers}
