"""Held-out evaluation helpers for binary model comparison."""
import numpy as np
import pandas as pd
import arviz as az


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def split_by_day_block(df, test_mod=5, remainder=0):
    """
    Whole-day split using day_of_month to avoid leakage across events from the same day.
    Default test set: days where day_of_month % 5 == 0  -> 5,10,15,20,25,30
    """
    df = df.copy()
    test_mask = (df["day_of_month"].astype(int) % test_mod) == remainder
    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()
    return train_df, test_df


def posterior_prob_matrix(idata, data, cat_cols, cont_cols):
    """
    Compute posterior predicted probabilities for arbitrary design data.
    Returns array of shape (n_samples, n_obs).
    Works for both event-level Bernoulli and grouped Binomial models
    because both use the same linear predictor structure.
    """
    posterior = idata.posterior
    sample_dim = ("chain", "draw")

    intercept = posterior["intercept"].stack(sample=sample_dim).values  # (S,)
    eta = intercept[:, None]  # (S, N)

    if len(cont_cols) > 0:
        beta = posterior["beta"].stack(sample=sample_dim).transpose("sample", "cont").values  # (S, C)
        X = data["X_cont"]  # (N, C)
        eta = eta + beta @ X.T  # (S, N)

    cat_idx = data.get("cat_idx", data.get("X_cat", {}))
    for col in cat_cols:
        eff = posterior[f"{col}_eff"].stack(sample=sample_dim).transpose("sample", col).values  # (S, K)
        idx = cat_idx[col]  # (N,)
        eta = eta + eff[:, idx]

    return sigmoid(eta)


def posterior_mean_prob(idata, data, cat_cols, cont_cols):
    return posterior_prob_matrix(idata, data, cat_cols, cont_cols).mean(axis=0)


def log_loss_binary(y_true, p, eps=1e-9):
    p = np.clip(np.asarray(p), eps, 1 - eps)
    y_true = np.asarray(y_true).astype(int)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier_score_binary(y_true, p):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p)
    return float(np.mean((p - y_true) ** 2))


def calibration_table(y_true, p, n_bins=10):
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(p)})
    q = min(n_bins, int(df["p"].nunique()))
    if q <= 1:
        return pd.DataFrame(
            [
                {
                    "bin": "all",
                    "n": int(df.shape[0]),
                    "mean_pred": float(df["p"].mean()),
                    "obs_rate": float(df["y"].mean()),
                }
            ]
        )

    df["bin"] = pd.qcut(df["p"], q=q, duplicates="drop")
    out = (
        df.groupby("bin", observed=False)
        .agg(
            n=("y", "size"),
            mean_pred=("p", "mean"),
            obs_rate=("y", "mean"),
        )
        .reset_index()
    )
    return out


def grouped_probs_to_events(test_events, grouped_prob_df, group_cols, prob_col="p_group"):
    return test_events.merge(grouped_prob_df[group_cols + [prob_col]], on=group_cols, how="left")


def save_holdout_scores(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


def hdi_bounds(samples, hdi_prob=0.95):
    arr = az.hdi(samples, hdi_prob=hdi_prob)
    return float(arr[0]), float(arr[1])


def posterior_group_effect_summary(idata, var_name, group_dim, hdi_prob=0.95):
    """
    Summarize a group-effect vector such as train_cat_eff.
    Returns one row per level with posterior mean, sd, and HDI.
    """
    da = idata.posterior[var_name]
    levels = da[group_dim].values
    stacked = da.stack(sample=("chain", "draw")).transpose(group_dim, "sample").values

    rows = []
    for level, vals in zip(levels, stacked):
        lo, hi = hdi_bounds(vals, hdi_prob=hdi_prob)
        rows.append(
            {
                group_dim: str(level),
                "post_mean": float(np.mean(vals)),
                "post_sd": float(np.std(vals, ddof=1)),
                "hdi_low": lo,
                "hdi_high": hi,
            }
        )
    return pd.DataFrame(rows)


def summarize_group_sigma(idata, sigma_name, hdi_prob=0.95):
    vals = idata.posterior[sigma_name].stack(sample=("chain", "draw")).values
    lo, hi = hdi_bounds(vals, hdi_prob=hdi_prob)
    return pd.DataFrame(
        [
            {
                "parameter": sigma_name,
                "post_mean": float(np.mean(vals)),
                "post_sd": float(np.std(vals, ddof=1)),
                "hdi_low": lo,
                "hdi_high": hi,
            }
        ]
    )


def observed_group_rate(df, group_col, outcome_col):
    out = (
        df.groupby(group_col, dropna=False)[outcome_col]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "obs_rate", "sum": "y", "count": "n"})
    )
    out[group_col] = out[group_col].astype(str)
    return out


def posterior_category_probability(
    idata,
    data_builder,
    base_df,
    category_col,
    category_levels,
    cat_cols,
    cont_cols,
    fixed_overrides=None,
    hdi_prob=0.95,
):
    """
    Build a small synthetic design table where only `category_col` changes level-by-level,
    then compute posterior mean probability and HDI for each level.
    `base_df` should contain one-row template with all required columns already prepared.
    """
    fixed_overrides = fixed_overrides or {}
    rows = []

    for level in category_levels:
        row = base_df.iloc[[0]].copy()
        row[category_col] = level
        for k, v in fixed_overrides.items():
            row[k] = v
        rows.append(row)

    pred_df = pd.concat(rows, ignore_index=True)
    data_pred = data_builder(pred_df)

    p_draws = posterior_prob_matrix(idata, data_pred, cat_cols=cat_cols, cont_cols=cont_cols)

    out_rows = []
    for i, level in enumerate(category_levels):
        vals = p_draws[:, i]
        lo, hi = hdi_bounds(vals, hdi_prob=hdi_prob)
        out_rows.append(
            {
                category_col: str(level),
                "post_prob_mean": float(np.mean(vals)),
                "hdi_low": lo,
                "hdi_high": hi,
            }
        )
    return pd.DataFrame(out_rows)
