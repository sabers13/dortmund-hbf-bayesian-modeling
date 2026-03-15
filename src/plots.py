"""Plotting and result export helpers."""
import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

from .config import RANDOM_SEED


def _apply_headers(fig, title=None, subtitle=None):
    subtitle_wrapped = None
    if subtitle:
        subtitle_wrapped = textwrap.fill(subtitle, width=140)

    if title:
        fig.suptitle(title, fontsize=12, y=0.995)
    if subtitle_wrapped:
        fig.text(
            0.5,
            0.94,
            subtitle_wrapped,
            ha="center",
            va="top",
            fontsize=8.5,
            color="dimgray",
            linespacing=1.2,
        )

    if title and subtitle_wrapped:
        top_rect = 0.80
    elif title or subtitle_wrapped:
        top_rect = 0.88
    else:
        top_rect = 0.95
    return top_rect


def save_trace(idata, path, var_names=None, coords=None, title=None, subtitle=None):
    az.plot_trace(idata, var_names=var_names, coords=coords)
    fig = plt.gcf()
    top_rect = _apply_headers(fig, title=title, subtitle=subtitle)
    fig.tight_layout(rect=(0, 0, 1, top_rect))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary(idata, path, var_names=None):
    summary = az.summary(idata, var_names=var_names, round_to=2)
    summary.to_csv(path)
    return summary


def _thin_idata(idata, max_draws):
    if not hasattr(idata, "posterior"):
        return idata
    if "draw" not in idata.posterior.dims:
        return idata
    n_draws = int(idata.posterior.sizes.get("draw", 0))
    if n_draws <= max_draws:
        return idata
    rng = np.random.default_rng(RANDOM_SEED)
    draw_idx = np.sort(rng.choice(n_draws, size=max_draws, replace=False))
    try:
        return idata.isel(draw=draw_idx)
    except Exception:
        return idata


def sample_ppc(model, idata, fast_dev, var_names):
    if fast_dev:
        idata = _thin_idata(idata, max_draws=200)
    with model:
        return pm.sample_posterior_predictive(
            idata, var_names=var_names, random_seed=RANDOM_SEED
        )


def ppc_rate_plot(
    model,
    idata,
    observed,
    path,
    title,
    fast_dev,
    n_trials=None,
    subtitle=None,
):
    ppc = sample_ppc(model, idata, fast_dev, var_names=["y"])
    y_ppc = ppc.posterior_predictive["y"].values
    obs_arr = np.asarray(observed, dtype=float).reshape(-1)
    if n_trials is not None:
        n_obs = np.asarray(n_trials, dtype=float).reshape(-1)
        if obs_arr.shape[0] != n_obs.shape[0]:
            raise ValueError("Observed and n_trials must have the same length for grouped PPC.")

        total_trials = float(n_obs.sum())
        if total_trials <= 0:
            raise ValueError("Total number of trials must be positive for grouped PPC.")

        # Posterior predictive overall event rate per draw:
        # total predicted successes / total trials
        rates = (y_ppc.sum(axis=2) / total_trials).ravel()

        if np.all((obs_arr >= 0.0) & (obs_arr <= 1.0)):
            # Observed provided as per-group rates -> convert to weighted overall rate
            obs_rate = float(np.average(obs_arr, weights=n_obs))
        else:
            # Observed provided as per-group counts -> overall observed rate
            obs_rate = float(obs_arr.sum() / total_trials)
    else:
        rates = y_ppc.mean(axis=2).ravel()
        obs_rate = float(obs_arr.mean())
    plt.figure()
    plt.hist(rates, bins=30, alpha=0.8, color="cornflowerblue")
    plt.axvline(obs_rate, color="red", linewidth=2, label="Observed")
    fig = plt.gcf()
    top_rect = _apply_headers(fig, title=title, subtitle=subtitle)
    plt.xlabel("Predicted rate")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout(rect=(0, 0, 1, top_rect))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def ppc_log_delay_plot(
    model,
    idata,
    observed_log,
    path,
    title,
    fast_dev,
    max_samples=10000,
    subtitle=None,
):
    ppc = sample_ppc(model, idata, fast_dev, var_names=["log_delay"])
    pred = ppc.posterior_predictive["log_delay"].values
    pred_flat = np.exp(pred).ravel()
    if pred_flat.size > max_samples:
        idx = np.random.choice(pred_flat.size, max_samples, replace=False)
        pred_flat = pred_flat[idx]
    obs_flat = np.exp(observed_log)
    if obs_flat.size > max_samples:
        idx = np.random.choice(obs_flat.size, max_samples, replace=False)
        obs_flat = obs_flat[idx]
    plt.figure()
    plt.hist(pred_flat, bins=40, alpha=0.6, label="PPC", color="steelblue")
    plt.hist(obs_flat, bins=40, alpha=0.6, label="Observed", color="orange")
    fig = plt.gcf()
    top_rect = _apply_headers(fig, title=title, subtitle=subtitle)
    plt.xlabel("Delay minutes")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout(rect=(0, 0, 1, top_rect))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def prior_rate_plot(model, observed, path, title, fast_dev, subtitle=None):
    samples = 200 if fast_dev else 500
    with model:
        prior = pm.sample_prior_predictive(samples=samples, random_seed=RANDOM_SEED, return_inferencedata=False)
    y_prior = prior["y"]
    rates = y_prior.mean(axis=1)
    obs_rate = observed.mean()
    plt.figure()
    plt.hist(rates, bins=30, alpha=0.8, color="gray")
    plt.axvline(obs_rate, color="red", linewidth=2, label="Observed")
    fig = plt.gcf()
    top_rect = _apply_headers(fig, title=title, subtitle=subtitle)
    plt.xlabel("Prior predictive rate")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout(rect=(0, 0, 1, top_rect))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def calibration_plot(calib_df, path, title, subtitle=None):
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Ideal")

    plt.plot(
        calib_df["mean_pred"],
        calib_df["obs_rate"],
        marker="o",
        linewidth=2,
        label="Observed vs predicted",
    )

    for _, row in calib_df.iterrows():
        plt.annotate(
            str(int(row["n"])),
            (row["mean_pred"], row["obs_rate"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            color="dimgray",
        )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")

    fig = plt.gcf()
    top_rect = _apply_headers(fig, title=title, subtitle=subtitle)
    plt.legend()
    plt.tight_layout(rect=(0, 0, 1, top_rect))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def holdout_metric_barplot(df, metric, path, title, subtitle=None):
    plot_df = df.copy().sort_values(["story", metric], ascending=[True, True])
    labels = plot_df["story"] + " | " + plot_df["model"]

    plt.figure(figsize=(8, 4.5))
    plt.barh(labels, plot_df[metric])
    plt.xlabel(metric)

    fig = plt.gcf()
    top_rect = _apply_headers(fig, title=title, subtitle=subtitle)
    plt.tight_layout(rect=(0, 0, 1, top_rect))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def group_effect_forest_plot(df, group_col, path, title, subtitle=None):
    plot_df = df.sort_values("post_mean").reset_index(drop=True)
    y = np.arange(len(plot_df))

    plt.figure(figsize=(7, 4.5))
    plt.hlines(y, plot_df["hdi_low"], plot_df["hdi_high"], linewidth=2)
    plt.plot(plot_df["post_mean"], y, "o")
    plt.axvline(0, linestyle="--", color="gray", linewidth=1)

    plt.yticks(y, plot_df[group_col])
    plt.xlabel("Posterior group effect")

    fig = plt.gcf()
    top_rect = _apply_headers(fig, title=title, subtitle=subtitle)
    plt.tight_layout(rect=(0, 0, 1, top_rect))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def observed_vs_reference_probability_plot(obs_df, post_df, group_col, path, title, subtitle=None):
    """
    Compare observed raw group rate vs posterior reference-scenario probability by group.
    """
    merged = obs_df.merge(post_df, on=group_col, how="inner").sort_values("n")

    plt.figure(figsize=(7, 5))
    sizes = np.sqrt(merged["n"].values) * 12

    plt.scatter(
        merged["obs_rate"],
        merged["post_prob_mean"],
        s=sizes,
        alpha=0.8,
    )

    for _, row in merged.iterrows():
        plt.annotate(
            row[group_col],
            (row["obs_rate"], row["post_prob_mean"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("Observed raw group rate")
    plt.ylabel("Posterior reference-scenario probability")

    fig = plt.gcf()
    top_rect = _apply_headers(fig, title=title, subtitle=subtitle)
    plt.tight_layout(rect=(0, 0, 1, top_rect))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def posterior_prob_by_group_plot(df, group_col, path, title, subtitle=None):
    plot_df = df.sort_values("post_prob_mean").reset_index(drop=True)
    y = np.arange(len(plot_df))

    plt.figure(figsize=(7, 4.5))
    plt.hlines(y, plot_df["hdi_low"], plot_df["hdi_high"], linewidth=2)
    plt.plot(plot_df["post_prob_mean"], y, "o")

    plt.yticks(y, plot_df[group_col])
    plt.xlabel("Posterior predicted probability")

    fig = plt.gcf()
    top_rect = _apply_headers(fig, title=title, subtitle=subtitle)
    plt.tight_layout(rect=(0, 0, 1, top_rect))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_loo_results(loo_obj, name, out_dir):
    def _get_any(keys, default=None):
        for key in keys:
            try:
                val = loo_obj[key]
                if val is not None:
                    return val
            except Exception:
                val = getattr(loo_obj, key, None)
                if val is not None:
                    return val
        return default

    def _safe_float(x):
        if x is None:
            return np.nan
        try:
            return float(x)
        except Exception:
            return np.nan

    elpd_loo = _get_any(["elpd_loo"])
    se = _get_any(["se", "elpd_loo_se", "se_elpd_loo"])
    p_loo = _get_any(["p_loo"])
    looic = _get_any(["looic"])
    scale = _get_any(["scale"], default="")

    if looic is None and elpd_loo is not None:
        looic = -2.0 * float(elpd_loo)

    loo_row = {
        "elpd_loo": _safe_float(elpd_loo),
        "se": _safe_float(se),
        "p_loo": _safe_float(p_loo),
        "looic": _safe_float(looic),
        "scale": str(scale),
    }

    pd.DataFrame([loo_row]).to_csv(
        os.path.join(out_dir, f"loo_{name}.csv"),
        index=False,
    )

    pareto = _get_any(["pareto_k"], default=None)
    if pareto is not None:
        try:
            pareto.to_pandas().to_csv(os.path.join(out_dir, f"pareto_k_{name}.csv"))
            pareto_vals = pareto.values.ravel()
        except Exception:
            pareto_vals = np.array(pareto).ravel()
            pd.Series(pareto_vals).to_csv(os.path.join(out_dir, f"pareto_k_{name}.csv"), index=False)

        pareto_summary = pd.Series(
            {
                "n_obs": int(pareto_vals.size),
                "pct_k_gt_0.7": float((pareto_vals > 0.7).mean()),
                "pct_k_gt_1.0": float((pareto_vals > 1.0).mean()),
                "max_k": float(np.nanmax(pareto_vals)),
            }
        )
        pareto_summary.to_csv(os.path.join(out_dir, f"pareto_summary_{name}.csv"))
    else:
        pd.Series({"note": "pareto_k missing. Run az.loo(..., pointwise=True)."}).to_csv(
            os.path.join(out_dir, f"pareto_summary_{name}.csv")
        )
