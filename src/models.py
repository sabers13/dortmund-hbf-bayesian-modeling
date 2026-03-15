"""Model builders and sampling helpers."""
import os

import numpy as np
import pymc as pm
import arviz as az

from .config import RANDOM_SEED, PRIORS_DEFAULT


def build_event_bernoulli_model(
    data,
    y,
    cat_cols,
    group_col="train_cat",
    cont_cols=None,
    priors=None,
):
    priors = priors or {}
    cont_cols = cont_cols or data.get("cont_cols", [])
    x_cat = data.get("X_cat", data.get("cat_idx", {}))

    coords = {"obs": np.arange(len(y))}
    for col in cat_cols:
        coords[col] = data["cat_levels"][col]
    if len(cont_cols) > 0:
        coords["cont"] = cont_cols

    with pm.Model(coords=coords) as model:
        eta = 0.0

        if len(cont_cols) > 0:
            X = pm.Data("X_cont", data["X_cont"], dims=("obs", "cont"))
            beta = pm.Normal("beta", 0, priors["beta_scale"], dims="cont")
            eta = eta + pm.math.dot(X, beta)

        intercept = pm.Normal(
            "intercept",
            priors.get("intercept_loc", 0.0),
            priors["intercept_scale"],
        )
        eta = eta + intercept

        for col in cat_cols:
            idx = pm.Data(f"{col}_idx", x_cat[col], dims="obs")

            if col == group_col:
                sigma = pm.HalfNormal(f"{col}_sigma", priors["group_sigma_scale"])
                offset = pm.Normal(f"{col}_offset", 0, 1, dims=col)
                raw_eff = pm.Deterministic(f"{col}_raw_eff", offset * sigma, dims=col)
                eff = pm.Deterministic(f"{col}_eff", raw_eff - pm.math.mean(raw_eff), dims=col)
            else:
                raw = pm.Normal(f"{col}_raw", 0, priors["cat_scale"], dims=col)
                eff = pm.Deterministic(f"{col}_eff", raw - pm.math.mean(raw), dims=col)

            eta = eta + eff[idx]

        p = pm.Deterministic("p", pm.math.sigmoid(eta), dims="obs")
        pm.Bernoulli("y", p=p, observed=y, dims="obs")

    return model


def build_grouped_binomial_model(
    data,
    y,
    n,
    cat_cols,
    group_col="train_cat",
    cont_cols=None,
    priors=None,
):
    priors = priors or {}
    cont_cols = cont_cols or data.get("cont_cols", [])
    x_cat = data.get("X_cat", data.get("cat_idx", {}))

    coords = {"obs": np.arange(len(y))}
    for col in cat_cols:
        coords[col] = data["cat_levels"][col]
    if len(cont_cols) > 0:
        coords["cont"] = cont_cols

    with pm.Model(coords=coords) as model:
        eta = 0.0

        if len(cont_cols) > 0:
            X = pm.Data("X_cont", data["X_cont"], dims=("obs", "cont"))
            beta = pm.Normal("beta", 0, priors["beta_scale"], dims="cont")
            eta = eta + pm.math.dot(X, beta)

        intercept = pm.Normal(
            "intercept",
            priors.get("intercept_loc", 0.0),
            priors["intercept_scale"],
        )
        eta = eta + intercept

        for col in cat_cols:
            idx = pm.Data(f"{col}_idx", x_cat[col], dims="obs")

            if col == group_col:
                sigma = pm.HalfNormal(f"{col}_sigma", priors["group_sigma_scale"])
                offset = pm.Normal(f"{col}_offset", 0, 1, dims=col)
                raw_eff = pm.Deterministic(f"{col}_raw_eff", offset * sigma, dims=col)
                eff = pm.Deterministic(f"{col}_eff", raw_eff - pm.math.mean(raw_eff), dims=col)
            else:
                raw = pm.Normal(f"{col}_raw", 0, priors["cat_scale"], dims=col)
                eff = pm.Deterministic(f"{col}_eff", raw - pm.math.mean(raw), dims=col)

            eta = eta + eff[idx]

        p = pm.Deterministic("p", pm.math.sigmoid(eta), dims="obs")
        pm.Binomial("y", n=n, p=p, observed=y, dims="obs")

    return model


def build_logistic_model(data, y, cat_cols, group_col=None, hierarchical=False, priors=None):
    if priors is None:
        priors = PRIORS_DEFAULT
    coords = data["coords"]
    with pm.Model(coords=coords) as model:
        X_cont = pm.Data("X_cont", data["X_cont"], dims=("obs_id", "cont"))
        beta = pm.Normal("beta", 0, priors["beta_scale"], dims="cont")
        intercept = pm.Normal(
            "intercept",
            priors.get("intercept_loc", 0.0),
            priors["intercept_scale"],
        )

        linear = intercept + pm.math.dot(X_cont, beta)

        for col in cat_cols:
            idx = pm.Data(f"{col}_idx", data["cat_idx"][col], dims="obs_id")
            if hierarchical and col == group_col:
                sigma = pm.HalfNormal(f"{col}_sigma", priors["group_sigma_scale"])
                offset = pm.Normal(f"{col}_offset", 0, 1, dims=col)
                raw_eff = pm.Deterministic(f"{col}_raw_eff", offset * sigma, dims=col)
                eff = pm.Deterministic(f"{col}_eff", raw_eff - pm.math.mean(raw_eff), dims=col)
            else:
                raw = pm.Normal(f"{col}_raw", 0, priors["cat_scale"], dims=col)
                eff = pm.Deterministic(f"{col}_eff", raw - pm.math.mean(raw), dims=col)
            linear = linear + eff[idx]

        pm.Bernoulli("y", logit_p=linear, observed=y, dims="obs_id")

    return model


def build_log_normal_model(data, y_log, cat_cols, group_col=None, hierarchical=False, priors=None):
    if priors is None:
        priors = PRIORS_DEFAULT
    coords = data["coords"]
    with pm.Model(coords=coords) as model:
        X_cont = pm.Data("X_cont", data["X_cont"], dims=("obs_id", "cont"))
        beta = pm.Normal("beta", 0, priors["beta_scale"], dims="cont")
        intercept = pm.Normal(
            "intercept",
            priors.get("intercept_loc", 0.0),
            priors["intercept_scale"],
        )
        sigma = pm.HalfNormal("sigma", priors["sigma_scale"])

        linear = intercept + pm.math.dot(X_cont, beta)

        for col in cat_cols:
            idx = pm.Data(f"{col}_idx", data["cat_idx"][col], dims="obs_id")
            if hierarchical and col == group_col:
                sigma_g = pm.HalfNormal(f"{col}_sigma", priors["group_sigma_scale"])
                offset = pm.Normal(f"{col}_offset", 0, 1, dims=col)
                raw_eff = pm.Deterministic(f"{col}_raw_eff", offset * sigma_g, dims=col)
                eff = pm.Deterministic(f"{col}_eff", raw_eff - pm.math.mean(raw_eff), dims=col)
            else:
                raw = pm.Normal(f"{col}_raw", 0, priors["cat_scale"], dims=col)
                eff = pm.Deterministic(f"{col}_eff", raw - pm.math.mean(raw), dims=col)
            linear = linear + eff[idx]

        pm.Normal("log_delay", mu=linear, sigma=sigma, observed=y_log, dims="obs_id")

    return model


def maybe_subsample(df, label, subsample_n):
    if subsample_n is None or len(df) <= subsample_n:
        print(f"{label}: {len(df)} rows")
        return df
    sub = df.sample(n=subsample_n, random_state=RANDOM_SEED).copy()
    print(f"{label}: {len(df)} -> {len(sub)} rows (FAST_DEV)")
    return sub


def sampling_config(fast_dev, cores, draws, tune, chains, target_accept):
    if draws is not None or tune is not None or chains is not None or target_accept is not None:
        return {
            "draws": draws or (500 if fast_dev else 1000),
            "tune": tune or (500 if fast_dev else 1000),
            "chains": chains or (2 if fast_dev else 4),
            "target_accept": target_accept or 0.9,
            "cores": cores,
        }
    if fast_dev:
        return {"draws": 500, "tune": 500, "chains": 2, "target_accept": 0.9, "cores": cores}
    return {"draws": 1000, "tune": 1000, "chains": 4, "target_accept": 0.9, "cores": cores}


def sample_model(model, name, run_tag, output_dir, fast_dev, cores, draws, tune, chains, target_accept):
    cache_path = os.path.join(output_dir, f"{name}_{run_tag}.nc")
    if os.path.exists(cache_path):
        print(f"Loading cached {name} from {cache_path}")
        return az.from_netcdf(cache_path)

    cfg = sampling_config(fast_dev, cores, draws, tune, chains, target_accept)
    with model:
        idata = pm.sample(
            draws=cfg["draws"],
            tune=cfg["tune"],
            chains=cfg["chains"],
            cores=cfg["cores"],
            target_accept=cfg["target_accept"],
            random_seed=RANDOM_SEED,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )

    divergences = int(idata.sample_stats["diverging"].sum())
    if divergences > 0:
        print(f"Refitting with higher target_accept due to {divergences} divergences.")
        cfg["target_accept"] = min(cfg["target_accept"] + 0.05, 0.99)
        with model:
            idata = pm.sample(
                draws=cfg["draws"],
                tune=cfg["tune"],
                chains=cfg["chains"],
                cores=cfg["cores"],
                target_accept=cfg["target_accept"],
                random_seed=RANDOM_SEED,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": True},
            )

    idata.to_netcdf(cache_path)
    return idata


def posterior_rate_from_intercept(idata):
    intercept = idata.posterior["intercept"].values.ravel()
    return 1 / (1 + np.exp(-intercept))
