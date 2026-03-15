"""Configuration and defaults."""
import os

RANDOM_SEED = 1234
CONT_COLS = ["hour_dec", "days_until_xmas"]
MODEL_VERSION = "phase2_v1"
PHASE2_PRIMARY_COMPARISON = "event_bernoulli_vs_grouped_binomial"

PHASE2_MODEL_PLAN = {
    "story1": ["1E_bern", "1G_binom"],
    "story2_delay": ["2E_delay", "2G_delay"],
    "story2_magnitude": ["2M_log"],
    "story3_delay": ["3E_delay", "3G_delay"],
    "story3_magnitude": ["3M_log"],
}

PHASE2_GROUPING_SPEC = {
    "story1": ["weekday", "hour_3h_bin", "train_cat", "event_type"],
    "story2_delay": ["weekday", "hour_3h_bin", "train_cat"],
    "story3_delay": ["weekday", "hour_3h_bin", "train_cat"],
}

PHASE2_EXCHANGEABLE_GROUP = "train_cat"

PHASE2_EXCHANGEABILITY_NOTE = (
    "Levels of train_cat are treated as exchangeable a priori: "
    "their group effects are modeled as draws from a shared Normal distribution "
    "with a common group-level scale."
)

PHASE2_PRIORS_BERNOULLI = {
    "story1": {
        "beta_scale": 0.5,
        "cat_scale": 0.5,
        "intercept_loc": -2.2,
        "intercept_scale": 0.7,
        "group_sigma_scale": 0.35,
    },
    "story2_delay": {
        "beta_scale": 0.5,
        "cat_scale": 0.5,
        "intercept_loc": 1.4,
        "intercept_scale": 0.7,
        "group_sigma_scale": 0.35,
    },
    "story3_delay": {
        "beta_scale": 0.5,
        "cat_scale": 0.5,
        "intercept_loc": 0.85,
        "intercept_scale": 0.7,
        "group_sigma_scale": 0.35,
    },
}

PHASE2_PRIORS_BINOMIAL = {
    "story1": {
        "beta_scale": 0.5,
        "cat_scale": 0.5,
        "intercept_loc": -2.2,
        "intercept_scale": 0.7,
        "group_sigma_scale": 0.35,
    },
    "story2_delay": {
        "beta_scale": 0.5,
        "cat_scale": 0.5,
        "intercept_loc": 1.4,
        "intercept_scale": 0.7,
        "group_sigma_scale": 0.35,
    },
    "story3_delay": {
        "beta_scale": 0.5,
        "cat_scale": 0.5,
        "intercept_loc": 0.85,
        "intercept_scale": 0.7,
        "group_sigma_scale": 0.35,
    },
}

PHASE2_PRIORS_LOG_MAGNITUDE = {
    "beta_scale": 0.5,
    "cat_scale": 0.5,
    "intercept_loc": 1.5,
    "intercept_scale": 0.8,
    "group_sigma_scale": 0.35,
    "sigma_scale": 0.7,
}

PRIORS_DEFAULT = {
    "beta_scale": 1.0,
    "cat_scale": 1.0,
    "intercept_scale": 1.5,
    "group_sigma_scale": 0.5,
    "sigma_scale": 1.0,
}

REQUIRED_COLS = [
    "delay_in_min",
    "is_canceled",
    "train_cat",
    "train_type",
    "final_destination_station",
    "hour_dec",
    "weekday",
    "day_of_month",
    "days_until_xmas",
    "arrival_planned_time",
    "arrival_change_time",
    "departure_planned_time",
    "departure_change_time",
]


def default_output_dirs(base_dir):
    return {
        "eda": os.path.join(base_dir, "eda"),
        "models": os.path.join(base_dir, "models"),
        "ppc": os.path.join(base_dir, "ppc"),
        "loo": os.path.join(base_dir, "loo"),
        "tables": os.path.join(base_dir, "tables"),
    }
