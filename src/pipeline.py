"""End-to-end pipeline for the Applied Bayesian project."""
import json
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

from .config import (
    RANDOM_SEED,
    CONT_COLS,
    MODEL_VERSION,
    PHASE2_PRIORS_BERNOULLI,
    PHASE2_PRIORS_BINOMIAL,
    PHASE2_PRIORS_LOG_MAGNITUDE,
    default_output_dirs,
    PHASE2_PRIMARY_COMPARISON,
    PHASE2_MODEL_PLAN,
    PHASE2_EXCHANGEABLE_GROUP,
    PHASE2_EXCHANGEABILITY_NOTE,
)
from .data import load_dataset, validate_and_prepare, clean_data
from .features import (
    add_standardized_columns,
    apply_standardized_columns,
    add_hour_3h_bin,
    apply_top_k,
    build_grouped_binomial_data,
    map_top_k,
    build_model_data,
    prepare_event_train_test,
)
from .evaluation import (
    split_by_day_block,
    posterior_mean_prob,
    log_loss_binary,
    brier_score_binary,
    calibration_table,
    grouped_probs_to_events,
    save_holdout_scores,
    posterior_group_effect_summary,
    summarize_group_sigma,
    observed_group_rate,
    posterior_category_probability,
)
from .models import (
    sample_model,
    build_event_bernoulli_model,
    build_grouped_binomial_model,
    build_log_normal_model,
)
from .plots import (
    save_trace,
    save_summary,
    ppc_rate_plot,
    ppc_log_delay_plot,
    prior_rate_plot,
    save_loo_results,
    calibration_plot,
    holdout_metric_barplot,
    group_effect_forest_plot,
    observed_vs_reference_probability_plot,
    posterior_prob_by_group_plot,
)


def run_pipeline(
    data_path,
    outputs_base,
    fast_dev=False,
    subsample_n=None,
    cores=None,
    draws=None,
    tune=None,
    chains=None,
    target_accept=None,
    run_tag=None,
):
    warnings.filterwarnings("ignore", category=FutureWarning)
    np.random.seed(RANDOM_SEED)

    mode_tag = "fast" if fast_dev else "full"
    draws_tag = draws if draws is not None else (500 if fast_dev else 1000)
    tune_tag = tune if tune is not None else (500 if fast_dev else 1000)
    chains_tag = chains if chains is not None else (2 if fast_dev else 4)
    accept_tag = target_accept if target_accept is not None else 0.9
    user_tag = run_tag if run_tag is not None else "untagged"
    cache_tag = (
        f"{MODEL_VERSION}__{user_tag}__{mode_tag}"
        f"__d{draws_tag}_t{tune_tag}_c{chains_tag}_a{accept_tag}"
    )

    phase2_plan = {
        "primary_comparison": PHASE2_PRIMARY_COMPARISON,
        "story1": PHASE2_MODEL_PLAN["story1"],
        "story2_delay": PHASE2_MODEL_PLAN["story2_delay"],
        "story2_magnitude": PHASE2_MODEL_PLAN["story2_magnitude"],
        "story3_delay": PHASE2_MODEL_PLAN["story3_delay"],
        "story3_magnitude": PHASE2_MODEL_PLAN["story3_magnitude"],
    }
    print("Phase 2 plan:", phase2_plan)

    cpu_count = os.cpu_count() or 2
    if cores is None:
        cores = 2 if fast_dev else min(4, cpu_count)

    output_dirs = default_output_dirs(outputs_base)
    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)
    report_dir = os.path.join(outputs_base, "report_bundle")
    os.makedirs(report_dir, exist_ok=True)

    with open(os.path.join(output_dirs["tables"], "exchangeability_note.txt"), "w") as f:
        f.write(PHASE2_EXCHANGEABILITY_NOTE + "\n")

    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    })

    df = load_dataset(data_path)
    print("Raw shape:", df.shape)

    df = validate_and_prepare(df)
    clean_df = clean_data(df)
    print("Clean shape:", clean_df.shape)

    def maybe_subsample(df, label, subsample_n):
        if subsample_n is None or len(df) <= subsample_n:
            print(f"{label}: {len(df)} rows")
            return df
        sub = df.sample(n=subsample_n, random_state=RANDOM_SEED).copy()
        print(f"{label}: {len(df)} -> {len(sub)} rows (FAST_DEV)")
        return sub

    def plot_tag_line(
        model_name,
        model_kind,
        story,
        n_rows,
        run_tag,
        is_fast_dev,
        group_factor="train_cat",
        grouped_rows=None,
    ):
        parts = [
            f"Story={story}",
            f"Model={model_name}",
            f"Type={model_kind}",
            f"Run={run_tag if run_tag is not None else 'untagged'}",
            f"Mode={'FAST' if is_fast_dev else 'FULL'}",
            f"n={n_rows}",
            f"Group={group_factor}",
        ]
        if grouped_rows is not None:
            parts.append(f"grouped_n={grouped_rows}")
        return " | ".join(parts)

    def report_tag_line(model_name, story, run_tag, is_fast_dev):
        return (
            f"Story={story} | Model={model_name} | "
            f"Run={run_tag if run_tag is not None else 'untagged'} | "
            f"Mode={'FAST' if is_fast_dev else 'FULL'} | "
            f"Exchangeable group=train_cat"
        )

    def pick_level(levels, preferred):
        for level in levels:
            if str(level) == preferred:
                return level
        return levels[0]

    def save_output_index(rows, path):
        pd.DataFrame(rows).to_csv(path, index=False)

    def diag_tag_line(idata):
        try:
            summ = az.summary(idata, round_to=3)
            max_rhat = float(summ["r_hat"].max()) if "r_hat" in summ.columns else float("nan")
            min_ess = float(summ["ess_bulk"].min()) if "ess_bulk" in summ.columns else float("nan")
        except Exception:
            max_rhat = float("nan")
            min_ess = float("nan")
        try:
            divergences = int(idata.sample_stats["diverging"].sum())
        except Exception:
            divergences = -1
        if divergences >= 0:
            return f"divergences={divergences} | max_rhat={max_rhat:.3f} | min_ess_bulk={min_ess:.1f}"
        return f"max_rhat={max_rhat:.3f} | min_ess_bulk={min_ess:.1f}"

    def score_binary_models_on_holdout(
        story_name,
        test_events,
        y_true,
        bern_probs,
        binom_probs,
        bern_model_name,
        binom_model_name,
        output_dirs,
        run_tag,
        fast_dev,
    ):
        _ = test_events
        rows = [
            {
                "story": story_name,
                "model": bern_model_name,
                "n_test_events": len(y_true),
                "log_loss": log_loss_binary(y_true, bern_probs),
                "brier": brier_score_binary(y_true, bern_probs),
            },
            {
                "story": story_name,
                "model": binom_model_name,
                "n_test_events": len(y_true),
                "log_loss": log_loss_binary(y_true, binom_probs),
                "brier": brier_score_binary(y_true, binom_probs),
            },
        ]

        save_holdout_scores(
            rows,
            os.path.join(output_dirs["tables"], f"{story_name}_holdout_scores.csv"),
        )

        calib_bern = calibration_table(y_true, bern_probs)
        calib_binom = calibration_table(y_true, binom_probs)

        calib_bern.to_csv(
            os.path.join(output_dirs["tables"], f"{story_name}_calibration_{bern_model_name}.csv"),
            index=False,
        )
        calib_binom.to_csv(
            os.path.join(output_dirs["tables"], f"{story_name}_calibration_{binom_model_name}.csv"),
            index=False,
        )

        subtitle = (
            f"Run={run_tag if run_tag is not None else 'untagged'} | "
            f"Mode={'FAST' if fast_dev else 'FULL'} | "
            f"n_test={len(y_true)}"
        )

        calibration_plot(
            calib_bern,
            os.path.join(output_dirs["ppc"], f"{story_name}_calibration_{bern_model_name}.png"),
            title=f"Calibration - {story_name} - {bern_model_name}",
            subtitle=subtitle,
        )
        calibration_plot(
            calib_binom,
            os.path.join(output_dirs["ppc"], f"{story_name}_calibration_{binom_model_name}.png"),
            title=f"Calibration - {story_name} - {binom_model_name}",
            subtitle=subtitle,
        )

    def align_to_training_levels(df, cat_levels, cat_cols):
        df = df.copy()
        for col in cat_cols:
            levels = [str(x) for x in cat_levels[col]]
            known = set(levels)
            fallback = "Other" if "Other" in known else ("Unknown" if "Unknown" in known else levels[0])
            values = df[col].fillna("Unknown").astype(str)
            df[col] = values.where(values.isin(known), fallback)
        return df

    # EDA outputs
    train_cat_counts = (
        clean_df["train_cat"]
        .fillna("Unknown")
        .value_counts(dropna=False)
        .rename_axis("train_cat")
        .reset_index(name="n_events")
    )
    train_cat_counts.to_csv(
        os.path.join(output_dirs["tables"], "train_cat_counts.csv"),
        index=False,
    )

    eda_summary = clean_df[["delay_in_min", "is_canceled", "is_delayed"]].describe()
    eda_summary.to_csv(os.path.join(output_dirs["tables"], "eda_summary.csv"))

    cancel_by_weekday = clean_df.groupby("weekday")["is_canceled"].mean().sort_index()
    plt.figure()
    plt.bar(cancel_by_weekday.index.astype(str), cancel_by_weekday.values)
    plt.title("Cancellation rate by weekday")
    plt.xlabel("Weekday")
    plt.ylabel("Cancellation rate")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs["eda"], "cancel_rate_by_weekday.png"), dpi=150)
    plt.close()

    pos_delays = clean_df.loc[clean_df["delay_in_min"] > 0, "delay_in_min"]
    plt.figure()
    plt.hist(np.log1p(pos_delays), bins=40, color="steelblue", alpha=0.8)
    plt.title("Positive delays (log1p minutes)")
    plt.xlabel("log1p(delay_in_min)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs["eda"], "delay_distribution_log1p.png"), dpi=150)
    plt.close()

    dest_counts = clean_df["final_destination_station"].value_counts().head(10)
    plt.figure()
    plt.barh(dest_counts.index[::-1], dest_counts.values[::-1], color="slategray")
    plt.title("Top 10 destination stations")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs["eda"], "top_destinations.png"), dpi=150)
    plt.close()

    cancel_by_cat = clean_df.groupby("train_cat")["is_canceled"].mean().sort_values(ascending=False)
    plt.figure()
    plt.barh(cancel_by_cat.index[:12][::-1], cancel_by_cat.values[:12][::-1], color="teal")
    plt.title("Cancellation rate by train_cat (top 12)")
    plt.xlabel("Cancellation rate")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dirs["eda"], "cancel_rate_by_train_cat.png"), dpi=150)
    plt.close()

    print("EDA outputs saved to", output_dirs["eda"])

    if subsample_n is None and fast_dev:
        subsample_n = 6000

    # Story 1: Cancellations
    story1_full = clean_df.copy()
    _, dest_top_levels_1 = map_top_k(story1_full["final_destination_station"], k=10)
    story1_full["dest_top"] = apply_top_k(
        story1_full["final_destination_station"],
        dest_top_levels_1,
    )
    story1_full, _ = add_standardized_columns(story1_full, CONT_COLS)
    story1_full["weekday_cat"] = story1_full["weekday"].astype(int).astype(str)

    group_col_1 = PHASE2_EXCHANGEABLE_GROUP
    group_stats_1 = {
        "chosen_group": group_col_1,
        "n_groups": int(story1_full[group_col_1].nunique(dropna=True)),
        "coverage": float(story1_full[group_col_1].notna().mean()),
    }
    print("Story 1 exchangeable grouping:", group_col_1, group_stats_1)
    story1_full = add_hour_3h_bin(story1_full)
    if "event_type" not in story1_full.columns:
        story1_full["event_type"] = "Unknown"
    story1_group_cols = ["weekday_cat", "hour_3h_bin", "train_cat", "event_type"]
    story1_grouped = build_grouped_binomial_data(
        story1_full,
        outcome_col="is_canceled",
        group_cols=story1_group_cols,
        cont_cols=["days_until_xmas_z"],
    )
    print("Story 1 grouped rows:", len(story1_grouped))
    story1_event = maybe_subsample(story1_full, "Story 1", subsample_n if fast_dev else None)

    data_1E = build_model_data(
        story1_event,
        cont_cols=CONT_COLS,
        cat_cols=["weekday_cat", "dest_top", "train_cat"],
    )
    y_1E = story1_event["is_canceled"].astype(int).values

    model_1E = build_event_bernoulli_model(
        data_1E,
        y_1E,
        cat_cols=["weekday_cat", "dest_top", "train_cat"],
        group_col=PHASE2_EXCHANGEABLE_GROUP,
        cont_cols=CONT_COLS,
        priors=PHASE2_PRIORS_BERNOULLI["story1"],
    )

    data_1G = build_model_data(
        story1_grouped,
        cont_cols=["days_until_xmas_z_mean"],
        cat_cols=["weekday_cat", "hour_3h_bin", "train_cat", "event_type"],
    )
    y_1G = story1_grouped["y"].astype(int).values
    n_1G = story1_grouped["n"].astype(int).values

    model_1G = build_grouped_binomial_model(
        data_1G,
        y_1G,
        n_1G,
        cat_cols=["weekday_cat", "hour_3h_bin", "train_cat", "event_type"],
        group_col=PHASE2_EXCHANGEABLE_GROUP,
        cont_cols=["days_until_xmas_z_mean"],
        priors=PHASE2_PRIORS_BINOMIAL["story1"],
    )

    # Story 2: Arrival delays
    story2_full = clean_df[(clean_df["is_canceled"] == 0) & (clean_df["arrival_planned_time"].notna())].copy()
    _, dest_top_levels_2 = map_top_k(story2_full["final_destination_station"], k=10)
    story2_full["dest_top"] = apply_top_k(
        story2_full["final_destination_station"],
        dest_top_levels_2,
    )
    story2_full, _ = add_standardized_columns(story2_full, CONT_COLS)
    story2_full["weekday_cat"] = story2_full["weekday"].astype(int).astype(str)

    group_col_2 = PHASE2_EXCHANGEABLE_GROUP
    group_stats_2 = {
        "chosen_group": group_col_2,
        "n_groups": int(story2_full[group_col_2].nunique(dropna=True)),
        "coverage": float(story2_full[group_col_2].notna().mean()),
    }
    print("Story 2 exchangeable grouping:", group_col_2, group_stats_2)
    story2_full = add_hour_3h_bin(story2_full)
    story2_group_cols = ["weekday_cat", "hour_3h_bin", "train_cat"]
    story2_grouped = build_grouped_binomial_data(
        story2_full,
        outcome_col="is_delayed",
        group_cols=story2_group_cols,
        cont_cols=["days_until_xmas_z"],
    )
    print("Story 2 grouped rows:", len(story2_grouped))
    story2_event = maybe_subsample(story2_full, "Story 2 (arrival)", subsample_n if fast_dev else None)

    data_2E = build_model_data(
        story2_event,
        cont_cols=CONT_COLS,
        cat_cols=["weekday_cat", "dest_top", "train_cat"],
    )
    y_2E = story2_event["is_delayed"].astype(int).values

    model_2E = build_event_bernoulli_model(
        data_2E,
        y_2E,
        cat_cols=["weekday_cat", "dest_top", "train_cat"],
        group_col=PHASE2_EXCHANGEABLE_GROUP,
        cont_cols=CONT_COLS,
        priors=PHASE2_PRIORS_BERNOULLI["story2_delay"],
    )

    data_2G = build_model_data(
        story2_grouped,
        cont_cols=["days_until_xmas_z_mean"],
        cat_cols=["weekday_cat", "hour_3h_bin", "train_cat"],
    )
    y_2G = story2_grouped["y"].astype(int).values
    n_2G = story2_grouped["n"].astype(int).values

    model_2G = build_grouped_binomial_model(
        data_2G,
        y_2G,
        n_2G,
        cat_cols=["weekday_cat", "hour_3h_bin", "train_cat"],
        group_col=PHASE2_EXCHANGEABLE_GROUP,
        cont_cols=["days_until_xmas_z_mean"],
        priors=PHASE2_PRIORS_BINOMIAL["story2_delay"],
    )

    story2_pos = story2_event[story2_event["delay_in_min"] > 0].copy()
    story2_pos = maybe_subsample(story2_pos, "Story 2 (arrival positive delays)", subsample_n if fast_dev else None)
    data_2M = build_model_data(
        story2_pos,
        cont_cols=CONT_COLS,
        cat_cols=["weekday_cat", "dest_top", "train_cat"],
    )
    y_2M = np.log(story2_pos["delay_in_min"].values)

    model_2M = build_log_normal_model(
        data_2M,
        y_2M,
        cat_cols=["weekday_cat", "dest_top", "train_cat"],
        group_col=PHASE2_EXCHANGEABLE_GROUP,
        hierarchical=True,
        priors=PHASE2_PRIORS_LOG_MAGNITUDE,
    )

    # Story 3: Departure delays
    story3_full = clean_df[(clean_df["is_canceled"] == 0) & (clean_df["departure_planned_time"].notna())].copy()
    _, dest_top_levels_3 = map_top_k(story3_full["final_destination_station"], k=10)
    story3_full["dest_top"] = apply_top_k(
        story3_full["final_destination_station"],
        dest_top_levels_3,
    )
    story3_full, _ = add_standardized_columns(story3_full, CONT_COLS)
    story3_full["weekday_cat"] = story3_full["weekday"].astype(int).astype(str)

    group_col_3 = PHASE2_EXCHANGEABLE_GROUP
    group_stats_3 = {
        "chosen_group": group_col_3,
        "n_groups": int(story3_full[group_col_3].nunique(dropna=True)),
        "coverage": float(story3_full[group_col_3].notna().mean()),
    }
    print("Story 3 exchangeable grouping:", group_col_3, group_stats_3)
    story3_full = add_hour_3h_bin(story3_full)
    story3_group_cols = ["weekday_cat", "hour_3h_bin", "train_cat"]
    story3_grouped = build_grouped_binomial_data(
        story3_full,
        outcome_col="is_delayed",
        group_cols=story3_group_cols,
        cont_cols=["days_until_xmas_z"],
    )
    print("Story 3 grouped rows:", len(story3_grouped))
    story1_grouped.to_csv(
        os.path.join(output_dirs["tables"], "story1_grouped_binomial_data.csv"),
        index=False,
    )
    story2_grouped.to_csv(
        os.path.join(output_dirs["tables"], "story2_grouped_binomial_data.csv"),
        index=False,
    )
    story3_grouped.to_csv(
        os.path.join(output_dirs["tables"], "story3_grouped_binomial_data.csv"),
        index=False,
    )
    pd.DataFrame(
        [
            {"story": "story1", "n_event_rows": len(story1_full), "n_grouped_rows": len(story1_grouped)},
            {"story": "story2", "n_event_rows": len(story2_full), "n_grouped_rows": len(story2_grouped)},
            {"story": "story3", "n_event_rows": len(story3_full), "n_grouped_rows": len(story3_grouped)},
        ]
    ).to_csv(
        os.path.join(output_dirs["tables"], "grouped_row_counts.csv"),
        index=False,
    )
    story3_event = maybe_subsample(story3_full, "Story 3 (departure)", subsample_n if fast_dev else None)

    data_3E = build_model_data(
        story3_event,
        cont_cols=CONT_COLS,
        cat_cols=["weekday_cat", "dest_top", "train_cat"],
    )
    y_3E = story3_event["is_delayed"].astype(int).values

    model_3E = build_event_bernoulli_model(
        data_3E,
        y_3E,
        cat_cols=["weekday_cat", "dest_top", "train_cat"],
        group_col=PHASE2_EXCHANGEABLE_GROUP,
        cont_cols=CONT_COLS,
        priors=PHASE2_PRIORS_BERNOULLI["story3_delay"],
    )

    data_3G = build_model_data(
        story3_grouped,
        cont_cols=["days_until_xmas_z_mean"],
        cat_cols=["weekday_cat", "hour_3h_bin", "train_cat"],
    )
    y_3G = story3_grouped["y"].astype(int).values
    n_3G = story3_grouped["n"].astype(int).values

    model_3G = build_grouped_binomial_model(
        data_3G,
        y_3G,
        n_3G,
        cat_cols=["weekday_cat", "hour_3h_bin", "train_cat"],
        group_col=PHASE2_EXCHANGEABLE_GROUP,
        cont_cols=["days_until_xmas_z_mean"],
        priors=PHASE2_PRIORS_BINOMIAL["story3_delay"],
    )

    story3_pos = story3_event[story3_event["delay_in_min"] > 0].copy()
    story3_pos = maybe_subsample(story3_pos, "Story 3 (departure positive delays)", subsample_n if fast_dev else None)
    data_3M = build_model_data(
        story3_pos,
        cont_cols=CONT_COLS,
        cat_cols=["weekday_cat", "dest_top", "train_cat"],
    )
    y_3M = np.log(story3_pos["delay_in_min"].values)

    model_3M = build_log_normal_model(
        data_3M,
        y_3M,
        cat_cols=["weekday_cat", "dest_top", "train_cat"],
        group_col=PHASE2_EXCHANGEABLE_GROUP,
        hierarchical=True,
        priors=PHASE2_PRIORS_LOG_MAGNITUDE,
    )

    idata_1E = sample_model(model_1E, "1E_bern", cache_tag, output_dirs["models"], fast_dev, cores, draws, tune, chains, target_accept)
    idata_1G = sample_model(model_1G, "1G_binom", cache_tag, output_dirs["models"], fast_dev, cores, draws, tune, chains, target_accept)
    idata_2E = sample_model(model_2E, "2E_delay", cache_tag, output_dirs["models"], fast_dev, cores, draws, tune, chains, target_accept)
    idata_2G = sample_model(model_2G, "2G_delay", cache_tag, output_dirs["models"], fast_dev, cores, draws, tune, chains, target_accept)
    idata_2M = sample_model(model_2M, "2M_log", cache_tag, output_dirs["models"], fast_dev, cores, draws, tune, chains, target_accept)
    idata_3E = sample_model(model_3E, "3E_delay", cache_tag, output_dirs["models"], fast_dev, cores, draws, tune, chains, target_accept)
    idata_3G = sample_model(model_3G, "3G_delay", cache_tag, output_dirs["models"], fast_dev, cores, draws, tune, chains, target_accept)
    idata_3M = sample_model(model_3M, "3M_log", cache_tag, output_dirs["models"], fast_dev, cores, draws, tune, chains, target_accept)

    # Held-out comparison for binary stories:
    # Bernoulli vs Binomial scored on the same event-level test set.
    def evaluate_binary_holdout(
        story_tag,
        story_label,
        story_df,
        outcome_col,
        event_model_name,
        grouped_model_name,
        event_cat_cols,
        grouped_cat_cols,
        grouped_group_cols,
        event_priors,
        grouped_priors,
    ):
        train_raw, test_raw = split_by_day_block(story_df, test_mod=5, remainder=0)
        print(f"{story_label} holdout split:", len(train_raw), len(test_raw))

        if len(train_raw) == 0 or len(test_raw) == 0:
            raise ValueError(f"{story_label}: empty train/test split.")

        if "event_type" in grouped_group_cols:
            if "event_type" not in train_raw.columns:
                train_raw["event_type"] = "Unknown"
            if "event_type" not in test_raw.columns:
                test_raw["event_type"] = "Unknown"

        train_ev, test_ev, _ = prepare_event_train_test(
            train_raw,
            test_raw,
            cont_cols=CONT_COLS,
            topk_source_col="final_destination_station",
            topk_k=10,
        )

        data_event_train = build_model_data(
            train_ev,
            cont_cols=CONT_COLS,
            cat_cols=event_cat_cols,
        )
        test_ev = align_to_training_levels(test_ev, data_event_train["cat_levels"], event_cat_cols)
        data_event_test = build_model_data(
            test_ev,
            cont_cols=CONT_COLS,
            cat_cols=event_cat_cols,
            cat_levels=data_event_train["cat_levels"],
        )
        y_event_train = train_ev[outcome_col].astype(int).values
        y_event_test = test_ev[outcome_col].astype(int).values

        holdout_event_model = build_event_bernoulli_model(
            data_event_train,
            y_event_train,
            cat_cols=event_cat_cols,
            group_col=PHASE2_EXCHANGEABLE_GROUP,
            cont_cols=CONT_COLS,
            priors=event_priors,
        )
        holdout_event_idata = sample_model(
            holdout_event_model,
            f"{event_model_name}_holdout",
            cache_tag,
            output_dirs["models"],
            fast_dev,
            cores,
            draws,
            tune,
            chains,
            target_accept,
        )
        p_event_test = posterior_mean_prob(
            holdout_event_idata,
            data_event_test,
            cat_cols=event_cat_cols,
            cont_cols=CONT_COLS,
        )

        train_ev = add_hour_3h_bin(train_ev)
        train_grp = build_grouped_binomial_data(
            train_ev,
            outcome_col=outcome_col,
            group_cols=grouped_group_cols,
            cont_cols=["days_until_xmas_z"],
        )

        grouped_cont_cols = ["days_until_xmas_z_mean"]
        data_group_train = build_model_data(
            train_grp,
            cont_cols=grouped_cont_cols,
            cat_cols=grouped_cat_cols,
        )

        test_ev = add_hour_3h_bin(test_ev)
        test_ev = align_to_training_levels(test_ev, data_group_train["cat_levels"], grouped_cat_cols)
        test_grp = build_grouped_binomial_data(
            test_ev,
            outcome_col=outcome_col,
            group_cols=grouped_group_cols,
            cont_cols=["days_until_xmas_z"],
        )
        data_group_test = build_model_data(
            test_grp,
            cont_cols=grouped_cont_cols,
            cat_cols=grouped_cat_cols,
            cat_levels=data_group_train["cat_levels"],
        )
        y_group_train = train_grp["y"].astype(int).values
        n_group_train = train_grp["n"].astype(int).values

        holdout_group_model = build_grouped_binomial_model(
            data_group_train,
            y_group_train,
            n_group_train,
            cat_cols=grouped_cat_cols,
            group_col=PHASE2_EXCHANGEABLE_GROUP,
            cont_cols=grouped_cont_cols,
            priors=grouped_priors,
        )
        holdout_group_idata = sample_model(
            holdout_group_model,
            f"{grouped_model_name}_holdout",
            cache_tag,
            output_dirs["models"],
            fast_dev,
            cores,
            draws,
            tune,
            chains,
            target_accept,
        )
        p_group_test = posterior_mean_prob(
            holdout_group_idata,
            data_group_test,
            cat_cols=grouped_cat_cols,
            cont_cols=grouped_cont_cols,
        )
        test_grp = test_grp.copy()
        test_grp["p_group"] = p_group_test

        test_events_scored = grouped_probs_to_events(
            test_ev,
            test_grp,
            group_cols=grouped_group_cols,
            prob_col="p_group",
        )
        p_group_test_events = test_events_scored["p_group"].to_numpy()
        if np.isnan(p_group_test_events).any():
            fallback_p = float(np.nanmean(p_group_test))
            if not np.isfinite(fallback_p):
                fallback_p = float(y_event_train.mean())
            p_group_test_events = np.where(np.isnan(p_group_test_events), fallback_p, p_group_test_events)

        score_binary_models_on_holdout(
            story_name=story_tag,
            test_events=test_events_scored,
            y_true=y_event_test,
            bern_probs=p_event_test,
            binom_probs=p_group_test_events,
            bern_model_name=event_model_name,
            binom_model_name=grouped_model_name,
            output_dirs=output_dirs,
            run_tag=user_tag,
            fast_dev=fast_dev,
        )

    eval_specs = [
        {
            "story_tag": "story1",
            "story_label": "Story 1",
            "story_df": clean_df.copy(),
            "outcome_col": "is_canceled",
            "event_model_name": "1E_bern",
            "grouped_model_name": "1G_binom",
            "event_cat_cols": ["weekday_cat", "dest_top", "train_cat"],
            "grouped_cat_cols": ["weekday_cat", "hour_3h_bin", "train_cat", "event_type"],
            "grouped_group_cols": ["weekday_cat", "hour_3h_bin", "train_cat", "event_type"],
            "event_priors": PHASE2_PRIORS_BERNOULLI["story1"],
            "grouped_priors": PHASE2_PRIORS_BINOMIAL["story1"],
            "filter": None,
        },
        {
            "story_tag": "story2",
            "story_label": "Story 2",
            "story_df": clean_df.copy(),
            "outcome_col": "is_delayed",
            "event_model_name": "2E_delay",
            "grouped_model_name": "2G_delay",
            "event_cat_cols": ["weekday_cat", "dest_top", "train_cat"],
            "grouped_cat_cols": ["weekday_cat", "hour_3h_bin", "train_cat"],
            "grouped_group_cols": ["weekday_cat", "hour_3h_bin", "train_cat"],
            "event_priors": PHASE2_PRIORS_BERNOULLI["story2_delay"],
            "grouped_priors": PHASE2_PRIORS_BINOMIAL["story2_delay"],
            "filter": (clean_df["is_canceled"] == 0) & (clean_df["arrival_planned_time"].notna()),
        },
        {
            "story_tag": "story3",
            "story_label": "Story 3",
            "story_df": clean_df.copy(),
            "outcome_col": "is_delayed",
            "event_model_name": "3E_delay",
            "grouped_model_name": "3G_delay",
            "event_cat_cols": ["weekday_cat", "dest_top", "train_cat"],
            "grouped_cat_cols": ["weekday_cat", "hour_3h_bin", "train_cat"],
            "grouped_group_cols": ["weekday_cat", "hour_3h_bin", "train_cat"],
            "event_priors": PHASE2_PRIORS_BERNOULLI["story3_delay"],
            "grouped_priors": PHASE2_PRIORS_BINOMIAL["story3_delay"],
            "filter": (clean_df["is_canceled"] == 0) & (clean_df["departure_planned_time"].notna()),
        },
    ]

    for spec in eval_specs:
        if spec["filter"] is not None:
            spec["story_df"] = spec["story_df"].loc[spec["filter"]].copy()
        try:
            evaluate_binary_holdout(
                story_tag=spec["story_tag"],
                story_label=spec["story_label"],
                story_df=spec["story_df"],
                outcome_col=spec["outcome_col"],
                event_model_name=spec["event_model_name"],
                grouped_model_name=spec["grouped_model_name"],
                event_cat_cols=spec["event_cat_cols"],
                grouped_cat_cols=spec["grouped_cat_cols"],
                grouped_group_cols=spec["grouped_group_cols"],
                event_priors=spec["event_priors"],
                grouped_priors=spec["grouped_priors"],
            )
        except Exception as e:
            err = str(e)
            pd.DataFrame(
                [{"story": spec["story_tag"], "error": err}]
            ).to_csv(
                os.path.join(output_dirs["tables"], f"{spec['story_tag']}_holdout_scores.csv"),
                index=False,
            )
            pd.DataFrame([{"error": err}]).to_csv(
                os.path.join(output_dirs["tables"], f"{spec['story_tag']}_calibration_{spec['event_model_name']}.csv"),
                index=False,
            )
            pd.DataFrame([{"error": err}]).to_csv(
                os.path.join(output_dirs["tables"], f"{spec['story_tag']}_calibration_{spec['grouped_model_name']}.csv"),
                index=False,
            )

    holdout_summary = pd.concat(
        [
            pd.read_csv(os.path.join(output_dirs["tables"], "story1_holdout_scores.csv")),
            pd.read_csv(os.path.join(output_dirs["tables"], "story2_holdout_scores.csv")),
            pd.read_csv(os.path.join(output_dirs["tables"], "story3_holdout_scores.csv")),
        ],
        ignore_index=True,
        sort=False,
    )
    holdout_summary.to_csv(
        os.path.join(output_dirs["tables"], "holdout_comparison_summary.csv"),
        index=False,
    )

    holdout_numeric = holdout_summary.copy()
    for metric_col in ["log_loss", "brier"]:
        holdout_numeric[metric_col] = pd.to_numeric(holdout_numeric.get(metric_col), errors="coerce")
    holdout_valid = holdout_numeric.dropna(subset=["log_loss", "brier"])

    winner_rows = []
    if not holdout_valid.empty:
        for story, sub in holdout_valid.groupby("story"):
            best_logloss = sub.loc[sub["log_loss"].idxmin()]
            best_brier = sub.loc[sub["brier"].idxmin()]

            winner_rows.append(
                {
                    "story": story,
                    "winner_log_loss": best_logloss["model"],
                    "winner_log_loss_value": best_logloss["log_loss"],
                    "winner_brier": best_brier["model"],
                    "winner_brier_value": best_brier["brier"],
                }
            )
    winner_df = pd.DataFrame(winner_rows)
    winner_df.to_csv(
        os.path.join(output_dirs["tables"], "holdout_winners.csv"),
        index=False,
    )

    subtitle = (
        f"Run={user_tag if user_tag is not None else 'untagged'} | "
        f"Mode={'FAST' if fast_dev else 'FULL'}"
    )
    if not holdout_valid.empty:
        holdout_metric_barplot(
            holdout_valid,
            metric="log_loss",
            path=os.path.join(output_dirs["ppc"], "holdout_log_loss_comparison.png"),
            title="Holdout comparison - log loss",
            subtitle=subtitle,
        )
        holdout_metric_barplot(
            holdout_valid,
            metric="brier",
            path=os.path.join(output_dirs["ppc"], "holdout_brier_comparison.png"),
            title="Holdout comparison - Brier score",
            subtitle=subtitle,
        )
    else:
        for metric_name, out_name in [
            ("log_loss", "holdout_log_loss_comparison.png"),
            ("brier", "holdout_brier_comparison.png"),
        ]:
            plt.figure(figsize=(8, 4.5))
            plt.text(0.5, 0.5, f"No valid rows for {metric_name} comparison", ha="center", va="center")
            plt.axis("off")
            plt.title(f"Holdout comparison - {metric_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dirs["ppc"], out_name), dpi=150, bbox_inches="tight")
            plt.close()

    comparison_report_rows = []
    if not holdout_valid.empty:
        for story, sub in holdout_valid.groupby("story"):
            sub_log = sub.sort_values("log_loss").reset_index(drop=True)
            if sub_log.shape[0] < 2:
                continue
            best = sub_log.iloc[0]
            second = sub_log.iloc[1]
            sub_brier = sub.sort_values("brier").reset_index(drop=True)

            comparison_report_rows.append(
                {
                    "story": story,
                    "best_model": best["model"],
                    "best_log_loss": best["log_loss"],
                    "runner_up_model": second["model"],
                    "runner_up_log_loss": second["log_loss"],
                    "delta_log_loss": second["log_loss"] - best["log_loss"],
                    "best_brier_model": sub_brier.iloc[0]["model"],
                    "best_brier": sub_brier.iloc[0]["brier"],
                }
            )
    comparison_report_df = pd.DataFrame(comparison_report_rows)
    comparison_report_df.to_csv(
        os.path.join(output_dirs["tables"], "holdout_report_summary.csv"),
        index=False,
    )

    print("\nHoldout comparison summary:")
    print(holdout_summary)

    # Diagnostics & PPC
    model_registry = {
        "1E_bern": {
            "model": model_1E,
            "idata": idata_1E,
            "kind": "bernoulli",
            "obs": y_1E,
            "story": "Story 1",
            "model_type": "Event-level Bernoulli",
            "n_rows": len(story1_event),
            "grouped_rows": None,
        },
        "1G_binom": {
            "model": model_1G,
            "idata": idata_1G,
            "kind": "binomial",
            "obs": y_1G,
            "n": n_1G,
            "story": "Story 1",
            "model_type": "Grouped Binomial",
            "n_rows": len(story1_grouped),
            "grouped_rows": len(story1_grouped),
        },
        "2E_delay": {
            "model": model_2E,
            "idata": idata_2E,
            "kind": "bernoulli",
            "obs": y_2E,
            "story": "Story 2",
            "model_type": "Event-level Bernoulli",
            "n_rows": len(story2_event),
            "grouped_rows": None,
        },
        "2G_delay": {
            "model": model_2G,
            "idata": idata_2G,
            "kind": "binomial",
            "obs": y_2G,
            "n": n_2G,
            "story": "Story 2",
            "model_type": "Grouped Binomial",
            "n_rows": len(story2_grouped),
            "grouped_rows": len(story2_grouped),
        },
        "2M_log": {
            "model": model_2M,
            "idata": idata_2M,
            "kind": "log_delay",
            "obs": y_2M,
            "story": "Story 2",
            "model_type": "Positive-delay log-normal",
            "n_rows": len(story2_pos),
            "grouped_rows": None,
        },
        "3E_delay": {
            "model": model_3E,
            "idata": idata_3E,
            "kind": "bernoulli",
            "obs": y_3E,
            "story": "Story 3",
            "model_type": "Event-level Bernoulli",
            "n_rows": len(story3_event),
            "grouped_rows": None,
        },
        "3G_delay": {
            "model": model_3G,
            "idata": idata_3G,
            "kind": "binomial",
            "obs": y_3G,
            "n": n_3G,
            "story": "Story 3",
            "model_type": "Grouped Binomial",
            "n_rows": len(story3_grouped),
            "grouped_rows": len(story3_grouped),
        },
        "3M_log": {
            "model": model_3M,
            "idata": idata_3M,
            "kind": "log_delay",
            "obs": y_3M,
            "story": "Story 3",
            "model_type": "Positive-delay log-normal",
            "n_rows": len(story3_pos),
            "grouped_rows": None,
        },
    }

    # Exchangeability-focused posterior summaries/plots for train_cat.
    obs_1 = observed_group_rate(story1_full, "train_cat", "is_canceled")
    obs_2 = observed_group_rate(story2_full, "train_cat", "is_delayed")
    obs_3 = observed_group_rate(story3_full, "train_cat", "is_delayed")
    obs_1.to_csv(os.path.join(output_dirs["tables"], "story1_train_cat_observed_rates.csv"), index=False)
    obs_2.to_csv(os.path.join(output_dirs["tables"], "story2_train_cat_observed_rates.csv"), index=False)
    obs_3.to_csv(os.path.join(output_dirs["tables"], "story3_train_cat_observed_rates.csv"), index=False)

    # Posterior train_cat effects and group scales.
    eff_1E = posterior_group_effect_summary(idata_1E, "train_cat_eff", "train_cat")
    eff_1G = posterior_group_effect_summary(idata_1G, "train_cat_eff", "train_cat")
    sig_1E = summarize_group_sigma(idata_1E, "train_cat_sigma")
    sig_1G = summarize_group_sigma(idata_1G, "train_cat_sigma")

    eff_2E = posterior_group_effect_summary(idata_2E, "train_cat_eff", "train_cat")
    eff_2G = posterior_group_effect_summary(idata_2G, "train_cat_eff", "train_cat")
    sig_2E = summarize_group_sigma(idata_2E, "train_cat_sigma")
    sig_2G = summarize_group_sigma(idata_2G, "train_cat_sigma")

    eff_3E = posterior_group_effect_summary(idata_3E, "train_cat_eff", "train_cat")
    eff_3G = posterior_group_effect_summary(idata_3G, "train_cat_eff", "train_cat")
    sig_3E = summarize_group_sigma(idata_3E, "train_cat_sigma")
    sig_3G = summarize_group_sigma(idata_3G, "train_cat_sigma")

    eff_1E.to_csv(os.path.join(output_dirs["tables"], "story1_train_cat_effects_1E_bern.csv"), index=False)
    eff_1G.to_csv(os.path.join(output_dirs["tables"], "story1_train_cat_effects_1G_binom.csv"), index=False)
    sig_1E.to_csv(os.path.join(output_dirs["tables"], "story1_train_cat_sigma_1E_bern.csv"), index=False)
    sig_1G.to_csv(os.path.join(output_dirs["tables"], "story1_train_cat_sigma_1G_binom.csv"), index=False)

    eff_2E.to_csv(os.path.join(output_dirs["tables"], "story2_train_cat_effects_2E_delay.csv"), index=False)
    eff_2G.to_csv(os.path.join(output_dirs["tables"], "story2_train_cat_effects_2G_delay.csv"), index=False)
    sig_2E.to_csv(os.path.join(output_dirs["tables"], "story2_train_cat_sigma_2E_delay.csv"), index=False)
    sig_2G.to_csv(os.path.join(output_dirs["tables"], "story2_train_cat_sigma_2G_delay.csv"), index=False)

    eff_3E.to_csv(os.path.join(output_dirs["tables"], "story3_train_cat_effects_3E_delay.csv"), index=False)
    eff_3G.to_csv(os.path.join(output_dirs["tables"], "story3_train_cat_effects_3G_delay.csv"), index=False)
    sig_3E.to_csv(os.path.join(output_dirs["tables"], "story3_train_cat_sigma_3E_delay.csv"), index=False)
    sig_3G.to_csv(os.path.join(output_dirs["tables"], "story3_train_cat_sigma_3G_delay.csv"), index=False)

    # Posterior reference-scenario probabilities by train_cat with a fixed reference scenario.
    def build_event_base(df, data_obj):
        base = df.iloc[[0]].copy()
        base["weekday_cat"] = pick_level(data_obj["cat_levels"]["weekday_cat"], "1")
        base["dest_top"] = pick_level(data_obj["cat_levels"]["dest_top"], "Other")
        for cont_name in CONT_COLS:
            z_col = f"{cont_name}_z"
            if z_col in base.columns:
                base[z_col] = 0.0
        return base

    def build_grouped_base(df, data_obj, with_event_type=False):
        base = df.iloc[[0]].copy()
        base["weekday_cat"] = pick_level(data_obj["cat_levels"]["weekday_cat"], "1")
        base["hour_3h_bin"] = pick_level(data_obj["cat_levels"]["hour_3h_bin"], "12-14")
        if with_event_type:
            base["event_type"] = pick_level(data_obj["cat_levels"]["event_type"], "Unknown")
        if "days_until_xmas_z_mean" in base.columns:
            base["days_until_xmas_z_mean"] = 0.0
        return base

    pred_1E = posterior_category_probability(
        idata_1E,
        data_builder=lambda df: build_model_data(
            df,
            cont_cols=CONT_COLS,
            cat_cols=["weekday_cat", "dest_top", "train_cat"],
            cat_levels=data_1E["cat_levels"],
        ),
        base_df=build_event_base(story1_event, data_1E),
        category_col="train_cat",
        category_levels=data_1E["cat_levels"]["train_cat"],
        cat_cols=["weekday_cat", "dest_top", "train_cat"],
        cont_cols=CONT_COLS,
    )
    pred_1G = posterior_category_probability(
        idata_1G,
        data_builder=lambda df: build_model_data(
            df,
            cont_cols=["days_until_xmas_z_mean"],
            cat_cols=["weekday_cat", "hour_3h_bin", "train_cat", "event_type"],
            cat_levels=data_1G["cat_levels"],
        ),
        base_df=build_grouped_base(story1_grouped, data_1G, with_event_type=True),
        category_col="train_cat",
        category_levels=data_1G["cat_levels"]["train_cat"],
        cat_cols=["weekday_cat", "hour_3h_bin", "train_cat", "event_type"],
        cont_cols=["days_until_xmas_z_mean"],
    )
    pred_2E = posterior_category_probability(
        idata_2E,
        data_builder=lambda df: build_model_data(
            df,
            cont_cols=CONT_COLS,
            cat_cols=["weekday_cat", "dest_top", "train_cat"],
            cat_levels=data_2E["cat_levels"],
        ),
        base_df=build_event_base(story2_event, data_2E),
        category_col="train_cat",
        category_levels=data_2E["cat_levels"]["train_cat"],
        cat_cols=["weekday_cat", "dest_top", "train_cat"],
        cont_cols=CONT_COLS,
    )
    pred_2G = posterior_category_probability(
        idata_2G,
        data_builder=lambda df: build_model_data(
            df,
            cont_cols=["days_until_xmas_z_mean"],
            cat_cols=["weekday_cat", "hour_3h_bin", "train_cat"],
            cat_levels=data_2G["cat_levels"],
        ),
        base_df=build_grouped_base(story2_grouped, data_2G),
        category_col="train_cat",
        category_levels=data_2G["cat_levels"]["train_cat"],
        cat_cols=["weekday_cat", "hour_3h_bin", "train_cat"],
        cont_cols=["days_until_xmas_z_mean"],
    )
    pred_3E = posterior_category_probability(
        idata_3E,
        data_builder=lambda df: build_model_data(
            df,
            cont_cols=CONT_COLS,
            cat_cols=["weekday_cat", "dest_top", "train_cat"],
            cat_levels=data_3E["cat_levels"],
        ),
        base_df=build_event_base(story3_event, data_3E),
        category_col="train_cat",
        category_levels=data_3E["cat_levels"]["train_cat"],
        cat_cols=["weekday_cat", "dest_top", "train_cat"],
        cont_cols=CONT_COLS,
    )
    pred_3G = posterior_category_probability(
        idata_3G,
        data_builder=lambda df: build_model_data(
            df,
            cont_cols=["days_until_xmas_z_mean"],
            cat_cols=["weekday_cat", "hour_3h_bin", "train_cat"],
            cat_levels=data_3G["cat_levels"],
        ),
        base_df=build_grouped_base(story3_grouped, data_3G),
        category_col="train_cat",
        category_levels=data_3G["cat_levels"]["train_cat"],
        cat_cols=["weekday_cat", "hour_3h_bin", "train_cat"],
        cont_cols=["days_until_xmas_z_mean"],
    )

    pred_1E.to_csv(os.path.join(output_dirs["tables"], "story1_train_cat_probs_1E_bern.csv"), index=False)
    pred_1G.to_csv(os.path.join(output_dirs["tables"], "story1_train_cat_probs_1G_binom.csv"), index=False)
    pred_2E.to_csv(os.path.join(output_dirs["tables"], "story2_train_cat_probs_2E_delay.csv"), index=False)
    pred_2G.to_csv(os.path.join(output_dirs["tables"], "story2_train_cat_probs_2G_delay.csv"), index=False)
    pred_3E.to_csv(os.path.join(output_dirs["tables"], "story3_train_cat_probs_3E_delay.csv"), index=False)
    pred_3G.to_csv(os.path.join(output_dirs["tables"], "story3_train_cat_probs_3G_delay.csv"), index=False)

    group_effect_forest_plot(
        eff_1E,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story1_train_cat_effects_1E_bern.png"),
        title="Story 1 - train_cat group effects - 1E_bern",
        subtitle=report_tag_line("1E_bern", "1", user_tag, fast_dev),
    )
    group_effect_forest_plot(
        eff_1G,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story1_train_cat_effects_1G_binom.png"),
        title="Story 1 - train_cat group effects - 1G_binom",
        subtitle=report_tag_line("1G_binom", "1", user_tag, fast_dev),
    )
    group_effect_forest_plot(
        eff_2E,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story2_train_cat_effects_2E_delay.png"),
        title="Story 2 - train_cat group effects - 2E_delay",
        subtitle=report_tag_line("2E_delay", "2", user_tag, fast_dev),
    )
    group_effect_forest_plot(
        eff_2G,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story2_train_cat_effects_2G_delay.png"),
        title="Story 2 - train_cat group effects - 2G_delay",
        subtitle=report_tag_line("2G_delay", "2", user_tag, fast_dev),
    )
    group_effect_forest_plot(
        eff_3E,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story3_train_cat_effects_3E_delay.png"),
        title="Story 3 - train_cat group effects - 3E_delay",
        subtitle=report_tag_line("3E_delay", "3", user_tag, fast_dev),
    )
    group_effect_forest_plot(
        eff_3G,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story3_train_cat_effects_3G_delay.png"),
        title="Story 3 - train_cat group effects - 3G_delay",
        subtitle=report_tag_line("3G_delay", "3", user_tag, fast_dev),
    )

    posterior_prob_by_group_plot(
        pred_1E,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story1_train_cat_probs_1E_bern.png"),
        title="Story 1 - posterior cancellation probability by train_cat - 1E_bern",
        subtitle=report_tag_line("1E_bern", "1", user_tag, fast_dev),
    )
    posterior_prob_by_group_plot(
        pred_1G,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story1_train_cat_probs_1G_binom.png"),
        title="Story 1 - posterior cancellation probability by train_cat - 1G_binom",
        subtitle=report_tag_line("1G_binom", "1", user_tag, fast_dev),
    )
    posterior_prob_by_group_plot(
        pred_2E,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story2_train_cat_probs_2E_delay.png"),
        title="Story 2 - posterior delay probability by train_cat - 2E_delay",
        subtitle=report_tag_line("2E_delay", "2", user_tag, fast_dev),
    )
    posterior_prob_by_group_plot(
        pred_2G,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story2_train_cat_probs_2G_delay.png"),
        title="Story 2 - posterior delay probability by train_cat - 2G_delay",
        subtitle=report_tag_line("2G_delay", "2", user_tag, fast_dev),
    )
    posterior_prob_by_group_plot(
        pred_3E,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story3_train_cat_probs_3E_delay.png"),
        title="Story 3 - posterior delay probability by train_cat - 3E_delay",
        subtitle=report_tag_line("3E_delay", "3", user_tag, fast_dev),
    )
    posterior_prob_by_group_plot(
        pred_3G,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story3_train_cat_probs_3G_delay.png"),
        title="Story 3 - posterior delay probability by train_cat - 3G_delay",
        subtitle=report_tag_line("3G_delay", "3", user_tag, fast_dev),
    )

    observed_vs_reference_probability_plot(
        obs_1,
        pred_1E,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story1_obs_vs_refprob_1E_bern.png"),
        title="Story 1 - observed raw rate vs reference-scenario probability - 1E_bern",
        subtitle=report_tag_line("1E_bern", "1", user_tag, fast_dev),
    )
    observed_vs_reference_probability_plot(
        obs_1,
        pred_1G,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story1_obs_vs_refprob_1G_binom.png"),
        title="Story 1 - observed raw rate vs reference-scenario probability - 1G_binom",
        subtitle=report_tag_line("1G_binom", "1", user_tag, fast_dev),
    )
    observed_vs_reference_probability_plot(
        obs_2,
        pred_2E,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story2_obs_vs_refprob_2E_delay.png"),
        title="Story 2 - observed raw rate vs reference-scenario probability - 2E_delay",
        subtitle=report_tag_line("2E_delay", "2", user_tag, fast_dev),
    )
    observed_vs_reference_probability_plot(
        obs_2,
        pred_2G,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story2_obs_vs_refprob_2G_delay.png"),
        title="Story 2 - observed raw rate vs reference-scenario probability - 2G_delay",
        subtitle=report_tag_line("2G_delay", "2", user_tag, fast_dev),
    )
    observed_vs_reference_probability_plot(
        obs_3,
        pred_3E,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story3_obs_vs_refprob_3E_delay.png"),
        title="Story 3 - observed raw rate vs reference-scenario probability - 3E_delay",
        subtitle=report_tag_line("3E_delay", "3", user_tag, fast_dev),
    )
    observed_vs_reference_probability_plot(
        obs_3,
        pred_3G,
        "train_cat",
        os.path.join(output_dirs["ppc"], "story3_obs_vs_refprob_3G_delay.png"),
        title="Story 3 - observed raw rate vs reference-scenario probability - 3G_delay",
        subtitle=report_tag_line("3G_delay", "3", user_tag, fast_dev),
    )

    exchangeability_summary = pd.concat(
        [
            sig_1E.assign(story="story1", model="1E_bern"),
            sig_1G.assign(story="story1", model="1G_binom"),
            sig_2E.assign(story="story2", model="2E_delay"),
            sig_2G.assign(story="story2", model="2G_delay"),
            sig_3E.assign(story="story3", model="3E_delay"),
            sig_3G.assign(story="story3", model="3G_delay"),
        ],
        ignore_index=True,
    )
    exchangeability_summary.to_csv(
        os.path.join(output_dirs["tables"], "exchangeability_summary.csv"),
        index=False,
    )

    # LOO per model
    loo_objects = {}
    for name, info in model_registry.items():
        try:
            loo_obj = az.loo(info["idata"], pointwise=True)
            loo_objects[name] = loo_obj
            save_loo_results(loo_obj, name, output_dirs["loo"])
        except Exception as e:
            pd.DataFrame(
                [{"model": name, "error": str(e)}]
            ).to_csv(
                os.path.join(output_dirs["loo"], f"loo_{name}_error.csv"),
                index=False,
            )

    not_applicable = pd.DataFrame(
        [
            {
                "comparison": "story1",
                "status": "not_applicable",
                "reason": "Bernoulli and Binomial use different observation units; compared instead on common held-out events.",
            },
            {
                "comparison": "story2",
                "status": "not_applicable",
                "reason": "Bernoulli and Binomial use different observation units; compared instead on common held-out events.",
            },
            {
                "comparison": "story3",
                "status": "not_applicable",
                "reason": "Bernoulli and Binomial use different observation units; compared instead on common held-out events.",
            },
        ]
    )
    not_applicable.to_csv(
        os.path.join(output_dirs["loo"], "comparison_not_applicable.csv"),
        index=False,
    )

    summary_rows = []
    for name, info in model_registry.items():
        summary_path = os.path.join(output_dirs["tables"], f"summary_{name}.csv")
        save_summary(info["idata"], summary_path)

        divergences = int(info["idata"].sample_stats["diverging"].sum())
        summary_rows.append({"model": name, "divergences": divergences})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dirs["tables"], "divergences_summary.csv"), index=False)

    for name, info in model_registry.items():
        save_summary(
            info["idata"],
            os.path.join(output_dirs["tables"], f"summary_{name}_train_cat_effects.csv"),
            var_names=["train_cat_sigma", "train_cat_eff"],
        )

    trace_diag = {name: diag_tag_line(info["idata"]) for name, info in model_registry.items()}

    # Trace plots for event-level Bernoulli models
    trace_event_models = [
        ("1E_bern", idata_1E, CONT_COLS),
        ("2E_delay", idata_2E, CONT_COLS),
        ("3E_delay", idata_3E, CONT_COLS),
    ]
    trace_grouped_models = [
        ("1G_binom", idata_1G, ["days_until_xmas_z_mean"]),
        ("2G_delay", idata_2G, ["days_until_xmas_z_mean"]),
        ("3G_delay", idata_3G, ["days_until_xmas_z_mean"]),
    ]
    for model_name, idata_obj, cont_names in trace_event_models + trace_grouped_models:
        info = model_registry[model_name]
        base_subtitle = plot_tag_line(
            model_name=model_name,
            model_kind=info["model_type"],
            story=info["story"],
            n_rows=info["n_rows"],
            run_tag=user_tag,
            is_fast_dev=fast_dev,
            group_factor=PHASE2_EXCHANGEABLE_GROUP,
            grouped_rows=info["grouped_rows"],
        )
        subtitle = f"{base_subtitle} | {trace_diag[model_name]}"
        for cont_name in cont_names:
            save_trace(
                idata_obj,
                os.path.join(output_dirs["models"], f"trace_{model_name}_{cont_name}.png"),
                var_names=["intercept", "beta"],
                coords={"cont": [cont_name]},
                title=f"Trace plot - {info['story']} {info['model_type']} ({cont_name})",
                subtitle=subtitle,
            )

    for model_name, idata_obj, _ in trace_event_models + trace_grouped_models:
        info = model_registry[model_name]
        base_subtitle = plot_tag_line(
            model_name=model_name,
            model_kind=info["model_type"],
            story=info["story"],
            n_rows=info["n_rows"],
            run_tag=user_tag,
            is_fast_dev=fast_dev,
            group_factor=PHASE2_EXCHANGEABLE_GROUP,
            grouped_rows=info["grouped_rows"],
        )
        subtitle = f"{base_subtitle} | {trace_diag[model_name]}"
        save_trace(
            idata_obj,
            os.path.join(output_dirs["models"], f"trace_{model_name}_beta_all.png"),
            var_names=["beta"],
            title=f"Trace plot - {info['story']} {info['model_type']} (beta_all)",
            subtitle=subtitle,
        )

    for name, info in model_registry.items():
        path = os.path.join(output_dirs["ppc"], f"ppc_{name}.png")
        subtitle = plot_tag_line(
            model_name=name,
            model_kind=info["model_type"],
            story=info["story"],
            n_rows=info["n_rows"],
            run_tag=user_tag,
            is_fast_dev=fast_dev,
            group_factor=PHASE2_EXCHANGEABLE_GROUP,
            grouped_rows=info["grouped_rows"],
        )
        if info["kind"] in ("bernoulli", "binomial"):
            ppc_rate_plot(
                info["model"],
                info["idata"],
                info["obs"],
                path,
                f"PPC - {info['story']} {info['model_type']} ({name})",
                fast_dev,
                n_trials=info.get("n"),
                subtitle=subtitle,
            )
        else:
            ppc_log_delay_plot(
                info["model"],
                info["idata"],
                info["obs"],
                path,
                f"PPC - {info['story']} {info['model_type']} ({name})",
                fast_dev,
                subtitle=subtitle,
            )

    print("Diagnostics and PPC outputs saved.")

    prior_rate_plot(
        model_1E,
        y_1E,
        os.path.join(output_dirs["ppc"], "prior_1E_bern.png"),
        "Prior predictive - Story 1 Event-level Bernoulli",
        fast_dev,
        subtitle=plot_tag_line(
            model_name="1E_bern",
            model_kind="Event-level Bernoulli",
            story="Story 1",
            n_rows=len(story1_event),
            run_tag=user_tag,
            is_fast_dev=fast_dev,
            group_factor=PHASE2_EXCHANGEABLE_GROUP,
        ),
    )
    prior_rate_plot(
        model_2E,
        y_2E,
        os.path.join(output_dirs["ppc"], "prior_2E_delay.png"),
        "Prior predictive - Story 2 Event-level Bernoulli",
        fast_dev,
        subtitle=plot_tag_line(
            model_name="2E_delay",
            model_kind="Event-level Bernoulli",
            story="Story 2",
            n_rows=len(story2_event),
            run_tag=user_tag,
            is_fast_dev=fast_dev,
            group_factor=PHASE2_EXCHANGEABLE_GROUP,
        ),
    )
    prior_rate_plot(
        model_3E,
        y_3E,
        os.path.join(output_dirs["ppc"], "prior_3E_delay.png"),
        "Prior predictive - Story 3 Event-level Bernoulli",
        fast_dev,
        subtitle=plot_tag_line(
            model_name="3E_delay",
            model_kind="Event-level Bernoulli",
            story="Story 3",
            n_rows=len(story3_event),
            run_tag=user_tag,
            is_fast_dev=fast_dev,
            group_factor=PHASE2_EXCHANGEABLE_GROUP,
        ),
    )

    # Story-level metadata and report-facing indexes.
    story_metadata = pd.DataFrame(
        [
            {
                "story": "story1",
                "event_rows": len(story1_full),
                "grouped_rows": len(story1_grouped),
                "outcome": "is_canceled",
                "event_model": "1E_bern",
                "grouped_model": "1G_binom",
                "support_model": "",
            },
            {
                "story": "story2",
                "event_rows": len(story2_full),
                "grouped_rows": len(story2_grouped),
                "outcome": "is_delayed",
                "event_model": "2E_delay",
                "grouped_model": "2G_delay",
                "support_model": "2M_log",
            },
            {
                "story": "story3",
                "event_rows": len(story3_full),
                "grouped_rows": len(story3_grouped),
                "outcome": "is_delayed",
                "event_model": "3E_delay",
                "grouped_model": "3G_delay",
                "support_model": "3M_log",
            },
        ]
    )
    story_metadata.to_csv(
        os.path.join(output_dirs["tables"], "story_metadata.csv"),
        index=False,
    )

    holdout_summary_path = os.path.join(output_dirs["tables"], "holdout_comparison_summary.csv")
    winners_path = os.path.join(output_dirs["tables"], "holdout_winners.csv")
    run_summary_path = os.path.join(output_dirs["tables"], "RUN_SUMMARY.md")

    summary_md = f"""# Phase 2 Run Summary

## Run identity
- Model version: {MODEL_VERSION}
- User run tag: {user_tag}
- Cache tag: {cache_tag}
- Mode: {"FAST" if fast_dev else "FULL"}

## Primary comparison
Event-level Bernoulli vs grouped Binomial, evaluated on the same held-out event sets.

## Exchangeability
- Exchangeable grouping factor: {PHASE2_EXCHANGEABLE_GROUP}

## Key files
- Holdout comparison summary: {holdout_summary_path}
- Holdout winners: {winners_path}
- Exchangeability summary: {os.path.join(output_dirs["tables"], "exchangeability_summary.csv")}
- Diagnostics summary: {os.path.join(output_dirs["tables"], "divergences_summary.csv")}
"""
    with open(run_summary_path, "w") as f:
        f.write(summary_md)

    main_comparison_rows = [
        {
            "section": "main_comparison",
            "story": "story1",
            "artifact": "holdout_scores",
            "file": os.path.join(output_dirs["tables"], "story1_holdout_scores.csv"),
            "purpose": "Event-level Bernoulli vs grouped Binomial comparison on held-out events",
        },
        {
            "section": "main_comparison",
            "story": "story2",
            "artifact": "holdout_scores",
            "file": os.path.join(output_dirs["tables"], "story2_holdout_scores.csv"),
            "purpose": "Event-level Bernoulli vs grouped Binomial comparison on held-out events",
        },
        {
            "section": "main_comparison",
            "story": "story3",
            "artifact": "holdout_scores",
            "file": os.path.join(output_dirs["tables"], "story3_holdout_scores.csv"),
            "purpose": "Event-level Bernoulli vs grouped Binomial comparison on held-out events",
        },
        {
            "section": "main_comparison",
            "story": "all",
            "artifact": "holdout_summary",
            "file": os.path.join(output_dirs["tables"], "holdout_comparison_summary.csv"),
            "purpose": "Combined holdout comparison across stories",
        },
        {
            "section": "main_comparison",
            "story": "all",
            "artifact": "holdout_winners",
            "file": os.path.join(output_dirs["tables"], "holdout_winners.csv"),
            "purpose": "Winner by story and metric",
        },
        {
            "section": "main_comparison",
            "story": "all",
            "artifact": "holdout_report_summary",
            "file": os.path.join(output_dirs["tables"], "holdout_report_summary.csv"),
            "purpose": "Compact narrative summary for report writing",
        },
    ]
    save_output_index(
        main_comparison_rows,
        os.path.join(output_dirs["tables"], "index_main_comparison.csv"),
    )

    exchangeability_rows = [
        {
            "section": "exchangeability",
            "story": "all",
            "artifact": "note",
            "file": os.path.join(output_dirs["tables"], "exchangeability_note.txt"),
            "purpose": "Explicit statement of the exchangeability assumption",
        },
        {
            "section": "exchangeability",
            "story": "all",
            "artifact": "summary",
            "file": os.path.join(output_dirs["tables"], "exchangeability_summary.csv"),
            "purpose": "Shared summary of train_cat group-level sigma across models",
        },
        {
            "section": "exchangeability",
            "story": "all",
            "artifact": "counts",
            "file": os.path.join(output_dirs["tables"], "train_cat_counts.csv"),
            "purpose": "Observed number of events per train category",
        },
    ]
    save_output_index(
        exchangeability_rows,
        os.path.join(output_dirs["tables"], "index_exchangeability.csv"),
    )

    diagnostic_rows = [
        {
            "section": "diagnostics",
            "story": "all",
            "artifact": "divergences_summary",
            "file": os.path.join(output_dirs["tables"], "divergences_summary.csv"),
            "purpose": "Sampling divergence summary across models",
        },
        {
            "section": "diagnostics",
            "story": "all",
            "artifact": "run_manifest",
            "file": os.path.join(output_dirs["tables"], "run_manifest.json"),
            "purpose": "Run configuration and metadata",
        },
        {
            "section": "diagnostics",
            "story": "all",
            "artifact": "missing_outputs",
            "file": os.path.join(output_dirs["tables"], "missing_outputs.csv"),
            "purpose": "Audit of expected outputs",
        },
    ]
    save_output_index(
        diagnostic_rows,
        os.path.join(output_dirs["tables"], "index_diagnostics.csv"),
    )

    report_bundle_files = [
        os.path.join(output_dirs["tables"], "holdout_comparison_summary.csv"),
        os.path.join(output_dirs["tables"], "holdout_winners.csv"),
        os.path.join(output_dirs["tables"], "holdout_report_summary.csv"),
        os.path.join(output_dirs["tables"], "exchangeability_summary.csv"),
        os.path.join(output_dirs["tables"], "RUN_SUMMARY.md"),
        os.path.join(output_dirs["ppc"], "holdout_log_loss_comparison.png"),
        os.path.join(output_dirs["ppc"], "holdout_brier_comparison.png"),
    ]
    for path in report_bundle_files:
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(report_dir, os.path.basename(path)))

    for story in ["story1", "story2", "story3"]:
        os.makedirs(os.path.join(report_dir, story), exist_ok=True)

    story1_bundle = [
        os.path.join(output_dirs["tables"], "story1_holdout_scores.csv"),
        os.path.join(output_dirs["tables"], "story1_calibration_1E_bern.csv"),
        os.path.join(output_dirs["tables"], "story1_calibration_1G_binom.csv"),
        os.path.join(output_dirs["tables"], "story1_train_cat_effects_1E_bern.csv"),
        os.path.join(output_dirs["tables"], "story1_train_cat_effects_1G_binom.csv"),
        os.path.join(output_dirs["tables"], "story1_train_cat_probs_1E_bern.csv"),
        os.path.join(output_dirs["tables"], "story1_train_cat_probs_1G_binom.csv"),
        os.path.join(output_dirs["ppc"], "story1_calibration_1E_bern.png"),
        os.path.join(output_dirs["ppc"], "story1_calibration_1G_binom.png"),
        os.path.join(output_dirs["ppc"], "story1_train_cat_effects_1E_bern.png"),
        os.path.join(output_dirs["ppc"], "story1_train_cat_effects_1G_binom.png"),
        os.path.join(output_dirs["ppc"], "story1_train_cat_probs_1E_bern.png"),
        os.path.join(output_dirs["ppc"], "story1_train_cat_probs_1G_binom.png"),
    ]
    for path in story1_bundle:
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(report_dir, "story1", os.path.basename(path)))

    story2_bundle = [
        os.path.join(output_dirs["tables"], "story2_holdout_scores.csv"),
        os.path.join(output_dirs["tables"], "story2_calibration_2E_delay.csv"),
        os.path.join(output_dirs["tables"], "story2_calibration_2G_delay.csv"),
        os.path.join(output_dirs["tables"], "story2_train_cat_effects_2E_delay.csv"),
        os.path.join(output_dirs["tables"], "story2_train_cat_effects_2G_delay.csv"),
        os.path.join(output_dirs["tables"], "story2_train_cat_probs_2E_delay.csv"),
        os.path.join(output_dirs["tables"], "story2_train_cat_probs_2G_delay.csv"),
        os.path.join(output_dirs["ppc"], "story2_calibration_2E_delay.png"),
        os.path.join(output_dirs["ppc"], "story2_calibration_2G_delay.png"),
        os.path.join(output_dirs["ppc"], "story2_train_cat_effects_2E_delay.png"),
        os.path.join(output_dirs["ppc"], "story2_train_cat_effects_2G_delay.png"),
        os.path.join(output_dirs["ppc"], "story2_train_cat_probs_2E_delay.png"),
        os.path.join(output_dirs["ppc"], "story2_train_cat_probs_2G_delay.png"),
    ]
    for path in story2_bundle:
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(report_dir, "story2", os.path.basename(path)))

    story3_bundle = [
        os.path.join(output_dirs["tables"], "story3_holdout_scores.csv"),
        os.path.join(output_dirs["tables"], "story3_calibration_3E_delay.csv"),
        os.path.join(output_dirs["tables"], "story3_calibration_3G_delay.csv"),
        os.path.join(output_dirs["tables"], "story3_train_cat_effects_3E_delay.csv"),
        os.path.join(output_dirs["tables"], "story3_train_cat_effects_3G_delay.csv"),
        os.path.join(output_dirs["tables"], "story3_train_cat_probs_3E_delay.csv"),
        os.path.join(output_dirs["tables"], "story3_train_cat_probs_3G_delay.csv"),
        os.path.join(output_dirs["ppc"], "story3_calibration_3E_delay.png"),
        os.path.join(output_dirs["ppc"], "story3_calibration_3G_delay.png"),
        os.path.join(output_dirs["ppc"], "story3_train_cat_effects_3E_delay.png"),
        os.path.join(output_dirs["ppc"], "story3_train_cat_effects_3G_delay.png"),
        os.path.join(output_dirs["ppc"], "story3_train_cat_probs_3E_delay.png"),
        os.path.join(output_dirs["ppc"], "story3_train_cat_probs_3G_delay.png"),
    ]
    for path in story3_bundle:
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(report_dir, "story3", os.path.basename(path)))

    expected_files = []

    # core summaries
    for name in model_registry.keys():
        expected_files.append(os.path.join(output_dirs["tables"], f"summary_{name}.csv"))
        expected_files.append(os.path.join(output_dirs["ppc"], f"ppc_{name}.png"))
        expected_files.append(os.path.join(output_dirs["loo"], f"loo_{name}.csv"))
        expected_files.append(os.path.join(output_dirs["loo"], f"pareto_summary_{name}.csv"))

    # trace plots for event-level and grouped binary models
    trace_model_cont_map = {
        "1E_bern": CONT_COLS,
        "2E_delay": CONT_COLS,
        "3E_delay": CONT_COLS,
        "1G_binom": ["days_until_xmas_z_mean"],
        "2G_delay": ["days_until_xmas_z_mean"],
        "3G_delay": ["days_until_xmas_z_mean"],
    }
    for model_name, cont_names in trace_model_cont_map.items():
        for cont_name in cont_names:
            expected_files.append(os.path.join(output_dirs["models"], f"trace_{model_name}_{cont_name}.png"))
        expected_files.append(os.path.join(output_dirs["models"], f"trace_{model_name}_beta_all.png"))

    # prior predictive plots
    expected_files.extend([
        os.path.join(output_dirs["ppc"], "prior_1E_bern.png"),
        os.path.join(output_dirs["ppc"], "prior_2E_delay.png"),
        os.path.join(output_dirs["ppc"], "prior_3E_delay.png"),
    ])

    # grouped binomial support tables
    expected_files.extend([
        os.path.join(output_dirs["tables"], "story1_grouped_binomial_data.csv"),
        os.path.join(output_dirs["tables"], "story2_grouped_binomial_data.csv"),
        os.path.join(output_dirs["tables"], "story3_grouped_binomial_data.csv"),
        os.path.join(output_dirs["tables"], "grouped_row_counts.csv"),
        os.path.join(output_dirs["tables"], "train_cat_counts.csv"),
        os.path.join(output_dirs["tables"], "exchangeability_note.txt"),
    ])

    # LOO comparability note for cross-unit model comparison
    expected_files.append(
        os.path.join(output_dirs["loo"], "comparison_not_applicable.csv"),
    )

    # holdout comparison outputs (main Bernoulli vs Binomial comparison)
    expected_files.extend([
        os.path.join(output_dirs["tables"], "story1_holdout_scores.csv"),
        os.path.join(output_dirs["tables"], "story2_holdout_scores.csv"),
        os.path.join(output_dirs["tables"], "story3_holdout_scores.csv"),
        os.path.join(output_dirs["tables"], "holdout_comparison_summary.csv"),
        os.path.join(output_dirs["tables"], "holdout_winners.csv"),
        os.path.join(output_dirs["tables"], "holdout_report_summary.csv"),
        os.path.join(output_dirs["tables"], "story1_calibration_1E_bern.csv"),
        os.path.join(output_dirs["tables"], "story1_calibration_1G_binom.csv"),
        os.path.join(output_dirs["tables"], "story2_calibration_2E_delay.csv"),
        os.path.join(output_dirs["tables"], "story2_calibration_2G_delay.csv"),
        os.path.join(output_dirs["tables"], "story3_calibration_3E_delay.csv"),
        os.path.join(output_dirs["tables"], "story3_calibration_3G_delay.csv"),
        os.path.join(output_dirs["ppc"], "holdout_log_loss_comparison.png"),
        os.path.join(output_dirs["ppc"], "holdout_brier_comparison.png"),
        os.path.join(output_dirs["ppc"], "story1_calibration_1E_bern.png"),
        os.path.join(output_dirs["ppc"], "story1_calibration_1G_binom.png"),
        os.path.join(output_dirs["ppc"], "story2_calibration_2E_delay.png"),
        os.path.join(output_dirs["ppc"], "story2_calibration_2G_delay.png"),
        os.path.join(output_dirs["ppc"], "story3_calibration_3E_delay.png"),
        os.path.join(output_dirs["ppc"], "story3_calibration_3G_delay.png"),
    ])

    # exchangeability-focused posterior summaries and visualizations
    expected_files.extend([
        os.path.join(output_dirs["tables"], "story1_train_cat_observed_rates.csv"),
        os.path.join(output_dirs["tables"], "story2_train_cat_observed_rates.csv"),
        os.path.join(output_dirs["tables"], "story3_train_cat_observed_rates.csv"),
        os.path.join(output_dirs["tables"], "story1_train_cat_effects_1E_bern.csv"),
        os.path.join(output_dirs["tables"], "story1_train_cat_effects_1G_binom.csv"),
        os.path.join(output_dirs["tables"], "story1_train_cat_sigma_1E_bern.csv"),
        os.path.join(output_dirs["tables"], "story1_train_cat_sigma_1G_binom.csv"),
        os.path.join(output_dirs["tables"], "story2_train_cat_effects_2E_delay.csv"),
        os.path.join(output_dirs["tables"], "story2_train_cat_effects_2G_delay.csv"),
        os.path.join(output_dirs["tables"], "story2_train_cat_sigma_2E_delay.csv"),
        os.path.join(output_dirs["tables"], "story2_train_cat_sigma_2G_delay.csv"),
        os.path.join(output_dirs["tables"], "story3_train_cat_effects_3E_delay.csv"),
        os.path.join(output_dirs["tables"], "story3_train_cat_effects_3G_delay.csv"),
        os.path.join(output_dirs["tables"], "story3_train_cat_sigma_3E_delay.csv"),
        os.path.join(output_dirs["tables"], "story3_train_cat_sigma_3G_delay.csv"),
        os.path.join(output_dirs["tables"], "story1_train_cat_probs_1E_bern.csv"),
        os.path.join(output_dirs["tables"], "story1_train_cat_probs_1G_binom.csv"),
        os.path.join(output_dirs["tables"], "story2_train_cat_probs_2E_delay.csv"),
        os.path.join(output_dirs["tables"], "story2_train_cat_probs_2G_delay.csv"),
        os.path.join(output_dirs["tables"], "story3_train_cat_probs_3E_delay.csv"),
        os.path.join(output_dirs["tables"], "story3_train_cat_probs_3G_delay.csv"),
        os.path.join(output_dirs["tables"], "exchangeability_summary.csv"),
        os.path.join(output_dirs["ppc"], "story1_train_cat_effects_1E_bern.png"),
        os.path.join(output_dirs["ppc"], "story1_train_cat_effects_1G_binom.png"),
        os.path.join(output_dirs["ppc"], "story2_train_cat_effects_2E_delay.png"),
        os.path.join(output_dirs["ppc"], "story2_train_cat_effects_2G_delay.png"),
        os.path.join(output_dirs["ppc"], "story3_train_cat_effects_3E_delay.png"),
        os.path.join(output_dirs["ppc"], "story3_train_cat_effects_3G_delay.png"),
        os.path.join(output_dirs["ppc"], "story1_train_cat_probs_1E_bern.png"),
        os.path.join(output_dirs["ppc"], "story1_train_cat_probs_1G_binom.png"),
        os.path.join(output_dirs["ppc"], "story2_train_cat_probs_2E_delay.png"),
        os.path.join(output_dirs["ppc"], "story2_train_cat_probs_2G_delay.png"),
        os.path.join(output_dirs["ppc"], "story3_train_cat_probs_3E_delay.png"),
        os.path.join(output_dirs["ppc"], "story3_train_cat_probs_3G_delay.png"),
        os.path.join(output_dirs["ppc"], "story1_obs_vs_refprob_1E_bern.png"),
        os.path.join(output_dirs["ppc"], "story1_obs_vs_refprob_1G_binom.png"),
        os.path.join(output_dirs["ppc"], "story2_obs_vs_refprob_2E_delay.png"),
        os.path.join(output_dirs["ppc"], "story2_obs_vs_refprob_2G_delay.png"),
        os.path.join(output_dirs["ppc"], "story3_obs_vs_refprob_3E_delay.png"),
        os.path.join(output_dirs["ppc"], "story3_obs_vs_refprob_3G_delay.png"),
    ])

    # report-facing organization files
    expected_files.extend([
        os.path.join(output_dirs["tables"], "index_main_comparison.csv"),
        os.path.join(output_dirs["tables"], "index_exchangeability.csv"),
        os.path.join(output_dirs["tables"], "index_diagnostics.csv"),
        os.path.join(output_dirs["tables"], "RUN_SUMMARY.md"),
        os.path.join(output_dirs["tables"], "story_metadata.csv"),
    ])

    for name in model_registry.keys():
        expected_files.append(os.path.join(output_dirs["tables"], f"summary_{name}_train_cat_effects.csv"))

    missing_files = [p for p in expected_files if not os.path.exists(p)]
    pd.DataFrame({"missing_file": missing_files}).to_csv(
        os.path.join(output_dirs["tables"], "missing_outputs.csv"),
        index=False,
    )

    manifest = {
        "run_tag": user_tag,
        "model_version": MODEL_VERSION,
        "user_run_tag": user_tag,
        "cache_tag": cache_tag,
        "draws_tag": draws_tag,
        "tune_tag": tune_tag,
        "chains_tag": chains_tag,
        "accept_tag": accept_tag,
        "fast_dev": fast_dev,
        "cont_cols": CONT_COLS,
        "phase2_exchangeable_group": PHASE2_EXCHANGEABLE_GROUP,
        "phase2_exchangeability_note": PHASE2_EXCHANGEABILITY_NOTE,
        "group_col_story1": group_col_1,
        "group_col_story2": group_col_2,
        "group_col_story3": group_col_3,
        "group_stats_story1": group_stats_1,
        "group_stats_story2": group_stats_2,
        "group_stats_story3": group_stats_3,
        "n_models": len(model_registry),
        "n_expected_files": len(expected_files),
        "n_missing_files": len(missing_files),
    }
    manifest.update(
        {
            "phase": "phase2",
            "primary_comparison": "event_bernoulli_vs_grouped_binomial",
            "exchangeable_group": PHASE2_EXCHANGEABLE_GROUP,
            "stories": {
                "story1": {
                    "event_model": "1E_bern",
                    "grouped_model": "1G_binom",
                    "support_model": None,
                },
                "story2": {
                    "event_model": "2E_delay",
                    "grouped_model": "2G_delay",
                    "support_model": "2M_log",
                },
                "story3": {
                    "event_model": "3E_delay",
                    "grouped_model": "3G_delay",
                    "support_model": "3M_log",
                },
            },
            "holdout_strategy": {
                "type": "whole_day_split",
                "rule": "day_of_month % 5 == 0 used as test set",
            },
            "report_bundle_dir": report_dir,
        }
    )

    with open(os.path.join(output_dirs["tables"], "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Expected files: {len(expected_files)}")
    print(f"Missing files: {len(missing_files)}")
    if missing_files:
        print("Missing outputs written to:", os.path.join(output_dirs["tables"], "missing_outputs.csv"))
    else:
        print("All expected outputs are present.")
