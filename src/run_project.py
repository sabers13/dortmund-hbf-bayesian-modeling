#!/usr/bin/env python3
"""CLI entrypoint for the Applied Bayesian project."""
import argparse
import os
from pathlib import Path
import sys


def _ensure_local_pytensor_cache():
    """Use project-local runtime dirs to avoid home-dir permission issues."""
    phase2_root = Path(__file__).resolve().parents[1]
    cache_root = phase2_root / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    compiledir = phase2_root / ".pytensor"
    compiledir.mkdir(parents=True, exist_ok=True)

    current_flags = os.environ.get("PYTENSOR_FLAGS", "").strip()
    compiledir_flag = f"compiledir={compiledir}"
    if "compiledir=" in current_flags:
        return
    if current_flags:
        os.environ["PYTENSOR_FLAGS"] = f"{current_flags},{compiledir_flag}"
    else:
        os.environ["PYTENSOR_FLAGS"] = compiledir_flag


def main():
    _ensure_local_pytensor_cache()

    from .pipeline import run_pipeline

    parser = argparse.ArgumentParser(description="Run the Applied Bayesian Final Project pipeline.")
    parser.add_argument("--data", default="Dortmund_HBF_December.csv", help="Path to the CSV dataset.")
    parser.add_argument("--outputs", default="outputs", help="Output base directory.")
    parser.add_argument("--fast", action="store_true", help="Enable FAST_DEV mode (subsample + fewer draws).")
    parser.add_argument("--subsample", type=int, default=None, help="Override subsample size for FAST_DEV.")
    parser.add_argument("--cores", type=int, default=None, help="Number of cores for sampling.")
    parser.add_argument("--draws", type=int, default=None, help="Number of posterior draws.")
    parser.add_argument("--tune", type=int, default=None, help="Number of tuning steps.")
    parser.add_argument("--chains", type=int, default=None, help="Number of chains.")
    parser.add_argument("--target-accept", type=float, default=None, help="Target accept probability.")
    parser.add_argument("--run-tag", default=None, help="Optional run label for cache/output versioning.")
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data,
        outputs_base=args.outputs,
        fast_dev=args.fast,
        subsample_n=args.subsample,
        cores=args.cores,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        run_tag=args.run_tag,
    )


if __name__ == "__main__":
    sys.exit(main())
