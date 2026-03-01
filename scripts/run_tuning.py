"""
Tunable evaluation on FIQA: sweep retrieval and metric parameters, run the
evaluation runner for each combination, and aggregate results to a single CSV
with best-by-accuracy and best-by-REM summaries.
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv

import pandas as pd
import yaml

# Load .env from project root so HF_TOKEN etc. are set for subprocess
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _flatten_sweep(sweep: dict, prefix: tuple = ()) -> list[tuple[tuple, list]]:
    """Convert nested sweep dict to list of (key_path_tuple, values_list)."""
    out = []
    for k, v in sweep.items():
        path = prefix + (k,)
        if isinstance(v, dict) and v:
            # Recurse into nested dict so we get (path + key, list) per leaf list
            out.extend(_flatten_sweep(v, path))
        elif isinstance(v, list):
            out.append((path, list(v)))
        else:
            out.append((path, [v]))
    return out


def _set_nested(cfg: dict, path: tuple, value) -> None:
    """Set cfg[path[0]][path[1]]... = value, creating intermediate dicts."""
    d = cfg
    for key in path[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[path[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tunable FIQA evaluation grid.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tuning_fiqa.yaml",
        help="Path to tuning config (default: configs/tuning_fiqa.yaml)",
    )
    args = parser.parse_args()

    tuning_path = Path(args.config)
    if not tuning_path.is_absolute():
        tuning_path = Path.cwd() / tuning_path
    tuning_path = tuning_path.resolve()
    if not tuning_path.exists():
        print(f"Tuning config not found: {tuning_path}", file=sys.stderr)
        sys.exit(1)

    # Project root: parent of configs/ when config path is under configs/
    config_dir = tuning_path.parent
    project_root = config_dir.parent if config_dir.name == "configs" else config_dir

    with open(tuning_path, "r") as f:
        tuning = yaml.safe_load(f)

    base_config_path = project_root / tuning["base_config"]
    if not base_config_path.exists():
        base_config_path = project_root / "configs" / tuning["base_config"]
    if not base_config_path.exists():
        print(f"Base config not found: {base_config_path}", file=sys.stderr)
        sys.exit(1)

    with open(base_config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    base_cfg["data"] = base_cfg.get("data") or {}
    base_cfg["data"]["dataset"] = tuning["dataset"]

    sweep_nested = tuning.get("sweep") or {}
    flat = _flatten_sweep(sweep_nested)
    keys = [p for p, _ in flat]
    value_lists = [v for _, v in flat]

    combinations = list(itertools.product(*value_lists))
    print(f"Running {len(combinations)} parameter combinations on FIQA.")

    tmp_dir = project_root / "configs" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dfs = []
    run_id = 0
    src_dir = project_root / "src"
    env = {**os.environ, "PYTHONPATH": str(src_dir)}

    for combo in combinations:
        run_id += 1
        overrides = dict(zip(keys, combo))
        cfg = deepcopy(base_cfg)

        for path, value in zip(keys, combo):
            _set_nested(cfg, path, value)
            if path == ("retrieval", "hybrid", "w_dense"):
                _set_nested(cfg, ("retrieval", "hybrid", "w_sparse"), 1.0 - value)

        logs_name = f"tuning_run_{run_id:03d}.csv"
        cfg["logging"] = cfg.get("logging") or {}
        cfg["logging"]["logs_csv"] = f"../logs/{logs_name}"

        temp_config_path = tmp_dir / f"tuning_run_{run_id:03d}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        cmd = [
            sys.executable,
            "-c",
            f"from runner import run_experiments; run_experiments(config_path={repr(str(temp_config_path.resolve()))})",
        ]
        subprocess.run(cmd, cwd=src_dir, env=env, check=True)

        run_log_path = logs_dir / logs_name
        df = pd.read_csv(run_log_path)
        df["run_id"] = run_id
        df["top_k"] = overrides.get(("retrieval", "top_k"), cfg["retrieval"]["top_k"])
        df["w_dense"] = overrides.get(("retrieval", "hybrid", "w_dense"), cfg["retrieval"]["hybrid"]["w_dense"])
        df["w_sparse"] = cfg["retrieval"]["hybrid"]["w_sparse"]
        df["accuracy_at_k"] = overrides.get(("metrics", "accuracy_at_k"), cfg["metrics"]["accuracy_at_k"])
        results_dfs.append(df)
        temp_config_path.unlink(missing_ok=True)
        run_log_path.unlink(missing_ok=True)

    combined = pd.concat(results_dfs, ignore_index=True)
    out_path = project_root / tuning["output"]["results_csv"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Wrote {len(combined)} rows to {out_path}.")

    # Per-run aggregates (one row per run_id): mean accuracy and mean REM across retrieval types
    run_agg = combined.groupby("run_id").agg(
        mean_accuracy=("accuracy", "mean"),
        mean_REM=("REM", "mean"),
    ).reset_index()
    run_params = combined.groupby("run_id")[["top_k", "w_dense", "w_sparse", "accuracy_at_k"]].first().reset_index()
    run_agg = run_agg.merge(run_params, on="run_id")

    # Save per-run summary table (one row per run)
    runs_path = out_path.with_stem(out_path.stem + "_runs")
    run_agg.to_csv(runs_path, index=False)
    print(f"Wrote run summary ({len(run_agg)} rows) to {runs_path}.")

    best_acc_run = run_agg.loc[run_agg["mean_accuracy"].idxmax()]
    best_rem_run = run_agg.loc[run_agg["mean_REM"].idxmax()]
    best_lines = [
        "Best run by mean accuracy:",
        f"  run_id={int(best_acc_run['run_id'])}, top_k={best_acc_run['top_k']}, w_dense={best_acc_run['w_dense']:.2f}, accuracy_at_k={best_acc_run['accuracy_at_k']}",
        f"  mean_accuracy={best_acc_run['mean_accuracy']:.4f}, mean_REM={best_acc_run['mean_REM']:.4f}",
        "",
        "Best run by mean REM:",
        f"  run_id={int(best_rem_run['run_id'])}, top_k={best_rem_run['top_k']}, w_dense={best_rem_run['w_dense']:.2f}, accuracy_at_k={best_rem_run['accuracy_at_k']}",
        f"  mean_accuracy={best_rem_run['mean_accuracy']:.4f}, mean_REM={best_rem_run['mean_REM']:.4f}",
    ]
    print("\n" + "\n".join(best_lines))

    # Save best-run summary to file
    best_path = out_path.with_stem(out_path.stem + "_best").with_suffix(".txt")
    with open(best_path, "w") as f:
        f.write("\n".join(best_lines) + "\n")
    print(f"Wrote best-run summary to {best_path}.")


if __name__ == "__main__":
    main()
