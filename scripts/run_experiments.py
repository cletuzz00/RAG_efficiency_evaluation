"""
Unified entry point for running RAG retrieval experiments.

Examples:

  # Single run on the dataset in configs/experiment_config.yaml
  python scripts/run_experiments.py --mode single

  # Single run for a specific dataset
  python scripts/run_experiments.py --mode single --datasets fiqa

  # Tuning on FIQA with the default tuning config
  python scripts/run_experiments.py --mode tuning --datasets fiqa

  # Tuning on FIQA and NFCorpus
  python scripts/run_experiments.py --mode tuning --datasets fiqa,nfcorpus

"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _load_default_dataset(project_root: Path, config_path: str) -> str:
    base_cfg_path = Path(config_path)
    if not base_cfg_path.is_absolute():
        base_cfg_path = project_root / base_cfg_path
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_cfg_path}")
    with open(base_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("data") or {}
    dataset = data_cfg.get("dataset")
    if not dataset:
        raise ValueError(
            "No datasets provided via --datasets and base config has no data.dataset"
        )
    return str(dataset)


def _clean_tuning_outputs(project_root: Path, dataset: str) -> None:
    """Remove per-dataset tuning outputs so each run starts from a clean slate."""
    logs_dir = project_root / "logs"
    targets = [
        logs_dir / f"tuning_{dataset}.csv",
        logs_dir / f"tuning_{dataset}_runs.csv",
        logs_dir / f"tuning_{dataset}_best.txt",
    ]
    for path in targets:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _run_single(project_root: Path, dataset: str, base_config: str) -> None:
    """Run a single experiment (no tuning) for one dataset."""
    base_cfg_path = Path(base_config)
    if not base_cfg_path.is_absolute():
        base_cfg_path = project_root / base_cfg_path
    with open(base_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("data", {})
    cfg["data"]["dataset"] = dataset

    tmp_dir = project_root / "configs" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg_path = tmp_dir / f"single_{dataset}.yaml"
    with open(tmp_cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    src_dir = project_root / "src"
    env = {**os.environ, "PYTHONPATH": str(src_dir)}

    cmd = [
        sys.executable,
        "-c",
        f"from runner import run_experiments; run_experiments(config_path={repr(str(tmp_cfg_path.resolve()))})",
    ]
    subprocess.run(cmd, cwd=src_dir, env=env, check=True)


def _run_tuning_for_dataset(
    project_root: Path,
    dataset: str,
    tuning_config: str | None,
) -> None:
    """Run tuning for a single dataset using scripts/run_tuning.py."""
    _clean_tuning_outputs(project_root, dataset)

    if tuning_config:
        tuning_path = Path(tuning_config)
    else:
        tuning_path = Path(f"configs/tuning_{dataset}.yaml")

    if not tuning_path.is_absolute():
        tuning_path = project_root / tuning_path

    if not tuning_path.exists():
        raise FileNotFoundError(f"Tuning config not found for dataset {dataset}: {tuning_path}")

    cmd = [
        sys.executable,
        "scripts/run_tuning.py",
        "--config",
        str(tuning_path.relative_to(project_root)),
    ]
    subprocess.run(cmd, cwd=project_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified RAG retrieval experiment runner.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "tuning"],
        default="single",
        help="Run a single config ('single') or a tuning sweep ('tuning').",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated list of datasets to run (e.g. 'fiqa,nfcorpus'). "
        "If omitted, falls back to data.dataset in the base config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Base experiment config path (for single-run mode).",
    )
    parser.add_argument(
        "--tuning-config",
        type=str,
        default="",
        help="Optional tuning config path for tuning mode; if omitted, "
        "expects configs/tuning_{dataset}.yaml for each dataset.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if args.datasets.strip():
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        default_dataset = _load_default_dataset(project_root, args.config)
        datasets = [default_dataset]

    if args.mode == "single":
        for ds in datasets:
            print(f"Running single experiment for dataset={ds}...")
            _run_single(project_root, ds, args.config)
    else:
        for ds in datasets:
            print(f"Running tuning for dataset={ds}...")
            _run_tuning_for_dataset(project_root, ds, args.tuning_config or None)


if __name__ == "__main__":
    main()

