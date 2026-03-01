"""Remove cache and log files for the dataset that is not currently active.

Run from the project root after switching data.dataset in config, to free space
and avoid confusion. Not required for correctness; paths are already per-dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

SUPPORTED_DATASETS = ("nfcorpus", "fiqa")


def main(config_path: str = "configs/experiment_config.yaml") -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    current = cfg.get("data", {}).get("dataset")
    if current not in SUPPORTED_DATASETS:
        print(f"Current config dataset is '{current}'; nothing to clean.")
        return
    other = "fiqa" if current == "nfcorpus" else "nfcorpus"
    root = Path(config_path).resolve().parent.parent
    cache_dir = root / "data" / "emb_cache"
    logs_dir = root / "logs"
    removed = []
    emb_path = cache_dir / f"doc_embeddings_{other}.npy"
    if emb_path.exists():
        emb_path.unlink()
        removed.append(str(emb_path))
    for p in logs_dir.glob(f"bm25_{other}.pkl"):
        p.unlink()
        removed.append(str(p))
    for p in logs_dir.glob(f"bm25_{other}_*.pkl"):
        p.unlink()
        removed.append(str(p))
    log_path = logs_dir / f"logs_{other}.csv"
    if log_path.exists():
        log_path.unlink()
        removed.append(str(log_path))
    if removed:
        print(f"Removed {len(removed)} file(s) for dataset '{other}':")
        for p in removed:
            print(f"  {p}")
    else:
        print(f"No cache/log files found for dataset '{other}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean caches and logs for the inactive dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to experiment config (used to read data.dataset).",
    )
    args = parser.parse_args()
    main(config_path=args.config)
