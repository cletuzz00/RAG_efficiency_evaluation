from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_logs(config_path: str = "../configs/experiment_config.yaml") -> pd.DataFrame:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    dataset = cfg["data"].get("dataset")
    if dataset is not None:
        logs_csv = cfg["logging"].get("logs_csv") or f"../logs/logs_{dataset}.csv"
    else:
        logs_csv = cfg["logging"]["logs_csv"]
    return pd.read_csv(logs_csv)


def plot_bars(df: pd.DataFrame, out_dir: str = "figures") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby("retrieval_type")[["accuracy", "latency_s", "cost_tokens", "REM"]].mean()

    metrics = ["accuracy", "latency_s", "cost_tokens", "REM"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), constrained_layout=True)

    for ax, metric in zip(axes, metrics):
        grouped[metric].plot(kind="bar", ax=ax)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xlabel("retrieval_type")

    fig.savefig(out / "bar_metrics.png", dpi=200)
    plt.close(fig)


def plot_radar(df: pd.DataFrame, out_dir: str = "figures") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby("retrieval_type")[["accuracy_norm", "latency_norm", "cost_norm", "REM"]].mean()

    # For visualization, we flip latency and cost (1 - x) to reflect "higher is better"
    grouped["latency_eff"] = 1.0 - grouped["latency_norm"]
    grouped["cost_eff"] = 1.0 - grouped["cost_norm"]

    metrics = ["accuracy_norm", "latency_eff", "cost_eff", "REM"]
    labels = ["Accuracy", "Latency (1 - norm)", "Cost (1 - norm)", "REM"]

    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for name, row in grouped.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, label=name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Retrieval Comparison (Radar)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.savefig(out / "radar_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df_logs = load_logs()
    plot_bars(df_logs)
    plot_radar(df_logs)

