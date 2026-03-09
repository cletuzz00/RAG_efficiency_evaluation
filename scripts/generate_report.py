"""
Generate the experiment report with actual results from tuning logs.

Run after: python scripts/run_tuning.py

Reads logs/tuning_fiqa.csv, logs/tuning_fiqa_runs.csv, logs/tuning_fiqa_best.txt
and replaces placeholders in report/experiment_report.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def load_best_sections(best_path: Path) -> tuple[str, str]:
    """Parse tuning_fiqa_best.txt into best-by-accuracy and best-by-REM text."""
    if not best_path.exists():
        return (
            "(Run `python scripts/run_tuning.py` and re-run this script to fill.)",
            "(Run `python scripts/run_tuning.py` and re-run this script to fill.)",
        )
    text = best_path.read_text()
    lines = text.strip().split("\n")
    best_acc: list[str] = []
    best_rem: list[str] = []
    current: list[str] | None = None
    for line in lines:
        if line.strip().startswith("Best run by mean accuracy"):
            current = best_acc
            continue
        if line.strip().startswith("Best run by mean REM"):
            current = best_rem
            continue
        if current is not None and line.strip():
            current.append(line)
    return (
        "\n".join(best_acc) if best_acc else "(No data)",
        "\n".join(best_rem) if best_rem else "(No data)",
    )


def _to_md_table(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown table (no tabulate dependency)."""
    lines = []
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    lines.append(header)
    lines.append(sep)
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def format_runs_table(runs_path: Path) -> str:
    """Format tuning_fiqa_runs.csv as markdown table."""
    if not runs_path.exists():
        return "(Run `python scripts/run_tuning.py` and re-run this script to fill.)"
    df = pd.read_csv(runs_path)
    cols = ["run_id", "top_k", "w_dense", "accuracy_at_k", "mean_accuracy", "mean_REM"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()
    df["mean_accuracy"] = df["mean_accuracy"].map(lambda x: f"{x:.4f}")
    df["mean_REM"] = df["mean_REM"].map(lambda x: f"{x:.4f}")
    df["w_dense"] = df["w_dense"].map(lambda x: f"{x:.2f}")
    return _to_md_table(df)


def format_per_type_table(combined_path: Path, run_id: int | None = None) -> str:
    """Format per-retrieval-type averages for one run from tuning_fiqa.csv."""
    if not combined_path.exists():
        return "(Run `python scripts/run_tuning.py` and re-run this script to fill.)"
    df = pd.read_csv(combined_path)
    if run_id is None and "run_id" in df.columns:
        run_id = int(df["run_id"].max())
    if run_id is not None:
        df = df[df["run_id"] == run_id]
    grouped = df.groupby("retrieval_type")[["accuracy", "latency_s", "cost_tokens", "REM"]].mean()
    grouped = grouped.round(4)
    return _to_md_table(grouped.reset_index())


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill experiment report with tuning results.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root (default: current directory)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="report/experiment_report.md",
        help="Report path (default: report/experiment_report.md)",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    logs_dir = root / "logs"
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = root / args.report

    best_path = logs_dir / "tuning_fiqa_best.txt"
    runs_path = logs_dir / "tuning_fiqa_runs.csv"
    combined_path = logs_dir / "tuning_fiqa.csv"

    best_acc, best_rem = load_best_sections(best_path)
    runs_table = format_runs_table(runs_path)

    run_id_for_type: int | None = None
    if runs_path.exists():
        run_agg = pd.read_csv(runs_path)
        run_id_for_type = int(run_agg.loc[run_agg["mean_REM"].idxmax(), "run_id"])
    per_type_table = format_per_type_table(combined_path, run_id=run_id_for_type)

    if not report_path.exists():
        print(f"Report not found: {report_path}", file=sys.stderr)
        sys.exit(1)

    content = report_path.read_text()
    replacements = {
        "{{BEST_BY_ACCURACY}}": best_acc,
        "{{BEST_BY_REM}}": best_rem,
        "{{RUNS_TABLE}}": runs_table,
        "{{PER_TYPE_TABLE}}": per_type_table,
    }
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)

    report_path.write_text(content)
    print(f"Updated {report_path} with results from {logs_dir}.")


if __name__ == "__main__":
    main()
