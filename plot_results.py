#!/usr/bin/env python3
"""
Create per-dataset comparison plots for the Risk-Aware Shift decision tree project.

Expected input files in the same directory as this script, unless overridden:
- test1_gain_ratio.csv
- test2_class_confidence.csv
- test3_risk_aware_shift_beta_0_6.csv
- test4_risk_aware_shift_beta_sweep.csv

Usage:
    python3 plot_results.py
    python3 plot_results.py --input-dir /path/to/results --output-dir /path/to/plots

This version creates separate plots for each dataset and annotates
every bar and every line point with its numeric value.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-dataset plots from experiment CSV files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing the experiment CSV files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where plot images will be saved."
    )
    parser.add_argument(
        "--risk-beta-file",
        type=str,
        default="test3_risk_aware_shift_beta_0_6.csv",
        help="Single-beta proposed-method CSV used for direct method comparison."
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def normalize_method_name(method: str) -> str:
    mapping = {
        "gain_ratio": "Gain Ratio",
        "class_confidence": "Class Confidence",
        "risk_aware_shift": "Risk-Aware Shift",
    }
    return mapping.get(method, method)


def safe_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def annotate_bars(ax) -> None:
    for container in ax.containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            if pd.isna(height):
                labels.append("")
            else:
                labels.append(f"{height:.3f}")
        ax.bar_label(container, labels=labels, padding=3, fontsize=8)


def save_dataset_bar_chart(dataset_df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    dataset_name = dataset_df["dataset"].iloc[0]
    ordered = dataset_df.sort_values("method_label")
    fig, ax = plt.subplots(figsize=(8, 6))
    ordered.plot(kind="bar", x="method_label", y=metric, legend=False, ax=ax)
    ax.set_title(f"{dataset_name}: {ylabel} by Method")
    ax.set_xlabel("Method")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_dataset_scatter(dataset_df: pd.DataFrame, x: str, y: str, title: str, output_path: Path) -> None:
    dataset_name = dataset_df["dataset"].iloc[0]
    plt.figure(figsize=(8, 6))
    for _, row in dataset_df.iterrows():
        label = row["method_label"]
        plt.scatter(row[x], row[y], label=label)
        plt.annotate(
            f"{label}\n({row[x]:.3f}, {row[y]:.3f})",
            (row[x], row[y]),
            fontsize=8
        )
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    plt.title(f"{dataset_name}: {title}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def annotate_line_points(x_values, y_values) -> None:
    for x, y in zip(x_values, y_values):
        if pd.isna(y):
            continue
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)


def save_dataset_beta_line_plot(dataset_df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    dataset_name = dataset_df["dataset"].iloc[0]
    ordered = dataset_df.sort_values("beta")
    x_values = ordered["beta"]
    y_values = ordered[metric]

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, marker="o")
    annotate_line_points(x_values, y_values)
    plt.xlabel("Beta")
    plt.ylabel(ylabel)
    plt.title(f"{dataset_name}: Beta vs {ylabel}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gain_ratio = load_csv(input_dir / "test1_gain_ratio.csv")
    class_confidence = load_csv(input_dir / "test2_class_confidence.csv")
    risk_single = load_csv(input_dir / args.risk_beta_file)
    risk_sweep = load_csv(input_dir / "test4_risk_aware_shift_beta_sweep.csv")

    compare_df = pd.concat([gain_ratio, class_confidence, risk_single], ignore_index=True)
    compare_df["method_label"] = compare_df["method"].map(normalize_method_name)

    risk_sweep = risk_sweep.copy()
    risk_sweep["method_label"] = risk_sweep["method"].map(normalize_method_name)

    bar_metrics = [
        ("accuracy", "Accuracy"),
        ("avg_shift", "Average Shift"),
        ("avg_risk", "Average Risk"),
        ("max_depth", "Max Depth"),
        ("total_nodes", "Total Nodes"),
    ]

    line_metrics = [
        ("accuracy", "Accuracy"),
        ("avg_shift", "Average Shift"),
        ("avg_risk", "Average Risk"),
        ("max_depth", "Max Depth"),
        ("total_nodes", "Total Nodes"),
    ]

    for dataset in sorted(compare_df["dataset"].unique()):
        dataset_compare = compare_df[compare_df["dataset"] == dataset].copy()
        dataset_sweep = risk_sweep[risk_sweep["dataset"] == dataset].copy()
        dataset_slug = safe_name(dataset)

        dataset_dir = output_dir / dataset_slug
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for metric, ylabel in bar_metrics:
            save_dataset_bar_chart(
                dataset_compare,
                metric,
                ylabel,
                dataset_dir / f"{metric}_bar.png"
            )

        save_dataset_scatter(
            dataset_compare,
            "avg_shift",
            "avg_risk",
            "Shift vs Risk",
            dataset_dir / "shift_vs_risk_scatter.png"
        )

        save_dataset_scatter(
            dataset_compare,
            "avg_risk",
            "accuracy",
            "Risk vs Accuracy",
            dataset_dir / "risk_vs_accuracy_scatter.png"
        )

        for metric, ylabel in line_metrics:
            save_dataset_beta_line_plot(
                dataset_sweep,
                metric,
                ylabel,
                dataset_dir / f"beta_vs_{metric}.png"
            )

    print(f"Saved per-dataset plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
