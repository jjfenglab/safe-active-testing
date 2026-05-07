"""Plot e-process statistics from history.csv for paper visualization."""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Plot e-process from history CSV")
    parser.add_argument("--history-csv", type=str, required=True, help="Path to history.csv")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for threshold")
    parser.add_argument("--m-min", type=int, default=None, help="Optional m_min for vertical line")
    parser.add_argument("--out-plot", type=str, required=True, help="Output plot path")
    return parser.parse_args()


def plot_e_process(history_df: pd.DataFrame, alpha: float, m_min: int | None, plot_path: str):
    """Plot the e-statistic history.

    Args:
        history_df: DataFrame with robustness_statistic and/or auditor_statistic columns
        alpha: Significance level for threshold line
        m_min: If provided, draw vertical line at this iteration
        plot_path: Path to output plot file
    """
    sns.set_context("paper", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(4.28, 2.4))

    # Plot robustness statistic if present
    ax.plot(
        history_df["robustness/futility_statistic"].values,
        marker="o",
        markersize=3,
        label="Model's Null",
        color=sns.color_palette()[1],
    )

    # Plot auditor statistic if present
    ax.plot(
        history_df["auditor/efficacy_statistic"].values,
        marker="s",
        markersize=3,
        label="Auditor's Null",
        color=sns.color_palette()[0],
    )

    # Add vertical line at m_min if provided
    if m_min is not None:
        ax.axvline(
            x=m_min,
            color="gray",
            linestyle=":",
            alpha=0.7,
        )

    # Add threshold line
    ax.axhline(
        y=1.0 / alpha,
        color="r",
        linestyle="--",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("E-Process")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    history_df = pd.read_csv(args.history_csv)
    plot_e_process(history_df, args.alpha, args.m_min, args.out_plot)


if __name__ == "__main__":
    main()
