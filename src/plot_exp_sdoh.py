"""Analyze SDoH experiment results: sampling frequency by category.

Reads the sdoh_summary.csv (category-level correctness rates) and history.csv
files from sr_adaptive runs across seeds. Outputs a CSV with sampling frequency
and average correctness rate per category.
"""
import logging
import argparse
import json

from adjustText import adjust_text
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# Set font sizes for paper figures
sns.set_context("paper", font_scale=1.5)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze SDoH sampling patterns by category"
    )
    parser.add_argument(
        "--sdoh-summary-csv",
        type=str,
        required=True,
        help="Path to sdoh_summary.csv from assemble_data",
    )
    parser.add_argument(
        "--history-csvs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to history.csv files from sr_adaptive runs",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        required=True,
        help="Output scatter plot path",
    )
    parser.add_argument(
        "--log",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(args.log),
    ])

    # Load category-level summary (avg correctness rates)
    summary_df = pd.read_csv(args.sdoh_summary_csv)
    assert "category" in summary_df.columns, "sdoh_summary.csv must have 'category' column"
    assert "avg_is_correct" in summary_df.columns, "sdoh_summary.csv must have 'avg_is_correct' column"

    # Load and concatenate history files
    history_dfs = []
    for csv_path in args.history_csvs:
        df = pd.read_csv(csv_path)
        history_dfs.append(df)
    history_df = pd.concat(history_dfs, ignore_index=True)

    # Extract category from metadata JSON
    history_df["category"] = history_df["metadata"].apply(
        lambda x: json.loads(x)["category"]
    )

    # Compute sampling frequency per category (proportion of samples)
    category_counts = history_df["category"].value_counts()
    total_samples = len(history_df)
    samp_freq_df = pd.DataFrame({
        "category": category_counts.index,
        "samp_freq": category_counts.values / total_samples,
    })

    # Merge with summary to get avg_is_correct
    out_df = samp_freq_df.merge(
        summary_df[["category", "avg_is_correct"]],
        on="category",
        how="left",
    )

    # Sort by sampling frequency descending
    out_df = out_df.sort_values("avg_is_correct", ascending=False).reset_index(drop=True)

    print("CORRELATION -- more negative the better!")
    logging.info(stats.pearsonr(out_df["samp_freq"], out_df["avg_is_correct"]))
    logging.info(stats.spearmanr(out_df["samp_freq"], out_df["avg_is_correct"]))

    # Create scatter plot
    plt.subplots(figsize=(4.28, 2.4))
    ax = sns.scatterplot(data=out_df, x="avg_is_correct", y="samp_freq", s=100)
    sns.despine()
    ax.set_xlabel("Average Correctness Rate")
    ax.set_ylabel("Sampling Frequency")

    # Label top 5 points by sampling frequency
    top5 = out_df.nlargest(5, "samp_freq")
    for _, row in top5.iterrows():
        ax.annotate(
            row["category"],
            (row["avg_is_correct"], row["samp_freq"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=12,
        )

    ax.figure.savefig(args.out_plot, bbox_inches="tight", dpi=150)

    out_df = out_df.round(3)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} categories to {args.out_csv}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
