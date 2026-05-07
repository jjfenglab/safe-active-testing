"""Aggregate multiple CSV result files with optional groupby summarization."""

import argparse
from os import path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate multiple CSV files and optionally compute group statistics"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        nargs="+",
        required=True,
        help="Input CSV files to aggregate",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--groupby-cols",
        type=str,
        nargs="+",
        default=None,
        help="Columns to group by for computing mean of remaining numeric columns",
    )
    parser.add_argument(
        "--add-col",
        type=str,
        default=None,
        help="Name of column to add with constant value",
    )
    parser.add_argument(
        "--add-val",
        type=str,
        default=None,
        help="Value for the added column",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Read and concatenate all input CSVs
    dfs = []
    for csv_file in args.input_csv:
        if path.exists(csv_file):
            df = pd.read_csv(csv_file)
            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Optionally add a constant column
    if args.add_col is not None:
        assert args.add_val is not None, "--add-val required when --add-col is specified"
        combined_df[args.add_col] = args.add_val

    # Optionally group by columns and compute mean
    if args.groupby_cols is not None:
        result_df = combined_df.groupby(args.groupby_cols, as_index=False).mean(numeric_only=True)
    else:
        result_df = combined_df

    result_df.to_csv(args.out_csv, index=False)
    print(f"Wrote aggregated results to {args.out_csv}")
    print(result_df)


if __name__ == "__main__":
    main()
