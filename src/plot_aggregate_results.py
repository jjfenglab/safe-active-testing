"""Plot distribution of rejection times to compare methods."""

import argparse
from os import path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set font sizes for paper figures
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
SPINE_COLOR = '#555'

# Method ordering and label mapping
METHOD_ORDER = [
    "sprt_stratified_abstain0",
    "sprt_learned_abstain0",
    "sprt_oracle_abstain0",
    # "sr_ewaf_stratified_abstain0",
    # "sr_ewaf_learned_abstain0",
    # "sr_ewaf_oracle_abstain0",
    "sprt_active_k0.0_abstain0",
    # "sprt_active_k0.0_abstain1",
    "sr_active_k0.0_abstain0",
    # "sr_active_k0.0_abstain1",
    # "sprt_mixture_active_k0.0_abstain0",
    # "sprt_mixture_active_k0.0_abstain1",
    # "sr_mixture_active_k0.0_abstain0",
    # "sr_mixture_active_k0.0_abstain1",
    "sprt_ewaf_active_k0.0_abstain0",
    # "sprt_ewaf_active_k0.0_abstain1",
    "sr_ewaf_active_k0.0_abstain0",
    # "sr_ewaf_active_k0.0_abstain1",
]
METHOD_LABELS = {
    "sprt_stratified_abstain0": "Stratified",
    "sprt_learned_abstain0": "Pre-learned",
    "sprt_oracle_abstain0": "Oracle",
    # "sr_ewaf_stratified_abstain0":"Stratified-SR-LR-a",
    # "sr_ewaf_learned_abstain0":"Pre-learned-SR-LR-a",
    # "sr_ewaf_oracle_abstain0": "Oracle-SR-LR-a",
    "sprt_active_k0.0_abstain0": "LR",
    # "sprt_active_k0.0_abstain1": "LR-label",
    "sr_active_k0.0_abstain0": "SR-LR",
    # "sr_active_k0.0_abstain1": "SR-LR-label",
    # "sprt_mixture_active_k0.0_abstain0": "LR-m",
    # "sprt_mixture_active_k0.0_abstain1": "LR-m-label",
    # "sr_mixture_active_k0.0_abstain0": "SR-LR-m",
    # "sr_mixture_active_k0.0_abstain1": "SR-LR-m-label",
    "sprt_ewaf_active_k0.0_abstain0": "LR-UI",
    # "sprt_ewaf_active_k0.0_abstain1": "LR-a-label",
    "sr_ewaf_active_k0.0_abstain0": "SR-LR-UI",
    # "sr_ewaf_active_k0.0_abstain1": "SR-LR-a-label",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot distribution of rejection times across methods"
    )
    parser.add_argument(
        "--data-setting",
        action="append",
        nargs="+",
        required=True,
        metavar=("NAME", "CSV"),
        help="Data setting: name followed by one or more CSV files. Can be repeated 1-4 times.",
    )
    parser.add_argument(
        "--y-max",
        type=int,
        default=200
    )
    parser.add_argument(
        "--out-plot-power",
        type=str,
        required=True,
        help="Output plot file path for power plot",
    )
    parser.add_argument(
        "--out-legend",
        type=str,
        required=True,
        help="Output plot file path for legend",
    )
    parser.add_argument(
        "--out-plot-obs",
        type=str,
        required=True,
        help="Output plot file path for num observations plot",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Output csv file path",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default="method",
        help="Column to group by for comparison (default: method)",
    )
    parser.add_argument(
        "--null-setting",
        type=str,
        default="No failure mode",
        help="Display name of the null setting (uses auditor rejection outcome)",
    )
    args = parser.parse_args()

    return args


def load_and_concat_csvs(csv_files):
    """Load multiple CSVs and concatenate them."""
    dfs = []
    for f in csv_files:
        if path.exists(f):
            print(f)
            df = pd.read_csv(f)
            dfs.append(df)
    print(f"Loaded {len(dfs)}")
    return pd.concat(dfs, ignore_index=True)


def compute_power_data(df, group_col):
    """Compute rejection probabilities for each method."""
    # Identify auditor and robustness rows by label
    df["test_category"] = df["label"].apply(
        lambda x: "auditor" if "auditor" in x.lower() else "robustness"
    )

    # Add an experiment index within each method
    df["exp_idx"] = df.groupby([group_col, "test_category"]).cumcount()

    pivot_df = df.pivot_table(
        index=[group_col, "exp_idx"],
        columns="test_category",
        values="is_rejected",
        aggfunc="first",
    ).reset_index()

    # Compute rejection categories
    pivot_df["rejected_auditor"] = pivot_df["auditor"] & ~pivot_df["robustness"]
    pivot_df["rejected_robustness"] = pivot_df["robustness"] & ~pivot_df["auditor"]
    pivot_df["no_rejection"] = ~pivot_df["auditor"] & ~pivot_df["robustness"]

    # Compute probabilities for each method
    power_data = []
    for method in pivot_df[group_col].unique():
        method_df = pivot_df[pivot_df[group_col] == method]
        n = len(method_df)
        for outcome, col in [("Reject Auditor's", "rejected_auditor"), ("Reject Model's", "rejected_robustness"), ("Inconclusive", "no_rejection")]:
            count = int(method_df[col].sum())
            power_data.append({
                group_col: method,
                "outcome": outcome,
                "count": count,
                "probability": count / n,
            })

    return pd.DataFrame(power_data), df


def get_method_order(available_methods):
    """Return ordered list of methods that exist in the data."""
    return [m for m in METHOD_ORDER if m in available_methods]


def plot_power_subplot(power_df, group_col, setting_name, ax, group_order, palette, hide_xlabel=False, hide_ylabel=False):
    """Plot power bar chart for one setting."""
    outcome_order = ["Reject Model's", "Inconclusive", "Reject Auditor's"]

    # Map method names to display labels
    plot_df = power_df.copy().sort_values("outcome", key=pd.Series({v: i for i, v in enumerate(outcome_order)}).get)
    plot_df[group_col] = plot_df[group_col].map(lambda x: METHOD_LABELS.get(x, x))
    display_order = [METHOD_LABELS.get(m, m) for m in group_order]
    plot_df = plot_df.assign(probability=plot_df.groupby(group_col)["probability"].cumsum())
    print(plot_df, group_col)
    sns.barplot(
        data=plot_df,
        y=group_col,
        x="probability",
        hue="outcome",
        hue_order=reversed(outcome_order),
        palette=palette,
        order=display_order,
        ax=ax,
        dodge=False
    )
    # ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel(setting_name if setting_name and not hide_ylabel else "", fontsize=12)
    ax.set_xlabel("Probability", fontsize=12)
    ax.tick_params(labelbottom=not hide_xlabel, labelsize=10, length=2, color=SPINE_COLOR)
    ax.set_xlim(0, 1)
    ax.legend_.remove()
    ax.spines['bottom'].set_color(SPINE_COLOR)
    ax.spines['left'].set_color(SPINE_COLOR)
    sns.despine()

def plot_obs_subplot(df, group_col, setting_name, ax, group_order, palette, y_max, null_setting="No failure mode", hide_xlabel=False, hide_ylabel=False):
    """Plot num observations box plot for one setting."""
    obs_df = df[[group_col, "test_category", "num_observations", "is_rejected"]].copy()
    relevant_test_category = "auditor" if setting_name == null_setting else "robustness"
    relevant_outcome = "Reject Auditor's" if setting_name == null_setting else "Reject Model's"
    obs_df = obs_df[obs_df["test_category"] == relevant_test_category]
    obs_df["outcome"] = relevant_outcome

    # Map method names to display labels
    obs_df[group_col] = obs_df[group_col].map(lambda x: METHOD_LABELS.get(x, x))
    display_order = [METHOD_LABELS.get(m, m) for m in group_order]

    if len(obs_df) > 0:
        sns.boxplot(
            data=obs_df,
            y=group_col,
            x="num_observations",
            hue="outcome",
            hue_order=[relevant_outcome],
            palette=palette,
            order=display_order,
            ax=ax,
            width=0.6
        )
    ax.set_ylabel(setting_name if setting_name and not hide_ylabel else "", fontsize=12)
    ax.set_xlabel("number labelled", fontsize=12)
    ax.tick_params(labelbottom=not hide_xlabel, labelsize=10, length=2, color=SPINE_COLOR)
    if hide_ylabel:
        ax.set_yticklabels([])
    ax.xaxis.grid(visible=True, linestyle='--', color='#ddd')
    ax.set_xlim(0, y_max)
    sns.despine()
    if ax.legend_:
        ax.legend_.remove()
    ax.spines['bottom'].set_color(SPINE_COLOR)
    ax.spines['left'].set_color(SPINE_COLOR)

    return obs_df


def main():
    args = parse_args()

    # Load all data settings
    all_dfs = {}
    all_power_dfs = {}
    all_group_values = set()
    setting_names = []

    hspace = 0.15 # vertical space between plots

    for setting in args.data_setting:
        name = setting[0]
        setting_names.append(name)
        csv_files = setting[1:]
        df = load_and_concat_csvs(csv_files)
        assert len(df) > 0, f"No data found for setting {name}"
        power_df, df = compute_power_data(df, args.group_col)
        all_dfs[name] = df
        all_power_dfs[name] = power_df
        all_group_values.update(df[args.group_col].unique())

    # Use predefined method order, filtered to available methods
    group_order = get_method_order(all_group_values)
    n_settings = len(args.data_setting)

    # Define consistent color palette for outcomes
    palette = {
        "Reject Auditor's": sns.color_palette()[0],
        "Reject Model's": sns.color_palette()[1],
        "Inconclusive": '#ccc',
    }

    # Create power plot: 1 row x n_settings cols
    fig_power, axes_power = plt.subplots(
        nrows=n_settings, ncols=1, figsize=(3, 1.4 * n_settings), squeeze=False, sharex=True
    )
    for i, name in enumerate(setting_names):
        plot_power_subplot(
            all_power_dfs[name],
            args.group_col,
            name if len(setting_names) > 1 else "",
            axes_power[i, 0],
            group_order,
            palette,
            hide_xlabel=i != len(setting_names) - 1,
            hide_ylabel=False
        )
    handles, labels = axes_power[0, 0].get_legend_handles_labels()
    fig_leg = plt.figure(figsize=(4, 0.5))
    # Add legend to the center of the new figure
    fig_leg.legend(reversed(handles), reversed(labels), ncols=3, loc='center')
    fig_leg.savefig(args.out_legend, dpi=150, bbox_inches='tight')
    plt.close(fig_leg)
    fig_power.subplots_adjust(hspace=hspace)
    fig_power.savefig(args.out_plot_power, dpi=150, bbox_inches="tight")
    plt.close(fig_power)
    print(f"Saved power plot to {args.out_plot_power}")

    # Create observations plot: 1 row x n_settings cols
    fig_obs, axes_obs = plt.subplots(
        nrows=n_settings, ncols=1, figsize=(3, 1.4 * n_settings), squeeze=False, sharex=True
    )
    all_obs_dfs = []
    for i, name in enumerate(setting_names):
        obs_df = plot_obs_subplot(
            all_dfs[name],
            args.group_col,
            name if len(setting_names) > 1 else "",
            axes_obs[i, 0],
            group_order,
            palette,
            y_max=args.y_max,
            hide_xlabel=i != len(setting_names) - 1,
            hide_ylabel=False,
            null_setting=args.null_setting,
        )
        obs_df["data_setting"] = name
        all_obs_dfs.append(obs_df)
    # handles, labels = axes_obs[0, 0].get_legend_handles_labels()
    # fig_obs.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    plt.subplots_adjust(hspace=hspace)
    plt.savefig(args.out_plot_obs, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved observations plot to {args.out_plot_obs}")

    # Print summary statistics
    summary_stats_dfs = []
    for setting in args.data_setting:
        name = setting[0]
        print(f"\n=== {name} ===")
        print("\nRejection probabilities by method:")
        summ_df = all_power_dfs[name].pivot(index=args.group_col, columns="outcome", values=["count", "probability"])
        summ_df["setting"] = name
        print(summ_df)
        summary_stats_dfs.append(summ_df)
    pd.concat(summary_stats_dfs).to_csv(args.out_csv)

    if all_obs_dfs:
        combined_obs = pd.concat(all_obs_dfs, ignore_index=True)
        print("\nObservation statistics:")
        summary = combined_obs.groupby(["data_setting", args.group_col, "outcome"])["num_observations"].agg(
            ["count", "mean", "std", "median", "min", "max"]
        )
        print(summary)


if __name__ == "__main__":
    main()
