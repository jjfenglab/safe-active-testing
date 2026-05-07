"""Main script for running e-testing on GPQA data with configurable samplers."""
import warnings
import argparse
import json
import logging
import sys
from pathlib import Path
import time

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.active_e_tester import ETesterSampler, ETesterGenerator
from src.history import SequentialTestHistory
from src.question_bank import QuestionBankSampler
from src.samplers import ActiveSampler, BaseSampler, IIDSampler, LearnedSampler, OracleSampler, StratifiedSampler
from src.sequential_tests import AdaptiveCombinedEDetector, BaseSequentialTest, CombinedEDetector, EWAFCombinedEDetector, MixtureCombinedEDetector


def load_and_shuffle_data(
    data_csv: str,
    embeddings_csv: str | None,
    rng: np.random.Generator,
    name: str = "data",
    id_col: str = "question_id",
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Load data CSV and optional embeddings, then shuffle both together.

    Args:
        data_csv: Path to the data CSV file
        embeddings_csv: Path to embeddings CSV file (optional)
        rng: NumPy random generator for shuffling
        name: Name for logging purposes
        id_col: Name of the ID column (default: "question_id")

    Returns:
        Tuple of (shuffled dataframe, shuffled embeddings array or None)
    """
    df = pd.read_csv(data_csv)
    assert len(df) > 0, f"{name} CSV is empty"

    # Load embeddings if provided
    embeddings = None
    if embeddings_csv is not None:
        embeddings_df = pd.read_csv(embeddings_csv, index_col=id_col)
        embeddings = embeddings_df.loc[df[id_col]].values
        assert len(embeddings) == len(df), (
            f"Embeddings count ({len(embeddings)}) != {name} count ({len(df)})"
        )

    # Shuffle both together
    shuffle_idx = rng.permutation(len(df))
    df = df.iloc[shuffle_idx].reset_index(drop=True)
    if embeddings is not None:
        embeddings = embeddings[shuffle_idx]

    logging.info(f"Loaded and shuffled {len(df)} {name} rows")
    if embeddings is not None:
        logging.info(f"Loaded pre-computed {name} embeddings with shape {embeddings.shape}")

    return df, embeddings


def create_sampler(args, embedding_dim: int, df: pd.DataFrame = None, id_col: str = "question_id") -> BaseSampler:
    """Create sampler based on command line arguments.

    Args:
        args: Parsed command line arguments
        embedding_dim: Dimension of embeddings
        df: DataFrame with question data (required for stratified sampler)
        id_col: Name of the ID column (default: "question_id")

    Returns:
        Initialized sampler
    """
    if args.sampler_type == "iid":
        return IIDSampler(seed=args.seed)
    elif args.sampler_type == "active":
        return ActiveSampler(
            embedding_dim=embedding_dim,
            num_nns=args.num_nns,
            hidden_dims=args.hidden_dims,
            epsilon=args.epsilon,
            num_epochs_list=args.num_epochs,
            learning_rate=args.learning_rate,
            l2_penalty=args.l2_penalty,
            seed=args.seed,
            k_std_dev=args.k_std_dev,
            target_threshold=args.null_prob if args.adaptive else args.alt_prob,
            class_weight_0=args.class_weight_0,
            class_weight_1=args.class_weight_1,
        )
    elif args.sampler_type == "stratified":
        assert df is not None, "df required for stratified sampler"
        assert args.stratify_by is not None, "--stratify-by required for stratified sampler"
        # Combine multiple fields into composite stratum (e.g., "chemistry_hard")
        id_to_stratum = {
            row[id_col]: "_".join(json.loads(row["metadata"])[field] for field in args.stratify_by)
            for _, row in df.iterrows()
        }
        return StratifiedSampler(id_to_stratum=id_to_stratum, seed=args.seed)
    elif args.sampler_type == "oracle":
        assert df is not None, "df required for oracle sampler"
        assert args.stratify_by is not None, "--stratify-by required for oracle sampler"
        assert args.oracle_stratum is not None, "--oracle-stratum required for oracle sampler"
        assert len(args.stratify_by) == len(args.oracle_stratum), (
            f"--stratify-by and --oracle-stratum must have same length: "
            f"{len(args.stratify_by)} vs {len(args.oracle_stratum)}"
        )
        id_to_strata = {
            row[id_col]: {field: json.loads(row["metadata"])[field] for field in args.stratify_by}
            for _, row in df.iterrows()
        }
        oracle_strata_dict = dict(zip(args.stratify_by, args.oracle_stratum))
        for k in oracle_strata_dict.keys():
            oracle_strata_dict[k] = oracle_strata_dict[k].split("+")
        return OracleSampler(
            id_to_strata=id_to_strata,
            oracle_strata=oracle_strata_dict,
            seed=args.seed,
        )
    elif args.sampler_type == "learned":
        return LearnedSampler(
            embedding_dim=embedding_dim,
            num_nns=args.num_nns,
            hidden_dims=args.hidden_dims,
            epsilon=args.epsilon,
            num_epochs_list=args.num_epochs,
            learning_rate=args.learning_rate,
            l2_penalty=args.l2_penalty,
            num_init=args.num_init,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown sampler type: {args.sampler_type}")

def write_result_csv(
    tester_name: str,
    test_type: str,
    sequential_test: BaseSequentialTest,
    alpha: float,
    result_csv_path: str,
    num_test_obs: int,
    num_label_only: int,
):
    """Write CSV with rejection results for aggregation across seeds.

    Args:
        tester_name: Identifier for the tester configuration (e.g., 'sr_active_abstain1')
        test_type: Type of test (e.g., 'robustness', 'auditor', 'dual')
        sequential_test: Sequential test with final state
        alpha: Significance level
        result_csv_path: Path to output CSV file
        num_test_obs: Number of observations that went through the sequential test
        num_label_only: Number of observations that only got labels (skipped test)
    """
    stat_history = sequential_test.get_statistic_history(alpha)

    rows = []
    for stat in stat_history:
        rows.append({
            "label": stat["label"],
            "method": tester_name,
            "test_type": test_type,
            "is_rejected": stat["is_rejected"],
            "num_observations": num_test_obs + num_label_only,
            "num_test_obs": num_test_obs,
            "num_label_only": num_label_only,
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(result_csv_path, index=False)


def plot_subgroup_histogram(history: SequentialTestHistory, plot_path: str):
    """Plot histogram of subgroup percentages.

    Args:
        history: SequentialTestHistory object with per-round information
        plot_path: Path to save the histogram plot
    """
    subgroup_pcts = history.to_dataframe()["subgroup_pct"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(subgroup_pcts, bins=10, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Subgroup Size (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Subgroup Sizes (Prevalences)")
    ax.axvline(np.mean(subgroup_pcts), color="red", linestyle="--", label=f"Mean: {np.mean(subgroup_pcts):.1f}%")
    ax.axvline(np.median(subgroup_pcts), color="orange", linestyle="--", label=f"Median: {np.median(subgroup_pcts):.1f}%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run sequential test")

    # General arguments
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--in-data-csv", type=str, default=None, help="Input CSV path (required for standard tester)")
    parser.add_argument("--in-embeddings-csv", type=str, default=None, help="Input embeddings CSV path (optional, skips embedding computation if provided)")
    parser.add_argument("--id-col", type=str, default="question_id", help="Name of ID column in data/embeddings CSVs (default: question_id)")
    parser.add_argument("--log", type=str, required=True, help="Output log file path")
    parser.add_argument("--history", type=str, required=True, help="Output history file path")
    parser.add_argument("--plot", type=str, required=False, help="Output plot file path")
    parser.add_argument("--result-csv", type=str, required=True, help="Output CSV with rejection results")
    parser.add_argument("--subgroup-plot", type=str, default=None, help="Output plot for subgroup size histogram (optional)")
    parser.add_argument("--debug-pred-dir", type=str, default=None, help="Directory to save per-iteration prediction CSVs (optional, enables debug mode)")
    parser.add_argument("--debug-sampler-dir", type=str, default=None, help="Directory to save per-iteration sampler pickles (optional, enables debug mode)")

    # E-detector arguments
    parser.add_argument("--method", type=str, default="sr", choices=["cusum", "sr", "sprt"], help="E-detector method")
    parser.add_argument("--null-prob", type=float, required=True, help="Null hypothesis probability")
    parser.add_argument("--alt-prob", type=float, required=True, help="Alternative hypothesis probability")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--max-samples", type=int, required=True, help="Maximum number of samples")
    parser.add_argument(
        "--test-type",
        type=str,
        default="dual",
        choices=["robustness", "auditor", "dual"],
        help="Which test(s) to run: 'robustness' only, 'auditor' only, or 'dual' (both)",
    )
    parser.add_argument(
        "--m-min",
        type=int,
        default=None,
        help="Minimum observations before auditor test starts. Required for 'auditor' and 'dual' test types.",
    )
    parser.add_argument(
        "--iter-require-test",
        type=int,
        default=None,
        help="At which iteration do we require testing and no longer allow label-only",
    )
    detector_group = parser.add_mutually_exclusive_group()
    detector_group.add_argument(
        "--adaptive",
        action="store_true",
        help="Use adaptive e-detector with dynamic alternative probabilities from sampler predictions.",
    )
    detector_group.add_argument(
        "--mixture",
        action="store_true",
        help="Use mixture e-detector with a fixed grid of candidate alternative probabilities.",
    )
    detector_group.add_argument(
        "--ewaf",
        action="store_true",
        help="Use EWAF e-detector with exponentially weighted average forecaster over a grid.",
    )
    parser.add_argument("--auditor-alt-prob", type=float, default=None, help="Auditor alternative probability (> null-prob); required for nonadaptive dual/auditor tests")
    parser.add_argument("--grid-low", type=float, default=None, help="Lower bound of robustness mixture grid (required with --mixture)")
    parser.add_argument("--grid-high", type=float, default=None, help="Upper bound of robustness mixture grid; must be < null-prob (required with --mixture)")
    parser.add_argument("--grid-step", type=float, default=None, help="Step size for robustness mixture grid (required with --mixture)")
    parser.add_argument("--auditor-grid-low", type=float, default=None, help="Lower bound of auditor mixture grid; must be > null-prob (required with --mixture)")
    parser.add_argument("--auditor-grid-high", type=float, default=None, help="Upper bound of auditor mixture grid (required with --mixture)")
    parser.add_argument("--auditor-grid-step", type=float, default=None, help="Step size for auditor mixture grid (required with --mixture)")
    parser.add_argument("--ewaf-learning-rate", type=float, default=None, help="Learning rate eta for EWAF weight updates (required with --ewaf)")
    parser.add_argument(
        "--no-pred-filter",
        action="store_true",
        help="Disable pred_prob filter: pass all observations to sequential test, not just predicted-bad ones.",
    )
    parser.add_argument(
        "--result-label-suffix",
        type=str,
        default="",
        help="Suffix appended to method name in result CSV (e.g., '_nofilter')",
    )

    # Sampler arguments
    parser.add_argument(
        "--sampler-type",
        type=str,
        default="active",
        choices=["iid", "active", "stratified", "oracle", "learned"],
        help="Type of sampler to use",
    )
    parser.add_argument(
        "--stratify-by",
        type=str,
        nargs="+",
        default=None,
        help="Metadata field(s) to stratify by (e.g., 'domain' or 'domain difficulty'). Required for stratified/oracle sampler.",
    )
    parser.add_argument(
        "--oracle-stratum",
        type=str,
        nargs="+",
        default=None,
        help="Stratum value(s) to sample from exclusively, matching --stratify-by fields (e.g., 'chemistry' or 'chemistry hard'). Required for oracle sampler.",
    )

    # Active sampler specific arguments (also used by RAG sampler for embedding model)
    parser.add_argument("--num-nns", type=int, default=5, help="Number of NNs in ensemble (active/rag sampler)")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32], help="Hidden layer dimensions (active/rag sampler)")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Percentile for sampling (active sampler only)")
    parser.add_argument("--k-std-dev", type=float, default=0.0, help="Multiplier for std dev in UCB-style selection (0 = greedy, active sampler only)")
    parser.add_argument("--num-epochs", type=int, nargs="+", default=[0, 10], help="Grid of training epoch values to tune over (active/rag sampler)")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate for Adam optimizer (active/rag sampler)")
    parser.add_argument(
        "--l2-penalty",
        type=float,
        default=0.0001,
        help="L2 penalty value for regularization (active/rag sampler)",
    )
    parser.add_argument(
        "--num-init",
        type=int,
        default=0,
        help="Number of random samples to initialize sampler (not passed to test)",
    )

    # Bayesian GP sampler specific arguments
    parser.add_argument(
        "--gp-latent-dim",
        type=int,
        default=2,
        help="Latent dimension for GP in Bayesian GP sampler",
    )
    parser.add_argument(
        "--gp-num-epochs",
        type=int,
        default=50,
        help="Number of training epochs for Bayesian GP sampler",
    )
    parser.add_argument(
        "--gp-num-inducing",
        type=int,
        default=50,
        help="Number of inducing points for Bayesian GP sampler",
    )

    # Class weight arguments for active sampler
    parser.add_argument(
        "--class-weight-0",
        type=float,
        default=1.0,
        help="Weight for Y=0 class in loss function (active sampler)",
    )
    parser.add_argument(
        "--class-weight-1",
        type=float,
        default=1.0,
        help="Weight for Y=1 class in loss function (active sampler)",
    )

    # RAG sampler specific arguments
    parser.add_argument(
        "--rag-num-candidates",
        type=int,
        default=10,
        help="Number of top candidates to present to LLM (rag sampler only)",
    )
    parser.add_argument(
        "--rag-context-k",
        type=int,
        default=3,
        help="Number of recent correct AND incorrect questions to show LLM (rag sampler only)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="sentence transformer model",
    )
    parser.add_argument(
        "--precomputed-embeddings",
        type=str,
        default=None,
        help="Path to precomputed embeddings .npy file. Skips embedding step if provided.",
    )
    parser.add_argument(
        "--rag-llm-model",
        type=str,
        default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        help="LLM model for RAG selection (rag sampler only)",
    )
    parser.add_argument(
        "--rag-llm-cache-path",
        type=str,
        default=None,
        help="Path to LLM cache database (rag sampler only)",
    )
    parser.add_argument(
        "--rag-prompt-template",
        type=str,
        default=None,
        help="Path to prompt template file (rag sampler only). Required for rag sampler.",
    )

    # Tester type arguments
    parser.add_argument(
        "--tester-type",
        type=str,
        default="standard",
        choices=["standard", "generator"],
        help="Type of tester: 'standard' selects from all candidates, 'generator' selects from sampled candidates each round",
    )
    parser.add_argument(
        "--question-bank-csv",
        type=str,
        default=None,
        help="Path to assembled question bank CSV (required for generator tester)",
    )
    parser.add_argument(
        "--question-bank-embeddings-csv",
        type=str,
        default=None,
        help="Path to question bank embeddings CSV (optional for generator tester, skips embedding computation if provided)",
    )
    parser.add_argument(
        "--sample-by",
        type=str,
        nargs="+",
        default=None,
        help="Metadata fields to stratify sampling by (e.g., 'domain difficulty'). Required for generator tester.",
    )
    parser.add_argument(
        "--allow-label-only",
        action="store_true",
        default=False,
        help="Allow auditor to choose label-only vs test (default: False, must test every observation)",
    )

    # Suppress this specific scikit-learn warning
    warnings.filterwarnings(
        "ignore", 
        message="Setting penalty=None will ignore the C and l1_ratio parameters"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(args.log),
    ])

    st_time = time.time()

    # Set seeds
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Load embedding model to get embedding dimension
    logging.info(f"Loading embedding model: {args.embedding_model}")
    embedding_model = None
    if args.embedding_model:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(args.embedding_model, local_files_only=True)
        embedding_dim = embedding_model.get_sentence_embedding_dimension()
        logging.info(f"Embedding dimension: {embedding_dim}")

    # Load data (only needed for standard tester)
    df = None
    precomputed_embeddings = None
    if args.tester_type == "standard":
        assert args.in_data_csv is not None, "--in-data-csv required for standard tester"
        df, precomputed_embeddings = load_and_shuffle_data(
            args.in_data_csv, args.in_embeddings_csv, rng, name="questions", id_col=args.id_col
        )
        if "is_correct" not in df.columns and "is_correct_prob" in df.columns:
            df["is_correct"] = np.random.binomial(n=1, p=df["is_correct_prob"], size=df.shape[0])
        if precomputed_embeddings is not None:
            embedding_dim = precomputed_embeddings.shape[1]

    # Load question bank data if provided
    qb_df = None
    qb_precomputed_embeddings = None
    if args.question_bank_csv is not None:
        qb_df, qb_precomputed_embeddings = load_and_shuffle_data(
            args.question_bank_csv, args.question_bank_embeddings_csv, rng, name="question bank", id_col=args.id_col
        )
        if qb_precomputed_embeddings is not None:
            embedding_dim = qb_precomputed_embeddings.shape[1]

    # Initialize combined e-detector
    if args.adaptive:
        sequential_test = AdaptiveCombinedEDetector(
            method=args.method,
            null_prob=args.null_prob,
            test_type=args.test_type,
            m_min=args.m_min,
            num_init=args.num_init,
        )
        logging.info("Using adaptive e-detector with dynamic alternative probabilities")
    elif args.mixture:
        assert all(v is not None for v in [args.grid_low, args.grid_high, args.grid_step,
                                            args.auditor_grid_low, args.auditor_grid_high, args.auditor_grid_step]), \
            "--grid-low/high/step and --auditor-grid-low/high/step are all required with --mixture"
        sequential_test = MixtureCombinedEDetector(
            method=args.method,
            null_prob=args.null_prob,
            grid_low=args.grid_low,
            grid_high=args.grid_high,
            grid_step=args.grid_step,
            auditor_grid_low=args.auditor_grid_low,
            auditor_grid_high=args.auditor_grid_high,
            auditor_grid_step=args.auditor_grid_step,
            test_type=args.test_type,
            m_min=args.m_min,
            num_init=args.num_init,
        )
        logging.info(f"Using mixture e-detector: rob grid [{args.grid_low}, {args.grid_high}] step {args.grid_step}, "
                     f"aud grid [{args.auditor_grid_low}, {args.auditor_grid_high}] step {args.auditor_grid_step}")
    elif args.ewaf:
        assert all(v is not None for v in [args.grid_low, args.grid_high, args.grid_step,
                                            args.auditor_grid_low, args.auditor_grid_high, args.auditor_grid_step,
                                            args.ewaf_learning_rate]), \
            "--grid-low/high/step, --auditor-grid-low/high/step, and --ewaf-learning-rate are all required with --ewaf"
        sequential_test = EWAFCombinedEDetector(
            method=args.method,
            null_prob=args.null_prob,
            grid_low=args.grid_low,
            grid_high=args.grid_high,
            grid_step=args.grid_step,
            auditor_grid_low=args.auditor_grid_low,
            auditor_grid_high=args.auditor_grid_high,
            auditor_grid_step=args.auditor_grid_step,
            ewaf_learning_rate=args.ewaf_learning_rate,
            test_type=args.test_type,
            m_min=args.m_min,
            num_init=args.num_init,
        )
        logging.info(f"Using EWAF e-detector: rob grid [{args.grid_low}, {args.grid_high}] step {args.grid_step}, "
                     f"aud grid [{args.auditor_grid_low}, {args.auditor_grid_high}] step {args.auditor_grid_step}, "
                     f"learning_rate={args.ewaf_learning_rate}")
    else:
        sequential_test = CombinedEDetector(
            method=args.method,
            null_prob=args.null_prob,
            alt_prob=args.alt_prob,
            auditor_alt_prob=args.auditor_alt_prob,
            test_type=args.test_type,
            m_min=args.m_min,
            num_init=args.num_init,
        )
    if args.test_type == "dual":
        logging.info(f"Using dual robustness/auditor test with m_min={args.m_min}")
    elif args.test_type == "robustness":
        logging.info("Using robustness-only test")
    else:
        logging.info(f"Using auditor-only test with m_min={args.m_min}")

    # Initialize sampler
    sampler = create_sampler(args, embedding_dim, df=df if df is not None else qb_df, id_col=args.id_col)
    if args.sampler_type == "stratified":
        logging.info(f"Using {args.sampler_type} sampler (stratify-by={args.stratify_by})")
    elif args.sampler_type == "oracle":
        oracle_criteria = ", ".join(f"{f}={v}" for f, v in zip(args.stratify_by, args.oracle_stratum))
        logging.info(f"Using {args.sampler_type} sampler ({oracle_criteria})")
    elif args.sampler_type == "rag":
        logging.info(f"Using {args.sampler_type} sampler (num_candidates={args.rag_num_candidates}, context_k={args.rag_context_k}, llm={args.rag_llm_model})")
    elif args.sampler_type == "learned":
        logging.info(f"Using {args.sampler_type} sampler (learned_num_init={args.num_init})")
    elif args.sampler_type == "bayesian_gp":
        logging.info(f"Using {args.sampler_type} sampler (latent_dim={args.gp_latent_dim}, num_epochs={args.gp_num_epochs}, num_inducing={args.gp_num_inducing})")
    else:
        logging.info(f"Using {args.sampler_type} sampler")

    # Run sequential testing
    logging.info("Starting sequential testing...")
    logging.info(args)

    if args.tester_type == "generator":
        # Generator mode: sample from question bank each round
        assert args.question_bank_csv is not None, "--question-bank-csv required for generator tester"
        assert args.sample_by is not None, "--sample-by required for generator tester"

        question_bank_sampler = QuestionBankSampler(
            qb_df=qb_df,
            sample_by=args.sample_by,
            seed=args.seed,
            precomputed_embeddings=qb_precomputed_embeddings,
        )

        tester = ETesterGenerator(
            question_bank_sampler=question_bank_sampler,
            sequential_test=sequential_test,
            sampler=sampler,
            alpha=args.alpha,
            max_samples=args.max_samples,
            num_init=args.num_init,
            rng=rng,
            adaptive=args.adaptive,
            iter_require_auditor_test=args.iter_require_test,
            allow_label_only=args.allow_label_only,
            debug_pred_dir=args.debug_pred_dir,
            debug_sampler_dir=args.debug_sampler_dir,
        )
    else:
        # Standard mode: select from all candidates
        tester = ETesterSampler(
            df=df,
            embedding_model=embedding_model,
            sequential_test=sequential_test,
            sampler=sampler,
            alpha=args.alpha,
            max_samples=args.max_samples,
            num_init=args.num_init,
            rng=rng,
            adaptive=args.adaptive,
            iter_require_auditor_test=args.iter_require_test,
            allow_label_only=args.allow_label_only,
            precomputed_embeddings=precomputed_embeddings,
            debug_pred_dir=args.debug_pred_dir,
            debug_sampler_dir=args.debug_sampler_dir,
            id_col=args.id_col,
        )

    history = tester.run()
    logging.info(f"Observations tested: {history.num_test_obs}, label-only: {history.num_label_only}")

    # Write outputs
    history.to_csv(args.history)
    logging.info(f"Wrote history CSV to {args.history}")

    if args.plot:
        sequential_test.plot(args.alpha, args.plot)
        logging.info(f"Wrote plot to {args.plot}")

    if args.subgroup_plot:
        plot_subgroup_histogram(history, args.subgroup_plot)
        logging.info(f"Wrote subgroup histogram to {args.subgroup_plot}")

    write_result_csv(
        tester.name,
        args.test_type,
        sequential_test,
        args.alpha,
        args.result_csv,
        history.num_test_obs,
        history.num_label_only,
    )
    logging.info(f"Wrote result CSV to {args.result_csv}")
    logging.info(f"Finish time {time.time() - st_time}")


if __name__ == "__main__":
    main()
