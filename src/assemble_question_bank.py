"""Assemble question bank CSVs into a single CSV with standardized format."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from assemble_gpqa_data import compute_embeddings

def load_question_bank(
    data_dir: str,
    domains: list[str],
    difficulties: list[str],
) -> pd.DataFrame:
    """Load all question bank CSVs and combine into a single DataFrame.

    Args:
        data_dir: Directory containing {difficulty}_{domain}_questions.csv files
        domains: List of domains to load
        difficulties: List of difficulties to load

    Returns:
        DataFrame with question, domain, difficulty columns
    """
    data_dir = Path(data_dir)
    all_rows = []

    for difficulty in difficulties:
        for domain in domains:
            filename = f"{difficulty}_{domain}_questions.csv"
            filepath = data_dir / filename
            assert filepath.exists(), f"Question bank file not found: {filepath}"

            df = pd.read_csv(filepath)
            assert "question" in df.columns, f"Missing 'question' column in {filepath}"

            # Add domain/difficulty if not present
            df["domain"] = domain
            df["difficulty"] = difficulty
            all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)


def assemble_question_bank(
    data_dir: str,
    config_path: str,
    simulate_by: str = "domain",
) -> pd.DataFrame:
    """Assemble question bank into standardized format matching assemble_gpqa_data output.

    Args:
        data_dir: Directory containing question bank CSVs
        config_path: Path to JSON config with domains, difficulties, and correctness probs
        simulate_by: Field to use for simulation ("difficulty", "domain", or "domain_difficulty")

    Returns:
        DataFrame with question_id, question, is_correct_prob, metadata columns
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    domains = config["domains"]
    difficulties = config["difficulties"]
    correctness_probs_key = f"{simulate_by}_correctness_probs"
    assert correctness_probs_key in config, (
        f"Expected key '{correctness_probs_key}' in config, found: {list(config.keys())}"
    )
    correctness_probs = config[correctness_probs_key]

    # Load raw question bank
    raw_df = load_question_bank(data_dir, domains, difficulties)

    rows = []
    for idx, row in raw_df.iterrows():
        question = row["question"]
        domain = row["domain"]
        difficulty = row["difficulty"]

        # Look up probability based on simulate_by
        if simulate_by == "domain_difficulty":
            assert domain in correctness_probs, (
                f"Domain '{domain}' not found in config"
            )
            assert difficulty in correctness_probs[domain], (
                f"Difficulty '{difficulty}' not found for domain '{domain}' in config"
            )
            prob_correct = correctness_probs[domain][difficulty]
        elif simulate_by == "domain":
            assert domain in correctness_probs, (
                f"Domain '{domain}' not found in config"
            )
            prob_correct = correctness_probs[domain]
        else:  # difficulty
            assert difficulty in correctness_probs, (
                f"Difficulty '{difficulty}' not found in config"
            )
            prob_correct = correctness_probs[difficulty]

        # Create metadata JSON
        metadata = json.dumps({"domain": domain, "difficulty": difficulty})

        rows.append({
            "question_id": f"qb_{idx}_{domain}_{difficulty}",
            "question": question,
            "is_correct_prob": prob_correct,
            "metadata": metadata,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Assemble question bank into standardized format"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing question bank CSVs",
    )
    parser.add_argument(
        "--embed-question",
        action="store_true",
        default=False,
        help="whether to embed question",
    )
    parser.add_argument(
        "--embed-metadata",
        action="store_true",
        default=False,
        help="whether to embed metadata instead",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config with domains, difficulties, correctness probs",
    )
    parser.add_argument(
        "--simulate-by",
        type=str,
        default="domain",
        choices=["difficulty", "domain", "domain_difficulty"],
        help="Field to use for simulation: 'difficulty', 'domain', or 'domain_difficulty'",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        help="Output CSV path",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Sentence transformer model for computing embeddings. If provided, outputs embeddings CSV.",
    )
    parser.add_argument(
        "--out-embeddings-csv",
        type=str,
        default=None,
        help="Output path for embeddings CSV (required if --embedding-model is provided)",
    )
    args = parser.parse_args()

    df = assemble_question_bank(args.data_dir, args.config, args.simulate_by)
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"Wrote {len(df)} rows to {args.out_csv}")

    # Compute and save embeddings if requested
    if args.embedding_model is not None:
        assert args.out_embeddings_csv is not None, "--out-embeddings-csv required when --embedding-model is provided"
        embeddings = compute_embeddings(df, args.embedding_model, embed_metadata=args.embed_metadata)
        # Save embeddings with question_id as index for alignment
        embeddings_df = pd.DataFrame(
            embeddings,
            index=df["question_id"],
        )
        embeddings_df.index.name = "question_id"
        embeddings_df.to_csv(args.out_embeddings_csv)
        print(f"Wrote embeddings ({embeddings.shape}) to {args.out_embeddings_csv}")


if __name__ == "__main__":
    main()
