"""Assemble GPQA data for a specific model into a CSV for active testing."""

import argparse
import json
import re
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_simulation_config(
    config_path: str, simulate_by: str
) -> dict[str, float] | dict[str, dict[str, float]]:
    """Load simulation config mapping category to correctness probabilities.

    Args:
        config_path: Path to JSON config file
        simulate_by: Which field to simulate by ("difficulty", "domain", or "domain_difficulty")

    Returns:
        For "difficulty" or "domain": Dict mapping category string to probability
        For "domain_difficulty": Nested dict mapping domain -> difficulty -> probability
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    key = f"{simulate_by}_correctness_probs"
    assert key in config, f"Expected key '{key}' in config, found: {list(config.keys())}"
    return config[key]


def normalize_text(text: str) -> str:
    """Normalize text for matching by removing extra whitespace and lowercasing."""
    return re.sub(r"\s+", " ", text.lower().strip())


def load_metadata(metadata_path: str) -> dict[str, dict]:
    """Load metadata from GPQA CSV and create lookup by normalized question text.

    Args:
        metadata_path: Path to gpqa_main.csv

    Returns:
        Dict mapping normalized question text to {"domain": ..., "difficulty": ...}
    """
    df = pd.read_csv(metadata_path)
    metadata_lookup = {}
    for _, row in df.iterrows():
        question_text = str(row["Question"])
        normalized = normalize_text(question_text)
        metadata_lookup[normalized] = {
            "domain": row["High-level domain"],
            "difficulty": row["Writer's Difficulty Estimate"],
        }
    return metadata_lookup


def load_gpqa_data(
    data_path: str,
    metadata_path: str,
    model: str,
    simulation_config: Optional[dict[str, float] | dict[str, dict[str, float]]] = None,
    simulate_by: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Load GPQA data and filter for a specific model.

    Args:
        data_path: Path to the GPQA JSON file
        metadata_path: Path to the GPQA metadata CSV file
        model: Model ID to filter responses for
        simulation_config: Optional config for simulating is_correct labels.
            For "difficulty"/"domain": dict mapping category to probability.
            For "domain_difficulty": nested dict mapping domain -> difficulty -> probability.
        simulate_by: Field to use for simulation ("difficulty", "domain", or "domain_difficulty").
            Required if simulation_config is provided.
        rng: Random number generator for simulation (required if simulation_config provided)

    Returns:
        DataFrame with question_id, question, candidate_answer, is_correct_prob, metadata
    """
    if simulation_config is not None:
        assert rng is not None, "rng required when simulation_config is provided"
        assert simulate_by is not None, "simulate_by required when simulation_config is provided"

    with open(data_path, "r") as f:
        data = json.load(f)

    metadata_lookup = load_metadata(metadata_path)

    rows = []
    for question_id, question_data in data.items():
        models = question_data.get("models", {})
        assert model in models, f"Model {model} not found in question {question_id}"

        question_text = question_data["input_text"]
        normalized_question = normalize_text(question_text)

        # Look up metadata
        metadata = metadata_lookup.get(normalized_question, {"domain": None, "difficulty": None})

        model_response = models[model]

        # Determine is_correct label
        if simulate_by == "domain_difficulty":
            domain = metadata["domain"]
            difficulty = metadata["difficulty"]
            assert domain in simulation_config, (
                f"Domain '{domain}' not found in simulation config"
            )
            if difficulty in simulation_config[domain]:
                prob_correct = simulation_config[domain][difficulty]
            else:
                prob_correct = 0.9
        else:
            category = metadata[simulate_by]
            assert category in simulation_config, (
                f"Category '{category}' not found in simulation config for {simulate_by}"
            )
            prob_correct = simulation_config[category]

        rows.append(
            {
                "question_id": question_id,
                "question": question_text,
                "candidate_answer": model_response["answer"],
                "is_correct_prob": prob_correct,
                "metadata": json.dumps(metadata),
            }
        )

    return pd.DataFrame(rows)

def compute_embeddings(
    df: pd.DataFrame,
    embedding_model_name: str,
    embed_metadata: bool = False,
    embed_question: bool = True,
) -> np.ndarray:
    """Compute embeddings for questions in dataframe.

    Args:
        df: DataFrame with 'question' column
        embedding_model_name: Name of sentence-transformers model

    Returns:
        Array of embeddings with shape (num_questions, embedding_dim)
    """
    embedding_model = SentenceTransformer(embedding_model_name, local_files_only=True)
    if embed_metadata:
        metadata_list = []
        metadata_categorical = []
        for metadata_dict in df["metadata"].apply(json.loads):
            if isinstance(metadata_dict['difficulty'], str):
                diff_str = metadata_dict['difficulty'].split(" (")[0]
            else:
                diff_str = "Post-graduate level or harder"
            metadata_str = f"Domain: {metadata_dict['domain']}, Difficulty: {diff_str}"
            metadata_list.append(metadata_str)
        
        print(pd.Series(metadata_list).value_counts())
        print("NUM OBS", len(metadata_list))

        if not embed_question:
            # embeddings = np.array(metadata_categorical)
            embeddings = embedding_model.encode(metadata_list, show_progress_bar=True)
        else:
            # embed both
            metadata_question_list = [f"{metadata_str}, Question: {question_str}" for metadata_str, question_str in zip(metadata_list, df['question'].tolist())]
            embeddings = embedding_model.encode(metadata_question_list, show_progress_bar=True)
    else:
        embeddings = embedding_model.encode(df["question"].tolist(), show_progress_bar=True)
    return np.array(embeddings)


def main():
    parser = argparse.ArgumentParser(
        description="Assemble GPQA data for active testing"
    )
    parser.add_argument("--data-path", type=str, required=True, help="Path to GPQA JSON file")
    parser.add_argument("--metadata-path", type=str, required=True, help="Path to GPQA metadata CSV file")
    parser.add_argument("--model", type=str, required=True, help="Model ID to filter")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for shuffling")
    parser.add_argument("--out-csv", type=str, help="Output CSV path")
    parser.add_argument(
        "--embed-metadata",
        action="store_true",
        default=False,
        help="whether to embed metadata instead",
    )
    parser.add_argument(
        "--embed-question",
        action="store_true",
        default=False,
        help="whether to embed question",
    )
    parser.add_argument(
        "--simulate-labels",
        type=str,
        default=None,
        help="Path to JSON config for simulating is_correct labels",
    )
    parser.add_argument(
        "--simulate-by",
        type=str,
        default="difficulty",
        choices=["difficulty", "domain", "domain_difficulty"],
        help="Field to use for simulation: 'difficulty', 'domain', or 'domain_difficulty'",
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

    rng = np.random.default_rng(args.seed)

    simulation_config = None
    if args.simulate_labels is not None:
        simulation_config = load_simulation_config(args.simulate_labels, args.simulate_by)

    df = load_gpqa_data(
        args.data_path,
        args.metadata_path,
        args.model,
        simulation_config=simulation_config,
        simulate_by=args.simulate_by if simulation_config else None,
        rng=rng,
    )
    assert len(df) > 0, "No data found for the specified model"
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"Wrote {len(df)} rows to {args.out_csv}")

    # Compute and save embeddings if requested
    if args.embedding_model is not None:
        assert args.out_embeddings_csv is not None, "--out-embeddings-csv required when --embedding-model is provided"
        embeddings = compute_embeddings(df, args.embedding_model, embed_metadata=args.embed_metadata, embed_question=args.embed_question)
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
