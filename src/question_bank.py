"""Question bank sampler for mocking LLM question generation."""

import json
import logging

import numpy as np
import pandas as pd


class QuestionBankSampler:
    """Samples questions from a pre-assembled question bank.

    Loads an assembled CSV (with question_id, question, is_correct, metadata columns)
    and provides methods to sample N×K matrices of questions (one per category cell).
    Sampling is without replacement across rounds.
    """

    def __init__(
        self,
        qb_df: pd.DataFrame,
        sample_by: list[str],
        seed: int,
        precomputed_embeddings: np.ndarray | None = None,
    ):
        """Initialize the question bank sampler.

        Args:
            sample_by: List of metadata fields to stratify sampling by (e.g., ["domain", "difficulty"])
            seed: Random seed for reproducibility
            precomputed_embeddings: Optional pre-computed embeddings array aligned with CSV rows.
                                    If provided, sample_candidates returns embeddings.
        """
        self.sample_by = sample_by
        self.rng = np.random.default_rng(seed)
        self.precomputed_embeddings = precomputed_embeddings

        # Load and organize question bank
        self.df = qb_df

        if precomputed_embeddings is not None:
            assert len(precomputed_embeddings) == len(self.df), (
                f"Embeddings length ({len(precomputed_embeddings)}) != CSV length ({len(self.df)})"
            )
        assert "question_id" in self.df.columns, "Missing 'question_id' column"
        assert "question" in self.df.columns, "Missing 'question' column"
        assert "is_correct_prob" in self.df.columns, "Missing 'is_correct_prob' column"
        assert "metadata" in self.df.columns, "Missing 'metadata' column"

        # Parse metadata and extract sample_by fields
        self.df["_parsed_metadata"] = self.df["metadata"].apply(json.loads)
        for field in sample_by:
            self.df[f"__{field}"] = self.df["_parsed_metadata"].apply(lambda m, f=field: m.get(f))

        # Get unique categories for each sample_by field
        self.categories = {
            field: sorted(self.df[f"__{field}"].unique().tolist())
            for field in sample_by
        }

        # Build cell keys (tuples of category values)
        self._build_cell_indices()

        logging.info(
            f"Loaded {len(self.df)} questions with categories: "
            f"{', '.join(f'{k}={v}' for k, v in self.categories.items())}"
        )

    def _build_cell_indices(self):
        """Build index mapping from cell key to list of row indices."""
        self.cell_indices: dict[tuple, list[int]] = {}
        self.sampled_indices: dict[tuple, set[int]] = {}

        # Generate all cell keys
        from itertools import product
        cell_values = [self.categories[field] for field in self.sample_by]
        all_cells = list(product(*cell_values))

        for cell_key in all_cells:
            # Find rows matching this cell
            mask = np.ones(len(self.df), dtype=bool)
            for field, value in zip(self.sample_by, cell_key):
                mask &= (self.df[f"__{field}"] == value)

            indices = self.df.index[mask].tolist()
            self.cell_indices[cell_key] = indices
            self.sampled_indices[cell_key] = set()

    def sample_candidates(self) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Sample one question from each category cell.

        Returns:
            Tuple of (candidates_df, embeddings) where:
            - candidates_df: DataFrame with columns question_id, question, is_correct, is_correct_prob, metadata
            - embeddings: Array of embeddings for sampled candidates, or None if no precomputed embeddings

        Raises:
            AssertionError: If any cell has no remaining questions
        """
        candidates = []
        sampled_df_indices = []

        for cell_key, indices in self.cell_indices.items():
            sampled = self.sampled_indices[cell_key]

            # Get available indices
            available = [i for i in indices if i not in sampled]
            assert len(available) > 0, f"No remaining questions for cell {cell_key}"

            # Sample one question
            idx = self.rng.choice(available)
            self.sampled_indices[cell_key].add(idx)
            sampled_df_indices.append(idx)

            row = self.df.loc[idx]
            candidates.append({
                "question_id": row["question_id"],
                "question": row["question"],
                "is_correct": np.random.binomial(n=1, p=row["is_correct_prob"]),
                "is_correct_prob": row["is_correct_prob"],
                "metadata": row["metadata"],
            })

        embeddings = None
        if self.precomputed_embeddings is not None:
            embeddings = self.precomputed_embeddings[sampled_df_indices]

        return pd.DataFrame(candidates), embeddings

    def get_remaining_counts(self) -> dict[tuple, int]:
        """Get the number of remaining questions per cell."""
        return {
            cell_key: len(indices) - len(self.sampled_indices[cell_key])
            for cell_key, indices in self.cell_indices.items()
        }

    def has_candidates(self) -> bool:
        """Check if there are remaining questions in all cells."""
        for cell_key, indices in self.cell_indices.items():
            if len(self.sampled_indices[cell_key]) >= len(indices):
                return False
        return True

    @property
    def num_candidates_per_round(self) -> int:
        """Number of candidates sampled per round."""
        return len(self.cell_indices)
