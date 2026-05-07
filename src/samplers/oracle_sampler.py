"""Oracle sampler that samples only from a specified stratum."""

import numpy as np

from src.samplers.base import BaseSampler


class OracleSampler(BaseSampler):
    """Sampler that samples only from a specific stratum (e.g., a known weak domain).

    This serves as an upper bound baseline - if you knew exactly which stratum
    had the worst performance, you would sample only from that stratum.

    Supports multiple stratification fields with AND logic across fields and OR logic within each field.
    E.g., (domain=chemistry OR domain=physics) AND (difficulty=hard)
    """

    def __init__(self, id_to_strata: dict[str, dict[str, str]], oracle_strata: dict[str, list[str]], seed: int):
        """Initialize the oracle sampler.

        Args:
            id_to_strata: Dict mapping question_id to dict of {field: value} (e.g., {"domain": "chemistry", "difficulty": "hard"})
            oracle_strata: Dict of {field: [values]} to match. OR logic within each field, AND logic across fields.
                           E.g., {"domain": ["chemistry", "physics"], "difficulty": ["hard"]}
            seed: Random seed for reproducibility
        """
        super().__init__()
        assert len(id_to_strata) > 0, "id_to_strata cannot be empty"
        assert len(oracle_strata) > 0, "oracle_strata cannot be empty"
        self.sampler_name = "oracle"
        self.id_to_strata = id_to_strata
        self.oracle_strata = oracle_strata
        self.rng = np.random.default_rng(seed)

        # Verify each oracle field/value exists in the data
        for field, values in oracle_strata.items():
            assert len(values) > 0, f"oracle_strata['{field}'] cannot be empty"
            values_in_data = set(strata.get(field) for strata in id_to_strata.values() if field in strata)
            for value in values:
                assert value in values_in_data, (
                    f"oracle_strata['{field}'] contains '{value}' not found in data. "
                    f"Available values for '{field}': {values_in_data}"
                )

    def _matches_oracle(self, id_: str) -> bool:
        """Check if a question matches oracle strata criteria.

        AND logic across fields, OR logic within each field.
        E.g., (domain in ["chemistry", "physics"]) AND (difficulty in ["hard"])
        """
        strata = self.id_to_strata.get(id_, {})
        return all(strata.get(field) in values for field, values in self.oracle_strata.items())

    def _sample_impl(self, embeddings: np.ndarray, ids: np.ndarray) -> tuple[object, int]:
        """Select a random candidate from the oracle stratum.

        If the oracle stratum is exhausted, falls back to uniform random sampling
        from remaining candidates.

        Args:
            embeddings: Array of shape (n_candidates, embedding_dim) - unused
            ids: Array of candidate identifiers

        Returns:
            Tuple of (sampled_id, subgroup_size) where subgroup is the oracle stratum
        """
        # Find candidates matching all oracle strata (AND logic)
        oracle_indices = [i for i, id_ in enumerate(ids) if self._matches_oracle(id_)]

        if oracle_indices:
            # Sample from oracle stratum
            subgroup_size = len(oracle_indices)
            chosen_idx = self.rng.choice(oracle_indices)
        else:
            # Fallback to uniform random if oracle stratum exhausted
            subgroup_size = len(ids)
            chosen_idx = self.rng.choice(len(ids))

        return chosen_idx, subgroup_size

    def update(self, embedding: np.ndarray, label: int) -> None:
        """No-op for oracle sampler (no state to update).

        Args:
            embedding: Embedding vector (unused)
            label: Binary label (unused)
        """
        pass
