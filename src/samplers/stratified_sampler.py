"""Stratified sampler that samples uniformly across metadata strata."""

import numpy as np

from src.samplers.base import BaseSampler


class StratifiedSampler(BaseSampler):
    """Sampler that gives equal probability to each stratum, then samples uniformly within.

    For example, if stratifying by domain with 3 domains (Physics, Chemistry, Biology),
    each sample() call picks a domain uniformly at random (1/3 each), then picks a
    random item within that domain. When a stratum is exhausted, its probability is
    redistributed equally among remaining strata.
    """

    def __init__(self, id_to_stratum: dict[str, str], seed: int):
        """Initialize the stratified sampler.

        Args:
            id_to_stratum: Dict mapping question_id to stratum value (e.g., domain)
            seed: Random seed for reproducibility
        """
        super().__init__()
        assert len(id_to_stratum) > 0, "id_to_stratum cannot be empty"
        self.sampler_name = "stratified"
        self.id_to_stratum = id_to_stratum
        self.rng = np.random.default_rng(seed)

    def _sample_impl(self, embeddings: np.ndarray, ids: np.ndarray) -> tuple[object, int]:
        """Select a candidate using stratified sampling.

        First picks a stratum uniformly at random from available strata,
        then picks a random candidate within that stratum.

        Args:
            embeddings: Array of shape (n_candidates, embedding_dim) - unused
            ids: Array of candidate identifiers

        Returns:
            Tuple of (sampled_id, subgroup_size) where subgroup is the chosen stratum
        """
        # Group remaining ids by stratum
        strata: dict[str, list[int]] = {}
        for i, id_ in enumerate(ids):
            stratum = self.id_to_stratum.get(id_)
            assert stratum is not None, f"ID '{id_}' not found in id_to_stratum mapping"
            if stratum not in strata:
                strata[stratum] = []
            strata[stratum].append(i)

        # Pick a stratum uniformly at random
        stratum_names = list(strata.keys())
        chosen_stratum = self.rng.choice(stratum_names)

        # Pick a random candidate within that stratum
        candidates_in_stratum = strata[chosen_stratum]
        subgroup_size = len(candidates_in_stratum)
        chosen_idx = self.rng.choice(candidates_in_stratum)

        return chosen_idx, subgroup_size

    def update(self, embedding: np.ndarray, label: int) -> None:
        """No-op for stratified sampler (no state to update).

        Args:
            embedding: Embedding vector (unused)
            label: Binary label (unused)
        """
        pass
