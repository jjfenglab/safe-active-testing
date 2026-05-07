"""IID (uniform random) sampler."""

import numpy as np

from src.samplers.base import BaseSampler


class IIDSampler(BaseSampler):
    """Uniform random sampler that selects candidates with equal probability."""
    sampler_name = "iid"

    def __init__(self, seed: int):
        """Initialize the IID sampler.

        Args:
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def _sample_impl(self, embeddings: np.ndarray, ids: np.ndarray) -> tuple[object, int]:
        """Select a random candidate uniformly.

        Args:
            embeddings: Array of shape (n_candidates, embedding_dim) - unused
            ids: Array of candidate identifiers

        Returns:
            Tuple of (sampled_id, subgroup_size) - subgroup is all candidates for IID
        """
        idx = self.rng.choice(len(ids))
        return idx, len(ids)

    def update(self, embedding: np.ndarray, label: int) -> None:
        """No-op for IID sampler (no state to update).

        Args:
            embedding: Embedding vector (unused)
            label: Binary label (unused)
        """
        pass
