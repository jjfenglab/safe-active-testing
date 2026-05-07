"""Base class for samplers."""

from abc import ABC, abstractmethod

import numpy as np


class BaseSampler(ABC):
    """Abstract base class for samplers used in sequential testing."""

    def __init__(self):
        """Initialize base sampler with tracking for previously sampled IDs."""
        self._sampled_ids: set = set()

    def sample(self, embeddings: np.ndarray, ids: np.ndarray) -> tuple[object, int]:
        """Select next sample from candidates, excluding previously sampled.

        Args:
            embeddings: Array of shape (n_candidates, embedding_dim)
            ids: Array of candidate identifiers

        Returns:
            Tuple of (sampled_id, subgroup_size) where:
            - sampled_id: The id of the sampled candidate
            - subgroup_size: Size of the subgroup from which the sample was drawn
        """
        # Filter out previously sampled IDs
        mask = np.array([id_ not in self._sampled_ids for id_ in ids])
        filtered_embeddings = embeddings[mask]
        filtered_ids = ids[mask]

        assert len(filtered_ids) > 0, "No unsampled candidates remaining"

        # Call subclass implementation
        sampled_id, subgroup_size = self._sample_impl(filtered_embeddings, filtered_ids)

        # Track the sampled ID
        self._sampled_ids.add(sampled_id)

        return sampled_id, subgroup_size

    @abstractmethod
    def _sample_impl(self, embeddings: np.ndarray, ids: np.ndarray) -> tuple[object, int]:
        """Select next sample from candidates (subclass implementation).

        Args:
            embeddings: Array of shape (n_candidates, embedding_dim)
            ids: Array of candidate identifiers (already filtered)

        Returns:
            Tuple of (sampled_id, subgroup_size) where:
            - sampled_id: The id of the sampled candidate
            - subgroup_size: Size of the subgroup from which the sample was drawn
        """
        pass

    @abstractmethod
    def update(self, embedding: np.ndarray, label: int) -> None:
        """Update internal state after observing a sample.

        Args:
            embedding: Embedding vector of the sampled candidate
            label: Binary label (0 or 1)
        """
        pass

    def get_training_summary(self) -> dict | None:
        """Return latest training info, or None if sampler has no model."""
        return None
