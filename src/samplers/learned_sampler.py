"""Learned sampler that stops updating after initialization."""
import logging
import numpy as np

from src.samplers.active_sampler import ActiveSampler


class LearnedSampler(ActiveSampler):
    """Sampler that learns from initial observations but stops updating thereafter.

    Inherits from ActiveSampler but only updates the neural network ensemble
    during the first num_init observations. After that, the model is frozen
    and used for sampling without further updates.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_nns: int,
        hidden_dims: list[int],
        epsilon: float,
        num_epochs_list: list[int],
        learning_rate: float,
        l2_penalty: float,
        num_init: int,
        seed: int,
    ):
        """Initialize the learned sampler.

        Args:
            embedding_dim: Dimension of input embeddings
            num_nns: Number of neural networks in the ensemble
            hidden_dims: List of hidden dimension values to tune over (each creates a single hidden layer)
            epsilon: Percentile threshold (e.g., 0.1 means bottom 10%)
            num_epochs_list: Grid of training epoch values to tune over
            learning_rate: Learning rate for Adam optimizer
            l2_penalty: L2 penalty value for regularization
            num_init: Number of initial observations to train on before freezing
            seed: Random seed for reproducibility
        """
        assert num_init > 0
        super().__init__(
            embedding_dim=embedding_dim,
            num_nns=num_nns,
            hidden_dims=hidden_dims,
            epsilon=epsilon,
            num_epochs_list=num_epochs_list,
            learning_rate=learning_rate,
            l2_penalty=l2_penalty,
            seed=seed,
        )
        self.sampler_name = "learned"
        self.num_init = num_init
        self.num_observations = 0

    def update(self, embedding: np.ndarray, label: int) -> None:
        """Update the ensemble with a new observation, only if within num_init.

        Args:
            embedding: Embedding vector of shape (embedding_dim,)
            label: Binary label (0 or 1)
        """
        self.num_observations += 1
        if self.num_observations <= self.num_init:
            logging.info(f"fitting learned model on num obs {self.num_observations}")
            super().update(embedding, label)
