"""Active sampler using an ensemble of neural networks."""

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LogisticRegression

from src.common import to_safe_prob
from src.samplers.base import BaseSampler


class InterceptOnlyLR(nn.Module):
    """Logistic regression with only an intercept term (no input features)."""

    def __init__(self, seed: int):
        """Initialize intercept-only logistic regression.

        Args:
            seed: Random seed for initialization
        """
        super().__init__()
        torch.manual_seed(seed)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns sigmoid of bias for all inputs.

        Args:
            x: Input tensor of shape (batch_size, input_dim) - ignored

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        batch_size = x.shape[0]
        return torch.sigmoid(self.bias.expand(batch_size, 1))


class MLP(nn.Module):
    """Simple MLP for predicting from embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        seed: int,
        output_dim: int = 1,
        output_activation: str | None = "sigmoid",
    ):
        """Initialize MLP.

        Args:
            input_dim: Dimension of input embeddings
            hidden_dims: List of hidden layer dimensions
            seed: Random seed for weight initialization
            output_dim: Dimension of output (default 1 for binary classification)
            output_activation: Activation for output layer ("sigmoid", "relu", or None)
        """
        super().__init__()
        torch.manual_seed(seed)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            if hidden_dim > 0:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "relu":
            layers.append(nn.ReLU())
        elif output_activation is not None:
            raise ValueError(f"Unknown output_activation: {output_activation}")

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)


class SklearnLRWrapper:
    """Wrapper around sklearn LogisticRegression to match the MLP interface."""
    _default_prob = 1
    eps = 0.05

    def __init__(self, l2_penalty: float, seed: int):
        self.l2_penalty = l2_penalty
        self.seed = seed
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        if np.unique(y).size < 2:
            return
        # C is inverse of regularization strength
        # C = 1.0 / max(self.l2_penalty, 1e-10)
        self.model = LogisticRegression(
            C=np.inf, solver="lbfgs", max_iter=1000, random_state=self.seed
        )
        self.model.fit(X, y, sample_weight=sample_weight)

    def eval(self):
        pass

    def train(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            return torch.full((x.shape[0], 1), self._default_prob)
        probs = self.model.predict_proba(x.numpy())[:, 1]
        return np.clip(
            torch.tensor(probs, dtype=torch.float32).unsqueeze(-1),
            a_min=self.eps,
            a_max=1-self.eps)


class NNEnsemble:
    """Neural network ensemble for predicting accuracy from embeddings.

    Manages multiple MLP models with online tuning over hidden_dims and num_epochs.
    Shared by ActiveSampler and RAGSampler.

    When hidden_dim=0, uses sklearn LogisticRegression instead of MLP for faster training.
    """
    past_weight = 0.98

    def __init__(
        self,
        embedding_dim: int,
        num_nns: int,
        hidden_dims: list[int],
        num_epochs_list: list[int],
        learning_rate: float,
        l2_penalty: float,
        seed: int,
        batch_size: int = 32,
        class_weight_0: float = 1.0,
        class_weight_1: float = 1.0,
    ):
        """Initialize the ensemble.

        Args:
            embedding_dim: Dimension of input embeddings
            num_nns: Number of neural networks in the ensemble
            hidden_dims: List of hidden dimension values to tune over (each creates a single hidden layer).
                         If hidden_dim=0, uses sklearn LogisticRegression instead of MLP.
            num_epochs_list: Grid of training epoch values to tune over
            learning_rate: Learning rate for Adam optimizer
            l2_penalty: L2 penalty value for regularization
            seed: Random seed for reproducibility
            batch_size: Minibatch size for training
            class_weight_0: Weight for Y=0 class in loss function
            class_weight_1: Weight for Y=1 class in loss function
        """
        self.embedding_dim = embedding_dim
        self.num_nns = num_nns
        self.hidden_dims = hidden_dims
        self.num_epochs_list = num_epochs_list
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty
        self.seed = seed
        self.batch_size = batch_size
        self.class_weight_0 = class_weight_0
        self.class_weight_1 = class_weight_1

        # Build tuning grid: (hidden_dim, num_epochs) combinations
        # For num_epochs=0, hidden_dim is ignored (intercept-only model)
        # For hidden_dim=0 with num_epochs>0, uses sklearn LogisticRegression
        self.tuning_grid = []
        for hidden_dim in hidden_dims:
            for num_epochs in num_epochs_list:
                self.tuning_grid.append((hidden_dim, num_epochs))

        # Initialize ensemble for each grid point
        self.ensembles = []
        for hidden_dim, num_epochs in self.tuning_grid:
            if num_epochs == 0:
                self.ensembles.append([InterceptOnlyLR(seed=seed)])
            elif hidden_dim == 0:
                # Use sklearn LogisticRegression for hidden_dim=0
                self.ensembles.append(
                    [SklearnLRWrapper(l2_penalty=l2_penalty, seed=seed + i) for i in range(num_nns)]
                )
            else:
                self.ensembles.append(
                    [MLP(embedding_dim, [hidden_dim], seed=seed + i) for i in range(num_nns)]
                )

        # Running log likelihood for each grid point (lower is better)
        self.running_loss = np.zeros(len(self.tuning_grid))

        # Pre-allocated storage for training data
        self._initial_capacity = 512
        self._embedding_buffer = np.empty((self._initial_capacity, embedding_dim), dtype=np.float32)
        self._label_buffer = np.empty(self._initial_capacity, dtype=np.float32)
        self._n_obs = 0

        # Latest training info for progress reporting
        self.last_training_info: dict | None = None

    def _get_best_idx(self) -> int:
        """Get the index of the tuning grid point with lowest running log likelihood."""
        return int(np.argmin(self.running_loss))

    def predict_with_ensemble(
        self, embeddings: np.ndarray, ensemble: list[MLP], return_mean: bool = True
    ) -> np.ndarray:
        """Get mean predicted accuracy across a specific ensemble.

        Args:
            embeddings: Array of shape (n_candidates, embedding_dim)
            ensemble: List of MLP models

        Returns:
            Mean predictions of shape (n_candidates,)
        """
        x = torch.tensor(embeddings, dtype=torch.float32)
        predictions = []
        for model in ensemble:
            model.eval()
            with torch.no_grad():
                pred = model.forward(x).squeeze(-1).numpy()
            predictions.append(pred)
        if return_mean:
            return to_safe_prob(np.mean(predictions, axis=0))
        else:
            return predictions

    def predict_with_std(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get mean and std of predicted accuracy across best ensemble.

        Args:
            embeddings: Array of shape (n_candidates, embedding_dim)

        Returns:
            Tuple of (mean predictions, std predictions), each of shape (n_candidates,)
        """
        best_idx = self._get_best_idx()
        ensemble = self.ensembles[best_idx]

        predictions = self.predict_with_ensemble(embeddings, ensemble, return_mean=False)
        return to_safe_prob(np.mean(predictions, axis=0)), np.std(predictions, axis=0)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Get mean predicted accuracy across best ensemble.

        Args:
            embeddings: Array of shape (n_candidates, embedding_dim)

        Returns:
            Mean predictions of shape (n_candidates,)
        """
        best_idx = self._get_best_idx()
        return self.predict_with_ensemble(embeddings, self.ensembles[best_idx])

    def update(self, embedding: np.ndarray, label: int) -> None:
        """Update the ensemble with a new observation.

        Args:
            embedding: Embedding vector of shape (embedding_dim,)
            label: Binary label (0 or 1)
        """
        # Compute online log likelihood before training (predictive loss on new point)
        embedding_batch = embedding.reshape(1, -1)
        for epochs_idx, ensemble in enumerate(self.ensembles):
            pred = self.predict_with_ensemble(embedding_batch, ensemble)[0]
            loss = (-label * np.log(pred) - (1 - label) * np.log(1 - pred)) * (self.class_weight_0 * (label == 0) + self.class_weight_1 * (label == 1))
            self.running_loss[epochs_idx] = (1 - self.past_weight) * loss + self.past_weight * self.running_loss[epochs_idx]

        best_idx = self._get_best_idx()
        best_hidden_dim, best_num_epochs = self.tuning_grid[best_idx]
        logging.info(
            f"Tuning: best=(hidden_dim={best_hidden_dim}, num_epochs={best_num_epochs}), "
            f"running_loss={self.running_loss}"
        )

        # Store observation in pre-allocated buffer
        if self._n_obs >= len(self._embedding_buffer):
            new_capacity = len(self._embedding_buffer) * 2
            new_emb = np.empty((new_capacity, self.embedding_dim), dtype=np.float32)
            new_emb[:self._n_obs] = self._embedding_buffer[:self._n_obs]
            self._embedding_buffer = new_emb
            new_lab = np.empty(new_capacity, dtype=np.float32)
            new_lab[:self._n_obs] = self._label_buffer[:self._n_obs]
            self._label_buffer = new_lab
        self._embedding_buffer[self._n_obs] = embedding
        self._label_buffer[self._n_obs] = label
        self._n_obs += 1

        # Prepare training data from buffer slices with per-sample weights
        X = torch.from_numpy(self._embedding_buffer[:self._n_obs].copy())
        y = torch.from_numpy(self._label_buffer[:self._n_obs].copy()).unsqueeze(-1)
        # Compute per-sample weights: weight = class_weight_1 * y + class_weight_0 * (1 - y)
        sample_weights = (self.class_weight_1 * y + self.class_weight_0 * (1 - y))
        dataset = TensorDataset(X, y, sample_weights)
        dataloader = DataLoader(
            dataset, batch_size=min(self.batch_size, self._n_obs//2) + 1, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed)
        )

        # Prepare numpy arrays for sklearn models
        X_np = self._embedding_buffer[:self._n_obs].copy()
        y_np = self._label_buffer[:self._n_obs].copy()
        sample_weights_np = self.class_weight_1 * y_np + self.class_weight_0 * (1 - y_np)

        # Reinitialize all ensembles from scratch and train on all data
        for grid_idx, (hidden_dim, num_epochs) in enumerate(self.tuning_grid):
            if num_epochs == 0:
                # Intercept-only logistic regression baseline
                self.ensembles[grid_idx] = [InterceptOnlyLR(seed=self.seed)]
                model = self.ensembles[grid_idx][0]
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
                # Train intercept-only model to convergence (fixed number of epochs)
                num_intercept_epochs = 100
                for epoch in range(num_intercept_epochs):
                    for X_batch, y_batch, w_batch in dataloader:
                        optimizer.zero_grad()
                        output = model(X_batch)
                        # Weighted BCE loss
                        bce = nn.functional.binary_cross_entropy(output, y_batch, reduction='none')
                        loss = bce.mean()
                        loss.backward()
                        optimizer.step()
                if grid_idx == best_idx:
                    logging.info(
                        f"Intercept-only LR training: n={self._n_obs}, "
                        f"final_loss={loss.item():.4f}"
                    )
            elif hidden_dim == 0:
                # Use sklearn LogisticRegression for hidden_dim=0
                self.ensembles[grid_idx] = [
                    SklearnLRWrapper(l2_penalty=self.l2_penalty, seed=self.seed + i)
                    for i in range(self.num_nns)
                ]
                for nn_idx, model in enumerate(self.ensembles[grid_idx]):
                    model.fit(X_np, y_np.astype(int), sample_weight=sample_weights_np)
                if grid_idx == best_idx:
                    logging.info(
                        f"Sklearn LR training (hidden_dim=0): n={self._n_obs}"
                    )
                    self.last_training_info = {
                        "best_epochs": num_epochs,
                        "final_loss": np.nan,
                        "n": self._n_obs,
                    }
            else:
                self.ensembles[grid_idx] = [
                    MLP(self.embedding_dim, [hidden_dim], seed=self.seed + i)
                    for i in range(self.num_nns)
                ]

                for nn_idx, model in enumerate(self.ensembles[grid_idx]):
                    model.train()
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
                    initial_loss = np.inf
                    final_loss = np.inf
                    for epoch in range(num_epochs):
                        for X_batch, y_batch, w_batch in dataloader:
                            optimizer.zero_grad()
                            l2_reg = sum(torch.norm(p, p=2) ** 2 for p in model.parameters())
                            output = model(X_batch)
                            # Weighted BCE loss
                            bce = nn.functional.binary_cross_entropy(output, y_batch, reduction='none')
                            loss = bce.mean() + self.l2_penalty * l2_reg
                            if epoch == 0:
                                initial_loss = loss.item()
                            loss.backward()
                            optimizer.step()
                            final_loss = loss.item()
                    if nn_idx == 0 and grid_idx == best_idx:
                        logging.info(
                            f"NN training (hidden_dim={hidden_dim}, epochs={num_epochs}): n={self._n_obs}, "
                            f"loss {initial_loss:.4f} -> {final_loss:.4f}"
                        )
                        self.last_training_info = {
                            "best_epochs": num_epochs,
                            "final_loss": final_loss,
                            "n": self._n_obs,
                        }


class ActiveSampler(BaseSampler):
    """Active sampler using an ensemble of neural networks.

    Predicts accuracy for candidates and samples from the lowest-performing
    percentile to find problematic subgroups.
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
        seed: int,
        k_std_dev: float = 0.0,
        target_threshold: float = 1.0,
        batch_size: int = 32,
        class_weight_0: float = 1.0,
        class_weight_1: float = 1.0,
    ):
        """Initialize the active sampler.

        Args:
            embedding_dim: Dimension of input embeddings
            num_nns: Number of neural networks in the ensemble
            hidden_dims: List of hidden dimension values to tune over (each creates a single hidden layer)
            epsilon: Percentile threshold (e.g., 0.1 means bottom 10%)
            num_epochs_list: Grid of training epoch values to tune over
            learning_rate: Learning rate for Adam optimizer
            l2_penalty: L2 penalty value for regularization
            seed: Random seed for reproducibility
            k_std_dev: Multiplier for std dev in UCB-style selection (0 = greedy)
            batch_size: Minibatch size for training
            class_weight_0: Weight for Y=0 class in loss function
            class_weight_1: Weight for Y=1 class in loss function
        """
        super().__init__()
        assert 0 < epsilon <= 1, f"epsilon must be in (0, 1], got {epsilon}"

        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.k_std_dev = k_std_dev

        # Initialize shared NN ensemble
        self.nn_ensemble = NNEnsemble(
            embedding_dim=embedding_dim,
            num_nns=num_nns,
            hidden_dims=hidden_dims,
            num_epochs_list=num_epochs_list,
            learning_rate=learning_rate,
            l2_penalty=l2_penalty,
            seed=seed,
            batch_size=batch_size,
            class_weight_0=class_weight_0,
            class_weight_1=class_weight_1,
        )
        self.target_threshold = target_threshold

        # Random generator for sampling
        self.rng = np.random.default_rng(seed)
        self.sampler_name = f"active_k{self.k_std_dev}"

    def _sample_impl(
        self, embeddings: np.ndarray, ids: np.ndarray
    ) -> tuple[object, int]:
        """Sample a candidate from the lowest-performing percentile.

        Uses UCB-style selection when k_std_dev > 0:
        score = predicted_accuracy - k_std_dev * std_dev

        Args:
            embeddings: Array of shape (n_candidates, embedding_dim)
            ids: Array of candidate identifiers

        Returns:
            Tuple of (sampled_id, subgroup_size) where subgroup is the bottom percentile
        """
        assert len(embeddings) == len(ids)
        assert (
            embeddings.shape[1] == self.embedding_dim
        ), f"Expected embedding dim {self.embedding_dim}, got {embeddings.shape[1]}"

        # Find bottom epsilon percentile
        scores, predictions, std_predictions = self.predict(embeddings, return_full=True)
        threshold = min(self.target_threshold, np.percentile(scores, self.epsilon * 100))
        bottom_mask = scores <= threshold
        logging.info(
            f"FULL pred range=[{predictions.min():.3f}, {predictions.max():.3f}], threshold={threshold}"
        )
        
        
        if np.mean(scores <= self.target_threshold) < self.epsilon:
            # HACK: hard coded
            logging.info(f"Falling back to half population")
            # use_epsilon = 0.5
            # We can't find any observations predicted to perform poorly
            # In this case, just optimize for exploration
            # bottom_mask = np.ones(scores.size, dtype=bool)
            # logging.info(f"Exploration fallback: {np.mean(scores <= self.target_threshold):.3f} below threshold {self.target_threshold}")
            scores = predictions - std_predictions
            threshold = np.percentile(scores, 50)
            # if np.sum(bottom_mask) == 0:
            #     logging.info(f"Falling back to full population")
            #     bottom_mask = np.ones(scores.size, dtype=bool)
        else:
            # Find bottom epsilon percentile
            threshold = np.percentile(scores, self.epsilon * 100)
            logging.info(f"THRESHOLD {self.epsilon} {threshold} (max {scores.max()}) (median {np.median(scores)})")
        
        bottom_mask = scores <= threshold

        # Ensure at least one candidate is selected
        assert np.sum(bottom_mask)

        # IID sample from the bottom percentile
        bottom_indices = np.where(bottom_mask)[0]
        subgroup_size = len(bottom_indices)
        sampled_idx = self.rng.choice(bottom_indices)

        # Cache prediction for the sampled candidate to avoid redundant forward pass
        self._cached_pred_prob = float(predictions[sampled_idx])

        return sampled_idx, subgroup_size
    
    def predict(self, embeddings, return_full=False):
        predictions, std_predictions = self.nn_ensemble.predict_with_std(embeddings)
        
        scores = predictions - self.k_std_dev * std_predictions
        logging.info(
            f"scores: [{scores.min():.3f}, {scores.max():.3f}]"
            f"pred range=[{predictions.min():.3f}, {predictions.max():.3f}], "
            f"std range=[{std_predictions.min():.3f}, {std_predictions.max():.3f}]"
        )
    
        if return_full:
            return scores, predictions, std_predictions
        else:
            scores = predictions
            return scores

    def get_training_summary(self) -> dict | None:
        return self.nn_ensemble.last_training_info

    def update(self, embedding: np.ndarray, label: int) -> None:
        """Update the ensemble with a new observation.

        Args:
            embedding: Embedding vector of shape (embedding_dim,)
            label: Binary label (0 or 1)
        """
        assert embedding.shape == (self.embedding_dim,), (
            f"Expected shape ({self.embedding_dim},), got {embedding.shape}"
        )
        assert label in (0, 1), f"label must be 0 or 1, got {label}"

        # Update calibration model
        pred_prob, _ = self.nn_ensemble.predict_with_std(embedding.reshape((1,-1)))
        pred_logit = np.log(pred_prob/(1 - pred_prob))

        self.nn_ensemble.update(embedding, label)
