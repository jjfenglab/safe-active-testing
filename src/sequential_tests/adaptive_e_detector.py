"""Adaptive e-detector classes with dynamic alternative probabilities.

These classes extend the base e-detectors to use predicted probabilities
from a sampler model instead of fixed alternative probabilities.
"""
import logging
from typing import Literal

import numpy as np

from src.sequential_tests.e_detector import (
    CombinedEDetector,
    EDetectorAuditor,
    EDetectorRobustness,
)


class AdaptiveEDetectorRobustness(EDetectorRobustness):
    """E-detector for robustness with dynamic alternative probabilities.

    Instead of using a fixed alt_prob, this class accepts a predicted
    probability from a sampler model at each update step.

    The likelihood ratio at each step becomes:
        S(y) = pred_prob(y) / null_prob(y)

    where pred_prob varies per observation based on model predictions.
    """

    def __init__(
        self,
        method: Literal["cusum", "sr", "sprt"],
        null_prob: float,
        m_min: int | None = None,
        eps: float = 0.05,
    ):
        """Initialize the adaptive robustness e-detector.

        Args:
            method: Either 'cusum', 'sr' (Shiryaev-Roberts), or 'sprt'
            null_prob: Probability under null hypothesis (e.g., 0.7)
            m_min: If specified, limits changepoints to first m_min observations
            eps: Small constant for numerical stability when clipping pred_prob
        """
        # Initialize parent with a dummy alt_prob (won't be used)
        # We pass null_prob - eps as alt_prob to satisfy the assertion
        super().__init__(
            method=method,
            null_prob=null_prob,
            alt_prob=null_prob - eps,
            m_min=m_min,
        )
        self.eps = eps
    
    def _likelihood_ratio_dynamic(self, y: int, pred_prob: float) -> float:
        """Compute the likelihood ratio using dynamic predicted probability.

        Args:
            y: Binary observation (0 or 1)
            pred_prob: Predicted probability from sampler model (must be < null_prob)

        Returns:
            Likelihood ratio for the observation
        """
        if y == 1:
            return pred_prob / self.null_prob
        else:
            return (1 - pred_prob) / (1 - self.null_prob)

    def update(self, y: int, pred_prob: float) -> float:
        """Update the e-statistic with a new observation and predicted probability.

        Args:
            y: Binary observation (0 or 1)
            pred_prob: Predicted probability from sampler model

        Returns:
            Updated e-statistic value
        """
        assert y in (0, 1), f"y must be 0 or 1, got {y}"

        
        # Clip pred_prob to valid range: (eps, null_prob)
        pred_prob_clipped = np.clip(pred_prob, self.eps, self.null_prob)
        logging.info(f"adaptive ROBUSTNESS pred_prob_clipped {pred_prob_clipped}")

        self._t += 1
        s = self._likelihood_ratio_dynamic(y, pred_prob_clipped)

        # Update all existing lambdas: Lambda_n^{(j)} *= S_n
        for i in range(len(self._lambdas)):
            self._lambdas[i] *= s

        # Add new changepoint if allowed
        if self.m_min is None or self._t <= self.m_min:
            self._lambdas.append(s)
            self._weights.append(self._get_weight(self._t))

        # Compute statistic
        if len(self._lambdas) == 0:
            new_statistic = 0.0
        elif self.method == "sr":
            new_statistic = np.dot(self._weights, self._lambdas)
        elif self.method == "cusum":
            weighted_lambdas = np.array(self._weights) * np.array(self._lambdas)
            new_statistic = np.max(weighted_lambdas)
        else:  # sprt
            new_statistic = self._lambdas[0]

        self.statistic_history.append(new_statistic)
        return new_statistic


class AdaptiveEDetectorAuditor(EDetectorAuditor):
    """E-detector for auditor test with dynamic alternative probabilities.

    Instead of using a fixed alt_prob, this class accepts a predicted
    probability from a sampler model at each update step.

    For the auditor test:
        - null_prob: low accuracy (failure mode exists)
        - pred_prob: predicted accuracy (used as alternative)

    The likelihood ratio at each step becomes:
        S(y) = pred_prob(y) / null_prob(y)

    When pred_prob is high (model is performing well), LR > 1 for y=1,
    accumulating evidence that the model is robust.
    """

    def __init__(
        self,
        null_prob: float,
        m_min: int,
        eps: float = 0.05,
    ):
        """Initialize the adaptive auditor e-detector.

        Args:
            null_prob: Probability under null (low accuracy, e.g., 0.5)
            m_min: Number of observations before auditor test starts
            eps: Small constant for numerical stability when clipping pred_prob
        """
        # Initialize parent with a dummy alt_prob (won't be used)
        # We pass null_prob + eps as alt_prob to satisfy the assertion
        super().__init__(
            null_prob=null_prob,
            alt_prob=null_prob + eps, # this is actually ignored
            m_min=m_min,
        )
        self.eps = eps

    def _likelihood_ratio_dynamic(self, y: int, pred_prob: float) -> float:
        """Compute the likelihood ratio using dynamic predicted probability.

        Args:
            y: Binary observation (0 or 1)
            pred_prob: Predicted probability from sampler model (should be > null_prob)

        Returns:
            Likelihood ratio for the observation
        """
        if y == 1:
            return pred_prob / self.null_prob
        else:
            return (1 - pred_prob) / (1 - self.null_prob)

    def update(self, y: int, pred_prob: float, is_adaptive: bool = True) -> float:
        """Update the e-statistic with a new observation and predicted probability.

        Args:
            y: Binary observation (0 or 1)
            pred_prob: Predicted probability from sampler model

        Returns:
            Updated e-statistic value
        """
        assert y in (0, 1), f"y must be 0 or 1, got {y}"

        self._t += 1

        if self._t < self.m_min:
            print("SKIPPING")
            # Before m_min, statistic stays at 1
            self.statistic_history.append(1.0)
            return 1.0

        # Clip pred_prob to valid range: (null_prob, 1-eps)
        # Truncated below at null_prob to ensure LR >= 1 for y=1
        pred_prob = np.clip(pred_prob, self.null_prob, 1 - self.eps)
        logging.info(f"adaptive AUDITOR clipped {pred_prob}")

        s = self._likelihood_ratio_dynamic(y, pred_prob)

        if self._t == self.m_min:
            new_statistic = s
        else:
            new_statistic = self.statistic * s

        self.statistic_history.append(new_statistic)
        return new_statistic


class AdaptiveCombinedEDetector(CombinedEDetector):
    """Combined e-detector with adaptive tests using dynamic predicted probabilities.

    Both the robustness and auditor tests use dynamic predicted probabilities
    from a sampler model instead of fixed alternative probabilities.
    """
    def __init__(
        self,
        method: Literal["cusum", "sr", "sprt"],
        null_prob: float,
        # alt_prob: float,
        test_type: Literal["robustness", "auditor", "dual"] = "dual",
        m_min: int | None = None,
        num_init: int = 0,
        eps: float = 0.05,
    ):
        """Initialize adaptive combined e-detector.

        Args:
            method: Either 'cusum', 'sr', or 'sprt' for the robustness detector
            null_prob: Probability under robustness null (high accuracy, e.g., 0.7)
            alt_prob: Probability under robustness alt (low accuracy, e.g., 0.5)
            test_type: Which test(s) to run: 'robustness', 'auditor', or 'dual'
            m_min: Required for 'auditor' and 'dual' modes
            num_init: Number of initial observations (for plotting offset)
            eps: Small constant for numerical stability
        """
        # Store parameters without calling parent __init__ to avoid creating
        # non-adaptive detectors
        self.method = method
        self.method_name = f"{method}_adaptive"
        self.null_prob = null_prob
        # self.alt_prob = null_prob
        self.test_type = test_type
        self.m_min = m_min
        self.num_init = num_init
        self.eps = eps

        # Validate arguments
        if test_type in ("auditor", "dual"):
            assert m_min is not None, f"m_min is required for test_type='{test_type}'"

        # Create adaptive robustness detector if needed
        if test_type in ("robustness", "dual"):
            self.robustness = AdaptiveEDetectorRobustness(
                method=method,
                null_prob=null_prob,
                m_min=m_min,
                eps=eps,
            )
        else:
            self.robustness = None

        # Create adaptive auditor detector if needed
        # For auditor: null = low accuracy (failure exists), alt = high accuracy (robust)
        # So auditor null_prob = robustness alt_prob
        if test_type in ("auditor", "dual"):
            self.auditor = AdaptiveEDetectorAuditor(
                null_prob=null_prob,  # auditor null = low accuracy = robustness alt
                # alt_prob=null_prob,
                m_min=m_min,
                eps=eps,
            )
        else:
            self.auditor = None

        self._t = 0

    def update(self, observation: int, pred_prob: float, is_adaptive_auditor: bool = True) -> None:
        """Update enabled detectors with a new observation.

        Args:
            observation: Binary observation (0 or 1)
            pred_prob: Predicted probability from sampler model
        """
        self._t += 1
        logging.info(f"INCOMING {pred_prob}")
        if self.robustness is not None:
            self.robustness.update(observation, pred_prob)
        if self.auditor is not None:
            self.auditor.update(observation, pred_prob, is_adaptive=is_adaptive_auditor)
