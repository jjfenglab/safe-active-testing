"""E-detector classes for sequential hypothesis testing.

Implements e-detectors from the testing-by-betting framework:
- EDetectorRobustness: Tests if AI has a failure mode (futility/early rejection)
- EDetectorAuditor: Tests if AI is robust (efficacy/declare success)
- CombinedEDetector: Runs both simultaneously
"""
import logging
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from src.sequential_tests.base import BaseSequentialTest


class EDetectorRobustness:
    """E-detector for the robustness hypothesis test (futility bound).

    Tests:
        H0 (robustness): All subgroups have accuracy >= null_prob
        H1: Some subgroup has accuracy < null_prob

    Rejection means we found evidence of a failure mode.

    Uses weighted CUSUM or Shiryaev-Roberts e-detectors with uniform weights:
        SR:    M_n = sum_{j=1}^n w_j * Lambda_n^{(j)}
        CUSUM: M_n = max_{j in [n]} w_j * Lambda_n^{(j)}

    Implementation maintains a running list of all Lambda_n^{(j)} values, updating
    each by multiplying by S_n on every observation, then computes the weighted
    sum (SR) or weighted max (CUSUM) via dot product.
    """

    def __init__(
        self,
        method: Literal["cusum", "sr", "sprt"],
        null_prob: float,
        alt_prob: float,
        m_min: int | None = None,
    ):
        """Initialize the robustness e-detector.

        Args:
            method: Either 'cusum' or 'sr' (Shiryaev-Roberts)
            null_prob: Probability under null hypothesis (e.g., 0.7)
            alt_prob: Probability under alternative hypothesis (e.g., 0.5), must be < null_prob
            m_min: If specified, limits changepoints to first m_min observations
        """
        assert 0 < null_prob < 1, f"null_prob must be in (0, 1), got {null_prob}"
        assert 0 < alt_prob < 1, f"alt_prob must be in (0, 1), got {alt_prob}"
        assert alt_prob < null_prob, f"alt_prob must be < null_prob for robustness test"

        self.method = method
        self.null_prob = null_prob
        self.alt_prob = alt_prob
        self.m_min = m_min
        assert self.m_min is not None

        # Track Lambda_n^{(j)} for each changepoint j
        # lambdas[i] = Lambda_n^{(i+1)} (0-indexed)
        self._lambdas: list[float] = []
        # Corresponding weights w_j
        self._weights: list[float] = []

        self.statistic_history = [1]
        self._t = 0

    @property
    def statistic(self) -> float:
        """Return the current e-statistic value."""
        return self.statistic_history[-1]

    def _likelihood_ratio(self, y: int) -> float:
        """Compute the likelihood ratio S(y) = alt(y) / null(y).

        Args:
            y: Binary observation (0 or 1)

        Returns:
            Likelihood ratio for the observation
        """
        if y == 1:
            return self.alt_prob / self.null_prob
        else:
            return (1 - self.alt_prob) / (1 - self.null_prob)

    def _get_weight(self, t: int) -> float:
        """Get the weight for changepoint at time t.

        Args:
            t: Time step (1-indexed)

        Returns:
            Mixture weight w_t
        """
        if self.method == "sprt":
            return None
        else:
            if t - 1 > self.m_min:
                return 0
            else:
                return 1/self.m_min
                

    def update(self, y: int) -> float:
        """Update the e-statistic with a new observation.

        Maintains a list of Lambda_n^{(j)} = prod_{k=j}^n S_k for each changepoint j.
        On each update:
        1. Multiply all existing lambdas by the new likelihood ratio S_n
        2. Add new changepoint with Lambda_n^{(n)} = S_n (if allowed by m_min)
        3. Compute statistic as weighted sum (SR) or weighted max (CUSUM)

        Args:
            y: Binary observation (0 or 1)

        Returns:
            Updated e-statistic value
        """
        assert y in (0, 1), f"y must be 0 or 1, got {y}"

        self._t += 1
        s = self._likelihood_ratio(y)

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
            # Weighted SR: M_n = sum_{j=1}^n w_j * Lambda_n^{(j)}
            new_statistic = np.dot(self._weights, self._lambdas)
        elif self.method == "cusum":  # cusum
            # Weighted CUSUM: M_n = max_{j in [n]} w_j * Lambda_n^{(j)}
            weighted_lambdas = np.array(self._weights) * np.array(self._lambdas)
            new_statistic = np.max(weighted_lambdas)
        else:  # sprt
            new_statistic = self._lambdas[0]

        logging.info(f"TEST ITER {self._t}: new_statistic {new_statistic}")
        self.statistic_history.append(new_statistic)
        return new_statistic


class EDetectorAuditor:
    """E-detector for the auditor hypothesis test.

    Tests:
        H0 (auditor): Auditor will find a failure mode (accuracy < threshold)
        H1: No failure mode exists (accuracy >= threshold)

    Rejection means we have evidence the AI is robust (no failure mode found).
    Only starts computing after m_min observations.

    The likelihood ratio is alt_prob(y) / null_prob(y) where:
        - null_prob: low accuracy (failure mode exists, e.g., q - delta)
        - alt_prob: high accuracy (no failure mode, e.g., q + delta)

    So for y=1: LR = alt_prob / null_prob > 1 (evidence for robustness)
    """

    def __init__(
        self,
        null_prob: float,
        alt_prob: float,
        m_min: int,
    ):
        """Initialize the auditor e-detector.

        Args:
            null_prob: Probability under null (low accuracy, e.g., 0.5 = q - delta)
            alt_prob: Probability under alternative (high accuracy, e.g., 0.7 = q + delta)
            m_min: Number of observations before auditor test starts
        """
        assert 0 < null_prob < 1, f"null_prob must be in (0, 1), got {null_prob}"
        assert 0 < alt_prob < 1, f"alt_prob must be in (0, 1), got {alt_prob}"
        assert null_prob < alt_prob, f"null_prob must be < alt_prob for auditor test"
        assert m_min >= 1, f"m_min must be >= 1, got {m_min}"

        self.null_prob = null_prob
        self.alt_prob = alt_prob
        self.m_min = m_min
        self.statistic_history = [1.0]
        self._t = 0

    @property
    def statistic(self) -> float:
        """Return the current e-statistic value."""
        return self.statistic_history[-1]

    def _likelihood_ratio(self, y: int) -> float:
        """Compute the likelihood ratio S(y) = alt(y) / null(y).

        For the auditor test:
            - null_prob is low (failure mode exists)
            - alt_prob is high (no failure mode)

        So LR > 1 when y=1, accumulating evidence for robustness.

        Args:
            y: Binary observation (0 or 1)

        Returns:
            Likelihood ratio for the observation
        """
        if y == 1:
            return self.alt_prob / self.null_prob
        else:
            return (1 - self.alt_prob) / (1 - self.null_prob)

    def update(self, y: int) -> float:
        """Update the e-statistic with a new observation.

        Args:
            y: Binary observation (0 or 1)

        Returns:
            Updated e-statistic value
        """
        assert y in (0, 1), f"y must be 0 or 1, got {y}"

        self._t += 1

        if self._t < self.m_min:
            # Before m_min, statistic stays at 1
            self.statistic_history.append(1.0)
            return 1.0

        # At or after m_min, compute product from m_min to t
        # M_t = prod_{i=m_min}^t S_i^{auditor} = (S_{m_min:t}^robustness)^{-1}
        s = self._likelihood_ratio(y)

        if self._t == self.m_min:
            new_statistic = s
        else:
            new_statistic = self.statistic * s

        self.statistic_history.append(new_statistic)
        return new_statistic


class CombinedEDetector(BaseSequentialTest):
    """Wrapper that can run robustness, auditor, or both e-detectors.

    Test types:
    - "robustness": Only run robustness test (tests for failure modes)
    - "auditor": Only run auditor test (tests for robustness, requires m_min)
    - "dual": Run both tests simultaneously

    Testing procedure for dual mode:
    - Before m_min: only robustness test is active (futility bound)
      Reject robustness null if statistic > 1/alpha
    - After m_min: both tests are active
      - Robustness rejection: found failure mode
      - Auditor rejection: AI is robust
    - The two tests are mutually exclusive: if one rejects, the other cannot.
    """
    def __init__(
        self,
        method: Literal["cusum", "sr", "sprt"],
        null_prob: float,
        alt_prob: float,
        auditor_alt_prob: float,
        test_type: Literal["robustness", "auditor", "dual"] = "dual",
        m_min: int | None = None,
        num_init: int = 0,
    ):
        """Initialize combined e-detector.

        Args:
            method: Either 'cusum' or 'sr' for the robustness detector
            null_prob: Probability under null hypothesis
            alt_prob: Probability under robustness alternative (< null_prob)
            auditor_alt_prob: Probability under auditor alternative (> null_prob)
            test_type: Which test(s) to run: 'robustness', 'auditor', or 'dual'
            m_min: Required for 'auditor' and 'dual' modes. For 'robustness' mode,
                   optionally limits changepoints.
        """
        self.method = method
        self.method_name = method
        self.null_prob = null_prob
        self.alt_prob = alt_prob
        self.test_type = test_type
        self.m_min = m_min
        self.num_init = num_init

        if test_type in ("auditor", "dual"):
            assert m_min is not None, f"m_min is required for test_type='{test_type}'"
            assert null_prob < auditor_alt_prob < 1, f"auditor_alt_prob must be in (null_prob={null_prob}, 1)"

        if test_type in ("robustness", "dual"):
            self.robustness = EDetectorRobustness(
                method=method,
                null_prob=null_prob,
                alt_prob=alt_prob,
                m_min=m_min,
            )
        else:
            self.robustness = None

        if test_type in ("auditor", "dual"):
            self.auditor = EDetectorAuditor(
                null_prob=null_prob,
                alt_prob=auditor_alt_prob,
                m_min=m_min,
            )
        else:
            self.auditor = None


        self._t = 0

    @property
    def robustness_statistic(self) -> float | None:
        """Return current robustness e-statistic, or None if not enabled."""
        if self.robustness is None:
            return None
        return self.robustness.statistic

    @property
    def auditor_statistic(self) -> float | None:
        """Return current auditor e-statistic, or None if not enabled."""
        if self.auditor is None:
            return None
        return self.auditor.statistic

    def update(self, observation: int) -> None:
        """Update enabled detectors with a new observation.

        Args:
            observation: Binary observation (0 or 1)
        """
        self._t += 1
        if self.robustness is not None:
            self.robustness.update(observation)
        if self.auditor is not None:
            self.auditor.update(observation)
    
    def init_update(self, observation: int) -> None:
        # Do nothing by default
        return

    def get_statistic_history(self, alpha: float) -> list[dict]:
        """Get full history of statistics.

        Args:
            alpha: Significance level (used to compute threshold)

        Returns:
            List of dicts with keys: 'label', 'statistics', 'threshold'
        """
        threshold = 1.0 / alpha
        result = []
        if self.robustness is not None:
            result.append(
                {
                    "label": "robustness/futility",
                    "statistics": self.robustness.statistic_history,
                    "threshold": threshold,
                    "is_rejected": self.robustness.statistic > threshold,
                    "num_test_observations": self.num_init + np.arange(len(self.robustness.statistic_history)),
                }
            )
        if self.auditor is not None:
            result.append(
                {
                    "label": "auditor/efficacy",
                    "statistics": self.auditor.statistic_history,
                    "threshold": threshold,
                    "is_rejected": self.auditor.statistic > threshold,
                    "num_test_observations": self.num_init + np.arange(len(self.robustness.statistic_history)),
                }
            )
        return result

    def check_rejection(self, alpha: float) -> tuple[bool, str | None]:
        """Check if any enabled test has crossed the rejection threshold.

        Args:
            alpha: Significance level

        Returns:
            Tuple of (rejected, which_test)
            which_test is 'robustness', 'auditor', or None
        """
        threshold = 1.0 / alpha

        if self.robustness is not None and self.robustness_statistic >= threshold:
            return True, "robustness"

        if self.auditor is not None and self._t >= self.m_min:
            if self.auditor_statistic >= threshold:
                return True, "auditor"

        return False, None

    def plot(self, alpha: float, plot_path: str) -> None:
        """Plot the e-statistic history for enabled detectors.

        Args:
            alpha: Significance level for threshold line
            plot_path: Path to output plot file
        """
        sns.set_context("paper", font_scale=1.5)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot robustness statistic if enabled
        if self.robustness is not None:
            ax.plot(
                self.robustness.statistic_history,
                marker="o",
                markersize=3,
                label="Model's Null",
                color="blue",
            )

        # Plot auditor statistic if enabled
        if self.auditor is not None:
            ax.plot(
                self.auditor.statistic_history,
                marker="s",
                markersize=3,
                label="Auditor's Null",
                color="green",
            )
            # Add vertical line at m_min
            ax.axvline(
                x=self.m_min,
                color="gray",
                linestyle=":",
                alpha=0.7,
            )

        # Add threshold line
        ax.axhline(
            y=1.0 / alpha,
            color="r",
            linestyle="--",
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("E-Process")
        ax.legend()
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
