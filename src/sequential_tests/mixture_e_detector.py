"""Mixture e-detector classes using a grid of candidate alternative probabilities.

Instead of a single (adaptive) alternative, these detectors average over K fixed
candidate alternatives, so the statistic grows whenever any grid point fits the data.
"""
import logging
import numpy as np
from typing import Literal

from src.sequential_tests.e_detector import (
    CombinedEDetector,
    EDetectorAuditor,
    EDetectorRobustness,
)


class MixtureEDetectorRobustness(EDetectorRobustness):
    """Robustness e-detector using a mixture over a grid of alternative probabilities.

    Maintains K independent processes — one per grid point p^(k). For each k and
    each changepoint hypothesis j, tracks Lambda_t^{(j,k)} = prod_{s=j}^t S_s(p^(k)).

    Reported statistic = (1/K) * sum_k stat_t^(k), where stat_t^(k) is the SR, CUSUM,
    or SPRT statistic computed from {Lambda_t^{(j,k)}}_j exactly as in EDetectorRobustness.
    """

    def __init__(
        self,
        method: Literal["cusum", "sr", "sprt"],
        null_prob: float,
        grid_low: float,
        grid_high: float,
        grid_step: float,
        m_min: int,
    ):
        """Initialize mixture robustness e-detector.

        Args:
            method: 'sr', 'cusum', or 'sprt'
            null_prob: Robustness null probability q; all grid points are < q.
            grid_low: Lower bound of robustness grid (e.g. 0.50).
            grid_high: Upper bound; must be < null_prob (e.g. 0.80).
            grid_step: Step size (e.g. 0.05).
            m_min: Max number of SR starting points (changepoint prior support).
        """
        assert grid_high < null_prob, f"grid_high={grid_high} must be < null_prob={null_prob}"
        # alt_prob=grid_high satisfies parent assertion alt_prob < null_prob; self.alt_prob
        # is otherwise unused since we override update().
        super().__init__(method=method, null_prob=null_prob, alt_prob=grid_high, m_min=m_min)
        self._rob_grid = np.arange(grid_low, grid_high + grid_step / 2, grid_step)

        # HACK: arbitrary grid weights
        self._grid_weight = np.arange(1, self._rob_grid.size + 1, 1).astype(float)
        self._grid_weight /= float(np.sum(self._grid_weight))
        # K separate lambda lists; _weights is inherited from parent and shared across all k
        self._lambdas_grid: list[list[float]] = [[] for _ in self._rob_grid]

    def _likelihood_ratio_at(self, y: int, p: float) -> float:
        """Compute likelihood ratio S(y) = p(y) / null_prob(y) at alternative p."""
        if y == 1:
            return p / self.null_prob
        else:
            return (1 - p) / (1 - self.null_prob)

    def update(self, y: int) -> float:
        assert y in (0, 1), f"y must be 0 or 1, got {y}"
        self._t += 1

        for k, p in enumerate(self._rob_grid):
            s = self._likelihood_ratio_at(y, p)
            for i in range(len(self._lambdas_grid[k])):
                self._lambdas_grid[k][i] *= s
            if self._t <= self.m_min:
                self._lambdas_grid[k].append(s)

        if self._t <= self.m_min:
            self._weights.append(self._get_weight(self._t))

        per_grid_stats = []
        for k in range(len(self._rob_grid)):
            _lambdas = self._lambdas_grid[k]
            if len(_lambdas) == 0:
                per_grid_stats.append(0.0)
            elif self.method == "sr":
                per_grid_stats.append(np.dot(self._weights, _lambdas))
            elif self.method == "cusum":
                per_grid_stats.append(np.max(np.array(self._weights) * np.array(_lambdas)))
            else:  # sprt
                per_grid_stats.append(_lambdas[0])

        logging.info(f"TEST ITER {self._t}: per_grid_stats {np.array(per_grid_stats)}")
        new_statistic = np.sum(np.array(per_grid_stats) * self._grid_weight)
        self.statistic_history.append(new_statistic)
        return new_statistic


class MixtureEDetectorAuditor(EDetectorAuditor):
    """Auditor e-detector using a mixture over a directly specified grid of alternatives.

    Maintains K independent running products Pi_t^(k) = prod_{s=m_min}^t S_s(p_aud^(k)).
    Reported statistic = (1/K) * sum_k Pi_t^(k).
    """

    def __init__(
        self,
        null_prob: float,
        auditor_grid_low: float,
        auditor_grid_high: float,
        auditor_grid_step: float,
        m_min: int,
    ):
        """Initialize mixture auditor e-detector.

        Args:
            null_prob: Auditor null probability q; all grid points must be > q.
            auditor_grid_low: Lower bound of auditor grid (must be > null_prob).
            auditor_grid_high: Upper bound of auditor grid.
            auditor_grid_step: Step size for auditor grid.
            m_min: Number of observations before auditor test starts.
        """
        self._aud_grid = np.arange(auditor_grid_low, auditor_grid_high + auditor_grid_step / 2, auditor_grid_step)
        assert len(self._aud_grid) > 0, "Auditor grid is empty; check auditor_grid_low/high/step"
        assert self._aud_grid[0] > null_prob, f"auditor_grid_low={auditor_grid_low} must be > null_prob={null_prob}"
        # alt_prob=_aud_grid[0] satisfies parent assertion null_prob < alt_prob; otherwise unused.
        super().__init__(null_prob=null_prob, alt_prob=self._aud_grid[0], m_min=m_min)
        self._grid_weight = 1 - self._aud_grid
        self._grid_weight /= np.sum(self._grid_weight)
        print("self._aud_grid", self._aud_grid)
        print("self._grid_weight", self._grid_weight)
        self._running_products: list[float] = []  # initialized at t == m_min

    def _likelihood_ratio_at(self, y: int, p: float) -> float:
        """Compute likelihood ratio S(y) = p(y) / null_prob(y) at alternative p."""
        if y == 1:
            return p / self.null_prob
        else:
            return (1 - p) / (1 - self.null_prob)

    def update(self, y: int) -> float:
        assert y in (0, 1), f"y must be 0 or 1, got {y}"
        self._t += 1

        if self._t < self.m_min:
            self.statistic_history.append(1.0)
            return 1.0

        if self._t == self.m_min:
            self._running_products = [self._likelihood_ratio_at(y, p) for p in self._aud_grid]
        else:
            for k, p in enumerate(self._aud_grid):
                self._running_products[k] *= self._likelihood_ratio_at(y, p)

        new_statistic = np.sum(np.array(self._running_products) * self._grid_weight)
        self.statistic_history.append(new_statistic)
        return new_statistic


class EWAFEDetectorRobustness(EDetectorRobustness):
    """Robustness e-detector using EWAF (Exponentially Weighted Average Forecaster) over a grid.

    Maintains EWAF weights over K grid points. At each step, selects the grid point
    with highest weight and uses that alternative probability for the likelihood ratio.
    Weights are updated using the multiplicative weights rule based on observed LRs.
    """

    def __init__(
        self,
        method: Literal["cusum", "sr", "sprt"],
        null_prob: float,
        grid_low: float,
        grid_high: float,
        grid_step: float,
        m_min: int,
        ewaf_learning_rate: float,
    ):
        """Initialize EWAF robustness e-detector.

        Args:
            method: 'sr', 'cusum', or 'sprt'
            null_prob: Robustness null probability q; all grid points are < q.
            grid_low: Lower bound of robustness grid (e.g. 0.50).
            grid_high: Upper bound; must be < null_prob (e.g. 0.80).
            grid_step: Step size (e.g. 0.05).
            m_min: Max number of SR starting points (changepoint prior support).
            ewaf_learning_rate: Learning rate eta for EWAF weight updates.
        """
        assert grid_high < null_prob, f"grid_high={grid_high} must be < null_prob={null_prob}"
        super().__init__(method=method, null_prob=null_prob, alt_prob=grid_high, m_min=m_min)
        self._rob_grid = np.arange(grid_low, grid_high + grid_step / 2, grid_step)
        logging.info(f"self._rob_grid {self._rob_grid}")
        self._ewaf_eta = ewaf_learning_rate
        # Initialize EWAF weights uniformly
        self._ewaf_weights = np.ones(len(self._rob_grid)) / len(self._rob_grid)

    def _likelihood_ratio_at(self, y: int, p: float) -> float:
        """Compute likelihood ratio S(y) = p(y) / null_prob(y) at alternative p."""
        if y == 1:
            return p / self.null_prob
        else:
            return (1 - p) / (1 - self.null_prob)

    def init_update(self, y: int) -> float:
        assert y in (0, 1), f"y must be 0 or 1, got {y}"
        
        # Compute likelihood ratios for all grid points
        lrs = np.array([self._likelihood_ratio_at(y, p) for p in self._rob_grid])

        # Update EWAF weights: w_k <- w_k * exp(eta * lr_k), then normalize
        self._ewaf_weights *= np.exp(self._ewaf_eta * np.log(lrs))
        self._ewaf_weights /= np.sum(self._ewaf_weights)

    def update(self, y: int) -> float:
        assert y in (0, 1), f"y must be 0 or 1, got {y}"
        self._t += 1

        # Compute predicted probability as weighted average over grid
        ewaf_pred_prob = np.dot(self._ewaf_weights, self._rob_grid)
        s = self._likelihood_ratio_at(y, ewaf_pred_prob)
        logging.info(f"ITER {self._t}: MODEL EWAF {self._ewaf_weights}, final pred prob {ewaf_pred_prob}")

        # Update EWAF weights: w_k <- w_k * exp(eta * log(lr_k)), then normalize
        lrs = np.array([self._likelihood_ratio_at(y, p) for p in self._rob_grid])
        self._ewaf_weights *= np.exp(self._ewaf_eta * np.log(lrs))
        self._ewaf_weights /= np.sum(self._ewaf_weights)

        # Update lambdas using the selected likelihood ratio
        for i in range(len(self._lambdas)):
            self._lambdas[i] *= s

        if self._t <= self.m_min:
            self._lambdas.append(s)
            self._weights.append(self._get_weight(self._t))

        # Compute statistic
        if len(self._lambdas) == 0:
            new_statistic = 0.0
        elif self.method == "sr":
            new_statistic = np.dot(self._weights, self._lambdas)
        elif self.method == "cusum":
            new_statistic = np.max(np.array(self._weights) * np.array(self._lambdas))
        else:  # sprt
            new_statistic = self._lambdas[0]

        self.statistic_history.append(new_statistic)
        return new_statistic


class EWAFEDetectorAuditor(EDetectorAuditor):
    """Auditor e-detector using EWAF (Exponentially Weighted Average Forecaster) over a grid.

    Maintains EWAF weights over K grid points. At each step, selects the grid point
    with highest weight and uses that alternative probability for the likelihood ratio.
    Weights are updated using the multiplicative weights rule based on observed LRs.
    """

    def __init__(
        self,
        null_prob: float,
        auditor_grid_low: float,
        auditor_grid_high: float,
        auditor_grid_step: float,
        m_min: int,
        ewaf_learning_rate: float,
    ):
        """Initialize EWAF auditor e-detector.

        Args:
            null_prob: Auditor null probability q; all grid points must be > q.
            auditor_grid_low: Lower bound of auditor grid (must be > null_prob).
            auditor_grid_high: Upper bound of auditor grid.
            auditor_grid_step: Step size for auditor grid.
            m_min: Number of observations before auditor test starts.
            ewaf_learning_rate: Learning rate eta for EWAF weight updates.
        """
        self._aud_grid = np.arange(auditor_grid_low, auditor_grid_high + auditor_grid_step / 2, auditor_grid_step)
        assert len(self._aud_grid) > 0, "Auditor grid is empty; check auditor_grid_low/high/step"
        assert self._aud_grid[0] > null_prob, f"auditor_grid_low={auditor_grid_low} must be > null_prob={null_prob}"
        super().__init__(null_prob=null_prob, alt_prob=self._aud_grid[0], m_min=m_min)
        self._ewaf_eta = ewaf_learning_rate
        # Initialize EWAF weights uniformly
        self._ewaf_weights = np.ones(len(self._aud_grid)) / len(self._aud_grid)
        self._running_product: float = 1.0

    def _likelihood_ratio_at(self, y: int, p: float) -> float:
        """Compute likelihood ratio S(y) = p(y) / null_prob(y) at alternative p."""
        if y == 1:
            return p / self.null_prob
        else:
            return (1 - p) / (1 - self.null_prob)

    def init_update(self, y: int) -> float:
        assert y in (0, 1), f"y must be 0 or 1, got {y}"
        
        # Compute likelihood ratios for all grid points
        lrs = np.array([self._likelihood_ratio_at(y, p) for p in self._aud_grid])

        # Update EWAF weights: w_k <- w_k * exp(eta * lr_k), then normalize
        self._ewaf_weights *= np.exp(self._ewaf_eta * np.log(lrs))
        self._ewaf_weights /= np.sum(self._ewaf_weights)


    def update(self, y: int) -> float:
        assert y in (0, 1), f"y must be 0 or 1, got {y}"
        self._t += 1

        if self._t < self.m_min:
            self.statistic_history.append(1.0)
            return 1.0

        # Compute predicted probability as weighted average over grid
        ewaf_pred_prob = np.dot(self._ewaf_weights, self._aud_grid)
        s = self._likelihood_ratio_at(y, ewaf_pred_prob)
        logging.info(f"ITER {self._t}: AUDITOR EWAF {self._ewaf_weights}, final pred prob {ewaf_pred_prob}")

        # Update EWAF weights: w_k <- w_k * exp(eta * lr_k), then normalize
        lrs = np.array([self._likelihood_ratio_at(y, p) for p in self._aud_grid])
        self._ewaf_weights *= np.exp(self._ewaf_eta * np.log(lrs))
        self._ewaf_weights /= np.sum(self._ewaf_weights)

        # Update running product
        if self._t == self.m_min:
            self._running_product = s
        else:
            self._running_product *= s

        self.statistic_history.append(self._running_product)
        return self._running_product


class EWAFCombinedEDetector(CombinedEDetector):
    """Combined e-detector using EWAF robustness and auditor tests.

    Uses Exponentially Weighted Average Forecaster to adaptively select
    the best alternative probability from a grid at each time step.
    """

    def __init__(
        self,
        method: Literal["cusum", "sr", "sprt"],
        null_prob: float,
        grid_low: float,
        grid_high: float,
        grid_step: float,
        auditor_grid_low: float,
        auditor_grid_high: float,
        auditor_grid_step: float,
        ewaf_learning_rate: float,
        test_type: Literal["robustness", "auditor", "dual"] = "dual",
        m_min: int | None = None,
        num_init: int = 0,
    ):
        """Initialize EWAF combined e-detector.

        Args:
            method: 'sr', 'cusum', or 'sprt' for the robustness detector.
            null_prob: Null probability shared by both detectors.
            grid_low: Lower bound of robustness grid (< null_prob).
            grid_high: Upper bound of robustness grid (< null_prob).
            grid_step: Step size for robustness grid.
            auditor_grid_low: Lower bound of auditor grid (> null_prob).
            auditor_grid_high: Upper bound of auditor grid.
            auditor_grid_step: Step size for auditor grid.
            ewaf_learning_rate: Learning rate eta for EWAF weight updates.
            test_type: Which test(s) to run.
            m_min: Required for 'auditor' and 'dual' modes.
            num_init: Number of initial observations (for plotting offset).
        """
        self.method = method
        self.method_name = f"{method}_ewaf"
        self.null_prob = null_prob
        self.test_type = test_type
        self.m_min = m_min
        self.num_init = num_init
        self.ewaf_learning_rate = ewaf_learning_rate

        if test_type in ("auditor", "dual"):
            assert m_min is not None, f"m_min is required for test_type='{test_type}'"

        self.robustness = (
            EWAFEDetectorRobustness(
                method=method, null_prob=null_prob, m_min=m_min,
                grid_low=grid_low, grid_high=grid_high, grid_step=grid_step,
                ewaf_learning_rate=ewaf_learning_rate)
            if test_type in ("robustness", "dual")
            else None
        )
        self.auditor = (
            EWAFEDetectorAuditor(
                null_prob=null_prob, m_min=m_min,
                auditor_grid_low=auditor_grid_low, auditor_grid_high=auditor_grid_high,
                auditor_grid_step=auditor_grid_step,
                ewaf_learning_rate=ewaf_learning_rate)
            if test_type in ("auditor", "dual")
            else None
        )
        self._t = 0

    def update(self, observation: int) -> None:
        """Update enabled detectors with a new observation."""
        self._t += 1
        if self.robustness is not None:
            self.robustness.update(observation)
        if self.auditor is not None:
            self.auditor.update(observation)


    def init_update(self, observation: int) -> None:
        """Update enabled detectors with a new observation."""
        self._t += 1
        if self.robustness is not None:
            self.robustness.init_update(observation)
        if self.auditor is not None:
            self.auditor.init_update(observation)


class MixtureCombinedEDetector(CombinedEDetector):
    """Combined e-detector using mixture robustness and auditor tests.

    Mirrors AdaptiveCombinedEDetector but uses fixed-grid mixture detectors
    instead of online-learned predicted probabilities.
    """

    def __init__(
        self,
        method: Literal["cusum", "sr", "sprt"],
        null_prob: float,
        grid_low: float,
        grid_high: float,
        grid_step: float,
        auditor_grid_low: float,
        auditor_grid_high: float,
        auditor_grid_step: float,
        test_type: Literal["robustness", "auditor", "dual"] = "dual",
        m_min: int | None = None,
        num_init: int = 0,
    ):
        """Initialize mixture combined e-detector.

        Args:
            method: 'sr', 'cusum', or 'sprt' for the robustness detector.
            null_prob: Null probability shared by both detectors.
            grid_low: Lower bound of robustness grid (< null_prob).
            grid_high: Upper bound of robustness grid (< null_prob).
            grid_step: Step size for robustness grid.
            auditor_grid_low: Lower bound of auditor grid (> null_prob).
            auditor_grid_high: Upper bound of auditor grid.
            auditor_grid_step: Step size for auditor grid.
            test_type: Which test(s) to run.
            m_min: Required for 'auditor' and 'dual' modes.
            num_init: Number of initial observations (for plotting offset).
        """
        self.method = method
        self.method_name = f"{method}_mixture"
        self.null_prob = null_prob
        self.test_type = test_type
        self.m_min = m_min
        self.num_init = num_init

        if test_type in ("auditor", "dual"):
            assert m_min is not None, f"m_min is required for test_type='{test_type}'"

        self.robustness = (
            MixtureEDetectorRobustness(method=method, null_prob=null_prob, m_min=m_min,
                                       grid_low=grid_low, grid_high=grid_high, grid_step=grid_step)
            if test_type in ("robustness", "dual")
            else None
        )
        self.auditor = (
            MixtureEDetectorAuditor(null_prob=null_prob, m_min=m_min,
                                    auditor_grid_low=auditor_grid_low, auditor_grid_high=auditor_grid_high,
                                    auditor_grid_step=auditor_grid_step)
            if test_type in ("auditor", "dual")
            else None
        )
        self._t = 0

    def update(self, observation: int) -> None:
        """Update enabled detectors with a new observation."""
        self._t += 1
        if self.robustness is not None:
            self.robustness.update(observation)
        if self.auditor is not None:
            self.auditor.update(observation)


    def init_update(self, observation: int) -> None:
        """Update enabled detectors with a new observation."""
        self._t += 1
        if self.robustness is not None:
            self.robustness.init_update(observation)
        if self.auditor is not None:
            self.auditor.init_update(observation)
