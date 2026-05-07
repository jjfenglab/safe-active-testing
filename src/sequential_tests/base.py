"""Base class for sequential tests."""

from abc import ABC, abstractmethod


class BaseSequentialTest(ABC):
    """Abstract base class for sequential hypothesis tests."""

    @abstractmethod
    def update(self, observation: int) -> None:
        """Update test with new observation.

        Args:
            observation: Binary observation (0 or 1)
        """
        pass

    @abstractmethod
    def get_statistic_history(self, alpha: float) -> list[dict]:
        """Get full history of statistics.

        Args:
            alpha: Significance level (used to compute thresholds)

        Returns:
            List of dicts with keys: 'label', 'statistics', 'threshold'
            Example: [
                {"label": "futility", "statistics": [1.0, 0.3, 1.4, ...], "threshold": 20.0},
                {"label": "efficacy", "statistics": [0.5, 0.8, ...], "threshold": 20.0},
            ]
        """
        pass

    def get_statistics(self, alpha: float) -> list[dict]:
        """Get current (last) statistics.

        Default implementation derives from get_statistic_history.

        Args:
            alpha: Significance level

        Returns:
            List of dicts with keys: 'label', 'statistic', 'threshold'
        """
        history = self.get_statistic_history(alpha)
        return [
            {
                "label": h["label"],
                "statistic": h["statistics"][-1] if h["statistics"] else None,
                "threshold": h["threshold"],
            }
            for h in history
        ]

    @abstractmethod
    def check_rejection(self, alpha: float) -> tuple[bool, str | None]:
        """Check if any test rejects.

        Args:
            alpha: Significance level

        Returns:
            Tuple of (rejected, which_label)
            which_label identifies which test rejected, or None if no rejection
        """
        pass

    @abstractmethod
    def plot(self, alpha: float, plot_path: str) -> None:
        """Create and save visualization.

        Args:
            alpha: Significance level for threshold line
            plot_path: Path to output plot file
        """
        pass
