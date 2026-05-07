"""History tracking for sequential testing rounds."""
import logging
import numpy as np
import pandas as pd
from scipy import stats

class SequentialTestHistory:
    """Tracks per-round information from sequential testing for later analysis."""

    def __init__(self):
        self.records = []
        self.num_test_obs = 0
        self.num_label_only = 0
        self.num_init_samples = 0

    def increment_test_obs(self):
        """Increment the count of observations that went through the sequential test."""
        self.num_test_obs += 1

    def increment_label_only(self):
        """Increment the count of observations that only got labels (skipped test)."""
        self.num_label_only += 1

    def set_num_init_samples(self, n: int):
        """Set the number of initialization samples (labeled but not tested)."""
        self.num_init_samples = n
        self.num_label_only += n

    @property
    def num_labeled(self) -> int:
        """Total number of labeled observations (test + label_only, includes init)."""
        return self.num_test_obs + self.num_label_only

    def add_record(
        self,
        iteration: int,
        method_name: str,
        question_id: str,
        label: int,
        subgroup_pct: float,
        metadata: str,
        pred_prob: float | None,
        statistics: dict[str, float],
        is_label_only: bool,
        oracle_prob: float | None = None,
    ):
        """Add a record for one round of sequential testing.

        Args:
            iteration: Current iteration number
            method_name: Identifier for the tester configuration
            question_id: ID of the sampled question
            label: Binary label (0 or 1)
            subgroup_pct: Percentage of remaining candidates in subgroup
            metadata: Metadata string for the question
            pred_prob: Predicted probability from sampler (None if not available)
            statistics: Dict mapping statistic label to value (e.g., {'robustness': 1.23})
            is_label_only: Whether this observation was label-only (skipped test)
            oracle_prob: Oracle true probability for the sampled observation (None if not available)
        """
        record = {
            "iteration": iteration,
            "method": method_name,
            "question_id": question_id,
            "label": label,
            "subgroup_pct": subgroup_pct,
            "pred_prob": pred_prob,
            "is_label_only": is_label_only,
            "oracle_prob": oracle_prob,
        }
        for stat_label, stat_value in statistics.items():
            record[f"{stat_label}_statistic"] = stat_value
        record["metadata"] = metadata
        self.records.append(record)

        if len(self.records) > 5:
            oracle_probs = np.array([r["oracle_prob"] for r in self.records])
            pred_probs = np.array([r["pred_prob"] for r in self.records])
            if pred_probs[0] is not None and oracle_probs[0] is not None:
                pearson_corr_probs = stats.pearsonr(oracle_probs, pred_probs)
                spearman_corr_probs = stats.spearmanr(oracle_probs, pred_probs)
                mse = np.mean(np.power(oracle_probs - pred_probs, 2))
                logging.info(f"CORRELATION {pearson_corr_probs}")
                logging.info(f"CORRELATION {spearman_corr_probs}")
                logging.info(f"RMSE {np.sqrt(mse)}")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to a DataFrame."""
        return pd.DataFrame(self.records)

    def to_csv(self, path: str):
        """Write history to a CSV file."""
        history_df = self.to_dataframe()
        if 'oracle_prob' in history_df.columns:
            logging.info(f"Auditor's average oracle prob {np.mean(history_df['oracle_prob'])}")
        history_df.to_csv(path, index=False)

    def __len__(self) -> int:
        return len(self.records)
