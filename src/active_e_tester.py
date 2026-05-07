"""Active E-Tester classes for sequential testing."""
import time
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.history import SequentialTestHistory
from src.question_bank import QuestionBankSampler
from src.samplers import BaseSampler
from src.sequential_tests import BaseSequentialTest


@dataclass
class ObservationInfo:
    """Info about a sampled observation."""
    label: int
    embedding: np.ndarray
    metadata: str
    question_id: str
    oracle_prob: float | None = None


class ETesterSampler:
    """Runs sequential e-testing with any sampler.

    Replaces the run_sequential_test function with a class-based approach.
    The auditor selects from all remaining candidates in the pool.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embedding_model,
        sequential_test: BaseSequentialTest,
        sampler: BaseSampler,
        alpha: float,
        max_samples: int,
        num_init: int,
        rng: np.random.Generator,
        adaptive: bool = False,
        iter_require_auditor_test: int | None = None,
        allow_label_only: bool = False,
        precomputed_embeddings: np.ndarray | None = None,
        test_column_name: str = "question",
        debug_pred_dir: str | None = None,
        debug_sampler_dir: str | None = None,
        id_col: str = "question_id",
    ):
        """Initialize the active e-tester.

        Args:
            df: DataFrame with id_col, question, is_correct, metadata columns
            embedding_model: SentenceTransformer model for embedding questions (unused if precomputed_embeddings provided)
            sequential_test: Initialized sequential test
            sampler: Initialized sampler
            alpha: Significance level
            max_samples: Maximum number of samples
            num_init: Number of random samples to initialize the sampler (not passed to test)
            rng: Random number generator
            adaptive: If True, use predicted probabilities from sampler for e-detector
            iter_require_auditor_test: If not None, require testing after this iteration
            allow_label_only: If True, auditor can choose to label-only vs test.
                              If False, auditor must test every observation.
            precomputed_embeddings: Optional pre-computed embeddings array aligned with df rows.
                                    If provided, skips embedding computation.
            debug_pred_dir: If provided, save CSV with predicted probabilities for remaining
                            candidates at each iteration to this directory.
            debug_sampler_dir: If provided, save sampler pickle at each iteration to this directory.
        """
        self.df = df

        if precomputed_embeddings is not None:
            logging.info("Using pre-computed embeddings")
            assert len(precomputed_embeddings) == len(df), (
                f"Embeddings length ({len(precomputed_embeddings)}) != df length ({len(df)})"
            )
            self.embeddings = precomputed_embeddings
        elif embedding_model is not None:
            # Embed questions
            logging.info("Embedding questions...")
            embeddings = embedding_model.encode(df[test_column_name].tolist(), show_progress_bar=False)
            self.embeddings = np.array(embeddings)
        else:
            raise ValueError("embedding model not available")
        self.sequential_test = sequential_test
        self.sampler = sampler
        self.alpha = alpha
        self.max_samples = max_samples
        self.num_init = num_init
        self.rng = rng
        self.adaptive = adaptive
        self.iter_require_auditor_test = iter_require_auditor_test
        self.allow_label_only = allow_label_only

        if adaptive:
            assert hasattr(sampler, "predict"), "Adaptive mode requires sampler with predict() method"

        # Debug settings
        self.debug_pred_dir = debug_pred_dir
        self.debug_sampler_dir = debug_sampler_dir
        if debug_pred_dir:
            Path(debug_pred_dir).mkdir(parents=True, exist_ok=True)
        if debug_sampler_dir:
            Path(debug_sampler_dir).mkdir(parents=True, exist_ok=True)

        # Extract data arrays
        self.id_col = id_col
        self.question_ids = df[id_col].values
        self.labels = df["is_correct"].values
        self.metadata_list = df["metadata"].values

        # Extract oracle probabilities if available
        self.oracle_probs = df["is_correct_prob"].values if "is_correct_prob" in df.columns else None

        # Track remaining candidates
        self.remaining_mask = np.ones(len(df), dtype=bool)
        self.history = SequentialTestHistory()
    
    @property
    def name(self):
        return f"{self.sequential_test.method_name}_{self.sampler.sampler_name}_abstain{int(self.allow_label_only)}"

    @property
    def name(self):
        return f"{self.sequential_test.method_name}_{self.sampler.sampler_name}_abstain{int(self.allow_label_only)}"

    def _initialize_sampler(self):
        """Initialize sampler with random samples (not passed to sequential test)."""
        if self.num_init <= 0:
            return

        n_init_actual = 0
        for i in range(self.num_init):
            embeddings, ids, context = self._get_round_candidates()
            if len(ids) == 0:
                break

            # Pick randomly instead of using sampler
            random_idx = self.rng.integers(len(ids))
            obs_info = self._extract_observation(random_idx, context, embeddings, ids)

            # Only update sampler (not sequential test) and state
            self.sampler.update(obs_info.embedding, obs_info.label)
            # HACK: we dont need the ewaf to start as early -- currently hard coded
            if i >= (self.num_init - 5):
                self.sequential_test.init_update(obs_info.label)
            self._post_sample_update(obs_info)
            n_init_actual += 1

        self.history.set_num_init_samples(n_init_actual)
        logging.info(f"Initialized sampler with {n_init_actual} random samples")

    def _check_stopping(self, iteration: int) -> tuple[bool, str | None]:
        """Check if we should stop the test.

        Returns:
            Tuple of (should_stop, reason) where reason is None if not stopping
        """
        rejected, which_test = self.sequential_test.check_rejection(self.alpha)
        if rejected:
            return True, f"Rejected {which_test}'s null at iteration {iteration}"

        if iteration >= self.max_samples:
            return True, f"Reached max samples ({self.max_samples})"

        if not np.any(self.remaining_mask):
            return True, "No candidates left"

        return False, None

    def _get_round_candidates(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Get candidates for this round of sampling.

        Returns:
            Tuple of (embeddings, ids, context) where context contains oracle_probs if available
        """
        remaining_embeddings = self.embeddings[self.remaining_mask]
        remaining_ids = self.question_ids[self.remaining_mask]
        remaining_oracle_probs = self.oracle_probs[self.remaining_mask] if self.oracle_probs is not None else None
        return remaining_embeddings, remaining_ids, {"oracle_probs": remaining_oracle_probs}

    def _extract_observation(
        self, sampled_idx: int, context: dict, embeddings: np.ndarray, ids: np.ndarray
    ) -> ObservationInfo:
        """Extract observation info after sampling.

        Args:
            sampled_idx: Index of sampled candidate in the candidate arrays
            context: Round-specific context from _get_round_candidates
            embeddings: Candidate embeddings array
            ids: Candidate IDs array

        Returns:
            ObservationInfo with label, embedding, metadata, question_id, oracle_prob
        """
        sampled_id = ids[sampled_idx]
        idx = np.where(self.question_ids == sampled_id)[0][0]
        oracle_prob = float(self.oracle_probs[idx]) if self.oracle_probs is not None else None
        return ObservationInfo(
            label=int(self.labels[idx]),
            embedding=self.embeddings[idx],
            metadata=self.metadata_list[idx],
            question_id=sampled_id,
            oracle_prob=oracle_prob,
        )

    def _post_sample_update(self, obs_info: ObservationInfo):
        """Update state after sampling (e.g., remove from pool). Override in subclass if needed."""
        idx = np.where(self.question_ids == obs_info.question_id)[0][0]
        self.remaining_mask[idx] = False

    def _save_debug_info(self, iteration: int, embeddings: np.ndarray, ids: np.ndarray, context: dict):
        """Save debug information: predicted probabilities and sampler state.

        Args:
            iteration: Current iteration number
            embeddings: Embeddings of remaining candidates
            ids: IDs of remaining candidates
            context: Context dict from _get_round_candidates (unused in base class, but subclasses may use)
        """
        if not self.debug_pred_dir and not self.debug_sampler_dir:
            return

        # Save predicted probabilities CSV
        if self.debug_pred_dir and hasattr(self.sampler, "predict"):
            pred_probs = self.sampler.predict(embeddings)

            # Get metadata for remaining candidates
            remaining_indices = np.where(self.remaining_mask)[0]
            metadata_list = [self.metadata_list[i] for i in remaining_indices]
            labels = [int(self.labels[i]) for i in remaining_indices]

            rows = []
            for i, (qid, pred_prob, metadata, label) in enumerate(zip(ids, pred_probs, metadata_list, labels)):
                row = {
                    "question_id": qid,
                    "pred_prob": pred_prob,
                    "is_correct": label,
                }
                # Parse metadata JSON and add fields
                try:
                    meta_dict = json.loads(metadata)
                    row.update(meta_dict)
                except (json.JSONDecodeError, TypeError):
                    row["metadata"] = metadata
                rows.append(row)

            pred_df = pd.DataFrame(rows)
            pred_path = Path(self.debug_pred_dir) / f"iter_{iteration:04d}_predictions.csv"
            pred_df.to_csv(pred_path, index=False)

        # Save sampler pickle
        if self.debug_sampler_dir:
            sampler_path = Path(self.debug_sampler_dir) / f"iter_{iteration:04d}_sampler.pkl"
            with open(sampler_path, "wb") as f:
                pickle.dump(self.sampler, f)

    def _should_test_model_null(self, pred_prob: float | None, iteration: int) -> bool:
        """Determine if observation should be tested vs label-only."""
        # no label-only option
        if not self.allow_label_only:
            return True
        
        # if there is no predicted probability, then the label-only option
        if pred_prob is None:
            return True

        return pred_prob < self.sequential_test.null_prob

    def _should_test_audit_null(self, pred_prob: float | None, iteration: int) -> bool:
        """Determine if observation should be tested vs label-only."""
        return self.iter_require_auditor_test is not None and iteration >= self.iter_require_auditor_test


    def _process_and_record(
        self,
        obs_info: ObservationInfo,
        subgroup_size: int,
        num_candidates: int,
        iteration: int,
    ) -> bool:
        """Process observation: predict, test/label, update sampler, record history.

        Args:
            obs_info: Observation info
            subgroup_size: Size of subgroup from which sample was drawn
            num_candidates: Total candidates this round
            iteration: Current iteration

        Returns:
            is_label_only: True if observation was label-only (skipped test)
        """
        subgroup_pct = 100.0 * subgroup_size / num_candidates

        # Get predicted probability (use cached value from sampling if available)
        pred_prob = None
        if hasattr(self.sampler, "_cached_pred_prob"):
            pred_prob = self.sampler._cached_pred_prob
        elif hasattr(self.sampler, "predict"):
            pred_prob = self.sampler.predict(obs_info.embedding.reshape(1, -1))[0]

        # Determine if we should test
        is_label_only = True
        if self._should_test_audit_null(pred_prob, iteration):
            is_label_only = False
            # We have to test now, because we have abstained too long (implicitly or explictly)
            # Cannot be adaptive anymore either for testing the auditor's null
            logging.info(f"NOT ADAPTIVE {iteration}")
            if self.adaptive:
                self.sequential_test.update(obs_info.label, pred_prob, is_adaptive_auditor=False)
            else:
                self.sequential_test.update(obs_info.label)
        elif self._should_test_model_null(pred_prob, iteration):
            is_label_only = False
            if self.adaptive:
                self.sequential_test.update(obs_info.label, pred_prob, is_adaptive_auditor=True)
            else:
                self.sequential_test.update(obs_info.label)
        else:
            logging.info("NOT TESTING")

        # Update sampler with label
        # st_time = time.time()
        self.sampler.update(obs_info.embedding, obs_info.label)
        # logging.info(f"SAMPLER UPDATE {time.time() - st_time}")

        # Update state (e.g., remove from pool)
        # st_time = time.time()
        self._post_sample_update(obs_info)
        # logging.info(f"POST UPDATE {time.time() - st_time}")

        # Log and record
        stats = self.sequential_test.get_statistics(self.alpha)
        statistics = {s["label"]: s["statistic"] for s in stats}

        self.history.add_record(
            iteration=iteration,
            method_name=self.name,
            question_id=obs_info.question_id,
            label=obs_info.label,
            subgroup_pct=subgroup_pct,
            metadata=obs_info.metadata,
            pred_prob=pred_prob,
            statistics=statistics,
            is_label_only=is_label_only,
            oracle_prob=obs_info.oracle_prob,
        )

        stats_str = ", ".join(f"{s['label']}={s['statistic']:.4f}" for s in stats)
        logging.info(
            f"Iter {iteration}: id={obs_info.question_id}, label={obs_info.label}, "
            f"{stats_str}, label_only={is_label_only}, metadata={obs_info.metadata}"
        )

        return is_label_only

    def run(self) -> SequentialTestHistory:
        """Run the sequential testing procedure.

        Returns:
            SequentialTestHistory object with per-round information and counts
        """
        self._initialize_sampler()

        iteration = 0

        while True:
            should_stop, reason = self._check_stopping(iteration)
            if should_stop:
                logging.info(reason)
                break

            # Get candidates for this round
            embeddings, ids, context = self._get_round_candidates()

            # Save debug info before sampling (captures predictions for all remaining candidates)
            if iteration % 20 == 0:
                self._save_debug_info(iteration, embeddings, ids, context)

            # Sample from candidates
            sampled_idx, subgroup_size = self.sampler._sample_impl(embeddings, ids)

            # Extract observation info
            obs_info = self._extract_observation(sampled_idx, context, embeddings, ids)

            # Process: predict, test/label, update, record
            is_label_only = self._process_and_record(
                obs_info, subgroup_size, len(ids), iteration
            )

            if is_label_only:
                self.history.increment_label_only()
            else:
                self.history.increment_test_obs()

            iteration += 1

        return self.history


class ETesterGenerator(ETesterSampler):
    """E-Tester that samples from generated question candidates.

    Instead of selecting from all remaining questions, each round:
    1. Sample N×K candidates from question bank (one per domain-difficulty cell)
    2. Get pre-computed embeddings (or compute if not available)
    3. Use sampler to rank among only these candidates
    4. Select one candidate for labeling/testing
    """

    def __init__(
        self,
        question_bank_sampler: QuestionBankSampler,
        sequential_test: BaseSequentialTest,
        sampler: BaseSampler,
        alpha: float,
        max_samples: int,
        num_init: int,
        rng: np.random.Generator,
        adaptive: bool = False,
        iter_require_auditor_test: int | None = None,
        allow_label_only: bool = True,
        debug_pred_dir: str | None = None,
        debug_sampler_dir: str | None = None,
    ):
        """Initialize the generator-based active e-tester.

        Simulates generation of questions by sampling from a question bank.

        Args:
            question_bank_sampler: Sampler for question bank (may include precomputed embeddings)
            sequential_test: Initialized sequential test
            sampler: Initialized sampler
            alpha: Significance level
            max_samples: Maximum number of samples
            num_init: Number of random samples to initialize the sampler
            rng: Random number generator
            adaptive: If True, use predicted probabilities from sampler for e-detector
            iter_require_auditor_test: If not None, require testing after this iteration
            allow_label_only: If True, auditor can choose to label-only vs test
            debug_pred_dir: If provided, save CSV with predicted probabilities for candidates
                            at each iteration to this directory.
            debug_sampler_dir: If provided, save sampler pickle at each iteration to this directory.
        """
        # Don't call parent __init__ since we don't have df/embeddings upfront
        self.question_bank_sampler = question_bank_sampler
        self.sequential_test = sequential_test
        self.sampler = sampler
        self.alpha = alpha
        self.max_samples = max_samples
        self.num_init = num_init
        self.rng = rng
        self.adaptive = adaptive
        self.iter_require_auditor_test = iter_require_auditor_test
        self.allow_label_only = allow_label_only

        # Debug settings
        self.debug_pred_dir = debug_pred_dir
        self.debug_sampler_dir = debug_sampler_dir
        if debug_pred_dir:
            Path(debug_pred_dir).mkdir(parents=True, exist_ok=True)
        if debug_sampler_dir:
            Path(debug_sampler_dir).mkdir(parents=True, exist_ok=True)

        if adaptive:
            assert hasattr(sampler, "predict"), "Adaptive mode requires sampler with predict() method"

        self.history = SequentialTestHistory()
        self._iteration_counter = 0  # For generating unique question IDs

    def _check_stopping(self, iteration: int) -> tuple[bool, str | None]:
        """Check if we should stop the test."""
        rejected, which_test = self.sequential_test.check_rejection(self.alpha)
        if rejected:
            return True, f"Rejected {which_test}'s null at iteration {iteration}"

        if iteration >= self.max_samples:
            return True, f"Reached max samples ({self.max_samples})"

        if not self.question_bank_sampler.has_candidates():
            return True, "No candidates left in question bank"

        return False, None

    def _get_round_candidates(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Sample candidates from question bank and get embeddings.

        Returns:
            Tuple of (embeddings, ids, context) where context contains candidates_df and oracle_probs
        """
        candidates_df, embeddings = self.question_bank_sampler.sample_candidates()
        assert embeddings is not None
        ids = candidates_df['question_id']
        oracle_probs = candidates_df['is_correct_prob'].values if 'is_correct_prob' in candidates_df.columns else None
        return embeddings, ids, {"candidates_df": candidates_df, "oracle_probs": oracle_probs}

    def _extract_observation(
        self, sampled_idx: int, context: dict, embeddings: np.ndarray, ids: np.ndarray
    ) -> ObservationInfo:
        """Extract observation info from sampled candidate."""
        candidates_df = context["candidates_df"]
        selected_row = candidates_df.iloc[sampled_idx]

        self._iteration_counter += 1
        question_id = f"gen_{self._iteration_counter}"
        oracle_prob = float(selected_row["is_correct_prob"]) if "is_correct_prob" in selected_row else None
        return ObservationInfo(
            label=selected_row["is_correct"],
            embedding=embeddings[sampled_idx],
            metadata=selected_row["metadata"],
            question_id=question_id,
            oracle_prob=oracle_prob,
        )

    def _post_sample_update(self, obs_info: ObservationInfo):
        """No pool to update for generator mode - candidates are sampled fresh each round."""
        pass

    def _save_debug_info(self, iteration: int, embeddings: np.ndarray, ids: np.ndarray, context: dict):
        """Save debug information for generator mode using candidates_df from context."""
        if not self.debug_pred_dir and not self.debug_sampler_dir:
            return

        # Save predicted probabilities CSV
        if self.debug_pred_dir and hasattr(self.sampler, "predict"):
            pred_probs = self.sampler.predict(embeddings)
            candidates_df = context["candidates_df"]

            rows = []
            for i, (qid, pred_prob) in enumerate(zip(ids, pred_probs)):
                row_data = candidates_df.iloc[i]
                row = {
                    "question_id": qid,
                    "pred_prob": pred_prob,
                    "is_correct": int(row_data["is_correct"]),
                }
                # Parse metadata JSON and add fields
                metadata = row_data["metadata"]
                try:
                    meta_dict = json.loads(metadata)
                    row.update(meta_dict)
                except (json.JSONDecodeError, TypeError):
                    row["metadata"] = metadata
                rows.append(row)

            pred_df = pd.DataFrame(rows)
            pred_path = Path(self.debug_pred_dir) / f"iter_{iteration:04d}_predictions.csv"
            pred_df.to_csv(pred_path, index=False)

        # Save sampler pickle
        if self.debug_sampler_dir:
            sampler_path = Path(self.debug_sampler_dir) / f"iter_{iteration:04d}_sampler.pkl"
            with open(sampler_path, "wb") as f:
                pickle.dump(self.sampler, f)
