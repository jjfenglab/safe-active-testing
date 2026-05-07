"""GPQA dataset loader."""

import json
import logging
from typing import Dict, List, Optional

import pandas as pd

from src.data.dataloader import Dataloader
from src.data.dataloader_factory import normalize_for_matching

logger = logging.getLogger(__name__)


class GPQADataLoader(Dataloader):
    """Data loader for GPQA dataset."""

    # Mapping from full difficulty names to shortened versions
    DIFFICULTY_MAPPING = {
        "Easy undergraduate level (or easier)": "easy_undergrad",
        "Hard undergraduate level (could be a question on a hard undergraduate exam for students majoring in the subject)": "hard_undergrad",
        "Hard graduate level (could be a question on a hard graduate exam for PhD students in the domain)": "hard_graduate",
        "Post-graduate level or harder (only individuals with years of highly specialized expertise could reliably answer correctly)": "post_graduate",
    }

    def __init__(self, data_path: str, metadata_path: Optional[str] = None, **kwargs):
        """Initialize GPQA adapter.

        Args:
            data_path: Path to the GPQA JSON data file
            metadata_path: Optional path to the GPQA metadata CSV file
            **kwargs: Additional parameters
        """
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.data = self._load_data()
        self.metadata = self._load_metadata() if metadata_path else None

        # Cache for extracted data
        self._questions_df = None
        self._responses_df = None

    def _load_data(self) -> Dict:
        """Load the GPQA JSON data.

        Returns:
            Dictionary with GPQA data
        """
        with open(self.data_path, "r") as f:
            return json.load(f)

    def _load_metadata(self) -> pd.DataFrame:
        """Load the GPQA metadata.

        Returns:
            DataFrame with GPQA metadata
        """
        metadata_df = pd.read_csv(self.metadata_path)

        # Prepare the metadata for merging
        metadata_df["difficulty"] = metadata_df["Writer's Difficulty Estimate"]
        metadata_df["domain"] = metadata_df["High-level domain"]
        metadata_df["normalized_question"] = metadata_df["Question"].apply(
            normalize_for_matching
        )

        # Drop rows where difficulty is NaN
        metadata_df = metadata_df.dropna(subset=["difficulty"])

        # Add shortened difficulty column
        metadata_df["difficulty_level"] = metadata_df["difficulty"].map(
            self.DIFFICULTY_MAPPING
        )

        return metadata_df

    def extract_questions_with_answers(self) -> pd.DataFrame:
        """Extract questions and correct answers from GPQA dataset.

        Returns:
            DataFrame with questions and correct answers
        """
        if self._questions_df is not None:
            return self._questions_df

        questions = []

        for q_id, q_data in self.data.items():
            question_text = q_data.get("input_text", "")

            # Get answer options and map to proper format if needed
            answer_options = q_data.get("references", [])
            if answer_options:
                answer_options = {
                    chr(65 + int(i)): v
                    for i, v in enumerate(answer_options)
                    if isinstance(i, int)
                }
            correct_answer = q_data.get("correct_answer", "")

            questions.append(
                {
                    "question_id": q_id,
                    "question": question_text,
                    "answer_options": answer_options,
                    "correct_answer": correct_answer,
                }
            )

        questions_df = pd.DataFrame(questions)

        # Merge with original GPQA metadata if available
        if self.metadata is not None:
            # Add normalized question column for matching
            questions_df["normalized_question"] = questions_df["question"].apply(
                normalize_for_matching
            )

            # Merge based on normalized question text
            merged_df = pd.merge(
                questions_df,
                self.metadata[
                    [
                        "normalized_question",
                        "difficulty",
                        "domain",
                        "difficulty_level",
                    ]
                ],
                on="normalized_question",
                how="inner",  # ensure we drop questions without difficulty/domain info
            )

            # Remove the normalized_question column as it's no longer needed
            merged_df = merged_df.drop(columns=["normalized_question"])
            self._questions_df = merged_df
            return merged_df

        self._questions_df = questions_df
        return questions_df

    def extract_candidate_responses(
        self, candidate_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Extract responses from candidate models in GPQA dataset.

        Args:
            candidate_ids: Optional list of candidate IDs to filter by

        Returns:
            DataFrame with candidate responses
        """
        if self._responses_df is not None and candidate_ids is None:
            return self._responses_df

        responses = []

        for q_id, q_data in self.data.items():
            for model_id, model_data in q_data.get("models", {}).items():
                if candidate_ids and model_id not in candidate_ids:
                    continue

                candidate_answer = model_data.get("answer", "")
                candidate_rationale = model_data.get("output_text", "")
                is_correct = model_data.get("is_correct", None)

                responses.append(
                    {
                        "question_id": q_id,
                        "candidate_id": model_id,
                        "candidate_answer": candidate_answer,
                        "candidate_rationale": candidate_rationale,
                        "is_correct": is_correct,
                    }
                )

        responses_df = pd.DataFrame(responses)

        if candidate_ids is None:
            self._responses_df = responses_df

        return responses_df

    def get_data(self) -> pd.DataFrame:
        """Get all GPQA questions and metadata.

        Returns:
            DataFrame with questions and metadata
        """
        return self.extract_questions_with_answers()

    def get_candidate_responses(
        self, candidate_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get all candidate responses to GPQA questions.

        Args:
            candidate_ids: Optional list of candidate IDs to filter by

        Returns:
            DataFrame with candidate responses
        """
        return self.extract_candidate_responses(candidate_ids)

    def get_score_type(self) -> str:
        """Return score type (binary for GPQA).

        Returns:
            Score type string
        """
        return "binary"

    def get_metadata_fields(self) -> List[str]:
        """Return available metadata fields for stratification.

        Returns:
            List of metadata field names
        """
        questions_df = self.extract_questions_with_answers()
        fields = []

        if "difficulty_level" in questions_df.columns:
            fields.append("difficulty_level")
        if "domain" in questions_df.columns:
            fields.append("domain")

        return fields

    def get_ground_truth_scores(self) -> pd.DataFrame:
        """Get ground truth scores for GPQA dataset.

        Returns:
            DataFrame with ground truth scores for each candidate
        """
        if self._responses_df is None:
            responses_df = self.extract_candidate_responses()
        else:
            responses_df = self._responses_df

        # Get unique question-candidate pairs
        question_candidate_pairs = responses_df[
            ["question_id", "candidate_id", "is_correct"]
        ].drop_duplicates()

        # Group by candidate_id to get the average score
        avg_ground_truth_scores = (
            question_candidate_pairs.groupby("candidate_id")["is_correct"]
            .mean()
            .reset_index()
            .rename(columns={"is_correct": "ground_truth_score"})
        )

        return avg_ground_truth_scores
