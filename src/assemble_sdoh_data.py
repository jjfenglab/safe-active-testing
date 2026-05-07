"""Assemble SDoH data for active testing by comparing LLM extractions to human annotations.

Uses the comparison logic from the Social Wayfinder project's eval_annotation.py
to compare LLM extraction output against human annotations. Each observation is a
(note, code) pair where is_correct = 1 if LLM and human agree on attribute presence,
0 otherwise.

Output format matches do_e_test.py expectations:
    question_id, question, is_correct, metadata
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Add exp_sdoh to path for local schemas import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp_sdoh"))
from schemas import GUIDEBOOK_MAPPINGS, ASNoteSummary


def clean_annotations(annotation_df: pd.DataFrame) -> pd.DataFrame:
    """Clean human annotation codes by splitting multi-code entries.

    Ported from eval_annotation.py in Social Wayfinder.
    """
    annotation_df = annotation_df.copy()
    annotation_df["Codes"] = annotation_df["Codes"].str.split("\n")
    annotation_df = annotation_df.explode("Codes").reset_index(drop=True)
    annotation_df["Codes"] = annotation_df["Codes"].str.strip()
    return annotation_df


def process_llm_human_codes(
    llm_annotation_df: pd.DataFrame,
    human_annotation_df: pd.DataFrame,
    uniq_codes: list[str],
    guidebook_mappings: dict,
    note_summary_cls,
) -> pd.DataFrame:
    """Compare LLM extractions to human annotations per (note, code).

    Ported from eval_annotation.py in Social Wayfinder.

    Returns DataFrame with columns: note_id, code, ann_0 (LLM), ann_1 (human)
    """
    long_df = []
    for note_id in llm_annotation_df.note_id.unique():
        llm_ann_data = llm_annotation_df[llm_annotation_df["note_id"] == note_id].iloc[0]
        human_ann_data = human_annotation_df[human_annotation_df["note_id"] == note_id]

        data = {
            "note_id": [note_id] * len(uniq_codes),
            "code": uniq_codes,
        }

        llm_summary = note_summary_cls.model_validate(json.loads(llm_ann_data["summary"]))
        llm_annotate_values = []
        for code in uniq_codes:
            attr_data = [
                value for value in guidebook_mappings.values() if value["code"] == code
            ][0]
            annotation = 1
            if (
                not attr_data["func"](llm_summary)
                or attr_data["func"](llm_summary) == "no"
                or attr_data["func"](llm_summary) == "unknown"
            ):
                annotation = 0
            llm_annotate_values.append(annotation)
        data["ann_0"] = pd.Series(llm_annotate_values, index=uniq_codes)

        human_codes = human_ann_data["Codes"].unique()
        human_values = pd.Series(
            [int(code in human_codes) for code in uniq_codes], index=uniq_codes
        )
        data["ann_1"] = human_values

        long_df.append(pd.DataFrame(data))

    return pd.concat(long_df).reset_index(drop=True)


def assemble_sdoh_data(
    llm_annotations_csv: str,
    human_annotations_csv: str,
    embed_mode: str = "definition_note",
) -> pd.DataFrame:
    """Compare LLM extractions to human annotations and produce pipeline-format CSV.

    Args:
        llm_annotations_csv: Path to llm_annotated_notes.csv (pipe-delimited)
        human_annotations_csv: Path to all_annotations.csv
        embed_mode: What text to use for the question field that gets embedded.
            - "definition_note": Full definition + clinical note text
            - "definition_only": Just the definition (no clinical note)

    Returns:
        DataFrame with columns: question_id, question, is_correct, metadata
    """
    assert embed_mode in ("definition_note", "definition_only"), f"Invalid embed_mode: {embed_mode}"
    # Load data
    llm_df = pd.read_csv(llm_annotations_csv, delimiter="|")
    human_df = pd.read_csv(human_annotations_csv)
    human_df = clean_annotations(human_df)

    # Filter LLM annotations to only notes with human annotations
    human_note_ids = human_df["note_id"].unique()
    filtered_llm_df = llm_df[llm_df["note_id"].isin(human_note_ids)]

    # Get the canonical code list from GUIDEBOOK_MAPPINGS
    uniq_codes = [value["code"] for value in GUIDEBOOK_MAPPINGS.values()]

    # Compare LLM vs human: produces ann_0 (LLM), ann_1 (human) per (note, code)
    comparison_df = process_llm_human_codes(
        filtered_llm_df, human_df, uniq_codes, GUIDEBOOK_MAPPINGS, ASNoteSummary
    )

    # Remove codes where both LLM and human are always absent across all notes
    # (e.g. aortic stenosis symptom codes that are irrelevant to social work notes)
    code_activity = comparison_df.groupby("code")[["ann_0", "ann_1"]].sum()
    active_codes = code_activity[
        (code_activity["ann_0"] > 0) | (code_activity["ann_1"] > 0)
    ].index
    comparison_df = comparison_df[comparison_df["code"].isin(active_codes)].reset_index(
        drop=True
    )

    # is_correct = LLM agrees with human
    comparison_df["is_correct"] = (comparison_df["ann_0"] == comparison_df["ann_1"]).astype(int)

    # Build the note text lookup
    note_text_lookup = dict(zip(
        filtered_llm_df["note_id"],
        filtered_llm_df["processed_note"].fillna(filtered_llm_df["note_text"])
    ))

    # Build code -> category mapping (reverse of GUIDEBOOK_MAPPINGS)
    code_to_category = {v["code"]: k for k, v in GUIDEBOOK_MAPPINGS.items()}

    rows = []
    for _, row in comparison_df.iterrows():
        note_id = row["note_id"]
        code = row["code"]
        category = code_to_category.get(code, code)
        note_text = note_text_lookup.get(note_id, "")

        question_id = f"{note_id}_{code}"
        definition = GUIDEBOOK_MAPPINGS.get(category, {}).get('definition', category)
        definition_text = f"Extract the following information from the clinical note: {definition}"

        metadata = {
            "category": category,
            "code": code,
            "note_id": str(note_id),
            "llm_present": int(row["ann_0"]),
            "human_present": int(row["ann_1"]),
        }

        row_data = {
            "question_id": question_id,
            "question": definition_text,
            "is_correct": int(row["is_correct"]),
            "metadata": json.dumps(metadata),
        }
        if embed_mode == "definition_note":
            row_data["question_aux"] = note_text
        rows.append(row_data)

    out_df = pd.DataFrame(rows)

    return out_df


def main():
    parser = argparse.ArgumentParser(
        description="Assemble SDoH data for active testing (human vs LLM comparison)"
    )
    parser.add_argument(
        "--llm-annotations-csv",
        type=str,
        required=True,
        help="Path to llm_annotated_notes.csv (pipe-delimited)",
    )
    parser.add_argument(
        "--human-annotations-csv",
        type=str,
        required=True,
        help="Path to all_annotations.csv",
    )
    parser.add_argument("--out-csv", type=str, required=True, help="Output CSV path")
    parser.add_argument(
        "--out-summary-csv",
        type=str,
        required=True,
        help="Output CSV path for category-level summary (average label values)",
    )
    parser.add_argument(
        "--embed-mode",
        type=str,
        default="definition_note",
        choices=["definition_note", "definition_only"],
        help="What text to embed: 'definition_note' (full) or 'definition_only' (no clinical note)",
    )

    args = parser.parse_args()

    out_df = assemble_sdoh_data(
        args.llm_annotations_csv,
        args.human_annotations_csv,
        embed_mode=args.embed_mode,
    )
    assert len(out_df) > 0, "No data produced"

    n_correct = out_df["is_correct"].sum()
    n_incorrect = len(out_df) - n_correct
    n_notes = out_df["question_id"].str.rsplit("_", n=1).str[0].nunique()

    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} rows ({n_correct} correct, {n_incorrect} incorrect, "
          f"{n_notes} notes) to {args.out_csv}")

    # Compute category-level summary with average label values
    out_df["category"] = out_df["metadata"].apply(lambda x: json.loads(x)["category"])
    summary_df = out_df.groupby("category")["is_correct"].agg(["mean", "count"]).reset_index()
    summary_df.columns = ["category", "avg_is_correct", "n_observations"]
    summary_df.sort_values("avg_is_correct").to_csv(args.out_summary_csv, index=False)
    print(f"Wrote category summary ({len(summary_df)} categories) to {args.out_summary_csv}")


if __name__ == "__main__":
    main()
