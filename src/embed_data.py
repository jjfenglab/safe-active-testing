"""Pre-compute embeddings for a dataset and save as CSV file with question_id index."""

import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings")
    parser.add_argument("--in-csv", type=str, required=True, help="Input CSV with 'question' and 'question_id' columns")
    parser.add_argument("--out-csv", type=str, required=True, help="Output CSV file with question_id index")
    parser.add_argument("--model", type=str, default="sentence-transformers/allenai-specter")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    assert "question" in df.columns, "Input CSV must have 'question' column"
    assert "question_id" in df.columns, "Input CSV must have 'question_id' column"

    model = SentenceTransformer(args.model)
    embeddings = model.encode(df["question"].tolist(), show_progress_bar=True)

    # If question_aux column exists, embed it and concatenate
    if "question_aux" in df.columns:
        embeddings_aux = model.encode(df["question_aux"].tolist(), show_progress_bar=True)
        embeddings = np.concatenate([embeddings, embeddings_aux], axis=1)

    embeddings_df = pd.DataFrame(
        embeddings,
        index=df["question_id"],
    )
    embeddings_df.index.name = "question_id"
    embeddings_df.to_csv(args.out_csv)
    print(f"Saved {embeddings.shape} embeddings to {args.out_csv}")


if __name__ == "__main__":
    main()
