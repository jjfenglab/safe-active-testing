"""Assemble CUB-200-2011 data for active testing with simulated labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms


def load_simulation_config(config_path: str) -> dict[str, float]:
    """Load simulation config mapping species to correctness probabilities.

    Args:
        config_path: Path to JSON config file

    Returns:
        Dict mapping species string to probability
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    key = "species_correctness_probs"
    assert key in config, f"Expected key '{key}' in config, found: {list(config.keys())}"
    return config[key]


def load_cub_data(
    data_dir: str,
    simulation_config: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Load CUB-200-2011 data with simulated labels.

    Args:
        data_dir: Path to CUB_200_2011 directory
        simulation_config: Dict mapping species to correctness probability
        rng: Random number generator for simulation

    Returns:
        DataFrame with image_id, image_path, species, is_correct_prob, metadata
    """
    if simulation_config is not None:
        assert rng is not None, "rng required when simulation_config is provided"

    data_path = Path(data_dir)

    # Load images list
    images_df = pd.read_csv(
        data_path / "images.txt",
        sep=" ",
        header=None,
        names=["image_id", "image_path"],
    )

    # Load class labels
    labels_df = pd.read_csv(
        data_path / "image_class_labels.txt",
        sep=" ",
        header=None,
        names=["image_id", "class_id"],
    )

    # Load class names
    classes_df = pd.read_csv(
        data_path / "classes.txt",
        sep=" ",
        header=None,
        names=["class_id", "class_name"],
    )

    # Merge dataframes
    df = images_df.merge(labels_df, on="image_id")
    df = df.merge(classes_df, on="class_id")

    # Determine is_correct_prob based on species
    rows = []
    for _, row in df.iterrows():
        species = row["class_name"]

        if simulation_config is not None:
            assert species in simulation_config, (
                f"Species '{species}' not found in simulation config"
            )
            prob_correct = simulation_config[species]
        else:
            prob_correct = 0.9  # default

        # Determine if it's a "black" bird based on name
        is_black_bird = "black" in species.lower()

        metadata = {
            "species": species,
            "class_id": int(row["class_id"]),
            "is_black_bird": is_black_bird,
        }

        rows.append({
            "image_id": row["image_id"],
            "image_path": str(data_path / "images" / row["image_path"]),
            "species": species,
            "is_correct_prob": prob_correct,
            "metadata": json.dumps(metadata),
        })

    return pd.DataFrame(rows)


def compute_embeddings(
    df: pd.DataFrame,
    batch_size: int = 32,
) -> np.ndarray:
    """Compute ResNet-50 embeddings for images.

    Args:
        df: DataFrame with 'image_path' column
        batch_size: Batch size for processing

    Returns:
        Array of embeddings with shape (num_images, embedding_dim)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained ResNet-50 and remove classification head
    model = models.resnet18(weights=models.ResNet18_Weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
    model = model.to(device)
    model.eval()

    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    embeddings = []
    image_paths = df["image_path"].tolist()

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_tensors = []

            for path in batch_paths:
                img = Image.open(path).convert("RGB")
                tensor = preprocess(img)
                batch_tensors.append(tensor)

            batch = torch.stack(batch_tensors).to(device)
            features = model(batch)
            features = features.squeeze(-1).squeeze(-1)  # Remove spatial dims
            embeddings.append(features.cpu().numpy())

            print(f"Processed {i + len(batch_paths)}/{len(image_paths)} images")

    return np.vstack(embeddings)


def main():
    parser = argparse.ArgumentParser(
        description="Assemble CUB-200-2011 data for active testing"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to CUB_200_2011 directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        help="Output CSV path for data",
    )
    parser.add_argument(
        "--simulate-labels",
        type=str,
        default=None,
        help="Path to JSON config for simulating is_correct labels",
    )
    parser.add_argument(
        "--out-embeddings-csv",
        type=str,
        default=None,
        help="Output path for embeddings CSV",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for computing embeddings",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    simulation_config = None
    if args.simulate_labels is not None:
        simulation_config = load_simulation_config(args.simulate_labels)

    df = load_cub_data(
        args.data_dir,
        simulation_config=simulation_config,
        rng=rng,
    )
    assert len(df) > 0, "No data found"

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"Wrote {len(df)} rows to {args.out_csv}")

    # Compute and save embeddings if requested
    if args.out_embeddings_csv is not None:
        embeddings = compute_embeddings(df, batch_size=args.batch_size)
        # Save embeddings with image_id as index for alignment
        embeddings_df = pd.DataFrame(
            embeddings,
            index=df["image_id"],
        )
        embeddings_df.index.name = "image_id"
        embeddings_df.to_csv(args.out_embeddings_csv)
        print(f"Wrote embeddings ({embeddings.shape}) to {args.out_embeddings_csv}")


if __name__ == "__main__":
    main()
