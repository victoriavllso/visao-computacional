import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate confusion matrix for photo type evaluation."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=Path("data/evaluation/latest/back_dev.csv"),
        help="Path to the evaluation CSV file (default: data/evaluation/latest/back_dev.csv)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory to save output files (default: same directory as the input file)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_file: Path = args.input_file
    output_dir: Path = args.output_dir

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if output_dir is None:
        output_dir = input_file.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)

    # ============================================================
    # CONFUSION MATRIX: PHOTO TYPE
    # ============================================================

    typed = df.dropna(subset=["true_type", "photo_type"])

    labels = sorted(
        set(typed["true_type"].unique()) | set(typed["photo_type"].unique())
    )

    cm = confusion_matrix(
        typed["true_type"],
        typed["photo_type"],
        labels=labels,
    )

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels,
    )
    cm_df.index.name = "true_type"

    csv_file = output_dir / f"{input_file.stem}_photo_type_cm.csv"
    cm_df.to_csv(csv_file)
    print(f"Confusion matrix CSV saved to: {csv_file}")

    # Save confusion matrix as PNG
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.title(f"Photo Type Confusion Matrix ({input_file.stem})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    image_file = output_dir / f"{input_file.stem}_photo_type_confusion_matrix.png"
    plt.savefig(image_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix image saved to: {image_file}")


if __name__ == "__main__":
    main()
