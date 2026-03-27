from pathlib import Path
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image


def plot_class_distribution(dataframe: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=dataframe, x="Labels")
    plt.xlabel("Class Labels")
    plt.ylabel("Count")
    plt.title("Distribution of Class Labels")
    plt.tight_layout()
    plt.show()


def plot_class_samples(data_dir: Path, max_classes: int = 5, samples_per_class: int = 5) -> None:
    class_dirs = [path for path in sorted(data_dir.iterdir()) if path.is_dir()][:max_classes]
    if not class_dirs:
        return

    fig, axes = plt.subplots(max_classes, samples_per_class, figsize=(20, 20))

    for row in range(max_classes):
        for col in range(samples_per_class):
            axes[row, col].axis("off")

    for row, class_dir in enumerate(class_dirs):
        image_paths = [path for path in sorted(class_dir.iterdir()) if path.is_file()]
        for col, image_path in enumerate(image_paths[:samples_per_class]):
            with Image.open(image_path) as image:
                axes[row, col].imshow(image)
            axes[row, col].set_title(class_dir.name)
            axes[row, col].axis("off")

    fig.tight_layout()
    plt.show()


def plot_random_predictions(images, y_true, y_pred, rows: int = 10) -> None:
    if not images:
        return

    count = min(rows, len(images))
    fig = plt.figure(figsize=(10, 2 * count))

    for i in range(count):
        idx = random.randrange(len(images))
        fig.add_subplot(count, 1, i + 1)
        plt.xlabel(f"Prediction: {y_pred[idx]} | Original: {y_true[idx]}")
        plt.imshow(images[idx])
        plt.axis("off")

    fig.tight_layout()
    plt.show()


def plot_confusion(cm, classes) -> None:
    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, xticklabels=classes, yticklabels=classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Original")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

