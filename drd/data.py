from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_dataframe(data_dir: Path) -> pd.DataFrame:
    image_paths: list[str] = []
    labels: list[str] = []

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in class_dir.iterdir():
            if image_path.is_file():
                image_paths.append(str(image_path.resolve()))
                labels.append(class_dir.name)

    if not image_paths:
        raise ValueError(f"No images found under: {data_dir}")

    return pd.DataFrame({"Image": image_paths, "Labels": labels})


def split_dataframe(
    dataframe: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataframe = shuffle(dataframe, random_state=seed).reset_index(drop=True)
    train_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=seed,
        stratify=dataframe["Labels"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def create_generators(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_size: tuple[int, int],
    batch_size: int,
    val_split: float,
    seed: int,
):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        vertical_flip=True,
        validation_split=val_split,
    )

    # Keep validation deterministic and free from augmentation.
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=None,
        x_col="Image",
        y_col="Labels",
        target_size=image_size,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        subset="training",
        seed=seed,
    )

    validation_generator = val_datagen.flow_from_dataframe(
        train_df,
        directory=None,
        x_col="Image",
        y_col="Labels",
        target_size=image_size,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        subset="validation",
        shuffle=False,
        seed=seed,
    )

    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        directory=None,
        x_col="Image",
        y_col="Labels",
        target_size=image_size,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    return train_generator, validation_generator, test_generator

