import argparse
from pathlib import Path

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from drd.config import Config
from drd.data import build_dataframe, create_generators, split_dataframe
from drd.evaluate import evaluate_predictions, index_to_label, load_images
from drd.model import build_model
from drd.visualize import (
    plot_class_distribution,
    plot_class_samples,
    plot_confusion,
    plot_random_predictions,
)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train and evaluate DRD model")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Config.data_dir,
        help="Dataset directory with class subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Config.output_dir,
        help="Directory to store trained weights",
    )
    parser.add_argument("--epochs", type=int, default=Config.epochs, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=Config.batch_size, help="Batch size")
    parser.add_argument("--seed", type=int, default=Config.seed, help="Random seed")
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=Config.image_size,
        metavar=("WIDTH", "HEIGHT"),
        help="Input image size",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable all matplotlib visualizations",
    )
    args = parser.parse_args()

    return Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=(args.image_size[0], args.image_size[1]),
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    ), args.no_plots


def main() -> None:
    config, no_plots = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = build_dataframe(config.data_dir)
    if not no_plots:
        plot_class_distribution(dataframe)
        plot_class_samples(
            config.data_dir,
            max_classes=5,
            samples_per_class=config.samples_per_class_plot,
        )

    train_df, test_df = split_dataframe(dataframe, test_size=config.test_size, seed=config.seed)
    train_gen, val_gen, test_gen = create_generators(
        train_df=train_df,
        test_df=test_df,
        image_size=config.image_size,
        batch_size=config.batch_size,
        val_split=config.val_split,
        seed=config.seed,
    )

    model = build_model(
        input_shape=(config.image_size[0], config.image_size[1], 3),
        num_classes=len(train_gen.class_indices),
    )
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        verbose=1,
        patience=10,
        restore_best_weights=True,
    )
    checkpointer = ModelCheckpoint(
        filepath=str(config.checkpoint_path),
        verbose=1,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )

    model.fit(
        train_gen,
        epochs=config.epochs,
        validation_data=val_gen,
        callbacks=[checkpointer, early_stopping],
    )

    if config.checkpoint_path.exists():
        model.load_weights(str(config.checkpoint_path))

    evaluation = model.evaluate(test_gen, verbose=1)
    print(f"Test Accuracy (Keras): {evaluation[1]:.4f}")

    idx_to_label = index_to_label(train_gen.class_indices)
    accuracy, report, cm, y_true, y_pred = evaluate_predictions(model, test_gen, idx_to_label)
    print(f"Test Accuracy (sklearn): {accuracy:.4f}")
    print(report)

    if not no_plots:
        image_paths = test_df["Image"].tolist()
        images = load_images(image_paths, image_size=config.image_size)
        plot_random_predictions(images, y_true, y_pred, rows=20)
        ordered_classes = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
        plot_confusion(cm, ordered_classes)


if __name__ == "__main__":
    main()

