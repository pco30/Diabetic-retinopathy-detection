from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def index_to_label(class_indices: dict[str, int]) -> dict[int, str]:
    return {index: label for label, index in class_indices.items()}


def evaluate_predictions(model, test_generator, idx_to_label: dict[int, str]):
    test_generator.reset()
    probabilities = model.predict(test_generator, verbose=0)
    y_pred_idx = np.argmax(probabilities, axis=1)
    y_true_idx = test_generator.classes

    y_pred_labels = [idx_to_label[i] for i in y_pred_idx]
    y_true_labels = [idx_to_label[i] for i in y_true_idx]

    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    report = classification_report(y_true_labels, y_pred_labels, zero_division=0)
    ordered_classes = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=ordered_classes)

    return accuracy, report, cm, y_true_labels, y_pred_labels


def load_images(image_paths: list[str], image_size: tuple[int, int]):
    images = []
    for path in image_paths:
        with Image.open(Path(path)) as image:
            images.append(image.resize(image_size))
    return images

