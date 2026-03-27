from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("outputs")
    image_size: tuple[int, int] = (256, 256)
    batch_size: int = 32
    test_size: float = 0.30
    val_split: float = 0.15
    epochs: int = 10
    seed: int = 42
    samples_per_class_plot: int = 5

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / "best_weights.keras"

