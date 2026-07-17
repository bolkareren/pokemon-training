from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Single source of truth for one training run.

    Every tunable lives here exactly once. Add a field to add a hyperparameter:
    it is picked up by the CLI, logged to MLflow, and saved with the run
    automatically. Nothing needs to be repeated anywhere else.
    """

    # Data
    data_dir: Path | None = None  # defaults to <project_root>/data when unset
    batch_size: int = 16
    val_size: float = 0.1
    test_size: float = 0.1
    random_state: int = 42

    # Model
    model_name: str = "resnet18"
    weights: str | None = "DEFAULT"  # torchvision weights enum name, or None
    train_last_n_layers: int = 5
    train_batch_norm_affine: bool = False
    batch_norm_mode: str = "train"

    # Optimizer
    optimizer_type: str = "AdamW"
    backbone_lr: float = 5e-4
    classifier_lr: float = 1e-3
    weight_decay: float = 1e-3
    label_smoothing: float = 0.2

    # Training loop
    epochs: int = 18

    # Tracking
    experiment_name: str = "pokemon-classification"
    run_name: str | None = None
    save_model: bool = False
