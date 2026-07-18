from dataclasses import asdict
from pathlib import Path

import mlflow
import torch
import tyro

from pokemon_training.config import ExperimentConfig
from pokemon_training.data import create_data_loaders
from pokemon_training.evaluation import accuracy
from pokemon_training.experiment import get_device, set_random_seed
from pokemon_training.model import (
    create_optimizer,
    load_pretrained_model,
    resolve_weights,
    set_batch_norm_trainable,
    set_trainable_weights,
)
from pokemon_training.train import train_outer_loop

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Pin the tracking store to one absolute location so it is independent of the
# working directory, and so `mlflow ui --backend-store-uri <same>` always reads
# exactly what the script wrote. MLflow's `ui` command and its client library
# have different default stores, which silently splits runs otherwise.
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


def main(config: ExperimentConfig) -> None:
    data_dir = config.data_dir or PROJECT_ROOT / "data"
    weights_checkpoint = config.weights_checkpoint
    if weights_checkpoint is not None and not weights_checkpoint.is_absolute():
        weights_checkpoint = PROJECT_ROOT / weights_checkpoint

    mlflow.set_tracking_uri(TRACKING_URI)

    device = get_device()
    set_random_seed(config.random_state)

    train_loader, val_loader, test_loader, classes = create_data_loaders(
        data_dir=data_dir,
        batch_size=config.batch_size,
        val_size=config.val_size,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    num_classes = len(classes)

    print(f"Using device: {device}")
    print(f"num_classes: {num_classes}")
    print(f"train_batches: {len(train_loader)}")
    print(f"validation_batches: {len(val_loader)}")
    print(f"test_batches: {len(test_loader)}")

    # A custom checkpoint supplies its own pretrained backbone, so it takes
    # precedence over the torchvision weights enum rather than combining with it.
    weights = None if weights_checkpoint else resolve_weights(config.model_name, config.weights)
    model = load_pretrained_model(
        num_classes=num_classes,
        weights=weights,
        model_name=config.model_name,
        weights_checkpoint=weights_checkpoint,
    )
    model = set_trainable_weights(model, train_last_n_layers=config.train_last_n_layers)
    model = set_batch_norm_trainable(model, trainable=config.train_batch_norm_affine)

    optimizer = create_optimizer(
        model,
        type=config.optimizer_type,
        lr=config.backbone_lr,
        classifier_lr=config.classifier_lr,
        weight_decay=config.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())

    print(f"total_parameters: {total_parameters}")
    print(f"trainable_parameters: {trainable_parameters}")
    print(f"percentage: {trainable_parameters / total_parameters * 100:.2f}%")

    # asdict keeps Path fields as Path objects; stringify so both MLflow params
    # and the YAML artifact stay serializable.
    config_dict = asdict(config)
    config_dict["data_dir"] = str(data_dir)
    config_dict["weights_checkpoint"] = str(weights_checkpoint) if weights_checkpoint else None

    # MLflow bakes an experiment's artifact_location in as an absolute path at
    # creation time and never updates it - if the project directory is later
    # moved/renamed, every run silently writes artifacts outside the project.
    # Pin it to the current PROJECT_ROOT the first time an experiment is
    # created, so a future move can't reintroduce that. Existing experiments
    # are left alone; artifact_location is only settable at creation time.
    if mlflow.get_experiment_by_name(config.experiment_name) is None:
        mlflow.create_experiment(
            config.experiment_name,
            artifact_location=f"file://{PROJECT_ROOT / 'mlruns'}",
        )
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=config.run_name):
        # Params: the full config (single source of truth) plus resolved facts
        # that aren't tunable but matter for reproducibility.
        mlflow.log_params(config_dict)
        mlflow.log_params(
            {
                "device": str(device),
                "resolved_weights": str(weights),
                "num_classes": num_classes,
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
            }
        )

        history = train_outer_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=config.epochs,
            device=device,
            batch_norm_mode=config.batch_norm_mode,
            on_epoch_end=lambda epoch, metrics: mlflow.log_metrics(metrics, step=epoch),
        )

        metrics = {
            "validation_accuracy": accuracy(model, val_loader, device=device),
            "validation_top3_accuracy": accuracy(model, val_loader, k=3, device=device),
            "validation_top5_accuracy": accuracy(model, val_loader, k=5, device=device),
        }
        mlflow.log_metrics(metrics)

        # Raw artifacts attached to the run so each run is self-contained.
        mlflow.log_dict(config_dict | {"classes": classes}, "config.yaml")
        mlflow.log_dict(history, "history.json")

        if config.save_model:
            mlflow.pytorch.log_model(model, name="model")

        for name, value in metrics.items():
            print(f"{name}: {value}")

    # Port 5001, not MLflow's default 5000, which macOS AirPlay Receiver squats on.
    print(f"\nLogged to {TRACKING_URI}")
    print(f"View runs:  uv run mlflow ui --port 5001 --backend-store-uri {TRACKING_URI}")


if __name__ == "__main__":
    main(tyro.cli(ExperimentConfig))
