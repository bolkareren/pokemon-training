import statistics
from dataclasses import asdict
from pathlib import Path

import mlflow
import torch
import tyro

from pokemon_training.config import ExperimentConfig
from pokemon_training.data import (
	create_data_loaders,
	create_fold_data_loaders,
	load_dataset,
)
from pokemon_training.evaluation import (
	accuracy,
	predict_top_k,
	top_k_accuracy_from_predictions,
)
from pokemon_training.experiment import get_device, set_random_seed
from pokemon_training.model import (
	adapt_input_channels,
	apply_stem,
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


def build_model_and_optimizer(config, num_classes, weights_checkpoint):
	"""Build a freshly initialized model and its optimizer from `config`.

	Factored out because cross-validation needs an independent model per fold -
	reusing one across folds would carry the previous fold's validation data in
	as training signal.
	"""
	# A custom checkpoint supplies its own pretrained backbone, so it takes
	# precedence over the torchvision weights enum rather than combining with it.
	weights = None if weights_checkpoint else resolve_weights(config.model_name, config.weights)
	model = load_pretrained_model(
		num_classes=num_classes,
		weights=weights,
		model_name=config.model_name,
		weights_checkpoint=weights_checkpoint,
	)
	model = apply_stem(model, stem=config.stem)
	model = adapt_input_channels(model, len(config.input_channels))
	model = set_trainable_weights(model, train_last_n_layers=config.train_last_n_layers)
	model = set_batch_norm_trainable(model, trainable=config.train_batch_norm_affine)

	optimizer = create_optimizer(
		model,
		type=config.optimizer_type,
		lr=config.backbone_lr,
		classifier_lr=config.classifier_lr,
		weight_decay=config.weight_decay,
	)

	return model, optimizer, weights


def run_cross_validation(config, data_dir, weights_checkpoint, device):
	"""Train one model per fold; return metrics pooled over out-of-fold
	predictions, with the per-fold spread as the uncertainty."""
	fold_loaders, test_loader, classes = create_fold_data_loaders(
		data_dir=data_dir,
		folds=config.folds,
		test_size=config.test_size,
		test_dir=config.test_dir,
		batch_size=config.batch_size,
		random_state=config.random_state,
		exclude_shiny=config.exclude_shiny,
		group_aware=config.group_aware_folds,
		include_pose_variants=config.include_pose_variants,
		augmentations=config.augmentations,
	)
	num_classes = len(classes)
	criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

	oof = []
	fold_accuracies = []
	fold_gaps = []

	for fold, (train_loader, val_loader, val_idx) in enumerate(fold_loaders):
		# Reseed per fold so each starts from the same initialization state,
		# making fold-to-fold differences attributable to the data split alone.
		set_random_seed(config.random_state)
		model, optimizer, _ = build_model_and_optimizer(config, num_classes, weights_checkpoint)

		print(
			f"\n=== fold {fold + 1}/{config.folds} "
			f"(train {len(train_loader.dataset)}, val {len(val_loader.dataset)}) ==="
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
			scheduler_type=config.scheduler,
			warmup_epochs=config.warmup_epochs,
			restore_best_epoch=config.restore_best_epoch,
		)

		predictions, labels = predict_top_k(model, val_loader, k=5, device=device)
		fold_accuracy = top_k_accuracy_from_predictions(predictions, labels, k=1)
		# Measure the gap at the epoch actually scored - the restored one when
		# restoration is on. Runs logged before this fix used the final epoch.
		scored_epoch = history.get("best_epoch", len(history["val_loss"]) - 1)
		gap = history["val_loss"][scored_epoch] - history["train_loss"][scored_epoch]

		fold_accuracies.append(fold_accuracy)
		fold_gaps.append(gap)
		oof.extend(
			{"index": int(i), "label": int(label), "top5": prediction}
			for i, label, prediction in zip(val_idx, labels, predictions)
		)

		fold_metrics = {"fold_accuracy": fold_accuracy, "fold_gap": gap}
		if "best_epoch" in history:
			fold_metrics["fold_best_epoch"] = history["best_epoch"]
		mlflow.log_metrics(fold_metrics, step=fold)
		print(f"fold {fold + 1} accuracy: {fold_accuracy:.4f} (gap {gap:+.3f})")

	predictions = [entry["top5"] for entry in oof]
	labels = [entry["label"] for entry in oof]

	metrics = {
		"oof_accuracy": top_k_accuracy_from_predictions(predictions, labels, k=1),
		"oof_top3_accuracy": top_k_accuracy_from_predictions(predictions, labels, k=3),
		"oof_top5_accuracy": top_k_accuracy_from_predictions(predictions, labels, k=5),
		"fold_accuracy_mean": statistics.mean(fold_accuracies),
		"fold_accuracy_stdev": statistics.stdev(fold_accuracies),
		# Standard error of the mean across folds - the number to compare
		# configs against, rather than any single fold's score.
		"fold_accuracy_sem": statistics.stdev(fold_accuracies) / len(fold_accuracies) ** 0.5,
		"fold_gap_mean": statistics.mean(fold_gaps),
		"oof_n": len(oof),
	}

	return metrics, oof, classes, test_loader


def main(config: ExperimentConfig) -> None:
	data_dir = config.data_dir or PROJECT_ROOT / "data"
	if config.test_dir is not None:
		if not config.test_dir.is_absolute():
			config.test_dir = PROJECT_ROOT / config.test_dir
		if not config.folds:
			raise ValueError(
				"test_dir (on-disk test split) is supported in fold mode only; pass --folds 5. "
				"The single-split path carves its own val/test out of data_dir."
			)
	weights_checkpoint = config.weights_checkpoint
	if weights_checkpoint is not None and not weights_checkpoint.is_absolute():
		weights_checkpoint = PROJECT_ROOT / weights_checkpoint

	mlflow.set_tracking_uri(TRACKING_URI)

	device = get_device()
	set_random_seed(config.random_state)

	# Class list and parameter counts are identical across folds, so a throwaway
	# build is enough to log them before choosing a training path.
	reference_dataset = load_dataset(data_dir)
	classes = reference_dataset.classes
	num_classes = len(classes)

	reference_model, _, weights = build_model_and_optimizer(config, num_classes, weights_checkpoint)
	trainable_parameters = sum(p.numel() for p in reference_model.parameters() if p.requires_grad)
	total_parameters = sum(p.numel() for p in reference_model.parameters())

	print(f"Using device: {device}")
	print(f"num_classes: {num_classes}")
	print(f"mode: {f'{config.folds}-fold CV' if config.folds else 'single split'}")
	print(f"total_parameters: {total_parameters}")
	print(f"trainable_parameters: {trainable_parameters}")
	print(f"percentage: {trainable_parameters / total_parameters * 100:.2f}%")

	# asdict keeps Path fields as Path objects; stringify so both MLflow params
	# and the YAML artifact stay serializable.
	config_dict = asdict(config)
	config_dict["data_dir"] = str(data_dir)
	config_dict["weights_checkpoint"] = str(weights_checkpoint) if weights_checkpoint else None

	# artifact_location is baked in as an absolute path at experiment creation
	# and never updated; pin it so a moved project can't scatter artifacts.
	if mlflow.get_experiment_by_name(config.experiment_name) is None:
		mlflow.create_experiment(
			config.experiment_name,
			artifact_location=f"file://{PROJECT_ROOT / 'mlruns'}",
		)
	mlflow.set_experiment(config.experiment_name)
	with mlflow.start_run(run_name=config.run_name):
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

		if config.folds:
			metrics, oof, classes, _ = run_cross_validation(
				config, data_dir, weights_checkpoint, device
			)
			# Every image scored by a model that never trained on it - the
			# confusion study's input.
			mlflow.log_dict({"classes": classes, "predictions": oof}, "oof_predictions.json")
			history = None
			model = None
		elif config.include_pose_variants:
			# The single-split path has no pose-variant plumbing; running here
			# would silently ignore the flag and mislabel the MLflow run.
			raise ValueError("include_pose_variants requires fold mode (--folds > 0)")
		else:
			train_loader, val_loader, test_loader, classes = create_data_loaders(
				data_dir=data_dir,
				batch_size=config.batch_size,
				val_size=config.val_size,
				test_size=config.test_size,
				random_state=config.random_state,
				exclude_shiny=config.exclude_shiny,
				augmentations=config.augmentations,
			)
			print(f"train_batches: {len(train_loader)}")
			print(f"validation_batches: {len(val_loader)}")
			print(f"test_batches: {len(test_loader)}")

			model, optimizer, _ = build_model_and_optimizer(config, num_classes, weights_checkpoint)
			criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

			history = train_outer_loop(
				model=model,
				train_loader=train_loader,
				val_loader=val_loader,
				optimizer=optimizer,
				criterion=criterion,
				epochs=config.epochs,
				device=device,
				batch_norm_mode=config.batch_norm_mode,
				scheduler_type=config.scheduler,
				warmup_epochs=config.warmup_epochs,
				restore_best_epoch=config.restore_best_epoch,
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
		if history is not None:
			mlflow.log_dict(history, "history.json")

		if config.save_model and model is not None:
			mlflow.pytorch.log_model(model, name="model")

		for name, value in metrics.items():
			print(f"{name}: {value}")

	# Port 5001, not MLflow's default 5000, which macOS AirPlay Receiver squats on.
	print(f"\nLogged to {TRACKING_URI}")
	print(f"View runs:  uv run mlflow ui --port 5001 --backend-store-uri {TRACKING_URI}")


if __name__ == "__main__":
	main(tyro.cli(ExperimentConfig))
