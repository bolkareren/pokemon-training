"""Ensemble machinery for Phase 6: load saved fold checkpoints and combine them
by averaging their softmax distributions.

Why averaged probabilities rather than votes: with 151 classes a handful of hard
votes fragments (ties everywhere) and cannot produce the ranked distribution the
top-3/top-5 metrics need. Averaging the *probabilities* (arithmetic mean) rather
than the logits is deliberate - logit-averaging is a geometric mean in
probability space, so one confidently-wrong model can veto a class by
contributing a near-zero probability, while the arithmetic mean degrades
gracefully.

Checkpoints come from `--save-model` runs: weights/<run_name>/fold<k>.pt, holding
each fold's *restored best-epoch* weights.

    uv run python scripts/ensemble.py --runs p14-ref-extsplit-s42 ...   # evaluate on test

LEAKAGE NOTE: a fold ensemble is only valid on data held out from *every* member.
That is true of test_data/, and false of the OOF pool (each pool image is
in-training for K-1 of its seed's models). Never point this at OOF predictions
across folds; seed-ensembling within a single fold is the OOF-valid variant.
"""

import argparse

import torch
import torchvision.transforms.functional as TF

from pokemon_training.config import ExperimentConfig
from pokemon_training.data import PROJECT_ROOT, create_fold_data_loaders
from pokemon_training.evaluation import top_k_accuracy_from_predictions, top_k_from_probabilities
from scripts.training import build_model_and_optimizer

WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Test-time augmentation. Silhouettes are contour-only, so the useful views are
# mirror and small rotations; scale/crop jitter would fight the framing the model
# was trained on. "identity" alone reproduces the plain ensemble.
TTA_VIEWS = {
	"identity": lambda x: x,
	"hflip": lambda x: torch.flip(x, dims=[-1]),
	"rot+10": lambda x: TF.rotate(x, 10.0),
	"rot-10": lambda x: TF.rotate(x, -10.0),
}


def checkpoint_paths(run_names):
	"""Every fold checkpoint for the given runs, in a stable order."""
	paths = []
	for run in run_names:
		run_dir = WEIGHTS_DIR / run
		if not run_dir.is_dir():
			raise SystemExit(
				f"no checkpoints at {run_dir} - was the run trained with --save-model?"
			)
		found = sorted(run_dir.glob("fold*.pt"), key=lambda p: int(p.stem.removeprefix("fold")))
		if not found:
			raise SystemExit(f"{run_dir} contains no fold*.pt checkpoints")
		paths.extend(found)
	return paths


def load_models(paths, config, num_classes, device):
	"""Rebuild the training architecture and load each checkpoint into it."""
	models = []
	for path in paths:
		model, _optimizer, _weights = build_model_and_optimizer(config, num_classes, None)
		model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
		models.append(model.to(device).eval())
	return models


@torch.no_grad()
def ensemble_probabilities(models, data_loader, device, views=("identity",), per_model=False):
	"""Mean softmax over every (model, TTA view) pair, in data_loader order.

	Returns (mean_probabilities, labels) and, when `per_model` is set, also the
	per-model probability tensors so single-vs-ensemble deltas can be reported.
	"""
	transforms = [TTA_VIEWS[v] for v in views]
	totals = None
	each = []
	labels = []

	for position, model in enumerate(models):
		model_total = None
		labels_this = []
		for inputs, batch_labels in data_loader:
			inputs = inputs.to(device)
			# Average the views first so each model contributes equally regardless
			# of how many views it was evaluated under.
			probabilities = sum(
				torch.softmax(model(view(inputs)), dim=1) for view in transforms
			) / len(transforms)
			probabilities = probabilities.cpu()
			model_total = (
				probabilities if model_total is None else torch.cat([model_total, probabilities])
			)
			labels_this.extend(batch_labels.tolist())

		if position == 0:
			labels = labels_this
		totals = model_total if totals is None else totals + model_total
		if per_model:
			each.append(model_total)

	mean = totals / len(models)
	return (mean, labels, each) if per_model else (mean, labels)


def report(probabilities, labels, prefix=""):
	top5 = top_k_from_probabilities(probabilities, k=5)
	return {
		f"{prefix}top1": top_k_accuracy_from_predictions(top5, labels, k=1),
		f"{prefix}top3": top_k_accuracy_from_predictions(top5, labels, k=3),
		f"{prefix}top5": top_k_accuracy_from_predictions(top5, labels, k=5),
	}


def build_test_loader(config, data_dir, test_dir):
	"""The held-out test loader, built through the normal fold path so the eval
	transform and shiny filtering are identical to training-time evaluation."""
	_folds, test_loader, classes = create_fold_data_loaders(
		data_dir=data_dir,
		folds=config.folds or 5,
		test_size=config.test_size,
		batch_size=config.batch_size,
		random_state=config.random_state,
		exclude_shiny=config.exclude_shiny,
		group_aware=config.group_aware_folds,
		augmentations=config.augmentations,
		test_dir=test_dir,
	)
	return test_loader, classes


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--runs", nargs="+", required=True, help="run names under weights/")
	parser.add_argument("--views", nargs="+", default=["identity"], choices=list(TTA_VIEWS))
	parser.add_argument("--test-dir", default="test_data")
	args = parser.parse_args()

	config = ExperimentConfig()
	data_dir = PROJECT_ROOT / "data"
	test_dir = PROJECT_ROOT / args.test_dir
	device = "mps" if torch.backends.mps.is_available() else "cpu"

	test_loader, classes = build_test_loader(config, data_dir, test_dir)
	paths = checkpoint_paths(args.runs)
	print(f"{len(paths)} checkpoints | views={args.views} | test n={len(test_loader.dataset)}")

	models = load_models(paths, config, len(classes), device)
	probabilities, labels, each = ensemble_probabilities(
		models, test_loader, device, views=tuple(args.views), per_model=True
	)

	singles = [report(p, labels)["top1"] for p in each]
	metrics = report(probabilities, labels)
	print(
		f"\nsingle models: mean top1 {sum(singles) / len(singles):.4f} "
		f"(min {min(singles):.4f}, max {max(singles):.4f})"
	)
	print(
		f"ensemble:      top1 {metrics['top1']:.4f}  top3 {metrics['top3']:.4f}  "
		f"top5 {metrics['top5']:.4f}"
	)
	gain = metrics["top1"] - sum(singles) / len(singles)
	print(f"ensemble gain over mean single model: {gain:+.4f}")


if __name__ == "__main__":
	main()
