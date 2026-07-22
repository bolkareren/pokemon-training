"""CNN frozen-feature probe (cross-fit OOF): how separable are the classes in the
fine-tuned CNN's penultimate representation, and can a shallow classifier on those
features beat the CNN's own linear softmax head (0.7604 OOF)?

This is the fine-tuned-CNN analog of `dinov2_probe.py`. The difference that matters
is leakage: DINOv2 never saw this dataset, so caching a feature per image is clean;
a CNN fine-tuned on the data has *memorized* the images it trained on, so caching
features from a single model would make the training images a memorization test
(the old shiny-duplicate trap). We avoid that by **cross-fitting**: for each of the
K folds we train the reference CNN on that fold's train split (the exact
`run_cross_validation` protocol) and cache the 512-d post-avgpool feature only for
that fold's held-out val images. Concatenated over folds, every image's feature
comes from a backbone that never trained on it, so the out-of-fold probe is honest
and measured on the identical held-out partition as the CNN's own 0.7604 OOF.

The probe reuses the shape-descriptor / DINOv2 shallow-classifier sweep, on the
same `fold_indices` split as `p10-lastn6-s<seed>`, so its OOF is directly
comparable. Features are cached per (config-signature, seed): re-running the
classifier sweep is free; only the 5 CNN trainings cost.

    uv run python scripts/cnn_feature_probe.py                 # seed 42, reference config
    uv run python scripts/cnn_feature_probe.py --seed 123      # confirm at a second seed

Nothing here re-tunes the CNN; it is the reference model used as a fixed extractor.
"""

import argparse
import hashlib
import statistics
from dataclasses import asdict

import mlflow
import numpy as np
import torch

from pokemon_training.config import ExperimentConfig
from pokemon_training.data import (
	PROJECT_ROOT,
	create_fold_data_loaders,
	fold_indices,
	load_dataset,
)
from pokemon_training.evaluation import predict_top_k, top_k_accuracy_from_predictions
from pokemon_training.experiment import get_device, set_random_seed
from pokemon_training.train import train_outer_loop
from scripts.shape_descriptor_baseline import _aligned_top5, classifiers
from scripts.training import build_model_and_optimizer

TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT = "pokemon-classification-clean"
CACHE_DIR = PROJECT_ROOT / ".cnn-probe-cache"

# Config fields that change the trained extractor (and thus invalidate a cache).
_SIGNATURE_FIELDS = (
	"model_name",
	"weights",
	"train_last_n_layers",
	"train_batch_norm_affine",
	"stem",
	"input_channels",
	"invert_mask",
	"aspect_crop",
	"optimizer_type",
	"backbone_lr",
	"classifier_lr",
	"weight_decay",
	"label_smoothing",
	"epochs",
	"scheduler",
	"warmup_epochs",
	"restore_best_epoch",
	"batch_norm_mode",
	"exclude_shiny",
	"include_pose_variants",
	"group_aware_folds",
	"test_size",
	"folds",
)


def config_signature(config):
	"""Short stable hash of the load-bearing config, so a cache is only reused
	for a byte-identical extractor. Fold/seed live in the filename separately."""
	payload = "|".join(f"{f}={getattr(config, f)}" for f in _SIGNATURE_FIELDS)
	return hashlib.sha1(payload.encode()).hexdigest()[:8]


@torch.no_grad()
def extract_penultimate(model, loader, device):
	"""Post-avgpool 512-d features for a shuffle=False loader, in loader order.

	Hooks avgpool rather than surgically removing fc, so the model stays exactly
	the one that produced the reference val accuracy."""
	captured = []
	handle = model.avgpool.register_forward_hook(
		lambda _module, _inp, out: captured.append(out.flatten(1).cpu())
	)
	model.eval()
	try:
		for images, _labels in loader:
			model(images.to(device))
	finally:
		handle.remove()
	return torch.cat(captured).numpy()


def crossfit_features(config, data_dir, device, cache_path):
	"""Train the reference CNN per fold; return leak-free OOF features + the CNN's
	own OOF predictions (the 0.7604 anchor). Cached by config signature + seed."""
	if cache_path.exists():
		stored = np.load(cache_path, allow_pickle=True)
		feats = {int(i): stored["features"][k] for k, i in enumerate(stored["indices"])}
		cnn_oof = list(stored["cnn_oof"])
		classes = list(stored["classes"])
		print(f"  features loaded from {cache_path.name} ({len(feats)} images)")
		return feats, cnn_oof, classes

	fold_loaders, _test_loader, classes = create_fold_data_loaders(
		data_dir=data_dir,
		folds=config.folds,
		test_size=config.test_size,
		batch_size=config.batch_size,
		random_state=config.random_state,
		exclude_shiny=config.exclude_shiny,
		group_aware=config.group_aware_folds,
		include_pose_variants=config.include_pose_variants,
		augmentations=config.augmentations,
	)
	num_classes = len(classes)
	criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

	feats = {}
	cnn_oof = []
	cnn_fold_acc = []
	for fold, (train_loader, val_loader, val_idx) in enumerate(fold_loaders):
		# Match run_cross_validation exactly: reseed per fold so each backbone
		# starts from the same init and only the split differs.
		set_random_seed(config.random_state)
		model, optimizer, _ = build_model_and_optimizer(config, num_classes, None)
		print(
			f"\n=== fold {fold + 1}/{config.folds} "
			f"(train {len(train_loader.dataset)}, val {len(val_loader.dataset)}) ==="
		)
		train_outer_loop(
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

		# CNN's own head, for the sanity anchor (should reproduce ~0.7604 OOF).
		predictions, labels = predict_top_k(model, val_loader, k=5, device=device)
		cnn_fold_acc.append(top_k_accuracy_from_predictions(predictions, labels, k=1))
		cnn_oof.extend(
			{"index": int(i), "label": int(label), "top5": pred}
			for i, label, pred in zip(val_idx, labels, predictions)
		)

		# The point of the script: leak-free penultimate features for this fold's
		# held-out val, from the backbone that never trained on them.
		fold_feats = extract_penultimate(model, val_loader, device)
		for i, vec in zip(val_idx, fold_feats):
			feats[int(i)] = vec.astype(np.float32)

	print(
		f"\nCNN own head OOF: {statistics.mean(cnn_fold_acc):.4f} "
		f"(sem {statistics.stdev(cnn_fold_acc) / len(cnn_fold_acc) ** 0.5:.4f}) "
		f"[reference p10-lastn6 = 0.7604]"
	)

	CACHE_DIR.mkdir(exist_ok=True)
	idx = np.array(sorted(feats))
	np.savez(
		cache_path,
		indices=idx,
		features=np.stack([feats[int(i)] for i in idx]),
		cnn_oof=np.array(cnn_oof, dtype=object),
		classes=np.array(classes, dtype=object),
	)
	return feats, cnn_oof, classes


def probe(feats, targets, pool_idx, folds, seed, classes):
	"""Shallow-classifier sweep with OOF over the pool, on the same fold split as
	the CNN CV (so probe-val features come from the one held-out backbone and the
	OOF partition is identical to the CNN's 0.7604)."""

	def matrix(idx):
		return np.stack([feats[int(i)] for i in idx]), targets[idx]

	results = {}
	for name in classifiers():
		oof, fold_acc = [], []
		for train_idx, val_idx in fold_indices(dataset_ref, pool_idx, folds, random_state=seed):
			Xtr, ytr = matrix(train_idx)
			Xva, yva = matrix(val_idx)
			clf = classifiers()[name]
			clf.fit(Xtr, ytr)
			top5 = _aligned_top5(clf, Xva, len(classes))
			fold_acc.append(top_k_accuracy_from_predictions(top5, yva.tolist(), k=1))
			oof.extend(
				{"index": int(i), "label": int(y), "top5": t}
				for i, y, t in zip(val_idx, yva.tolist(), top5)
			)
		preds = [e["top5"] for e in oof]
		labels = [e["label"] for e in oof]
		results[name] = {
			"oof": oof,
			"oof_accuracy": top_k_accuracy_from_predictions(preds, labels, 1),
			"oof_top3_accuracy": top_k_accuracy_from_predictions(preds, labels, 3),
			"oof_top5_accuracy": top_k_accuracy_from_predictions(preds, labels, 5),
			"fold_accuracy_mean": statistics.mean(fold_acc),
			"fold_accuracy_sem": statistics.stdev(fold_acc) / len(fold_acc) ** 0.5,
			"fold_accuracies": fold_acc,
		}
	return results


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--folds", type=int, default=5)
	parser.add_argument("--no-mlflow", action="store_true")
	args = parser.parse_args()

	config = ExperimentConfig(random_state=args.seed, folds=args.folds)
	data_dir = config.data_dir or PROJECT_ROOT / "data"

	global dataset_ref
	dataset_ref = load_dataset(data_dir)
	targets = np.array(dataset_ref.targets)

	device = get_device()
	sig = config_signature(config)
	stem = f"{config.model_name}-lastn{config.train_last_n_layers}-{sig}-s{args.seed}"
	cache_path = CACHE_DIR / f"{stem}.npz"

	print(
		f"cross-fit CNN feature probe | {config.model_name} lastN{config.train_last_n_layers} "
		f"| seed {args.seed} | device {device}\n"
		f"reference: CNN own head 0.7604 OOF (p10-lastn6-s*) | DINOv2 frozen 0.618 | floor ~0.285\n"
	)
	feats, cnn_oof, classes = crossfit_features(config, data_dir, device, cache_path)

	pool_idx = np.array(sorted(feats))
	cnn_preds = [e["top5"] for e in cnn_oof]
	cnn_labels = [e["label"] for e in cnn_oof]
	cnn_oof_acc = top_k_accuracy_from_predictions(cnn_preds, cnn_labels, 1)

	results = probe(feats, targets, pool_idx, args.folds, args.seed, classes)
	best = max(results, key=lambda n: results[n]["oof_accuracy"])
	feat_dim = len(next(iter(feats.values())))

	print(f"\nCNN own softmax head OOF: {cnn_oof_acc:.4f}  (anchor)")
	print(f"probe on {feat_dim}-d penultimate features, per-classifier OOF:")
	for name in results:
		r = results[name]
		delta = r["oof_accuracy"] - cnn_oof_acc
		print(
			f"    {name:>14}: oof={r['oof_accuracy']:.4f} (sem {r['fold_accuracy_sem']:.4f}) "
			f"top5={r['oof_top5_accuracy']:.4f}  vs head {delta:+.4f}"
		)
	r = results[best]
	gap = r["oof_accuracy"] - cnn_oof_acc
	print(f"  best: {best}  oof {r['oof_accuracy']:.4f}  ({gap:+.4f} vs head)\n")

	if args.no_mlflow:
		return
	mlflow.set_tracking_uri(TRACKING_URI)
	mlflow.set_experiment(EXPERIMENT)
	run_name = f"cnnprobe-{config.model_name}-lastn{config.train_last_n_layers}-{best}-s{args.seed}"
	with mlflow.start_run(run_name=run_name):
		mlflow.log_params(
			{
				"model_family": "cnn_crossfit_feature_probe",
				"backbone": config.model_name,
				"train_last_n_layers": config.train_last_n_layers,
				"feature": "penultimate_avgpool",
				"feature_dim": feat_dim,
				"classifier": best,
				"folds": args.folds,
				"random_state": args.seed,
				"config_signature": sig,
			}
		)
		for fold, acc in enumerate(r["fold_accuracies"]):
			mlflow.log_metrics({"fold_accuracy": acc}, step=fold)
		mlflow.log_metrics(
			{k: v for k, v in r.items() if k not in ("oof", "fold_accuracies")}
			| {"oof_n": len(r["oof"]), "cnn_head_oof_accuracy": cnn_oof_acc}
		)
		mlflow.log_dict({"classes": classes, "predictions": r["oof"]}, "oof_predictions.json")
		mlflow.log_dict(asdict(config) | {"data_dir": str(data_dir)}, "extractor_config.json")


if __name__ == "__main__":
	main()
