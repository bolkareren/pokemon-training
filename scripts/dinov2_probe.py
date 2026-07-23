"""DINOv2 frozen-feature probe: does a strong self-supervised (non-ImageNet)
representation transfer to silhouettes, and how close does it get to the
fine-tuned CNN?

Frozen DINOv2 ViT-L/14 @ 518 features (CLS + mean patch token, 2048-d) → the same
shallow classifier sweep as the shape-descriptor floor, on the *same split* as
`p7-ref-26-s<seed>` (reuses `fold_indices`), so OOF is directly comparable to the
CNN's 0.7496 and the ~0.285 classical floor. Two input encodings are compared:
the raw binary silhouette (3-channel, in-distribution for DINOv2) and the CNN's
own (mask, sdt, mask). Features are cached per (encoding, model, resolution,
image-set), so re-runs are free.

    uv run python scripts/dinov2_probe.py                    # seed 42, both encodings

Model: torch.hub facebookresearch/dinov2 (weights from dl.fbaipublicfiles.com;
xformers is optional and falls back). Nothing is fine-tuned — this is a probe.
"""

import argparse
import statistics

import mlflow
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

from pokemon_training.data import (
	PROJECT_ROOT,
	_normal_sprite_indices,
	fold_indices,
	get_transforms,
	load_dataset,
)
from pokemon_training.evaluation import top_k_accuracy_from_predictions
from scripts.shape_descriptor_baseline import _aligned_top5, classifiers

TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT = "pokemon-classification-clean"
CACHE_DIR = PROJECT_ROOT / ".dinov2-cache"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RESOLUTION = 518  # 37x14; DINOv2 interpolates positional encodings to fit
HUB_MODEL = "dinov2_vitl14"


def build_transforms():
	"""Two (3, 518, 518) ImageNet-normalized encodings of a silhouette."""
	# Raw binary silhouette: creature dark-on-white, replicated to 3 channels.
	silhouette = transforms.Compose(
		[
			transforms.Grayscale(num_output_channels=3),
			transforms.Resize(
				(RESOLUTION, RESOLUTION),
				interpolation=transforms.InterpolationMode.NEAREST,
			),
			transforms.ToTensor(),
			transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
		]
	)
	# The CNN's own (mask, sdt, mask) encoding, applied at 518.
	eval_encode = get_transforms(input_channels=("mask", "sdt", "mask"))[1]

	def msm(image):
		resized = image.convert("L").resize((RESOLUTION, RESOLUTION), Image.NEAREST)
		return eval_encode(resized)

	return {"silhouette": silhouette, "msm": msm}


@torch.no_grad()
def extract_features(model, dataset, indices, transform, device, batch_size=8):
	"""index -> 2048-d DINOv2 feature (CLS token ++ mean patch token)."""
	feats = {}
	batch, batch_idx = [], []

	def flush():
		if not batch:
			return
		x = torch.stack(batch).to(device)
		out = model.forward_features(x)
		vec = torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(dim=1)], dim=-1)
		for j, i in enumerate(batch_idx):
			feats[i] = vec[j].float().cpu().numpy()
		batch.clear()
		batch_idx.clear()

	for n, i in enumerate(indices):
		image = Image.open(dataset.samples[i][0])
		batch.append(transform(image))
		batch_idx.append(int(i))
		if len(batch) == batch_size:
			flush()
		if (n + 1) % 200 == 0:
			print(f"  {n + 1}/{len(indices)} features")
	flush()
	return feats


def cached_features(model, dataset, indices, transform, device, encoding):
	CACHE_DIR.mkdir(exist_ok=True)
	path = CACHE_DIR / f"{HUB_MODEL}-{RESOLUTION}-{encoding}.npz"
	if path.exists():
		stored = np.load(path)
		cached = {int(i): stored["features"][k] for k, i in enumerate(stored["indices"])}
		if set(int(i) for i in indices) <= set(cached):
			print(f"  features loaded from {path.name}")
			return cached
	print(f"  extracting {encoding} features ({len(indices)} images)...")
	feats = extract_features(model, dataset, indices, transform, device)
	idx = np.array(sorted(feats))
	np.savez(path, indices=idx, features=np.stack([feats[int(i)] for i in idx]))
	return feats


def evaluate(feats, targets, pool_idx, test_idx, folds, seed, classes):
	"""Shallow-classifier sweep with OOF over the pool + a held-out test score."""

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
	best = max(results, key=lambda n: results[n]["oof_accuracy"])

	Xpool, ypool = matrix(pool_idx)
	Xtest, ytest = matrix(test_idx)
	clf = classifiers()[best]
	clf.fit(Xpool, ypool)
	test_top5 = _aligned_top5(clf, Xtest, len(classes))
	test_top1 = top_k_accuracy_from_predictions(test_top5, ytest.tolist(), 1)
	return results, best, test_top1


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--folds", type=int, default=5)
	parser.add_argument("--test-size", type=float, default=0.15)
	parser.add_argument("--encodings", nargs="+", default=["silhouette", "msm"])
	parser.add_argument("--no-mlflow", action="store_true")
	args = parser.parse_args()

	global dataset_ref
	dataset_ref = load_dataset(PROJECT_ROOT / "data")
	targets = np.array(dataset_ref.targets)
	classes = dataset_ref.classes

	normal = _normal_sprite_indices(dataset_ref)
	pool_idx, test_idx = train_test_split(
		normal, test_size=args.test_size, stratify=targets[normal], random_state=args.seed
	)
	needed = np.concatenate([pool_idx, test_idx])

	device = "mps" if torch.backends.mps.is_available() else "cpu"
	print(f"loading {HUB_MODEL} @ {RESOLUTION} on {device}...")
	model = torch.hub.load("facebookresearch/dinov2", HUB_MODEL).to(device).eval()

	tfs = build_transforms()
	print(f"\nreference: CNN 0.7496 (p7-ref-26-s{args.seed}) | classical floor ~0.285\n")
	for enc in args.encodings:
		feats = cached_features(model, dataset_ref, needed, tfs[enc], device, enc)
		results, best, test_top1 = evaluate(
			feats, targets, pool_idx, test_idx, args.folds, args.seed, classes
		)
		print(f"[{enc}] per-classifier OOF:")
		for name in results:
			r = results[name]
			print(
				f"    {name:>14}: oof={r['oof_accuracy']:.4f} (sem {r['fold_accuracy_sem']:.4f}) "
				f"top5={r['oof_top5_accuracy']:.4f}"
			)
		r = results[best]
		print(f"  best [{enc}]: {best}  oof {r['oof_accuracy']:.4f}  test {test_top1:.4f}\n")

		if args.no_mlflow:
			continue
		mlflow.set_tracking_uri(TRACKING_URI)
		mlflow.set_experiment(EXPERIMENT)
		with mlflow.start_run(run_name=f"dinov2L518-{enc}-{best}-s{args.seed}"):
			mlflow.log_params(
				{
					"model_family": "dinov2_frozen_probe",
					"backbone": HUB_MODEL,
					"resolution": RESOLUTION,
					"encoding": enc,
					"feature": "cls+meanpatch",
					"feature_dim": len(next(iter(feats.values()))),
					"classifier": best,
					"folds": args.folds,
					"random_state": args.seed,
				}
			)
			for fold, acc in enumerate(r["fold_accuracies"]):
				mlflow.log_metrics({"fold_accuracy": acc}, step=fold)
			mlflow.log_metrics(
				{k: v for k, v in r.items() if k not in ("oof", "fold_accuracies")}
				| {"oof_n": len(r["oof"]), "test_accuracy": test_top1, "test_n": len(test_idx)}
			)
			mlflow.log_dict({"classes": classes, "predictions": r["oof"]}, "oof_predictions.json")


if __name__ == "__main__":
	main()
