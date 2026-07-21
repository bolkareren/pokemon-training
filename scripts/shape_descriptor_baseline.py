"""Classical shape-descriptor + shallow-classifier floor baseline.

Answers "how far does pure global shape get, with no learned features?" — the
lower bound the ResNet50 pipeline is measured against. It reuses `data.py`'s
exact split machinery (normal-sprite indices, the 15% test carve-out, and the
grouped 5-fold `fold_indices`), so at a given seed the out-of-fold image set and
fold membership are byte-identical to `p7-ref-26-s<seed>` — the OOF accuracy is
directly comparable to the 0.7496 reference, and the paired-analysis / confusion
tooling runs on the logged `oof_predictions.json` unchanged.

Descriptors are scale- and translation-invariant by construction (normalized
elliptic Fourier coefficients, log Hu moments, dimensionless shape ratios): a
floor for *shape*, honouring the "no metadata shortcuts" rule — raw size is a
generation proxy and is kept out, exactly as the aspect-crop was motivated.
Rotation is only normalised away with the fully-invariant EFD; `--keep-orientation`
retains the canonical upright sprite pose as signal (the better floor: +1.8pt
OOF / +7.6pt test at seed 42).

    uv run python scripts/shape_descriptor_baseline.py --keep-orientation   # best floor
    uv run python scripts/shape_descriptor_baseline.py --seed 43 --no-mlflow
"""

import argparse
import statistics
from pathlib import Path

import cv2
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pokemon_training.data import (
	_normal_sprite_indices,
	fold_indices,
	load_dataset,
)
from pokemon_training.evaluation import top_k_accuracy_from_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT = "pokemon-classification-clean"

EFD_ORDER = 20  # 4*order coeffs; the first 3 are fixed by normalisation and dropped.


# ---------------------------------------------------------------------------
# Descriptors — all invariant to translation, rotation, and scale.
# ---------------------------------------------------------------------------
def _elliptic_fourier(contour, order=EFD_ORDER, keep_orientation=False):
	"""Kuhl-Giardina elliptic Fourier coefficients of a closed contour, normalised
	for start-point and scale. Object orientation is removed too (full invariance)
	unless `keep_orientation` — sprites are rendered in a canonical upright pose,
	so their orientation is real signal, not a nuisance to normalise away.

	Fully invariant: the first three coefficients become constant and are dropped
	(dim 4*order-3). Orientation-preserving: all coefficients are kept (dim
	4*order), since none is forced to a canonical value."""
	contour = np.asarray(contour, dtype=float)
	# Close the loop so the traversal returns to the start (T = full perimeter).
	contour = np.vstack([contour, contour[0]])
	dxy = np.diff(contour, axis=0)
	dt = np.hypot(dxy[:, 0], dxy[:, 1])
	dt[dt == 0] = 1e-9
	t = np.concatenate([[0.0], np.cumsum(dt)])
	T = t[-1]
	phi = (2.0 * np.pi * t) / T

	coeffs = np.zeros((order, 4))
	for n in range(1, order + 1):
		const = T / (2.0 * n * n * np.pi * np.pi)
		cos = np.cos(n * phi)
		sin = np.sin(n * phi)
		dcos, dsin = np.diff(cos), np.diff(sin)
		coeffs[n - 1] = [
			const * np.sum((dxy[:, 0] / dt) * dcos),
			const * np.sum((dxy[:, 0] / dt) * dsin),
			const * np.sum((dxy[:, 1] / dt) * dcos),
			const * np.sum((dxy[:, 1] / dt) * dsin),
		]

	# Phase rotation (right-multiply): aligns the arbitrary contour start point,
	# not the object's spatial orientation. Always applied.
	a, b, c, d = coeffs[0]
	theta = 0.5 * np.arctan2(2.0 * (a * b + c * d), a * a - b * b + c * c - d * d)
	for n in range(order):
		m = coeffs[n].reshape(2, 2)
		rot = np.array(
			[
				[np.cos((n + 1) * theta), -np.sin((n + 1) * theta)],
				[np.sin((n + 1) * theta), np.cos((n + 1) * theta)],
			]
		)
		coeffs[n] = m.dot(rot).flatten()

	scale = np.hypot(coeffs[0, 0], coeffs[0, 2]) or 1e-9
	if keep_orientation:
		# Scale-normalise only; the spatial rotation below is what would erase
		# orientation, so skip it and keep every coefficient.
		return (coeffs / scale).flatten()

	# Spatial rotation (left-multiply) to the first harmonic's major axis, then
	# scale — full rotation + scale invariance.
	psi = np.arctan2(coeffs[0, 2], coeffs[0, 0])
	psi_rot = np.array([[np.cos(psi), np.sin(psi)], [-np.sin(psi), np.cos(psi)]])
	for n in range(order):
		m = coeffs[n].reshape(2, 2)
		coeffs[n] = (psi_rot.dot(m) / scale).flatten()

	# coeffs[0] is now ~[1, 0, 0, d1]; the first three carry no information.
	return coeffs.flatten()[3:]


def _shape_features(gray):
	"""Log Hu moments (whole mask) + dimensionless contour ratios (largest blob).
	`gray` is the L-channel; creature is dark (<=127), matching data.py."""
	binary = (gray <= 127).astype(np.uint8) * 255
	moments = cv2.moments(binary)
	hu = cv2.HuMoments(moments).flatten()
	# Log-scale the vast dynamic range; keep the sign, guard zeros.
	hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

	contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contours = [c for c in contours if len(c) >= 5]
	if not contours:
		return None, np.concatenate([hu, np.zeros(6)])

	main = max(contours, key=cv2.contourArea).squeeze(1)
	area = cv2.contourArea(main.astype(np.int32))
	perimeter = cv2.arcLength(main.astype(np.int32), True) or 1e-9
	hull = cv2.convexHull(main.astype(np.int32))
	hull_area = cv2.contourArea(hull) or 1e-9
	hull_perim = cv2.arcLength(hull, True) or 1e-9
	x, y, w, h = cv2.boundingRect(main.astype(np.int32))
	(_, _), (ax1, ax2), _ = cv2.fitEllipse(main.astype(np.int32))
	major, minor = max(ax1, ax2), min(ax1, ax2)

	ratios = np.array(
		[
			area / hull_area,  # solidity
			4.0 * np.pi * area / (perimeter * perimeter),  # circularity
			area / (w * h) if w * h else 0.0,  # extent
			hull_perim / perimeter,  # convexity
			np.sqrt(max(0.0, 1.0 - (minor / major) ** 2)) if major else 0.0,  # eccentricity
			min(len(contours), 8) / 8.0,  # blob count (structure), scale-free & capped
		]
	)
	return main, np.concatenate([hu, ratios])


def descriptor(path, keep_orientation=False):
	gray = np.array(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
	main, extra = _shape_features(gray)
	efd_dim = 4 * EFD_ORDER if keep_orientation else 4 * EFD_ORDER - 3
	efd = (
		_elliptic_fourier(main, keep_orientation=keep_orientation)
		if main is not None
		else np.zeros(efd_dim)
	)
	return np.concatenate([efd, extra]).astype(np.float32)


# ---------------------------------------------------------------------------
# Classifiers — shallow only.
# ---------------------------------------------------------------------------
def classifiers():
	return {
		"logreg": make_pipeline(
			StandardScaler(),
			LogisticRegression(max_iter=2000, C=1.0),
		),
		"svm_rbf": make_pipeline(
			StandardScaler(),
			SVC(kernel="rbf", C=10.0, gamma="scale", probability=False),
		),
		"random_forest": RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1),
	}


def scores_for(clf, X):
	if hasattr(clf, "predict_proba"):
		return clf.predict_proba(X)
	return clf.decision_function(X)  # SVC ovr: (n, n_classes)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--folds", type=int, default=5)
	parser.add_argument("--test-size", type=float, default=0.15)
	parser.add_argument("--no-mlflow", action="store_true")
	parser.add_argument(
		"--keep-orientation",
		action="store_true",
		help="skip the EFD spatial-rotation normalisation; keep canonical sprite pose as signal",
	)
	args = parser.parse_args()

	data_dir = PROJECT_ROOT / "data"
	dataset = load_dataset(str(data_dir))
	targets = np.array(dataset.targets)
	classes = dataset.classes

	# --- Reproduce the neural split exactly (same calls, same params). ----------
	normal = _normal_sprite_indices(dataset)
	pool_idx, test_idx = train_test_split(
		normal, test_size=args.test_size, stratify=targets[normal], random_state=args.seed
	)

	# --- Descriptors for every image we will touch (cache once). ----------------
	needed = np.concatenate([pool_idx, test_idx])
	mode = "orientation-preserving" if args.keep_orientation else "fully rotation-invariant"
	print(f"extracting descriptors for {len(needed)} images ({mode})...")
	feats = {
		int(i): descriptor(dataset.samples[i][0], keep_orientation=args.keep_orientation)
		for i in needed
	}
	dim = len(next(iter(feats.values())))
	efd_dim = 4 * EFD_ORDER if args.keep_orientation else 4 * EFD_ORDER - 3
	print(f"descriptor dim: {dim} (EFD {efd_dim} + Hu 7 + ratios 6)")

	def matrix(idx):
		return np.stack([feats[int(i)] for i in idx]), targets[idx]

	# --- Out-of-fold CV, one shallow model per fold, per classifier. ------------
	results = {}
	for name in classifiers():
		oof = []
		fold_acc = []
		for train_idx, val_idx in fold_indices(
			dataset, pool_idx, args.folds, random_state=args.seed, group_aware=True
		):
			Xtr, ytr = matrix(train_idx)
			Xva, yva = matrix(val_idx)
			clf = classifiers()[name]  # fresh estimator per fold
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
			"fold_accuracy_stdev": statistics.stdev(fold_acc),
			"fold_accuracy_sem": statistics.stdev(fold_acc) / len(fold_acc) ** 0.5,
			"fold_accuracies": fold_acc,
		}
		print(
			f"{name:>14}: oof={results[name]['oof_accuracy']:.4f} "
			f"(sem {results[name]['fold_accuracy_sem']:.4f})  "
			f"top5={results[name]['oof_top5_accuracy']:.4f}"
		)

	best = max(results, key=lambda n: results[n]["oof_accuracy"])
	print(f"\nbest classifier: {best}  (oof {results[best]['oof_accuracy']:.4f})")

	# --- Held-out test split: train the best on the full pool, score test once. -
	Xpool, ypool = matrix(pool_idx)
	Xtest, ytest = matrix(test_idx)
	clf = classifiers()[best]
	clf.fit(Xpool, ypool)
	test_top5 = _aligned_top5(clf, Xtest, len(classes))
	test_top1 = top_k_accuracy_from_predictions(test_top5, ytest.tolist(), 1)
	test_top5_acc = top_k_accuracy_from_predictions(test_top5, ytest.tolist(), 5)
	print(f"held-out test ({len(test_idx)} imgs): top1={test_top1:.4f}  top5={test_top5_acc:.4f}")

	if args.no_mlflow:
		return

	mlflow.set_tracking_uri(TRACKING_URI)
	mlflow.set_experiment(EXPERIMENT)
	r = results[best]
	tag = "orient" if args.keep_orientation else "inv"
	with mlflow.start_run(run_name=f"sd-efd{tag}-{best}-s{args.seed}"):
		mlflow.log_params(
			{
				"model_family": "classical_shape_descriptor",
				"descriptor": f"efd{EFD_ORDER}+hu+ratios",
				"descriptor_dim": dim,
				"keep_orientation": args.keep_orientation,
				"classifier": best,
				"folds": args.folds,
				"random_state": args.seed,
				"test_size": args.test_size,
				"group_aware_folds": True,
				"exclude_shiny": True,
			}
		)
		for fold, acc in enumerate(r["fold_accuracies"]):
			mlflow.log_metrics({"fold_accuracy": acc}, step=fold)
		mlflow.log_metrics(
			{k: v for k, v in r.items() if k not in ("oof", "fold_accuracies")}
			| {
				"oof_n": len(r["oof"]),
				"test_accuracy": test_top1,
				"test_top5_accuracy": test_top5_acc,
				"test_n": len(test_idx),
			}
		)
		mlflow.log_dict({"classes": classes, "predictions": r["oof"]}, "oof_predictions.json")
		mlflow.log_dict(
			{n: {k: v for k, v in res.items() if k != "oof"} for n, res in results.items()},
			"classifier_sweep.json",
		)
	print(f"\nLogged sd-efd{tag}-{best}-s{args.seed} to {TRACKING_URI}")


def _aligned_top5(clf, X, n_classes):
	"""Top-5 class *labels* (0..n_classes-1) from a fitted estimator, remapping the
	estimator's own class ordering back to global label ids."""
	scores = scores_for(clf, X)
	order = np.argsort(-scores, axis=1)[:, :5]
	return clf.classes_[order].tolist()


if __name__ == "__main__":
	main()
