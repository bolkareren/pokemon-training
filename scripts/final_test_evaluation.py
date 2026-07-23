"""Phase 6 one-shot test evaluation: the ensemble, scored once on the never-trained
`test_data/` split, with the error studies that still apply to an ensemble.

This is the one-way door. `test_data/` was carved out on disk precisely so nothing
in exploration could see it; anything learned here becomes a *new hypothesis*, not
a tuning signal. Run it after the ensemble config is frozen on OOF.

Reports, beyond top-1/3/5:
  * Wilson 95% CIs - with n=197 the interval is wide (~+-6pt), which is the honest
    resolution of a single held-out split and the reason not to tune against it.
  * ensemble vs. its own members, so the ensembling gain is attributable.
  * evolution-line confusions and silhouette-collision similarity, reused verbatim
    from the confusion study - both are class-level properties, so they apply to
    ensemble predictions unchanged.
  * confidence calibration and member agreement, which only exist for an ensemble:
    the averaged softmax is a usable confidence signal for the guessing game.

    uv run python scripts/final_test_evaluation.py --runs p14-ref-extsplit-s42 ...
"""

import argparse
import math
from collections import Counter

import mlflow
import numpy as np
import torch

from pokemon_training.config import ExperimentConfig
from pokemon_training.data import PROJECT_ROOT, load_dataset
from pokemon_training.evaluation import top_k_from_probabilities
from scripts.confusion_study import (
	cached_class_similarity,
	report_evolution_line_confusions,
)
from scripts.ensemble import (
	TTA_VIEWS,
	build_test_loader,
	checkpoint_paths,
	ensemble_probabilities,
	load_models,
	report,
)

TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT = "pokemon-classification-clean"


def wilson_interval(correct, total, z=1.96):
	"""Wilson score interval - the right binomial CI at these n, where the normal
	approximation misbehaves near the tails."""
	if total == 0:
		return (float("nan"), float("nan"))
	p = correct / total
	denominator = 1 + z**2 / total
	center = (p + z**2 / (2 * total)) / denominator
	half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
	return (max(0.0, center - half), min(1.0, center + half))


def report_silhouette_collisions(errors, similarity, classes, top_pairs=10):
	"""How much of the error is explained by classes that genuinely collide in
	silhouette - the ~3% ceiling the confusion study measured for a single model."""
	pairs = [(e["label"], e["top5"][0]) for e in errors]
	sims = np.array([similarity[a, b] for a, b in pairs])

	off_diagonal = similarity[~np.eye(len(classes), dtype=bool)]
	print(
		f"\nsilhouette similarity of confused pairs: mean {sims.mean():.3f} "
		f"vs {off_diagonal.mean():.3f} for a random class pair "
		f"(lift {sims.mean() / off_diagonal.mean():.2f}x)"
	)
	for threshold in (0.9, 0.8, 0.7):
		share = (sims >= threshold).mean()
		count = int((sims >= threshold).sum())
		print(f"  errors with IoU >= {threshold}: {count:>3} ({share:.1%})")

	ranked = sorted(zip(pairs, sims), key=lambda item: -item[1])[:top_pairs]
	print("  most shape-collided confusions:")
	for (true, pred), sim in ranked:
		print(f"    IoU {sim:.3f}  {classes[true]:>14} -> {classes[pred]}")


def report_confidence(probabilities, labels, bins=5):
	"""Is the averaged softmax a usable confidence signal? Only meaningful for an
	ensemble, where the mean probability reflects member agreement."""
	confidence, predicted = probabilities.max(dim=1)
	correct = np.array([p == label for p, label in zip(predicted.tolist(), labels)])
	confidence = confidence.numpy()

	print("\nconfidence calibration (ensemble max-probability):")
	edges = np.quantile(confidence, np.linspace(0, 1, bins + 1))
	edges[-1] += 1e-9
	for low, high in zip(edges[:-1], edges[1:]):
		mask = (confidence >= low) & (confidence < high)
		if mask.sum() == 0:
			continue
		print(
			f"  conf {low:.2f}-{high:.2f}: n={int(mask.sum()):>3}  "
			f"accuracy {correct[mask].mean():.1%}  (mean conf {confidence[mask].mean():.2f})"
		)


def report_agreement(each, labels, classes):
	"""How often the members already agree - the ensemble can only add value where
	they do not."""
	votes = np.stack([p.argmax(dim=1).numpy() for p in each])
	unanimous = (votes == votes[0]).all(axis=0)
	labels_array = np.array(labels)
	unanimous_accuracy = (votes[0][unanimous] == labels_array[unanimous]).mean()
	print(
		f"\nmember agreement: unanimous on {unanimous.mean():.1%} of images "
		f"({unanimous.sum()}/{len(labels)}), accuracy there {unanimous_accuracy:.1%}"
	)
	split = ~unanimous
	if split.sum():
		majority = np.array([Counter(votes[:, i]).most_common(1)[0][0] for i in np.where(split)[0]])
		print(
			f"  where members disagree: n={int(split.sum())}, "
			f"majority-vote accuracy {(majority == labels_array[split]).mean():.1%}"
		)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--runs", nargs="+", required=True)
	parser.add_argument("--views", nargs="+", default=["identity"], choices=list(TTA_VIEWS))
	parser.add_argument("--test-dir", default="test_data")
	parser.add_argument("--skip-similarity", action="store_true", help="skip the IoU study")
	parser.add_argument("--no-mlflow", action="store_true")
	args = parser.parse_args()

	config = ExperimentConfig()
	data_dir = PROJECT_ROOT / "data"
	device = "mps" if torch.backends.mps.is_available() else "cpu"

	test_loader, classes = build_test_loader(config, data_dir, PROJECT_ROOT / args.test_dir)
	paths = checkpoint_paths(args.runs)
	n_test = len(test_loader.dataset)
	print("=== Phase 6 one-shot test evaluation ===")
	print(f"{len(paths)} checkpoints from {len(args.runs)} runs | views={args.views}")
	print(f"held-out test images: {n_test} | classes: {len(classes)}\n")

	models = load_models(paths, config, len(classes), device)
	probabilities, labels, each = ensemble_probabilities(
		models, test_loader, device, views=tuple(args.views), per_model=True
	)

	metrics = report(probabilities, labels)
	singles = [report(p, labels)["top1"] for p in each]
	single_mean = sum(singles) / len(singles)

	print("accuracy (Wilson 95% CI):")
	for k in (1, 3, 5):
		value = metrics[f"top{k}"]
		low, high = wilson_interval(round(value * n_test), n_test)
		print(f"  top-{k}: {value:.4f}  [{low:.4f}, {high:.4f}]")
	print(
		f"\nmembers: mean top1 {single_mean:.4f} (min {min(singles):.4f}, max {max(singles):.4f})"
	)
	print(f"ensemble gain over mean member: {metrics['top1'] - single_mean:+.4f}")

	top5 = top_k_from_probabilities(probabilities, k=5)
	errors = [
		{"label": label, "top5": prediction}
		for prediction, label in zip(top5, labels)
		if prediction[0] != label
	]
	print(f"\ntop-1 errors: {len(errors)}/{n_test}")

	report_evolution_line_confusions(errors, classes, top_pairs=10)
	if not args.skip_similarity:
		pool = load_dataset(data_dir)
		similarity = cached_class_similarity(pool, classes)
		report_silhouette_collisions(errors, similarity, classes)
	report_confidence(probabilities, labels)
	report_agreement(each, labels, classes)

	if args.no_mlflow:
		return
	mlflow.set_tracking_uri(TRACKING_URI)
	mlflow.set_experiment(EXPERIMENT)
	with mlflow.start_run(run_name=f"p6-final-test-{len(paths)}models"):
		mlflow.log_params(
			{
				"model_family": "fold_seed_ensemble",
				"members": len(paths),
				"runs": ",".join(args.runs),
				"views": ",".join(args.views),
				"combination": "mean_softmax",
				"test_n": n_test,
			}
		)
		low1, high1 = wilson_interval(round(metrics["top1"] * n_test), n_test)
		mlflow.log_metrics(
			{
				"test_top1": metrics["top1"],
				"test_top3": metrics["top3"],
				"test_top5": metrics["top5"],
				"test_top1_ci_low": low1,
				"test_top1_ci_high": high1,
				"member_top1_mean": single_mean,
				"ensemble_gain": metrics["top1"] - single_mean,
			}
		)
		mlflow.log_dict(
			{
				"classes": classes,
				"predictions": [
					{"label": label, "top5": prediction} for prediction, label in zip(top5, labels)
				],
			},
			"test_predictions.json",
		)
	print("\nlogged to MLflow. This split is now spent - further tuning against it is not valid.")


if __name__ == "__main__":
	main()
