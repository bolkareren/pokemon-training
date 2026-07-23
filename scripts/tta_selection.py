"""OOF-valid TTA selection: which test-time views help, decided without touching
the one-shot test split.

The validity rule that shapes this script: each fold's model is run on *its own*
held-out val fold only. Pooling the fold ensemble over OOF data would be leakage
(every pool image is in-training for K-1 of its seed's models), so nothing here
ensembles across folds - it ensembles only across *views of one image*, under the
one model that never saw it. Pooling the five folds' predictions afterwards
reconstructs a clean OOF set, exactly as `run_cross_validation` does.

Consequently this measures the TTA effect and nothing else; the fold-ensemble
gain stays unmeasurable until the test spend.

Pre-registered decision rule: each view set is compared to `identity` paired over
the 15 (seed, fold) pairs - mean delta, paired t-test, and the 2x SEM bar. Views
are adopted only if they clear it; otherwise the final ensemble runs identity-only.

    uv run python scripts/tta_selection.py --runs p14-ref-extsplit-s42 ... --seeds 42 43 44
"""

import argparse
import statistics

import torch
from scipy import stats

from pokemon_training.config import ExperimentConfig
from pokemon_training.data import PROJECT_ROOT
from scripts.ensemble import (
	checkpoint_paths,
	ensemble_probabilities,
	fold_loaders_for,
	load_models,
	report,
)

# Candidate view sets. identity is the baseline every other set is measured against.
# The last three were added as an exploratory follow-up once the pre-registered four
# produced an odd result (hflip null, rotations null, both together significant):
# "4v rot-only" is the view-count control that rules out "more views is all that
# matters", and it does not clear significance while the hflip-containing sets of the
# same size do. "6v all" is the measured winner and the config the final ensemble uses.
VIEW_SETS = {
	"identity": ("identity",),
	"+hflip": ("identity", "hflip"),
	"+rot10": ("identity", "rot+10", "rot-10"),
	"+hflip+rot10": ("identity", "hflip", "rot+10", "rot-10"),
	"4v rot-only": ("identity", "rot+10", "rot-10", "rot+20"),
	"4v hflip-heavy": ("identity", "hflip", "rot+20", "rot-20"),
	"6v all": ("identity", "hflip", "rot+10", "rot-10", "rot+20", "rot-20"),
}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--runs", nargs="+", required=True)
	parser.add_argument("--seeds", nargs="+", type=int, required=True)
	parser.add_argument("--test-dir", default="test_data")
	args = parser.parse_args()

	if len(args.runs) != len(args.seeds):
		raise SystemExit("--runs and --seeds must be parallel (each run's own fold split)")

	config = ExperimentConfig()
	data_dir = PROJECT_ROOT / "data"
	test_dir = PROJECT_ROOT / args.test_dir
	device = "mps" if torch.backends.mps.is_available() else "cpu"

	# fold_accuracy[view_set] = [acc per (seed, fold)]; oof[view_set][seed] = pooled
	fold_accuracy = {name: [] for name in VIEW_SETS}
	oof_accuracy = {name: {} for name in VIEW_SETS}

	for run, seed in zip(args.runs, args.seeds):
		fold_loaders, classes = fold_loaders_for(config, data_dir, test_dir, seed)
		paths = checkpoint_paths([run])
		if len(paths) != len(fold_loaders):
			raise SystemExit(f"{run}: {len(paths)} checkpoints vs {len(fold_loaders)} folds")
		print(f"\n=== {run} (seed {seed}) ===")

		pooled = {name: {"probs": [], "labels": []} for name in VIEW_SETS}
		for fold, ((_train, val_loader, _idx), path) in enumerate(zip(fold_loaders, paths)):
			# Load once, evaluate every view set against it.
			model = load_models([path], config, len(classes), device)[0]
			line = []
			for name, views in VIEW_SETS.items():
				probabilities, labels = ensemble_probabilities(
					[model], val_loader, device, views=views
				)
				accuracy = report(probabilities, labels)["top1"]
				fold_accuracy[name].append(accuracy)
				pooled[name]["probs"].append(probabilities)
				pooled[name]["labels"].extend(labels)
				line.append(f"{name} {accuracy:.4f}")
			print(f"  fold {fold + 1}: " + "  ".join(line))
			del model

		for name in VIEW_SETS:
			probabilities = torch.cat(pooled[name]["probs"])
			metrics = report(probabilities, pooled[name]["labels"])
			oof_accuracy[name][seed] = metrics
			print(f"  OOF [{name:>13}]: top1 {metrics['top1']:.4f}  top5 {metrics['top5']:.4f}")

	baseline = fold_accuracy["identity"]
	print(f"\n{'=' * 66}\nOOF TTA selection vs identity ({len(baseline)} paired folds)\n{'=' * 66}")
	for name in VIEW_SETS:
		means = [oof_accuracy[name][s]["top1"] for s in args.seeds]
		mean_oof = statistics.mean(means)
		if name == "identity":
			per_seed = [f"{m:.4f}" for m in means]
			print(f"{name:>13}: OOF {mean_oof:.4f}  (baseline)  per-seed {per_seed}")
			continue
		deltas = [a - b for a, b in zip(fold_accuracy[name], baseline)]
		mean_delta = statistics.mean(deltas)
		t_stat = mean_delta / (statistics.stdev(deltas) / len(deltas) ** 0.5)
		p_value = stats.t.sf(abs(t_stat), df=len(deltas) - 1) * 2
		positive = sum(1 for d in deltas if d > 0)
		print(
			f"{name:>13}: OOF {mean_oof:.4f}  delta {mean_delta:+.4f}  "
			f"t={t_stat:+.2f} p={p_value:.4f}  ({positive}/{len(deltas)} folds up)"
		)


if __name__ == "__main__":
	main()
