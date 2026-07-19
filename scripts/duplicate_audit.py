"""Audit silhouette duplication in `data/` and leakage across the train/val split.

One-off analysis script (like the scraping/processing scripts, not part of the
package). Motivation: pokemondb.net's per-Pokemon sprite page serves both the
*normal* and the *shiny* sprite for each generation. Shiny differs only in
palette, so after the binary silhouette threshold in `data.py` the two are
frequently pixel-identical - which makes a random split leak.

    uv run python scripts/duplicate_audit.py

Reports, for the processed 224x224 silhouettes:
  1. how many images are exact / near duplicates of another image in their class
  2. what fraction of each validation split has a twin sitting in train
"""

from pathlib import Path

import numpy as np
from PIL import Image

from pokemon_training.data import load_dataset, split_dataset_indices

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# IoU above which two silhouettes count as "near duplicates". 0.97 is well above
# the ~0.4-0.9 range genuine cross-generation redesigns of the same Pokemon land
# in, so this measures redundancy, not merely similarity.
NEAR_DUPLICATE_IOU = 0.97

SEEDS = (42, 43, 44)


def load_silhouette(path):
	"""Reproduce the binary silhouette `data.py` trains on, as a bool mask."""
	return np.array(Image.open(path).convert("L")) > 127


def iou(a, b):
	union = (a | b).sum()
	return (a & b).sum() / union if union else 0.0


def audit_duplicates(silhouettes, targets):
	"""Count redundant images per class: exact byte-equal, and near-duplicate."""
	exact = 0
	near = 0

	for label in np.unique(targets):
		members = [silhouettes[i] for i in np.flatnonzero(targets == label)]

		seen = set()
		for silhouette in members:
			key = silhouette.tobytes()
			if key in seen:
				exact += 1
			seen.add(key)

		kept = []
		for silhouette in members:
			if any(iou(silhouette, k) > NEAR_DUPLICATE_IOU for k in kept):
				near += 1
			else:
				kept.append(silhouette)

	return exact, near


def audit_split_leakage(silhouettes, targets, dataset, seed):
	"""Fraction of the val split whose twin image sits in the train split."""
	train_idx, val_idx, _ = split_dataset_indices(
		dataset, val_size=0.1, test_size=0.1, random_state=seed
	)

	train_hashes = {silhouettes[i].tobytes() for i in train_idx}

	train_by_class = {}
	for i in train_idx:
		train_by_class.setdefault(targets[i], []).append(silhouettes[i])

	exact = sum(1 for i in val_idx if silhouettes[i].tobytes() in train_hashes)
	near = sum(
		1
		for i in val_idx
		if any(
			iou(silhouettes[i], t) > NEAR_DUPLICATE_IOU for t in train_by_class.get(targets[i], [])
		)
	)

	return len(val_idx), exact, near


def main():
	dataset = load_dataset(PROJECT_ROOT / "data")
	targets = np.array(dataset.targets)
	silhouettes = [load_silhouette(path) for path, _ in dataset.samples]
	total = len(silhouettes)

	exact, near = audit_duplicates(silhouettes, targets)
	print(f"total images: {total}")
	print(f"exact-duplicate redundant images:  {exact:>5} ({exact / total:.1%})")
	print(f"near-duplicate redundant images:   {near:>5} ({near / total:.1%})")
	print(f"distinct silhouettes after dedup:  {total - near:>5}")
	print()

	print("leakage across the stratified train/val split:")
	for seed in SEEDS:
		n_val, exact, near = audit_split_leakage(silhouettes, targets, dataset, seed)
		print(
			f"  seed {seed}: val n={n_val} | "
			f"exact twin in train {exact:>3} ({exact / n_val:.1%}) | "
			f"near twin in train {near:>3} ({near / n_val:.1%})"
		)


if __name__ == "__main__":
	main()
