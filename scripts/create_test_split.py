"""Carve the seed-42 test split out of `data/` onto disk, into `test_data/`.

Phase 6 needs a test set that cannot leak into ensembling. Until now the split
was *derived* (a seeded `train_test_split` inside `create_fold_data_loaders`), so
the guarantee lived in code; this makes it physical.

Two things this moves, and why both:
  * the 197 seed-42 test normals, and
  * their 122 shiny recolours - near-pixel-identical silhouettes of test images.
    Leaving those in `data/` would keep the guarantee dependent on
    `exclude_shiny=True`, which is exactly the code-level protection an on-disk
    split is meant to replace. 319 files total.

Every move is recorded in `test_split_manifest.json` (committed), so the
operation is reversible with `--revert` and the split stays documented even
though `data/` is gitignored.

    uv run python scripts/create_test_split.py --dry-run   # inspect, move nothing
    uv run python scripts/create_test_split.py             # perform the move
    uv run python scripts/create_test_split.py --revert     # put everything back

NOTE: this changes ImageFolder indexing, so fold composition after the move no
longer matches `p10-lastn6-s*`. The reference must be re-baselined on the new
structure (see EXPERIMENTS.md Phase 6).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from pokemon_training.data import (
	PROJECT_ROOT,
	_normal_sprite_indices,
	load_dataset,
	sprite_groups,
)

DATA_DIR = PROJECT_ROOT / "data"
TEST_DIR = PROJECT_ROOT / "test_data"
MANIFEST = PROJECT_ROOT / "test_split_manifest.json"
SEED = 42
TEST_SIZE = 0.15


def plan_moves():
	"""Compute the seed-42 test normals + their shiny twins, as (src, dst, kind)."""
	dataset = load_dataset(DATA_DIR)
	targets = np.array(dataset.targets)
	normal = _normal_sprite_indices(dataset)

	_pool_idx, test_idx = train_test_split(
		normal, test_size=TEST_SIZE, stratify=targets[normal], random_state=SEED
	)

	all_idx = np.arange(len(dataset))
	group_of = dict(zip(all_idx.tolist(), sprite_groups(dataset, all_idx).tolist()))
	test_groups = {group_of[int(i)] for i in test_idx}
	shiny = set(all_idx.tolist()) - set(normal.tolist())
	twin_idx = sorted(i for i in shiny if group_of[i] in test_groups)

	moves = []
	for kind, indices in (("normal", sorted(int(i) for i in test_idx)), ("shiny_twin", twin_idx)):
		for i in indices:
			src = Path(dataset.samples[i][0])
			moves.append(
				{
					"kind": kind,
					"class": dataset.classes[dataset.targets[i]],
					"src": str(src.relative_to(PROJECT_ROOT)),
					"dst": str((TEST_DIR / src.parent.name / src.name).relative_to(PROJECT_ROOT)),
				}
			)
	return dataset, moves


def do_move(moves):
	for m in moves:
		dst = PROJECT_ROOT / m["dst"]
		dst.parent.mkdir(parents=True, exist_ok=True)
		shutil.move(PROJECT_ROOT / m["src"], dst)


def do_revert():
	if not MANIFEST.exists():
		raise SystemExit(f"{MANIFEST.name} not found - nothing to revert")
	manifest = json.loads(MANIFEST.read_text())
	for m in manifest["moves"]:
		src = PROJECT_ROOT / m["src"]
		dst = PROJECT_ROOT / m["dst"]
		if dst.exists():
			src.parent.mkdir(parents=True, exist_ok=True)
			shutil.move(dst, src)
	print(f"reverted {len(manifest['moves'])} files back into data/")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dry-run", action="store_true")
	parser.add_argument("--revert", action="store_true")
	args = parser.parse_args()

	if args.revert:
		do_revert()
		return

	if TEST_DIR.exists() and any(TEST_DIR.iterdir()):
		raise SystemExit(f"{TEST_DIR} already exists and is non-empty - refusing to overwrite")

	dataset, moves = plan_moves()
	normals = [m for m in moves if m["kind"] == "normal"]
	twins = [m for m in moves if m["kind"] == "shiny_twin"]
	classes = {m["class"] for m in normals}
	print(f"dataset: {len(dataset)} images, {len(dataset.classes)} classes")
	print(f"to move: {len(normals)} test normals + {len(twins)} shiny twins = {len(moves)} files")
	print(f"test normals cover {len(classes)}/{len(dataset.classes)} classes")

	if args.dry_run:
		print("\n--dry-run: nothing moved. First 5:")
		for m in moves[:5]:
			print(f"  {m['src']}  ->  {m['dst']}  ({m['kind']})")
		return

	do_move(moves)
	MANIFEST.write_text(
		json.dumps(
			{
				"seed": SEED,
				"test_size": TEST_SIZE,
				"n_test_normals": len(normals),
				"n_shiny_twins": len(twins),
				"note": (
					"Seed-42 test split carved onto disk for Phase 6. Shiny recolours of "
					"test normals moved too, so no near-duplicate of a test silhouette "
					"remains in data/. Reversible: scripts/create_test_split.py --revert."
				),
				"moves": moves,
			},
			indent=2,
		)
		+ "\n"
	)
	print(f"\nmoved {len(moves)} files -> {TEST_DIR.name}/")
	print(f"manifest written: {MANIFEST.name}")


if __name__ == "__main__":
	main()
