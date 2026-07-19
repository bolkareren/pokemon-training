"""Compare a run's out-of-fold confusions against silhouette shape similarity
and the Gen 1 evolution-line structure.

    uv run python scripts/confusion_study.py [--run-name <mlflow run name>]

Similarity is max IoU on bbox-cropped, scale-normalised masks - raw IoU would
be dominated by the generation sprite-scale artifact.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import mlflow
import numpy as np
from PIL import Image

from pokemon_training.data import PROJECT_ROOT, load_dataset

TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT = "pokemon-classification-clean"
NORMALIZED_SIZE = 64
# Run-independent, so persisted (gitignored) and keyed by an image-set digest.
SIMILARITY_CACHE = PROJECT_ROOT / ".similarity-cache.npz"

# Gen 1 evolution lines with 2+ members among the 151 classes (slugs match
# data/ directory names); unlisted classes are singleton families.
# Cross-generation relatives (Onix->Steelix, Tyrogue) do not count.
EVOLUTION_FAMILIES = [
	("bulbasaur", "ivysaur", "venusaur"),
	("charmander", "charmeleon", "charizard"),
	("squirtle", "wartortle", "blastoise"),
	("caterpie", "metapod", "butterfree"),
	("weedle", "kakuna", "beedrill"),
	("pidgey", "pidgeotto", "pidgeot"),
	("rattata", "raticate"),
	("spearow", "fearow"),
	("ekans", "arbok"),
	("pikachu", "raichu"),
	("sandshrew", "sandslash"),
	("nidoran-f", "nidorina", "nidoqueen"),
	("nidoran-m", "nidorino", "nidoking"),
	("clefairy", "clefable"),
	("vulpix", "ninetales"),
	("jigglypuff", "wigglytuff"),
	("zubat", "golbat"),
	("oddish", "gloom", "vileplume"),
	("paras", "parasect"),
	("venonat", "venomoth"),
	("diglett", "dugtrio"),
	("meowth", "persian"),
	("psyduck", "golduck"),
	("mankey", "primeape"),
	("growlithe", "arcanine"),
	("poliwag", "poliwhirl", "poliwrath"),
	("abra", "kadabra", "alakazam"),
	("machop", "machoke", "machamp"),
	("bellsprout", "weepinbell", "victreebel"),
	("tentacool", "tentacruel"),
	("geodude", "graveler", "golem"),
	("ponyta", "rapidash"),
	("slowpoke", "slowbro"),
	("magnemite", "magneton"),
	("doduo", "dodrio"),
	("seel", "dewgong"),
	("grimer", "muk"),
	("shellder", "cloyster"),
	("gastly", "haunter", "gengar"),
	("drowzee", "hypno"),
	("krabby", "kingler"),
	("voltorb", "electrode"),
	("exeggcute", "exeggutor"),
	("cubone", "marowak"),
	("koffing", "weezing"),
	("rhyhorn", "rhydon"),
	("horsea", "seadra"),
	("goldeen", "seaking"),
	("staryu", "starmie"),
	("magikarp", "gyarados"),
	("eevee", "vaporeon", "jolteon", "flareon"),
	("omanyte", "omastar"),
	("kabuto", "kabutops"),
	("dratini", "dragonair", "dragonite"),
]


def normalized_silhouette(path, size=NORMALIZED_SIZE):
	"""Bbox-crop the creature, then resize to a fixed square: shape without scale."""
	mask = np.array(Image.open(path).convert("L")) <= 127  # creature is dark on white
	rows, cols = np.where(mask)
	if len(rows) == 0:
		return np.zeros((size, size), dtype=bool)

	cropped = mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
	resized = Image.fromarray(cropped.astype(np.uint8) * 255).resize((size, size), Image.BILINEAR)
	return np.array(resized) > 127


def iou(a, b):
	union = (a | b).sum()
	return (a & b).sum() / union if union else 0.0


def load_oof(run_name):
	mlflow.set_tracking_uri(TRACKING_URI)
	client = mlflow.tracking.MlflowClient()
	experiment = mlflow.get_experiment_by_name(EXPERIMENT)
	matches = [
		r
		for r in client.search_runs([experiment.experiment_id], max_results=500)
		if r.data.tags.get("mlflow.runName") == run_name and r.info.status == "FINISHED"
	]
	if not matches:
		raise SystemExit(f"no FINISHED run named {run_name} in {EXPERIMENT}")

	path = client.download_artifacts(matches[0].info.run_id, "oof_predictions.json")
	with open(path) as handle:
		return json.load(handle)


def class_similarity(dataset, classes):
	"""Max IoU between any two images of two classes - max because one
	confusable pair of renderings is what makes classes confusable."""
	by_class = defaultdict(list)
	for path, target in dataset.samples:
		by_class[target].append(path)

	shapes = {label: [normalized_silhouette(p) for p in paths] for label, paths in by_class.items()}

	n = len(classes)
	similarity = np.zeros((n, n))
	for a in range(n):
		for b in range(a + 1, n):
			best = max(iou(x, y) for x in shapes[a] for y in shapes[b])
			similarity[a, b] = similarity[b, a] = best

	return similarity


def cached_class_similarity(dataset, classes):
	"""class_similarity with an npz cache; the paths+mtimes digest invalidates
	it whenever the image set changes."""
	fingerprint = "\n".join(
		f"{path}:{Path(path).stat().st_mtime_ns}" for path, _ in dataset.samples
	)
	digest = hashlib.sha1(f"{NORMALIZED_SIZE}\n{fingerprint}".encode()).hexdigest()

	if SIMILARITY_CACHE.exists():
		stored = np.load(SIMILARITY_CACHE, allow_pickle=False)
		if str(stored["digest"]) == digest:
			print(f"similarity matrix loaded from {SIMILARITY_CACHE.name}")
			return stored["similarity"]

	print("computing pairwise shape similarity (bbox-cropped, scale-normalised)...")
	similarity = class_similarity(dataset, classes)
	np.savez(SIMILARITY_CACHE, similarity=similarity, digest=digest)
	return similarity


def family_by_class(classes):
	"""Map class index -> family id (singletons for unlisted classes), failing
	loudly on any slug mismatch."""
	listed = [slug for family in EVOLUTION_FAMILIES for slug in family]
	duplicated = {slug for slug in listed if listed.count(slug) > 1}
	if duplicated:
		raise SystemExit(f"slugs in more than one family: {sorted(duplicated)}")
	unknown = set(listed) - set(classes)
	if unknown:
		raise SystemExit(f"family slugs not in dataset classes: {sorted(unknown)}")

	family_of_slug = {
		slug: family_id for family_id, family in enumerate(EVOLUTION_FAMILIES) for slug in family
	}
	return {
		index: family_of_slug.get(slug, len(EVOLUTION_FAMILIES) + index)
		for index, slug in enumerate(classes)
	}


def report_evolution_line_confusions(errors, classes, top_pairs):
	"""Share of top-1 errors landing inside the true class's evolution family,
	next to its chance rate so runs of different accuracy stay comparable."""
	family = family_by_class(classes)
	within = [e for e in errors if family[e["label"]] == family[e["top5"][0]]]

	family_size = Counter(family.values())
	chance = np.mean([(family_size[family[e["label"]]] - 1) / (len(classes) - 1) for e in errors])

	rate = len(within) / len(errors)
	print(
		f"\nevolution-line confusions: {len(within)} / {len(errors)} errors "
		f"({rate:.1%}, chance {chance:.1%}, lift {rate / chance:.0f}x)"
	)
	counts = Counter((e["label"], e["top5"][0]) for e in within)
	for (true, pred), count in counts.most_common(top_pairs):
		print(f"  {count:>3}x  {classes[true]:>14} -> {classes[pred]}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--run-name", default="c2-resnet50-standard")
	parser.add_argument("--top-pairs", type=int, default=15)
	args = parser.parse_args()

	oof = load_oof(args.run_name)
	classes = oof["classes"]
	predictions = oof["predictions"]

	dataset = load_dataset(PROJECT_ROOT / "data")
	# Restrict to the images the run actually scored, so similarity is computed
	# over the same non-shiny subset the model saw.
	scored = {entry["index"] for entry in predictions}
	dataset.samples = [s for i, s in enumerate(dataset.samples) if i in scored]

	print(f"run: {args.run_name}   images: {len(predictions)}   classes: {len(classes)}")
	similarity = cached_class_similarity(dataset, classes)

	upper = similarity[np.triu_indices(len(classes), k=1)]
	print(f"\ncross-class similarity over {len(upper):,} pairs:")
	print(f"  mean {upper.mean():.3f}   median {np.median(upper):.3f}   max {upper.max():.3f}")
	for threshold in (0.95, 0.90, 0.85, 0.80, 0.75):
		count = int((upper > threshold).sum())
		print(f"  pairs above IoU {threshold}: {count:>5}  ({count / len(upper):.2%})")

	print("\nmost similar class pairs (different Pokemon, near-identical shape):")
	order = np.argsort(upper)[::-1]
	pairs = [(i, j) for i in range(len(classes)) for j in range(i + 1, len(classes))]
	for rank in order[: args.top_pairs]:
		i, j = pairs[rank]
		print(f"  {upper[rank]:.3f}  {classes[i]} / {classes[j]}")

	# --- confusions -------------------------------------------------------
	errors = [e for e in predictions if e["top5"][0] != e["label"]]
	in_top3 = [e for e in errors if e["label"] in e["top5"][:3]]
	in_top5 = [e for e in errors if e["label"] in e["top5"]]

	error_rate = len(errors) / len(predictions)
	print(f"\ntop-1 errors: {len(errors)} / {len(predictions)} ({error_rate:.1%})")
	print(f"  recovered in top-3: {len(in_top3)} ({len(in_top3) / len(errors):.1%})")
	print(f"  recovered in top-5: {len(in_top5)} ({len(in_top5) / len(errors):.1%})")

	confused_sim = np.array([similarity[e["label"]][e["top5"][0]] for e in errors])
	correct = [e for e in predictions if e["top5"][0] == e["label"]]
	# Baseline: how similar is a class to a randomly chosen other class?
	rng = np.random.default_rng(0)
	random_sim = np.array(
		[similarity[e["label"]][rng.integers(len(classes))] for e in predictions for _ in range(3)]
	)

	print("\nshape similarity, wrong prediction vs true class:")
	print(f"  confused: mean {confused_sim.mean():.3f} median {np.median(confused_sim):.3f}")
	print(f"  random:   mean {random_sim.mean():.3f} median {np.median(random_sim):.3f}")
	print(f"  lift: {confused_sim.mean() / random_sim.mean():.2f}x")

	for threshold in (0.90, 0.85, 0.80):
		share = float((confused_sim > threshold).mean())
		base = float((random_sim > threshold).mean())
		print(f"  above IoU {threshold}: {share:.1%} (random {base:.1%})")

	print("\nmost frequent confusions (true -> predicted, with shape similarity):")
	counts = Counter((e["label"], e["top5"][0]) for e in errors)
	for (true, pred), count in counts.most_common(args.top_pairs):
		sim = similarity[true][pred]
		print(f"  {count:>3}x  {classes[true]:>14} -> {classes[pred]:<14} IoU {sim:.3f}")

	report_evolution_line_confusions(errors, classes, args.top_pairs)

	# Does high similarity predict "right answer demoted to top-3/5" rather than lost?
	recovered_sim = np.array([similarity[e["label"]][e["top5"][0]] for e in in_top5])
	lost = [e for e in errors if e["label"] not in e["top5"]]
	lost_sim = np.array([similarity[e["label"]][e["top5"][0]] for e in lost])
	print(f"\nerror, true class in top-5: mean IoU {recovered_sim.mean():.3f} (n={len(in_top5)})")
	if len(lost):
		print(f"error, fell outside top-5: mean IoU {lost_sim.mean():.3f} (n={len(lost)})")

	per_class_correct = Counter(e["label"] for e in correct)
	per_class_total = Counter(e["label"] for e in predictions)
	accuracy_by_class = {c: per_class_correct[c] / per_class_total[c] for c in per_class_total}
	worst = sorted(accuracy_by_class.items(), key=lambda kv: kv[1])[:12]
	print("\nworst classes (accuracy, closest other class):")
	for label, acc in worst:
		nearest = int(np.argmax(similarity[label]))
		sim = similarity[label][nearest]
		print(f"  {acc:.2f}  {classes[label]:>14}  nearest: {classes[nearest]} (IoU {sim:.3f})")


if __name__ == "__main__":
	main()
