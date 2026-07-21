"""Cross-seed agreement + a decomposition of *why* silhouettes get confused.

Two studies over the three reference seeds (`p7-ref-26-s{42,43,44}`):

1. **Agreement** — top-1/3/5 accuracy per seed, pairwise top-1 agreement, and the
   per-image "how many seeds got it / agreed" breakdown. Quantifies how much of
   the error is stable vs. seed-dependent (the ensemble headroom noted in
   EXPERIMENTS.md).
2. **Confusion reasons** — for every top-1 error (pooled over the three seeds),
   is the wrong guess the same evolution family / a type-sharing Pokemon / a
   similar-size Pokemon / a similar-shape Pokemon? Each factor's rate is shown
   next to its chance baseline (lift), so factors of different base rates stay
   comparable, plus a combined "explained by at least one".

    uv run python scripts/seed_agreement_study.py
    uv run python scripts/seed_agreement_study.py --runs p7-ref-26-s42 p7-ref-26-s43

Types are Gen-1 (Red/Blue) typings — era-consistent with the sprites; the later
Fairy/Steel retcons (Clefairy, Magnemite, Mr. Mime...) are not applied. Type is a
reference table for post-hoc analysis, not a preprocessing input. "Size" is
per-class median silhouette occupancy (creature pixels / 224x224 canvas); all
151 are Gen 1, so occupancy is a fair within-dataset size proxy.
"""

import argparse
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from pokemon_training.data import PROJECT_ROOT, load_dataset
from scripts.confusion_study import (
	cached_class_similarity,
	family_by_class,
	load_oof,
)

# Gen-1 type sets, keyed by dataset slug. Dual types are a 2-set; "share a type"
# is a non-empty intersection.
GEN1_TYPES = {
	"abra": {"psychic"},
	"aerodactyl": {"rock", "flying"},
	"alakazam": {"psychic"},
	"arbok": {"poison"},
	"arcanine": {"fire"},
	"articuno": {"ice", "flying"},
	"beedrill": {"bug", "poison"},
	"bellsprout": {"grass", "poison"},
	"blastoise": {"water"},
	"bulbasaur": {"grass", "poison"},
	"butterfree": {"bug", "flying"},
	"caterpie": {"bug"},
	"chansey": {"normal"},
	"charizard": {"fire", "flying"},
	"charmander": {"fire"},
	"charmeleon": {"fire"},
	"clefable": {"normal"},
	"clefairy": {"normal"},
	"cloyster": {"water", "ice"},
	"cubone": {"ground"},
	"dewgong": {"water", "ice"},
	"diglett": {"ground"},
	"ditto": {"normal"},
	"dodrio": {"normal", "flying"},
	"doduo": {"normal", "flying"},
	"dragonair": {"dragon"},
	"dragonite": {"dragon", "flying"},
	"dratini": {"dragon"},
	"drowzee": {"psychic"},
	"dugtrio": {"ground"},
	"eevee": {"normal"},
	"ekans": {"poison"},
	"electabuzz": {"electric"},
	"electrode": {"electric"},
	"exeggcute": {"grass", "psychic"},
	"exeggutor": {"grass", "psychic"},
	"farfetchd": {"normal", "flying"},
	"fearow": {"normal", "flying"},
	"flareon": {"fire"},
	"gastly": {"ghost", "poison"},
	"gengar": {"ghost", "poison"},
	"geodude": {"rock", "ground"},
	"gloom": {"grass", "poison"},
	"golbat": {"poison", "flying"},
	"goldeen": {"water"},
	"golduck": {"water"},
	"golem": {"rock", "ground"},
	"graveler": {"rock", "ground"},
	"grimer": {"poison"},
	"growlithe": {"fire"},
	"gyarados": {"water", "flying"},
	"haunter": {"ghost", "poison"},
	"hitmonchan": {"fighting"},
	"hitmonlee": {"fighting"},
	"horsea": {"water"},
	"hypno": {"psychic"},
	"ivysaur": {"grass", "poison"},
	"jigglypuff": {"normal"},
	"jolteon": {"electric"},
	"jynx": {"ice", "psychic"},
	"kabuto": {"rock", "water"},
	"kabutops": {"rock", "water"},
	"kadabra": {"psychic"},
	"kakuna": {"bug", "poison"},
	"kangaskhan": {"normal"},
	"kingler": {"water"},
	"koffing": {"poison"},
	"krabby": {"water"},
	"lapras": {"water", "ice"},
	"lickitung": {"normal"},
	"machamp": {"fighting"},
	"machoke": {"fighting"},
	"machop": {"fighting"},
	"magikarp": {"water"},
	"magmar": {"fire"},
	"magnemite": {"electric"},
	"magneton": {"electric"},
	"mankey": {"fighting"},
	"marowak": {"ground"},
	"meowth": {"normal"},
	"metapod": {"bug"},
	"mew": {"psychic"},
	"mewtwo": {"psychic"},
	"moltres": {"fire", "flying"},
	"mr-mime": {"psychic"},
	"muk": {"poison"},
	"nidoking": {"poison", "ground"},
	"nidoqueen": {"poison", "ground"},
	"nidoran-f": {"poison"},
	"nidoran-m": {"poison"},
	"nidorina": {"poison"},
	"nidorino": {"poison"},
	"ninetales": {"fire"},
	"oddish": {"grass", "poison"},
	"omanyte": {"rock", "water"},
	"omastar": {"rock", "water"},
	"onix": {"rock", "ground"},
	"paras": {"bug", "grass"},
	"parasect": {"bug", "grass"},
	"persian": {"normal"},
	"pidgeot": {"normal", "flying"},
	"pidgeotto": {"normal", "flying"},
	"pidgey": {"normal", "flying"},
	"pikachu": {"electric"},
	"pinsir": {"bug"},
	"poliwag": {"water"},
	"poliwhirl": {"water"},
	"poliwrath": {"water", "fighting"},
	"ponyta": {"fire"},
	"porygon": {"normal"},
	"primeape": {"fighting"},
	"psyduck": {"water"},
	"raichu": {"electric"},
	"rapidash": {"fire"},
	"raticate": {"normal"},
	"rattata": {"normal"},
	"rhydon": {"ground", "rock"},
	"rhyhorn": {"ground", "rock"},
	"sandshrew": {"ground"},
	"sandslash": {"ground"},
	"scyther": {"bug", "flying"},
	"seadra": {"water"},
	"seaking": {"water"},
	"seel": {"water"},
	"shellder": {"water"},
	"slowbro": {"water", "psychic"},
	"slowpoke": {"water", "psychic"},
	"snorlax": {"normal"},
	"spearow": {"normal", "flying"},
	"squirtle": {"water"},
	"starmie": {"water", "psychic"},
	"staryu": {"water"},
	"tangela": {"grass"},
	"tauros": {"normal"},
	"tentacool": {"water", "poison"},
	"tentacruel": {"water", "poison"},
	"vaporeon": {"water"},
	"venomoth": {"bug", "poison"},
	"venonat": {"bug", "poison"},
	"venusaur": {"grass", "poison"},
	"victreebel": {"grass", "poison"},
	"vileplume": {"grass", "poison"},
	"voltorb": {"electric"},
	"vulpix": {"fire"},
	"wartortle": {"water"},
	"weedle": {"bug", "poison"},
	"weepinbell": {"grass", "poison"},
	"weezing": {"poison"},
	"wigglytuff": {"normal"},
	"zapdos": {"electric", "flying"},
	"zubat": {"poison", "flying"},
}

SIZE_BAND = 0.05  # occupancy fraction; "similar size" = |occ_true - occ_pred| < this
SHAPE_IOU = 0.80  # scale-normalised IoU above which shapes count as "similar"


def occupancy_by_class(dataset, classes, scored):
	"""Per-class median silhouette occupancy over the scored (non-shiny) images."""
	occ = defaultdict(list)
	for i, (path, target) in enumerate(dataset.samples):
		if i not in scored:
			continue
		with Image.open(path) as image:
			mask = np.array(image.convert("L")) <= 127
		occ[target].append(mask.mean())
	return np.array([np.median(occ[c]) if occ[c] else np.nan for c in range(len(classes))])


def seed_agreement(seed_preds, seeds):
	"""Top-k accuracy per seed, pairwise top-1 agreement, and the per-image
	'n seeds correct' breakdown over the images all seeds scored in common."""
	common = set.intersection(*(set(p) for p in seed_preds.values()))
	common = sorted(common)
	labels = {i: seed_preds[seeds[0]][i]["label"] for i in common}

	print(f"common scored images: {len(common)}")
	print("\nper-seed accuracy:")
	print(f"  {'seed':>6} {'top-1':>7} {'top-3':>7} {'top-5':>7}")
	top1 = {}
	for s in seeds:
		p = seed_preds[s]
		t1 = np.mean([p[i]["top5"][0] == labels[i] for i in common])
		t3 = np.mean([labels[i] in p[i]["top5"][:3] for i in common])
		t5 = np.mean([labels[i] in p[i]["top5"] for i in common])
		top1[s] = {i: p[i]["top5"][0] for i in common}
		print(f"  {s:>6} {t1:>7.4f} {t3:>7.4f} {t5:>7.4f}")

	print("\npairwise top-1 agreement (fraction of images the two seeds predict alike):")
	for a in range(len(seeds)):
		for b in range(a + 1, len(seeds)):
			sa, sb = seeds[a], seeds[b]
			agree = np.mean([top1[sa][i] == top1[sb][i] for i in common])
			# Of the images where they agree, how often is the shared guess right?
			both = [i for i in common if top1[sa][i] == top1[sb][i]]
			right = np.mean([top1[sa][i] == labels[i] for i in both])
			print(f"  {sa} vs {sb}: {agree:.1%} agree; when they agree, {right:.1%} correct")

	all_agree = [i for i in common if len({top1[s][i] for s in seeds}) == 1]
	agree_right = np.mean([top1[seeds[0]][i] == labels[i] for i in all_agree])
	print(
		f"\nall three seeds predict the same class: {len(all_agree)}/{len(common)} "
		f"({len(all_agree) / len(common):.1%}); of those {agree_right:.1%} are correct"
	)

	n_correct = Counter(sum(top1[s][i] == labels[i] for s in seeds) for i in common)
	print(f"\nhow many of the {len(seeds)} seeds get each image right:")
	for k in range(len(seeds) + 1):
		c = n_correct.get(k, 0)
		print(f"  {k}/{len(seeds)} correct: {c:>4} ({c / len(common):.1%})")
	solved = len(common) - n_correct.get(0, 0)
	best_single = max(np.mean([top1[s][i] == labels[i] for i in common]) for s in seeds)
	print(
		f"solved by at least one seed: {solved}/{len(common)} ({solved / len(common):.1%})  "
		f"(vs {best_single:.1%} best single seed)"
	)
	return common, labels


def confusion_reasons(seed_preds, seeds, common, classes, similarity, occupancy):
	"""Pool top-1 errors across seeds; tag each with structural reasons and compare
	each reason's hit-rate to its chance baseline."""
	family = family_by_class(classes)
	types = [GEN1_TYPES[c] for c in classes]
	n = len(classes)
	others = n - 1

	# Per-class chance a random *other* class shares the property.
	family_size = Counter(family.values())
	fam_chance = {c: (family_size[family[c]] - 1) / others for c in range(n)}
	type_chance = {
		c: sum(1 for d in range(n) if d != c and types[c] & types[d]) / others for c in range(n)
	}
	size_chance = {
		c: (np.sum(np.abs(occupancy - occupancy[c]) < SIZE_BAND) - 1) / others for c in range(n)
	}
	shape_chance = {c: (np.sum(similarity[c] > SHAPE_IOU)) / others for c in range(n)}

	errors = [
		{"label": seed_preds[s][i]["label"], "pred": seed_preds[s][i]["top5"][0]}
		for s in seeds
		for i in common
		if seed_preds[s][i]["top5"][0] != seed_preds[s][i]["label"]
	]
	m = len(errors)

	def tally(hit, chance):
		rate = np.mean([hit(e) for e in errors])
		base = np.mean([chance[e["label"]] for e in errors])
		return rate, base

	print(f"\n=== confusion reasons (pooled top-1 errors over {len(seeds)} seeds, n={m}) ===")
	print(f"  {'reason':>22} {'rate':>7} {'chance':>7} {'lift':>6}")
	rows = [
		("same evolution family", lambda e: family[e["label"]] == family[e["pred"]], fam_chance),
		("shares a type", lambda e: bool(types[e["label"]] & types[e["pred"]]), type_chance),
		(
			f"similar size (<{SIZE_BAND:.0%} occ)",
			lambda e: abs(occupancy[e["label"]] - occupancy[e["pred"]]) < SIZE_BAND,
			size_chance,
		),
		(
			f"similar shape (IoU>{SHAPE_IOU})",
			lambda e: similarity[e["label"]][e["pred"]] > SHAPE_IOU,
			shape_chance,
		),
	]
	for name, hit, chance in rows:
		rate, base = tally(hit, chance)
		print(f"  {name:>22} {rate:>7.1%} {base:>7.1%} {rate / base:>5.1f}x")

	# Combined coverage + a mutually-exclusive priority decomposition.
	fam = lambda e: family[e["label"]] == family[e["pred"]]  # noqa: E731
	typ = lambda e: bool(types[e["label"]] & types[e["pred"]])  # noqa: E731
	shp = lambda e: similarity[e["label"]][e["pred"]] > SHAPE_IOU  # noqa: E731
	any_hit = np.mean([fam(e) or typ(e) or shp(e) for e in errors])
	print(f"\n  explained by >=1 of family/type/shape: {any_hit:.1%}")

	buckets = Counter()
	for e in errors:
		if fam(e):
			buckets["same family (evolution)"] += 1
		elif shp(e):
			buckets["near-identical shape (diff family)"] += 1
		elif typ(e):
			buckets["shares a type only"] += 1
		else:
			buckets["none of the above"] += 1
	print("\n  dominant reason per error (priority: family > shape > type):")
	for name, c in buckets.most_common():
		print(f"    {name:>36}: {c:>4} ({c / m:.1%})")

	# Size framing that doesn't depend on a band: mean occupancy gap.
	conf_gap = np.mean([abs(occupancy[e["label"]] - occupancy[e["pred"]]) for e in errors])
	rng = np.random.default_rng(0)
	rand_gap = np.mean(
		[abs(occupancy[e["label"]] - occupancy[rng.integers(n)]) for e in errors for _ in range(5)]
	)
	print(
		f"\n  mean occupancy gap: confused {conf_gap:.1%} vs random pair {rand_gap:.1%} "
		f"({rand_gap / conf_gap:.2f}x closer)"
	)

	print("\n  most frequent confusions (pooled), with reason tags:")
	pairs = Counter((e["label"], e["pred"]) for e in errors)
	for (t, p), c in pairs.most_common(15):
		tags = []
		if family[t] == family[p]:
			tags.append("family")
		if types[t] & types[p]:
			tags.append("type")
		if similarity[t][p] > SHAPE_IOU:
			tags.append("shape")
		print(
			f"    {c:>3}x  {classes[t]:>12} -> {classes[p]:<12} "
			f"IoU {similarity[t][p]:.2f}  [{', '.join(tags) or '-'}]"
		)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--runs", nargs="+", default=["p7-ref-26-s42", "p7-ref-26-s43", "p7-ref-26-s44"]
	)
	args = parser.parse_args()

	seeds = args.runs
	seed_preds = {}
	classes = None
	for s in seeds:
		oof = load_oof(s)
		classes = oof["classes"]
		seed_preds[s] = {e["index"]: e for e in oof["predictions"]}

	# Validate the type table against the dataset slugs, loudly.
	missing = set(classes) - set(GEN1_TYPES)
	if missing:
		raise SystemExit(f"type table missing slugs: {sorted(missing)}")

	print(f"seeds: {', '.join(seeds)}   classes: {len(classes)}")
	print("=" * 60)
	common, labels = seed_agreement(seed_preds, seeds)

	# Shape similarity + occupancy over the scored image set.
	dataset = load_dataset(PROJECT_ROOT / "data")
	scored = set(common)
	dataset.samples = [s for i, s in enumerate(dataset.samples) if i in scored]
	similarity = cached_class_similarity(dataset, classes)
	occupancy = occupancy_by_class(load_dataset(PROJECT_ROOT / "data"), classes, scored)

	confusion_reasons(seed_preds, seeds, common, classes, similarity, occupancy)


if __name__ == "__main__":
	main()
