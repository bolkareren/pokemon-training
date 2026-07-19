"""Record, per Pokemon, the image index at which the shiny sprite series begins.

pokemondb.net serves each Pokemon's *normal* sprite series (one per generation,
ascending in size 56x56 -> 128x128) followed by the *shiny* series repeating the
same generations from 56x56 again. Shiny differs only in palette, so after the
binary silhouette threshold the two series are near-identical - see
`scripts/duplicate_audit.py`.

The boundary is detectable from raw sprite dimensions but *not* from `data/`,
whose images are all resized to 224x224. This script writes the boundary out as
a manifest so training never needs `raw_data/` (which is gitignored and only
regenerated on demand).

    uv run python scripts/generate_shiny_manifest.py

The boundary is not a constant: Gen 1 had no shiny sprites, so the normal series
runs one longer than the shiny series, and classes hold differing numbers of
sprites. It lands at index 8 for 52 Pokemon and 9 for the other 99.
"""

import json
import re
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
MANIFEST_PATH = PROJECT_ROOT / "shiny_index.json"


def image_index(path):
	return int(re.search(r"image-(\d+)", path.name).group(1))


def find_shiny_start(class_dir):
	"""First index where sprite size drops back to the minimum: the series restart.

	Returns the image count when no restart is found, i.e. "no shiny series here",
	so callers can treat the result uniformly as "keep everything below this".
	"""
	paths = sorted(class_dir.glob("*.png"), key=image_index)
	areas = [Image.open(p).size[0] * Image.open(p).size[1] for p in paths]
	smallest = min(areas)

	for i in range(1, len(areas)):
		if areas[i] == smallest and areas[i - 1] > smallest:
			return i

	return len(areas)


def main():
	if not RAW_DATA_DIR.is_dir():
		raise SystemExit(
			f"{RAW_DATA_DIR} not found - rerun scripts/data_scraping.py first. "
			"Sprite dimensions are only recoverable from the raw sprites."
		)

	manifest = {
		class_dir.name: find_shiny_start(class_dir)
		for class_dir in sorted(RAW_DATA_DIR.iterdir())
		if class_dir.is_dir()
	}

	MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

	total = sum(len(list((RAW_DATA_DIR / name).glob("*.png"))) for name in manifest)
	kept = sum(manifest.values())
	print(f"wrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)} for {len(manifest)} classes")
	print(f"normal sprites kept: {kept} / {total} ({kept / total:.1%})")


if __name__ == "__main__":
	main()
