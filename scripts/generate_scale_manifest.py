"""Compute a per-generation rescale factor that removes canvas padding but keeps size.

Sprite scale in `data/` mixes two things:

  * **Artifact** - later generations draw on roomier canvases, so the creature
    fills ~45% of the frame in Gen 1 and ~13% in Gen 6. Median linear size
    ranges 2.06x across generations. This is about the sprite sheet, not the
    Pokemon.
  * **Signal** - within a generation, bigger Pokemon really are drawn bigger.
    Size increases monotonically along the evolution line in 95% of cases
    (Pidgey < Pidgeotto < Pidgeot), and the within-generation spread is 1.83x.

Those two are currently the same magnitude, so they are entangled: a Gen 6
Venusaur can occupy the same pixel area as a Gen 1 Bulbasaur. Size is therefore
unusable as a species cue unless the model first infers the generation.

This applies **one scale factor per sprite index**, so every Pokemon at a given
index is scaled identically. Between-generation differences collapse; the
within-generation size ordering is preserved exactly. That is the opposite of a
per-image bbox crop, which would normalise away the signal along with the
artifact.

    uv run python scripts/generate_scale_manifest.py
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = PROJECT_ROOT / "scale_index.json"

CANVAS = 224
# Leave a margin so the largest creature at any index cannot clip after scaling.
SAFETY = 0.95


def linear_size(path):
    """sqrt of filled pixel count - a scale measure robust to shape."""
    mask = np.array(Image.open(path).convert("L")) <= 127
    return float(np.sqrt(mask.sum()))


def main():
    classes = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())

    sizes_by_index = {}
    for index in range(16):
        sizes = [
            linear_size(DATA_DIR / name / f"image-{index}.png")
            for name in classes
            if (DATA_DIR / name / f"image-{index}.png").exists()
        ]
        if sizes:
            sizes_by_index[index] = np.array(sizes)

    # Target median size, chosen as large as possible without letting the most
    # oversized creature at any index run past the canvas.
    worst_ratio = max(s.max() / np.median(s) for s in sizes_by_index.values())
    target = CANVAS * SAFETY / worst_ratio

    manifest = {
        str(index): target / float(np.median(sizes))
        for index, sizes in sizes_by_index.items()
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"target median linear size: {target:.1f}px (canvas {CANVAS}, safety {SAFETY})")
    print(f"wrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)}\n")
    print("index  n   median  factor  -> median after")
    for index, sizes in sizes_by_index.items():
        factor = manifest[str(index)]
        median = float(np.median(sizes))
        after = median * factor
        print(f"  {index:>2}  {len(sizes):3d}  {median:6.1f}  {factor:5.2f}x  {after:6.1f}")


if __name__ == "__main__":
    main()
