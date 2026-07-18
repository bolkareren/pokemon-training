# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Trains image classifiers to recognize Pokémon (Gen 1, 151 classes) from
silhouettes, for a "Who's That Pokémon?" style guessing game. Sprites are
scraped from pokemondb.net, converted to silhouettes, and used to fine-tune a
torchvision backbone (ResNet-18/34/50).

**Current state: 0.653 out-of-fold accuracy**, single model under 5-fold grouped
CV over 1110 images. Read [EXPERIMENTS.md](EXPERIMENTS.md) before running or
interpreting any experiment — particularly "Results so far" and "Noise floor".
The active plan resumes at **Phase N1**.

## Commands

```bash
uv sync                              # install dependencies
make train                           # single split (folds=0)
make train ARGS="--folds 5"          # 5-fold grouped CV — the reporting standard
make train ARGS="--epochs 30 ..."    # override any ExperimentConfig field as a CLI flag
make ui                              # serve MLflow UI at localhost:5001
uv run ruff check .                  # lint (line-length 100, py311 target)

uv run python scripts/duplicate_audit.py           # dataset duplication + split leakage
uv run python scripts/generate_shiny_manifest.py   # regenerate shiny_index.json (needs raw_data/)
uv run python scripts/confusion_study.py           # confusions vs. silhouette similarity
```

There is no test suite yet.

## Architecture

- [pokemon_training/config.py](pokemon_training/config.py) — `ExperimentConfig`
  is the single source of truth for hyperparameters. Add a field here to add a
  hyperparameter; it's automatically picked up by the tyro CLI in
  `scripts/training.py` and logged as an MLflow param. Don't duplicate config
  values elsewhere.
- [pokemon_training/data.py](pokemon_training/data.py) — builds an
  `ImageFolder` dataset over `data/<pokemon>/*.png`, filters shiny sprites,
  applies a binary silhouette threshold (`x > 0.5`) plus augmentation, and
  produces either a stratified train/val/test split or grouped K-fold splits.
- [pokemon_training/model.py](pokemon_training/model.py) — loads a
  torchvision backbone, freezes all but the last N feature layers plus the
  classifier head, and builds an optimizer with separate backbone/classifier
  learning rates.
- [pokemon_training/train.py](pokemon_training/train.py) /
  [evaluation.py](pokemon_training/evaluation.py) — epoch loop, top-k accuracy,
  and `predict_top_k` for per-image predictions. Device resolution always
  prefers cuda, then mps, then cpu.
- [scripts/training.py](scripts/training.py) — entrypoint. Wires config →
  data loaders → model → optimizer → MLflow run, in either single-split or
  cross-validation mode. Every fold run logs `oof_predictions.json` — each image
  scored by a model that never trained on it — which is the input to the
  confusion study.
- [scripts/data_scraping.py](scripts/data_scraping.py) /
  [data_processing.py](scripts/data_processing.py) — one-off, run-manually
  data pipeline (not part of the package). Scraping is polite by design
  (`time.sleep(2)` between requests to pokemondb.net) — don't remove the
  delays.

## The duplicate leak — read this before trusting any old number

Each Pokémon's pokemondb page serves a **shiny** sprite alongside the normal one
for every generation. Shiny is a recolour, so after the binary threshold the two
silhouettes are usually pixel-identical. 32.7% of the raw dataset was exact
duplicates, and a random split put a pixel-identical twin in train for **~62% of
every validation set**. The configuration that validated at 0.906 scores **0.617**
once deduplicated.

- `exclude_shiny=True` (default) drops the shiny series using `shiny_index.json`.
  The boundary is **not** a constant — Gen 1 had no shiny sprites, so it is index
  8 for 52 Pokémon and 9 for the other 99. It is only derivable from raw sprite
  dimensions, hence the precomputed manifest.
- Everything in [LEAKY-EXPERIMENTS.md](LEAKY-EXPERIMENTS.md) was measured before
  this fix. Its numbers are not comparable to anything current and several of its
  conclusions are known to be backwards; it carries a "what this file got wrong"
  table. Don't quote it.

## Conventions and gotchas

- **`--folds 5` is the reporting standard, and omitting it fails silently.**
  `folds` defaults to 0, which runs a single split and logs `validation_accuracy`
  instead of `oof_accuracy`. The run still succeeds, so a forgotten flag produces
  a plausible number that isn't comparable to anything.
- **Compare on `fold_accuracy_sem`, not single folds.** The binomial floor at
  n=1110 is ±1.5pt; treat a difference as real only at ~2× SEM. Quoting a noise
  floor 3× too small is what turned ~40 runs of the previous investigation into a
  false narrative.
- **`val_size`/`test_size` must stay ≥ 0.15.** At 1307 images a 10% split is 131,
  fewer than the 151 classes, and the stratified split raises.
- **Don't reintroduce metadata shortcuts.** Preprocessing may only use
  information present in the image itself. Sprite index, generation, and source
  resolution are properties of the dataset, not of an arbitrary input silhouette.
  A per-generation scale normalisation was built and then deliberately removed
  for this reason — see EXPERIMENTS.md "The scale finding".
- **Don't ensemble during exploration.** Every experiment is a single model under
  5-fold grouped CV. Ensembling triples the cost of each cheap single-factor test
  and judges every later architecture as an ensemble on both accuracy and
  compute, which is not how any of them would ship. It belongs in the final phase
  only (D5). If a comparison is too close to call, use repeated CV or a second
  `random_state` — that measures the same model better, rather than changing it.
- **When the time does come: ensembling the K fold models on out-of-fold data is
  leakage.** Each image is out-of-fold for exactly one model and in-training for
  the other four. Use a seed ensemble within each fold, or the held-out test
  split. See Phase D5.
- Two MLflow experiments, deliberately not mixed: `pokemon-classification-clean`
  (current, the `ExperimentConfig` default) and `pokemon-classification` (the
  leaky-split runs). They use the same metric names for different things.
- MLflow tracking URI is pinned to an absolute `sqlite:///<repo>/mlflow.db` in
  both `Makefile` and `scripts/training.py`, so `make train` and `make ui`
  always agree regardless of cwd. Keep any new MLflow entrypoint consistent
  with this rather than relying on MLflow's default store resolution.
- MLflow run names are **not unique** — an interrupted run leaves a `RUNNING`
  record that a later rerun does not replace. Filter on `status == "FINISHED"`
  when looking runs up by name.
- MLflow UI runs on port 5001, not the default 5000 — 5000 is squatted by
  macOS AirPlay Receiver.
- `data/` and `raw_data/` are gitignored (large binary image sets). Do not add
  images to git; they're regenerated via the scripts above.
- Silhouette conversion assumes a white or transparent background per
  Pokémon sprite; grayscale, RGB, and RGBA images are each handled with a
  different masking branch in `scripts/data_processing.py`.
- `data/bulbasaur/` contains two strays from an older processing convention
  (`silhouette-image-*.png`, one a byte-identical copy of `image-1.png`). They're
  excluded by a canonical-filename match, but the class should be regenerated.

## Known properties of the data

Measured, and load-bearing for the current plan:

- **1307 images** after shiny filtering, ~8.7 per class over 151 classes. 56
  near-duplicate clusters remain within the normal series; grouped folds keep
  them from straddling a split.
- **Sprite scale mixes an artifact with a signal.** Later generations use roomier
  canvases (creature fills ~45% of the frame in Gen 1, ~13% in Gen 6 — a 2.06×
  range), while within a generation bigger Pokémon really are drawn bigger (1.83×
  spread; size rises along the evolution line in 95% of cases). Same magnitude,
  so they cancel.
- **The always-on `RandomAffine` includes `scale=(0.85, 1.15)`**, which randomises
  away ~74% of that size range — i.e. actively trains away the cue that separates
  Pidgey from Pidgeot. Phase N1 tests removing it.
- **The 3 input channels hold identical copies of the same binary mask**, so two
  thirds of input capacity is redundant. Phase N2 fills it.
- **Silhouette collisions are rare**: 3 of 11,325 class pairs exceed IoU 0.90
  (electrode/voltorb is the most similar at 0.969). They explain ~3% of errors,
  so the ceiling is not the binding constraint.
- **Models are high-variance.** Five configs scoring ~725/1110 each agree on only
  75-82% of predictions; 880 images are solved by at least one, 230 by none.
