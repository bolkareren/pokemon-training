# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Trains image classifiers to recognize Pokémon (Gen 1, 151 classes) from
silhouettes, for a "Who's That Pokémon?" style guessing game. Sprites are
scraped from pokemondb.net, converted to silhouettes, and used to fine-tune a
torchvision backbone (ResNet-18/34).

## Commands

```bash
uv sync                              # install dependencies
make train                           # run training with defaults
make train ARGS="--epochs 30 ..."    # override any ExperimentConfig field as a CLI flag
make ui                              # serve MLflow UI at localhost:5001
uv run ruff check .                  # lint (line-length 100, py311 target)
```

There is no test suite yet.

## Architecture

- [pokemon_training/config.py](pokemon_training/config.py) — `ExperimentConfig`
  is the single source of truth for hyperparameters. Add a field here to add a
  hyperparameter; it's automatically picked up by the tyro CLI in
  `scripts/training.py` and logged as an MLflow param. Don't duplicate config
  values elsewhere.
- [pokemon_training/data.py](pokemon_training/data.py) — builds an
  `ImageFolder` dataset over `data/<pokemon>/*.png`, applies a binary
  silhouette threshold (`x > 0.5`) plus augmentation, and does a stratified
  train/val/test split by index.
- [pokemon_training/model.py](pokemon_training/model.py) — loads a
  torchvision backbone, freezes all but the last N feature layers plus the
  classifier head, and builds an optimizer with separate backbone/classifier
  learning rates.
- [pokemon_training/train.py](pokemon_training/train.py) /
  [evaluation.py](pokemon_training/evaluation.py) — epoch loop and top-k
  accuracy. Device resolution always prefers cuda, then mps, then cpu.
- [scripts/training.py](scripts/training.py) — entrypoint. Wires config →
  data loaders → model → optimizer → MLflow run.
- [scripts/data_scraping.py](scripts/data_scraping.py) /
  [data_processing.py](scripts/data_processing.py) — one-off, run-manually
  data pipeline (not part of the package). Scraping is polite by design
  (`time.sleep(2)` between requests to pokemondb.net) — don't remove the
  delays.

## Conventions and gotchas

- MLflow tracking URI is pinned to an absolute `sqlite:///<repo>/mlflow.db` in
  both `Makefile` and `scripts/training.py`, so `make train` and `make ui`
  always agree regardless of cwd. Keep any new MLflow entrypoint consistent
  with this rather than relying on MLflow's default store resolution.
- MLflow UI runs on port 5001, not the default 5000 — 5000 is squatted by
  macOS AirPlay Receiver.
- `data/` and `raw_data/` are gitignored (large binary image sets). Do not add
  images to git; they're regenerated via the scripts above.
- Silhouette conversion assumes a white or transparent background per
  Pokémon sprite; grayscale, RGB, and RGBA images are each handled with a
  different masking branch in `scripts/data_processing.py`.
