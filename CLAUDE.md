# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Trains image classifiers to recognize Pokémon (Gen 1, 151 classes) from
silhouettes, for a "Who's That Pokémon?" style guessing game. Sprites are
scraped from pokemondb.net, converted to silhouettes, and used to fine-tune a
torchvision backbone.

**Current best: 0.716 out-of-fold accuracy** (single model, 5-fold grouped CV,
n=1110, two-seed mean of the config defaults — SDT input channel plus
cosine/warmup/restore-best at blr 4e-4 over 32 epochs, all confirmed). Read
[EXPERIMENTS.md](EXPERIMENTS.md) before running or interpreting any experiment
— it holds the protocol, all measured results, and the active roadmap
(see its "Next session" section: leak decomposition first, then the
data-vs-checkpoint fork it gates).

## Commands

```bash
uv sync                              # install dependencies
make train ARGS="--folds 5"          # 5-fold grouped CV — the reporting standard
make train ARGS="--epochs 30 ..."    # any ExperimentConfig field is a CLI flag
make ui                              # MLflow UI at localhost:5001
uv run ruff check .                  # lint (line-length 100, py311, tab indent)
uv run ruff format .                 # formatter (tabs, including wraps)

uv run python scripts/duplicate_audit.py           # duplication + split leakage
uv run python scripts/generate_shiny_manifest.py   # regenerate shiny_index.json
uv run python scripts/confusion_study.py           # confusions vs. similarity + evolution lines
```

There is no test suite yet.

## Architecture

- [pokemon_training/config.py](pokemon_training/config.py) — `ExperimentConfig`,
  the single source of truth for hyperparameters. Add a field to add a
  hyperparameter; it is picked up by the tyro CLI and logged to MLflow. Don't
  duplicate config values elsewhere.
- [pokemon_training/data.py](pokemon_training/data.py) — `ImageFolder` over
  `data/<pokemon>/*.png`, shiny filtering, binary threshold, augmentation,
  per-channel input encoding (`SilhouetteChannels`: mask/sdt/curv/edge),
  stratified or grouped-K-fold splits, and the decode/eval caches (both
  RNG-free, so training is byte-identical to the uncached pipeline).
- [pokemon_training/model.py](pokemon_training/model.py) — backbone loading,
  partial freezing, optimizer with split backbone/classifier LRs.
- [pokemon_training/train.py](pokemon_training/train.py) /
  [evaluation.py](pokemon_training/evaluation.py) — epoch loop, per-step
  cosine+warmup scheduler, best-epoch (val_loss) restoration, top-k metrics.
  Device: cuda → mps → cpu.
- [scripts/training.py](scripts/training.py) — entrypoint; wires config → data
  → model → MLflow. Fold runs log `oof_predictions.json`, the confusion
  study's input.
- [scripts/data_scraping.py](scripts/data_scraping.py) /
  [data_processing.py](scripts/data_processing.py) — one-off data pipeline.
  Scraping sleeps between requests to pokemondb.net — don't remove the delays.

## The duplicate leak

Shiny sprites are recolours; after thresholding their silhouettes duplicate the
normal series, which once made ~62% of validation a memorization test.
`exclude_shiny=True` (default) drops them via `shiny_index.json` — the
per-class boundary is not a constant and only derivable from raw sprite
dimensions, hence the precomputed manifest.
[LEAKY-EXPERIMENTS.md](LEAKY-EXPERIMENTS.md) is the tombstone of that era;
don't quote it.

## Conventions and gotchas

- **`--folds 5` on every experiment; omitting it fails silently** (single
  split, `validation_accuracy`, incomparable). Compare on `fold_accuracy_sem`
  at 2× combined SEM; confirm load-bearing results at a second seed.
- **`val_size`/`test_size` must stay ≥ 0.15** — below that a stratified split
  over 151 classes raises.
- **No metadata shortcuts.** Preprocessing may only use information present in
  the image itself; sprite index, generation, and source resolution are dataset
  properties. A generation-based scale normalisation was built and removed.
- **No ensembling during exploration**; it belongs to the final phase, and
  ensembling K fold models on out-of-fold data is leakage (see EXPERIMENTS.md
  Phase 6).
- Two MLflow experiments, never mixed: `pokemon-classification-clean`
  (current) and `pokemon-classification` (leaky era) — same metric names,
  different meanings.
- MLflow tracking URI is pinned to an absolute `sqlite:///<repo>/mlflow.db` in
  both `Makefile` and `scripts/training.py`; keep new entrypoints consistent.
- MLflow run names are not unique; filter on `status == "FINISHED"`.
- MLflow UI runs on port 5001 (5000 is squatted by macOS AirPlay Receiver).
- `data/` and `raw_data/` are gitignored; never add images to git.
- Silhouette conversion handles grayscale/RGB/RGBA with separate masking
  branches in `scripts/data_processing.py`; `data/bulbasaur/` contains two
  excluded strays and should be regenerated.
- Indentation is tabs, enforced by `ruff format` (`indent-style = "tab"`) and
  flagged by E101 on mixed lines.
