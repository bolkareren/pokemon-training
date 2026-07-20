# Who's That Pokémon?

Train image classifiers that recognize Pokémon from their silhouettes, in the
style of the "Who's That Pokémon?" segment from the anime.

The pipeline scrapes sprite images from [pokemondb.net](https://pokemondb.net),
converts them into binary silhouettes, and fine-tunes a torchvision ResNet-50
to classify the 151 Generation 1 Pokémon. The input is a three-channel encoding
derived from the silhouette alone (binary mask plus a signed distance
transform), so the model deploys against any silhouette, not just this
dataset's sprites.

**Current best: 0.716 top-1 out-of-fold accuracy** (0.875 top-5), measured as a
single model under 5-fold grouped cross-validation over 1,110 deduplicated
images — see [EXPERIMENTS.md](EXPERIMENTS.md) for the measurement protocol,
every result, and the active roadmap. The headline lesson of the project: an
early 0.906 turned out to be a duplicate-leakage artifact
([LEAKY-EXPERIMENTS.md](LEAKY-EXPERIMENTS.md)), and the honest number was
rebuilt from 0.596 through data hygiene, input encoding, and training-schedule
work.

## Project layout

```
pokemon_training/          # Library code
  config.py                # ExperimentConfig - single source of truth for hyperparameters
  data.py                  # Dataset, silhouette channel encoding, splits, caches
  model.py                 # Backbone loading, partial freezing, optimizer
  train.py                 # Epoch loop, cosine+warmup scheduler, best-epoch restore
  evaluation.py            # Top-k metrics and per-image predictions
  experiment.py            # Seeding and device helpers

scripts/
  data_scraping.py         # Scrapes sprites into raw_data/ (polite delays - keep them)
  data_processing.py       # Converts raw_data/ sprites into silhouettes in data/
  generate_shiny_manifest.py  # Regenerates shiny_index.json from raw sprite dimensions
  duplicate_audit.py       # Measures dataset duplication and split leakage
  confusion_study.py       # Confusions vs. shape similarity and evolution lines
  training.py              # CLI entrypoint; logs every run to MLflow

data/                      # Silhouette dataset, one folder per Pokémon (gitignored)
raw_data/                  # Raw scraped sprites (gitignored)
```

## Setup

Requires Python ≥ 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Usage

### 1. Collect and process data

```bash
uv run python scripts/data_scraping.py       # sprites -> raw_data/<pokemon>/
uv run python scripts/data_processing.py     # silhouettes -> data/<pokemon>/
uv run python scripts/generate_shiny_manifest.py   # shiny_index.json
```

The manifest matters: each Pokémon's page also serves *shiny* sprites, which
are recolours whose silhouettes duplicate the normal ones. Training excludes
them by default — leaving them in silently turns validation into a
memorization test.

### 2. Train

```bash
make train ARGS="--folds 5"     # the reporting standard: 5-fold grouped CV
make train ARGS="--folds 5 --epochs 16 --backbone-lr 2e-4"   # any config field is a flag
```

Every hyperparameter lives on `ExperimentConfig`
([pokemon_training/config.py](pokemon_training/config.py)) and is exposed
automatically as a `--flag` via [tyro](https://github.com/brentyi/tyro). A
no-flag `--folds 5` run reproduces the current best configuration (~40 min on
Apple Silicon).

### 3. Inspect and analyse

```bash
make ui                                        # MLflow at http://localhost:5001
uv run python scripts/confusion_study.py --run-name <mlflow run name>
```

Every cross-validated run logs `oof_predictions.json` — each image scored by a
model that never trained on it — which the confusion study turns into
confusion tables, shape-similarity statistics, and evolution-line confusion
rates.

## Contributing / conventions

[CLAUDE.md](CLAUDE.md) holds the working conventions (measurement rules,
gotchas, code style); [EXPERIMENTS.md](EXPERIMENTS.md) is the experiment log
and roadmap. The important ones: always `--folds 5`, compare on
`fold_accuracy_sem` at 2× combined SEM, confirm load-bearing results at a
second seed, no metadata shortcuts in preprocessing, and no ensembling during
exploration.
