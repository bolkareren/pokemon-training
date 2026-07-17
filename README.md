# Who's That Pokémon?

Train image classifiers that recognize Pokémon from their silhouettes, in the
style of the "Who's That Pokémon?" segment from the anime.

The pipeline scrapes sprite images from [pokemondb.net](https://pokemondb.net),
converts them into black silhouettes on a transparent/white background, and
fine-tunes a torchvision backbone (ResNet-18/34) to classify them. Currently
covers the 151 Generation 1 Pokémon.

## Project layout

```
pokemon_training/       # Library code
  config.py             # ExperimentConfig - single source of truth for hyperparameters
  data.py                # Dataset loading, silhouette transforms, train/val/test split
  model.py               # Backbone loading, freezing/unfreezing layers, optimizer setup
  train.py               # Training loop
  evaluation.py           # Top-k accuracy
  experiment.py           # Seeding and device resolution helpers

scripts/
  data_scraping.py        # Downloads sprites from pokemondb.net into dataset/
  data_processing.py      # Resizes sprites and converts them to silhouettes into data/
  training.py              # CLI entrypoint: builds a run from ExperimentConfig and logs to MLflow

data/                    # Processed silhouette dataset, one folder per Pokémon (gitignored)
raw_data/                 # Raw scraped sprites (gitignored)
```

## Setup

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Usage

### 1. Collect and process data

```bash
uv run python scripts/data_scraping.py     # scrapes dataset/<pokemon>/image-*.png
uv run python scripts/data_processing.py   # writes silhouettes to data/<pokemon>/
```

### 2. Train

```bash
make train
# or with overrides, e.g.:
make train ARGS="--epochs 30 --backbone-lr 1e-4"
```

Every hyperparameter lives on `ExperimentConfig`
([pokemon_training/config.py](pokemon_training/config.py)) and is exposed
automatically as a `--flag` via [tyro](https://github.com/brentyi/tyro).

### 3. Inspect runs

Training runs are logged to a local MLflow SQLite store (`mlflow.db`).

```bash
make ui
```

Then open http://localhost:5001.
