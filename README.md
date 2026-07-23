# Who's That Pokémon?

Train image classifiers that recognize Pokémon from their silhouettes, in the
style of the "Who's That Pokémon?" segment from the anime.

The pipeline scrapes sprite images from [pokemondb.net](https://pokemondb.net),
converts them into binary silhouettes, and fine-tunes a torchvision ResNet-50
to classify the 151 Generation 1 Pokémon. The input is a three-channel encoding
derived from the silhouette alone (binary mask plus a signed distance
transform), so the model deploys against any silhouette, not just this
dataset's sprites.

> **Status: complete.** The held-out test split has been spent and the project
> is closed. Everything below is a record of where it landed.

## Result

**0.8325 top-1 / 0.9442 top-5** on 197 never-trained images, Wilson 95% CI
[0.774, 0.878]. The final model is a 15-member ensemble (3 seeds × 5 CV folds)
combined by averaging softmax distributions, with 6-view test-time augmentation.

| stage | top-1 | |
|---|---|---|
| classical shape-descriptor floor | ~0.285 | elliptic Fourier + Hu moments |
| frozen DINOv2 ViT-L/14 probe | 0.618 | strong transfer, still 13pt short |
| single fine-tuned ResNet-50 (OOF) | 0.7643 | the exploration reference |
| + 6-view TTA | 0.7781 | +1.38pt, p=0.0001 |
| **+ 15-model ensemble (test)** | **0.8325** | **+4.74pt over mean member** |

Top-5 at 94% is the number the guessing game actually cares about: the right
Pokémon is nearly always in the shortlist.

Two results are worth more than the headline:

- **The honest number was rebuilt from a lie.** An early 0.906 was duplicate
  leakage — shiny sprites are recolours whose silhouettes are pixel-identical
  after thresholding, so ~62% of validation was a memorization test
  ([LEAKY-EXPERIMENTS.md](LEAKY-EXPERIMENTS.md)). The clean baseline restarted
  at 0.596 and was rebuilt through data hygiene, input encoding, and schedule
  work.
- **Most ideas didn't work, and measuring that carefully was the point.** The
  gains came from four things: the LR schedule (+5.1pt), full backbone unfreeze
  (+1.68pt), TTA (+1.38pt), and ensembling (+4.74pt). Nearly everything else —
  more data, aspect-ratio cropping, stem surgery, optimizer re-tuning, a
  single-channel stem — was measured and came back null. See
  [EXPERIMENTS.md](EXPERIMENTS.md).

## Project layout

```
pokemon_training/          # Library code
  config.py                # ExperimentConfig - single source of truth for hyperparameters
  data.py                  # Dataset, silhouette channel encoding, splits, caches
  model.py                 # Backbone loading, partial freezing, stem surgery, optimizer
  train.py                 # Epoch loop, cosine+warmup scheduler, best-epoch restore
  evaluation.py            # Top-k metrics, per-image predictions, softmax distributions
  experiment.py            # Seeding and device helpers

scripts/
  data_scraping.py         # Scrapes sprites into raw_data/ (polite delays - keep them)
  data_processing.py       # Converts raw_data/ sprites into silhouettes in data/
  generate_shiny_manifest.py  # Regenerates shiny_index.json from raw sprite dimensions
  create_test_split.py     # Moves the seed-42 test split onto disk (reversible)
  training.py              # CLI entrypoint; logs every run to MLflow

  tta_selection.py         # OOF-valid test-time-augmentation view selection
  ensemble.py              # Loads fold checkpoints, averages softmax, TTA views
  final_test_evaluation.py # The one-shot test spend, with CIs and error studies

  duplicate_audit.py       # Measures dataset duplication and split leakage
  confusion_study.py       # Confusions vs. shape similarity and evolution lines
  seed_agreement_study.py  # Cross-seed prediction agreement
  shape_descriptor_baseline.py  # Classical shape-descriptor floor (~0.285)
  dinov2_probe.py          # Frozen DINOv2 feature probe (0.618)
  cnn_feature_probe.py     # Cross-fit probe of the fine-tuned CNN's own features

data/                      # Silhouette pool, one folder per Pokémon (gitignored)
test_data/                 # Held-out test split, carved onto disk (gitignored)
weights/                   # Per-fold checkpoints from --save-model (gitignored)
raw_data/                  # Raw scraped sprites (gitignored)
test_split_manifest.json   # Exactly which files were moved to test_data/ (committed)
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

### 2. Carve the test split onto disk

```bash
uv run python scripts/create_test_split.py --dry-run   # inspect first
uv run python scripts/create_test_split.py             # move 319 files
```

This moves the 197 seed-42 test images **and their 122 shiny recolours** into
`test_data/`, so the held-out set cannot leak into ensembling regardless of any
runtime flag. Every move is recorded in `test_split_manifest.json` and is
reversible with `--revert`.

Afterwards, training runs need `--test-dir test_data`; omitting it raises
rather than silently carving a second test split out of the pool.

### 3. Train

```bash
make train ARGS="--folds 5 --test-dir test_data"                # 5-fold grouped CV
make train ARGS="--folds 5 --test-dir test_data --save-model"   # + per-fold checkpoints
```

Every hyperparameter lives on `ExperimentConfig`
([pokemon_training/config.py](pokemon_training/config.py)) and is exposed
automatically as a `--flag` via [tyro](https://github.com/brentyi/tyro). A
default `--folds 5` run reproduces the reference configuration (~40 min on
Apple Silicon). `--save-model` writes each fold's best-epoch weights to
`weights/<run_name>/foldN.pt` for ensembling.

### 4. Ensemble and evaluate

```bash
uv run python scripts/tta_selection.py --runs <run> ... --seeds 42 43 44
uv run python scripts/ensemble.py --runs <run> ... --views identity hflip rot+10 rot-10
uv run python scripts/final_test_evaluation.py --runs <run> ... --views <chosen views>
```

The ordering is deliberate. TTA views are selected on out-of-fold data, where
each fold's model only ever scores images it never trained on. A *fold*
ensemble is only valid on `test_data/` — averaging fold models over OOF
predictions is leakage, since each image is in-training for K−1 of them.

### 5. Inspect and analyse

```bash
make ui                                        # MLflow at http://localhost:5001
uv run python scripts/confusion_study.py --run-name <mlflow run name>
```

Every cross-validated run logs `oof_predictions.json` — each image scored by a
model that never trained on it — which the confusion study turns into
confusion tables, shape-similarity statistics, and evolution-line confusion
rates.

## What's left standing

The remaining error is structured, and one mode survived everything:

- **Evolution-line confusions are the dominant failure**: 12.1% of the
  ensemble's errors land inside the true Pokémon's evolution family, at 13×
  chance — essentially unchanged from the single model. `raichu→pikachu`,
  `pidgeot→pidgeotto`, `kadabra→alakazam`.
- **Silhouette collisions stopped mattering.** For a single model they explained
  ~3% of errors; the ensemble has zero errors on pairs with IoU ≥ 0.9 and gets
  electrode/voltorb (the dataset's worst collision at 0.969) right.

Two findings were generated *on the spent test split* and are therefore
hypotheses, not results — acting on them would need fresh measurement: the 15
members are unanimous on 50.8% of images and **100% accurate there**, and
ensemble confidence is sharply monotonic (42.5% → 76.9% → 97.4% → 100% → 100%
across ascending bins). If the guessing game ever wants to know when to commit
versus hedge, that is where to start.

## Conventions

[CLAUDE.md](CLAUDE.md) holds the working conventions (measurement rules,
gotchas, code style); [EXPERIMENTS.md](EXPERIMENTS.md) is the experiment log.
The important ones: always `--folds 5`, compare on `fold_accuracy_sem` at 2×
combined SEM, confirm load-bearing results at a second seed, no metadata
shortcuts in preprocessing, and no ensembling during exploration.

The measurement discipline earned its keep. Single-run resolution here is
~3.4pt, so several effects that looked like clear wins at one seed (aspect crop
at +1.89pt, weight decay at +0.99pt) evaporated under a three-seed paired
battery. Nearly every "obvious" improvement in this log is a null.
