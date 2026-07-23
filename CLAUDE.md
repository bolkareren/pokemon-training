# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Trains image classifiers to recognize Pokémon (Gen 1, 151 classes) from
silhouettes, for a "Who's That Pokémon?" style guessing game. Sprites are
scraped from pokemondb.net, converted to silhouettes, and used to fine-tune a
torchvision backbone.

**Current reference: 0.7586 out-of-fold accuracy** (single model, 5-fold grouped
CV, n=1110, config defaults at seed 42 on index-grouped folds; 3-seed spread
0.7586/0.7676/0.7550, mean 0.7604 — `p10-lastn6-s*`). The defaults are the SDT
input channel, cosine/warmup/restore-best at blr 4e-4 over 26 epochs, and
**lastN 6 (full feature unfreeze)** — depth was the win of 2026-07-22 (+1.68pt
over the old lastN-3 reference of 0.7496; the "depth plateau" was a distorted-fold
artifact). Read [EXPERIMENTS.md](EXPERIMENTS.md) before running or interpreting
any experiment — it holds the protocol, all measured results, and the active
roadmap. The full-unfreeze optimizer follow-ups are now both closed as null (BN
affine `p11-bnaffine-s*`; the `backbone_lr`/`weight_decay` re-tune `p12-lrwd-*` —
LR settled at 4e-4, wd null at 3 seeds, defaults stand). Next up (its "Next
session" section): the tail of the **input-encoding re-exploration**. Its sharpest
question is already closed negative — the single-channel stem loses (mono-mask
−1.17pt, mono-sdt −3.18pt, `p13-mono-*`), so a trainable stem does *not* learn to
replace the hand-designed channels and `(mask, sdt, mask)` is load-bearing. What
remains (edge redux, channel-position/dup-mask, re-confirming SDT) is now
low-prior checklist-closing, not an expected win.

## Workflow

- **Solo project: commit and push straight to `main`.** No branch/PR ceremony is
  required — push directly unless explicitly asked to open a PR. (This overrides
  the default "branch before committing on main" behaviour.)
- **Ask before implementing anything with several defensible designs** (selection
  criterion, budget semantics, scheduler, stem surgery, descriptor choices):
  present the concrete forks as a short `AskUserQuestion` (2–4 options, each with
  a recommendation and honest trade-offs) and wait. The maintainer treats design
  decisions as their call and engages with the trade-offs. Flags-only experiment
  runs need no questions.
- **Pre-register the decision rule before launching runs** — state how the result
  will be judged (the 2× SEM bar; paired t-test + McNemar for paired batteries)
  up front, and confirm any load-bearing result or default change at a second
  seed. Single-run resolution is ~3.4pt, so any effect under ~3pt needs a
  multi-seed paired battery, not one run per arm, or it reads as a null
  regardless of truth.

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
uv run python scripts/shape_descriptor_baseline.py --keep-orientation  # classical shape floor
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
- [scripts/shape_descriptor_baseline.py](scripts/shape_descriptor_baseline.py) —
  classical shape-descriptor floor (elliptic Fourier + Hu moments + shape ratios
  → shallow classifier). Reuses `data.py`'s split functions so its OOF set is
  byte-identical to `p7-ref-26-s<seed>` and its `oof_predictions.json` is
  directly comparable. Floor is ~0.285 OOF vs the CNN's 0.7496.

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
