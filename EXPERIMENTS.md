# Experiment reproduction log

This is a curated replay of the pre-MLflow experiments that were tracked under
the (now-deleted) `runs/` directory. Each run there was a `result.json` with a
config, a loss history, and a validation accuracy. There were **50 runs across
40 distinct configs**, in two sessions:

- **2026-05-16** — initial exploration: how deep to fine-tune, first pass at
  regularization and batch-norm handling. Single learning rate; no top-3/top-5.
- **2026-05-22** — refinement toward the config that is now the `ExperimentConfig`
  default: differential backbone/classifier LR, stronger label smoothing,
  epoch/depth sweeps.

The goal of rerunning these is twofold: **reproduce** the results under the new
MLflow-based pipeline, and **record the progression** from baseline to the final
config as a trail of MLflow runs. Check each box as you rerun it and confirm the
new accuracy lands near the historical one.

## How to read this

- Each item is a `make train ARGS="…"` command with a `--run-name` so it is easy
  to find in the MLflow UI (`make ui` → http://localhost:5001).
- **hist acc** is the original `validation_accuracy` from `runs/`. Match within a
  few points, not exactly — see caveats below.
- Anything not passed as a flag keeps the `ExperimentConfig` default
  (`resnet18`, ImageNet weights, `batch_size=16`, `AdamW`, stratified 80/10/10
  split at `random_state=42`).

### Reproducibility caveats

- The old runs read from `…/Playground/pokemon_training/data`. If the current
  `data/` has a different image count per class, the deterministic split shifts
  and accuracies won't be bit-identical.
- Run-to-run variance is real: the *final* config was run 4 times originally and
  scored **0.839–0.903**. Treat a ~5-point band as "reproduced."
- Old (2026-05-16) runs logged only top-1; the new pipeline also logs top-3/top-5
  for free, so reruns are strictly more informative.

---

## Phase 1 — How deep to fine-tune (baseline: single LR, no regularization)

Backbone and classifier both at `1e-3`, `weight_decay=1e-4`, no label smoothing,
15 epochs. Sweeps `--train-last-n-layers`. This is the foundational ablation:
too few unfrozen layers underfits, too many overfits/destabilizes.

- [x] **lastN=0** (head only) — hist acc **0.544** — floor; classifier alone is not enough
  ```bash
  make train ARGS="--run-name p1-lastN0 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.0 --epochs 15 --train-last-n-layers 0"
  ```
- [x] **lastN=1** — hist acc **0.843**
  ```bash
  make train ARGS="--run-name p1-lastN1 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.0 --epochs 15 --train-last-n-layers 1"
  ```
- [x] **lastN=2** — hist acc **0.880** (best of the baseline sweep)
  ```bash
  make train ARGS="--run-name p1-lastN2 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.0 --epochs 15 --train-last-n-layers 2"
  ```
- [x] **lastN=3** — hist acc **0.853**
  ```bash
  make train ARGS="--run-name p1-lastN3 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.0 --epochs 15 --train-last-n-layers 3"
  ```
- [x] **lastN=4** — hist acc **0.876**
  ```bash
  make train ARGS="--run-name p1-lastN4 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.0 --epochs 15 --train-last-n-layers 4"
  ```
- [x] **lastN=5** — hist acc **0.857**
  ```bash
  make train ARGS="--run-name p1-lastN5 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.0 --epochs 15 --train-last-n-layers 5"
  ```
- [x] **lastN=6** — hist acc **0.816** (too much unfrozen)
  ```bash
  make train ARGS="--run-name p1-lastN6 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.0 --epochs 15 --train-last-n-layers 6"
  ```

## Phase 2 — Add regularization (label smoothing + weight decay)

Fix `lastN=5`, single LR `1e-3`. Adds label smoothing, then bumps weight decay
`1e-4 → 1e-3`.

- [x] **ls=0.1, wd=1e-4** — hist acc **0.876**
  ```bash
  make train ARGS="--run-name p2-ls0.1-wd1e-4 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.1 --epochs 15 --train-last-n-layers 5"
  ```
- [x] **ls=0.1, wd=1e-3** — hist acc **0.885** (stronger weight decay helps)
  ```bash
  make train ARGS="--run-name p2-ls0.1-wd1e-3 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-3 --label-smoothing 0.1 --epochs 15 --train-last-n-layers 5"
  ```

## Phase 3 — Differential learning rate (slower backbone)

Drop the backbone LR to `1e-4` while keeping the classifier at `1e-3`.

- [x] **blr=1e-4, clr=1e-3** — hist acc **0.889** (best of session 1)
  ```bash
  make train ARGS="--run-name p3-diff-lr --backbone-lr 1e-4 --classifier-lr 1e-3 --weight-decay 1e-3 --label-smoothing 0.1 --epochs 15 --train-last-n-layers 5"
  ```

## Phase 4 — Batch-norm handling ablation

Confirms the current default (BN in **train** mode, affine params **frozen**) is
right.

- [x] **BN in eval mode** — hist acc **0.820** (clearly worse; don't freeze BN stats)
  ```bash
  make train ARGS="--run-name p4-bn-eval --batch-norm-mode eval --backbone-lr 1e-4 --classifier-lr 1e-3 --weight-decay 1e-3 --label-smoothing 0.1 --epochs 15 --train-last-n-layers 5"
  ```
- [x] **BN affine trainable** (lastN=3) — hist acc **0.885** (no clear gain from unfreezing affine)
  ```bash
  make train ARGS="--run-name p4-bn-affine --train-batch-norm-affine --backbone-lr 1e-4 --classifier-lr 1e-3 --weight-decay 1e-3 --label-smoothing 0.1 --epochs 15 --train-last-n-layers 3"
  ```

## Phase 5 — Session 2: stronger smoothing, LR and depth around the optimum

Label smoothing to `0.2`; find the backbone-LR sweet spot and re-check depth.

- [x] **blr=1e-4, ls=0.2** — hist acc **0.894**
  ```bash
  make train ARGS="--run-name p5-blr1e-4-ls0.2 --backbone-lr 1e-4 --classifier-lr 1e-3 --weight-decay 1e-3 --label-smoothing 0.2 --epochs 15 --train-last-n-layers 5"
  ```
- [x] **blr=5e-4, ls=0.2** — hist acc **0.903** (top result; `5e-4` beats both `1e-4` and `1e-3`)
  ```bash
  make train ARGS="--run-name p5-blr5e-4-ls0.2 --backbone-lr 5e-4 --classifier-lr 1e-3 --weight-decay 1e-3 --label-smoothing 0.2 --epochs 15 --train-last-n-layers 5"
  ```
- [x] **wd=3e-3** (weight-decay ablation) — hist acc **0.889** (more decay doesn't help)
  ```bash
  make train ARGS="--run-name p5-wd3e-3 --backbone-lr 1e-4 --classifier-lr 1e-3 --weight-decay 3e-3 --label-smoothing 0.2 --epochs 15 --train-last-n-layers 5"
  ```
- [x] **lastN=4, ep=18** — hist acc **0.903** (depth robustness near the optimum)
  ```bash
  make train ARGS="--run-name p5-lastN4-ep18 --backbone-lr 5e-4 --classifier-lr 1e-3 --weight-decay 1e-3 --label-smoothing 0.2 --epochs 18 --train-last-n-layers 4"
  ```
- [x] **lastN=6, ep=18** — hist acc **0.880** (deeper again regresses)
  ```bash
  make train ARGS="--run-name p5-lastN6-ep18 --backbone-lr 5e-4 --classifier-lr 1e-3 --weight-decay 1e-3 --label-smoothing 0.2 --epochs 18 --train-last-n-layers 6"
  ```

## Phase 6 — Final configuration (current `ExperimentConfig` default)

`blr=5e-4, clr=1e-3, wd=1e-3, ls=0.2, lastN=5, epochs=18, BN train/no-affine`.
This is exactly the default, so it runs with **no args**. Historically run 4×,
scoring **0.839 / 0.866 / 0.903 / 0.903** (top-3 ≈ 0.94, top-5 ≈ 0.95).

- [x] **Final config** — hist acc **0.90** (best), ~0.86 typical
  ```bash
  make train ARGS="--run-name final-default"
  ```

## Phase 7 — ResNet34 capacity check

All 50 historical runs and the phase 1-6 reruns above used `resnet18`. Before
jumping to `resnet50` (needed anyway for shape-biased/Stylized-ImageNet
weights), check whether the extra capacity of `resnet34` helps or overfits at
this dataset size (~14 images/class, 2162 total).

Two groups:
- **Baseline** — the exact current `ExperimentConfig` defaults
  (`lastN=2, blr=3e-4, wd=1e-3, ls=0.15, epochs=16`), just swapping the
  backbone to `resnet34`. Isolates the architecture change with nothing else
  varying.
- **Regularization sweep** — slightly stronger regularization
  (`wd=2e-3, ls=0.2`, backbone LR pulled back to `2e-4`) crossed with
  `train_last_n_layers` in `{1,2,3,4}`. Capped at 4, not 5-6 like the resnet18
  sweep, since resnet18 already showed diminishing/negative returns unfreezing
  more than ~2-3 layers - no reason to expect resnet34 tolerates more.

No hist acc column here - these are new, not reproductions.

- [x] **r34-baseline-default** (current defaults, resnet34)
  ```bash
  make train ARGS="--run-name r34-baseline-default --model-name resnet34"
  ```
- [x] **r34-reg-lastN1** (stronger reg, lastN=1)
  ```bash
  make train ARGS="--run-name r34-reg-lastN1 --model-name resnet34 --backbone-lr 2e-4 --classifier-lr 1e-3 --weight-decay 2e-3 --label-smoothing 0.2 --epochs 16 --train-last-n-layers 1"
  ```
- [x] **r34-reg-lastN2** (stronger reg, lastN=2)
  ```bash
  make train ARGS="--run-name r34-reg-lastN2 --model-name resnet34 --backbone-lr 2e-4 --classifier-lr 1e-3 --weight-decay 2e-3 --label-smoothing 0.2 --epochs 16 --train-last-n-layers 2"
  ```
- [x] **r34-reg-lastN3** (stronger reg, lastN=3)
  ```bash
  make train ARGS="--run-name r34-reg-lastN3 --model-name resnet34 --backbone-lr 2e-4 --classifier-lr 1e-3 --weight-decay 2e-3 --label-smoothing 0.2 --epochs 16 --train-last-n-layers 3"
  ```
- [x] **r34-reg-lastN4** (stronger reg, lastN=4)
  ```bash
  make train ARGS="--run-name r34-reg-lastN4 --model-name resnet34 --backbone-lr 2e-4 --classifier-lr 1e-3 --weight-decay 2e-3 --label-smoothing 0.2 --epochs 16 --train-last-n-layers 4"
  ```
- [x] **r18-baseline-default** (resnet18 counterpart to `r34-baseline-default`,
  added after the sweep to get a true apples-to-apples train/val-gap
  comparison at the exact same defaults - resnet18 is already the default
  `model_name`, so no flags needed beyond the run name)
  ```bash
  make train ARGS="--run-name r18-baseline-default"
  ```

**Train/val gap comparison** (final-epoch `val_loss - train_loss`; both use
identical defaults except backbone):

| run | trainable % | final gap | best-epoch gap | val acc |
|---|---|---|---|---|
| r18-baseline-default | 93.9% | 0.407 | 0.338 | 0.880 |
| r34-baseline-default | 93.6% | 0.447 | 0.357 | 0.876 |

At matched trainable-parameter fraction, resnet34 shows a ~10% larger
train/val gap and a lower validation accuracy than resnet18, despite having
~2x the parameters - the extra capacity buys overfitting, not accuracy, on
this dataset size. See conversation for the full breakdown including the
regularized resnet34 depth sweep.

## Phase 8 — ResNet50 with shape-biased weights (smoke test only)

Silhouettes have zero texture (binary threshold in
[data.py](pokemon_training/data.py)), so a backbone whose early filters
already emphasize contour/shape over texture is a plausible better fit than
standard ImageNet weights. Wired up `resnet50` plus a `weights_checkpoint`
config field that loads a raw state-dict checkpoint instead of a torchvision
weights enum (see [model.py](pokemon_training/model.py)'s
`load_checkpoint_weights`).

Checkpoint: `resnet50_finetune_60_epochs_lr_decay_after_30_start_..._IN_SF-ca06340c.pth.tar`
("SIN+IN then fine-tuned on IN", aka Shape-ResNet) from
[rgeirhos/texture-vs-shape](https://github.com/rgeirhos/texture-vs-shape)
(ICLR 2019), the paper's own recommended general-purpose drop-in replacement
for standard ResNet50. Downloaded to `weights/resnet50_shape_biased.pth.tar`
(gitignored, 195MB - not tracked, re-download from the URL above if needed).

- [x] **Smoke test** (3 epochs, just verifying the pipeline runs) - loss fell
  every epoch (train 5.37→4.70→3.98, val 4.85→4.60→3.84), checkpoint loaded
  and the 1000-class `fc` correctly swapped to 151. Not a real accuracy
  comparison - too few epochs. A full run at the standard 16-epoch default
  is the next step before comparing against the resnet18/34 baselines above.
  ```bash
  make train ARGS="--run-name r50-shape-biased-smoketest --model-name resnet50 --weights-checkpoint weights/resnet50_shape_biased.pth.tar --epochs 3"
  ```

## Phase 9 — ResNet50 sweep: shape-biased vs. standard weights

Full-length runs (16 epochs), mirroring the Phase 7 resnet34 sweep structure.
Two baselines isolate two variables separately, then a regularized depth
sweep on the shape-biased checkpoint:

- **r50-standard-baseline** — resnet50 + standard ImageNet weights, exact
  current defaults. Isolates "does resnet50's extra capacity alone help",
  same role as `r34-baseline-default` did for resnet34.
- **r50-shape-baseline** — resnet50 + the shape-biased checkpoint, same
  defaults otherwise. Directly comparable to `r50-standard-baseline` -
  the only difference is pretrained-weight origin, not architecture.
- **r50-shape-reg-lastN{1,2,3,4}** — shape-biased checkpoint, stronger
  regularization (`blr=2e-4, wd=2e-3, ls=0.2`, same as the resnet34 reg
  sweep), depth capped at 4 for the same reason as before.

- [x] **r50-standard-baseline**
  ```bash
  make train ARGS="--run-name r50-standard-baseline --model-name resnet50"
  ```
- [x] **r50-shape-baseline**
  ```bash
  make train ARGS="--run-name r50-shape-baseline --model-name resnet50 --weights-checkpoint weights/resnet50_shape_biased.pth.tar"
  ```
- [x] **r50-shape-reg-lastN1**
  ```bash
  make train ARGS="--run-name r50-shape-reg-lastN1 --model-name resnet50 --weights-checkpoint weights/resnet50_shape_biased.pth.tar --backbone-lr 2e-4 --classifier-lr 1e-3 --weight-decay 2e-3 --label-smoothing 0.2 --epochs 16 --train-last-n-layers 1"
  ```
- [x] **r50-shape-reg-lastN2**
  ```bash
  make train ARGS="--run-name r50-shape-reg-lastN2 --model-name resnet50 --weights-checkpoint weights/resnet50_shape_biased.pth.tar --backbone-lr 2e-4 --classifier-lr 1e-3 --weight-decay 2e-3 --label-smoothing 0.2 --epochs 16 --train-last-n-layers 2"
  ```
- [x] **r50-shape-reg-lastN3**
  ```bash
  make train ARGS="--run-name r50-shape-reg-lastN3 --model-name resnet50 --weights-checkpoint weights/resnet50_shape_biased.pth.tar --backbone-lr 2e-4 --classifier-lr 1e-3 --weight-decay 2e-3 --label-smoothing 0.2 --epochs 16 --train-last-n-layers 3"
  ```
- [x] **r50-shape-reg-lastN4**
  ```bash
  make train ARGS="--run-name r50-shape-reg-lastN4 --model-name resnet50 --weights-checkpoint weights/resnet50_shape_biased.pth.tar --backbone-lr 2e-4 --classifier-lr 1e-3 --weight-decay 2e-3 --label-smoothing 0.2 --epochs 16 --train-last-n-layers 4"
  ```

**Results:**

| run | trainable % | final gap | val acc |
|---|---|---|---|
| r18-baseline-default | 93.9% | 0.407 | 0.880 |
| r34-baseline-default | 93.6% | 0.447 | 0.876 |
| r50-standard-baseline | 93.8% | 0.373 | 0.871 |
| r50-shape-baseline | 93.8% | 0.435 | 0.862 |
| r50-shape-reg-lastN1 | 64.0% | 0.420 | 0.857 |
| r50-shape-reg-lastN2 | 93.8% | 0.401 | 0.866 |
| **r50-shape-reg-lastN3** | 98.8% | **0.279** | **0.908** |
| r50-shape-reg-lastN4 | 99.7% | 0.357 | 0.876 |

At matched shallow-depth HP, shape-biased weights underperformed standard
resnet50 weights (0.862 vs 0.871) - the "silhouettes have no texture" theory
doesn't show up at `lastN=2`. But `lastN=3` with stronger regularization is a
standout: **0.908**, the best single accuracy across this whole investigation,
also with the smallest train/val gap of the group. Not monotonic with depth
(lastN=2 → 0.866, lastN=4 → 0.876), so this reads as a genuine sweet spot
rather than "more unfrozen is better." Caveat: single-seed run on a ~216-image
val split - the identical resnet18 final config alone spanned 0.839-0.903

**Seed check on `r50-shape-reg-lastN3`** - same HP, `random_state` changed
(reshuffles the stratified split and reseeds every RNG), to see if 0.908
holds up or was a favorable split/init draw:

- [x] **r50-shape-reg-lastN3-seed43**
  ```bash
  make train ARGS="--run-name r50-shape-reg-lastN3-seed43 --model-name resnet50 --weights-checkpoint weights/resnet50_shape_biased.pth.tar --backbone-lr 2e-4 --classifier-lr 1e-3 --weight-decay 2e-3 --label-smoothing 0.2 --epochs 16 --train-last-n-layers 3 --random-state 43"
  ```
- [x] **r50-shape-reg-lastN3-seed44**
  ```bash
  make train ARGS="--run-name r50-shape-reg-lastN3-seed44 --model-name resnet50 --weights-checkpoint weights/resnet50_shape_biased.pth.tar --backbone-lr 2e-4 --classifier-lr 1e-3 --weight-decay 2e-3 --label-smoothing 0.2 --epochs 16 --train-last-n-layers 3 --random-state 44"
  ```
across 4 historical reruns, so a repeat with a different `random_state` is
needed before trusting this number.

**Seed check results:**

| seed | val acc | top3 | top5 | final gap |
|---|---|---|---|---|
| 42 | 0.908 | 0.940 | 0.945 | 0.279 |
| 43 | 0.899 | 0.922 | 0.945 | 0.308 |
| 44 | 0.912 | 0.959 | 0.968 | 0.292 |

**Mean 0.906, stdev 0.007, range 0.899-0.912** - a tight 1.3pt spread, versus
the 6.4pt spread (0.839-0.903) the resnet18 final config showed across its
4 historical reruns. This config is both higher-performing and meaningfully
more stable, not a lucky split: **resnet50 + shape-biased weights, lastN=3,
blr=2e-4, wd=2e-3, ls=0.2, epochs=16** is the strongest validated
configuration found in this investigation.

---

## Deliberately not reran

- **Exact duplicates.** Several configs were run 2–4× identically (e.g. the final
  config ×4, `blr=1e-4/ls=0.2/lastN=5` ×3). Rerun each once; the historical spread
  above already documents variance.
- **One transient failure.** The `blr=1e-3, clr=1e-3, wd=1e-3, ls=0.1, lastN=5`
  config scored **0.885** three times and **0.005** once (~1/151 = random). The
  0.005 was a one-off crash/mis-init, not the config — not worth reproducing.
