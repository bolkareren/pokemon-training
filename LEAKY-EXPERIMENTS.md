# Experiment reproduction log (SUPERSEDED — leaky splits)

> **Every accuracy in this file is invalid as a measure of generalization.**
>
> The dataset these runs trained on contained the *shiny* sprite of each Pokémon
> alongside the normal one. Shiny sprites are recolours, so after the binary
> silhouette threshold they are frequently pixel-identical to their normal
> counterpart. 32.7% of the dataset was exact duplicates, and the random split
> put a pixel-identical twin in train for **~62% of every validation set**.
>
> The headline result here — 0.906 across three seeds — is therefore mostly a
> memorization score. Re-measured on deduplicated data, the same configuration
> scores **0.617**. See [Phase 11](#phase-11--duplicateleakage-audit-blocking-audit-already-run)
> for the audit, and **[EXPERIMENTS.md](EXPERIMENTS.md)** for the clean-data
> replication that supersedes this file.
>
> This file is kept because the *relative* progression it records is still a
> useful hypothesis set, and because knowing which conclusions were artifacts is
> worth more than deleting them. But no number here should be quoted, and no
> conclusion here should be carried forward without re-testing on clean data —
> several are known to be backwards (see "What this file got wrong" below).

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
- **The defaults changed after Phase 11.** `exclude_shiny=True` and
  `val_size=test_size=0.15` are now the defaults, so the commands in Phases 1-10
  no longer reproduce the accuracies printed beside them. To rerun any of them
  as originally measured, append:
  `--no-exclude-shiny --val-size 0.1 --test-size 0.1`. Every number in Part I is
  a leaky-split number and is **not** comparable to clean-data results.

### What this file got wrong

Recorded so the same conclusions are not quietly re-inherited. Each of these was
stated confidently below and is now known to be an artifact of the leak.

| claim in this file | why it is wrong |
|---|---|
| "0.906 ± 0.007 is the strongest validated config" | ~62% of val was memorized. Honest score: **0.617**. |
| "tight seed variance means it isn't a lucky split" | Binomial SE at n=217 is ±0.020; an observed 0.007 is *below the noise floor*. Leaked images answer identically every seed, which suppresses variance. Stability was evidence *of* the leak. |
| "`lastN=3` has the smallest train/val gap (0.279), so it is well-regularized" | The gap was compressed mechanically — `val_loss` was partly re-measuring `train_loss`. Honest gap is **~0.80**. |
| "resnet50 + shape-biased beats resnet18 (0.908 vs 0.880)" | Selected in a regime that rewarded memorizing duplicates, which favours capacity. Untested honestly; may reverse. |
| "`lastN=3` is a genuine sweet spot" | Same. 98.8% of resnet50 unfrozen against 914 real training images is a memorization-friendly configuration, not a generalization-friendly one. |
| "`epochs` is flat past 16" | Measured on leaky val. On clean data `val_loss` was still falling at epoch 16. |
| "OFAT: anything >3-4pt is real signal" | Derived from the understated 0.007 spread. The honest threshold was ~±8pt, so every Phase 10 verdict except the `blr=4e-4` collapse was noise. |
| "lastN=4 (0.876) and lastN=5 (0.857) are different depths" | **They are the same configuration.** `lastN=5` only additionally unfreezes `bn1`, whose sole parameters are BN affine — which `set_batch_norm_trainable(trainable=False)` re-freezes immediately after. Identical trainable-parameter sets. The 1.9pt between them is a direct measurement of run-to-run noise on an identical setup, and on its own refutes the ±3-4pt threshold above. Same applies to `lastN=6` vs `lastN=7`. |

The one Phase 10 finding that **does** survive is the `backbone_lr=4e-4`
collapse (-49.8pt). An effect that large is not a sampling artifact, and it was
found precisely because the OFAT pass tested an assumption carried over from a
different architecture instead of trusting it.

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

## Phase 10 — Flatness re-check (OFAT) on the new winning config

`weight_decay`, `label_smoothing`, `backbone_lr`, and `epochs` were all found
flat on **resnet18 with standard weights** (Phases 1-6). That was never
re-verified on resnet50 + shape-biased weights at `lastN=3` - a different
backbone/weight regime could have a different sensitivity profile, and
`lastN=3`'s unusually tight train/val gap (0.279-0.308 across seeds, vs
0.35-0.45 everywhere else) specifically raises the question of whether
`epochs` still has headroom instead of being flat.

One-factor-at-a-time, not a grid: hold the winning config fixed
(`resnet50, shape-biased checkpoint, lastN=3, blr=2e-4, clr=1e-3, wd=2e-3,
ls=0.2, epochs=16, random_state=42`) and vary one axis per run, bracketing
each current value. Baseline is `r50-shape-reg-lastN3` above (0.908) - not
rerun here.

- [x] **r50-flat-wd1e-3** (weight_decay 2e-3 → 1e-3)
  ```bash
  make train ARGS="--run-name r50-flat-wd1e-3 --model-name resnet50 --weights-checkpoint weights/resnet50_shape_biased.pth.tar --backbone-lr 2e-4 --classifier-lr 1e-3 --weight-decay 1e-3 --label-smoothing 0.2 --epochs 16 --train-last-n-layers 3"
  ```
- [x] **r50-flat-wd4e-3** (weight_decay 2e-3 → 4e-3)
  ```bash
  make train ARGS="--run-name r50-flat-wd4e-3 --model-name resnet50 --weights-checkpoint weights/resnet50_shape_biased.pth.tar --backbone-lr 2e-4 --classifier-lr 1e-3 --weight-decay 4e-3 --label-smoothing 0.2 --epochs 16 --train-last-n-layers 3"
  ```
- [x] **r50-flat-ls0.1** (label_smoothing 0.2 → 0.1)
  ```bash
  make train ARGS="--run-name r50-flat-ls0.1 --model-name resnet50 --weights-checkpoint weights/resnet50_shape_biased.pth.tar --backbone-lr 2e-4 --classifier-lr 1e-3 --weight-decay 2e-3 --label-smoothing 0.1 --epochs 16 --train-last-n-layers 3"
  ```
- [x] **r50-flat-ls0.3** (label_smoothing 0.2 → 0.3)
  ```bash
  make train ARGS="--run-name r50-flat-ls0.3 --model-name resnet50 --weights-checkpoint weights/resnet50_shape_biased.pth.tar --backbone-lr 2e-4 --classifier-lr 1e-3 --weight-decay 2e-3 --label-smoothing 0.3 --epochs 16 --train-last-n-layers 3"
  ```
- [x] **r50-flat-blr1e-4** (backbone_lr 2e-4 → 1e-4)
  ```bash
  make train ARGS="--run-name r50-flat-blr1e-4 --model-name resnet50 --weights-checkpoint weights/resnet50_shape_biased.pth.tar --backbone-lr 1e-4 --classifier-lr 1e-3 --weight-decay 2e-3 --label-smoothing 0.2 --epochs 16 --train-last-n-layers 3"
  ```
- [x] **r50-flat-blr4e-4** (backbone_lr 2e-4 → 4e-4)
  ```bash
  make train ARGS="--run-name r50-flat-blr4e-4 --model-name resnet50 --weights-checkpoint weights/resnet50_shape_biased.pth.tar --backbone-lr 4e-4 --classifier-lr 1e-3 --weight-decay 2e-3 --label-smoothing 0.2 --epochs 16 --train-last-n-layers 3"
  ```
- [x] **r50-flat-ep12** (epochs 16 → 12, checking for premature-stop headroom)
  ```bash
  make train ARGS="--run-name r50-flat-ep12 --model-name resnet50 --weights-checkpoint weights/resnet50_shape_biased.pth.tar --backbone-lr 2e-4 --classifier-lr 1e-3 --weight-decay 2e-3 --label-smoothing 0.2 --epochs 12 --train-last-n-layers 3"
  ```
- [x] **r50-flat-ep20** (epochs 16 → 20, checking whether the tight gap means more training still helps)
  ```bash
  make train ARGS="--run-name r50-flat-ep20 --model-name resnet50 --weights-checkpoint weights/resnet50_shape_biased.pth.tar --backbone-lr 2e-4 --classifier-lr 1e-3 --weight-decay 2e-3 --label-smoothing 0.2 --epochs 20 --train-last-n-layers 3"
  ```

Single run per point, not multi-seed - this is a cheap directional check, not a
rigorous re-estimate. Given the seed check above put single-run noise at
roughly ±1.3pts for this specific config, any point landing >3-4pts from the
0.908 baseline is a real signal worth following up with seed repeats; smaller
deviations are noise.

**Results:**

| axis | value | val acc | delta vs 0.908 | verdict |
|---|---|---|---|---|
| weight_decay | 1e-3 | 0.885 | -2.3pt | flat-ish |
| weight_decay | 4e-3 | 0.899 | -0.9pt | flat |
| label_smoothing | 0.1 | 0.876 | -3.2pt | mildly not flat |
| label_smoothing | 0.3 | 0.889 | -1.8pt | mildly not flat |
| backbone_lr | 1e-4 | 0.903 | -0.5pt | flat |
| **backbone_lr** | **4e-4** | **0.410** | **-49.8pt** | **collapsed** |
| epochs | 12 | 0.880 | -2.8pt | slightly undertrained |
| epochs | 20 | 0.899 | -0.9pt | flat, no extra headroom |

**Headline finding: `backbone_lr` is not flat here.** `4e-4` didn't
underperform gently, it collapsed training (final train/val loss 3.21/3.54,
vs ~1.8/2.1 everywhere else; top-1 acc 0.41). This directly contradicts the
resnet18-derived assumption that backbone LR is flat across roughly `1e-4` to
`1e-3` - at `lastN=3` (98.8% of resnet50 unfrozen), somewhere between `2e-4`
and `4e-4` is a real instability cliff, not a gentle slope. An assumption
carried over from a different architecture/depth regime turned out to be
wrong until directly tested - exactly what this OFAT pass was for.

Everything else held up: `weight_decay` is genuinely flat (consistent with
resnet18). `epochs` shows no headroom past 16 despite the tight train/val gap
- saturates rather than keeps improving. `label_smoothing` shows a mild but
consistent signal that `0.2` is a real (if small) local optimum - both
alternatives underperformed in the same direction, though the effect size is
close to the seed-noise floor.

Not yet done: bisecting the `backbone_lr` cliff (e.g. `3e-4`) to find the
actual safe ceiling, and confirming the `label_smoothing` effect isn't noise
via seed repeats at `0.1`/`0.3`.

---

# Part II — The audit that invalidated Part I

Phases 1-10 above tuned hyperparameters on `resnet18/34/50` and landed at a
validated 0.906 ± 0.007. Before spending more runs on that number, this phase
asked whether the number means what it appears to mean. It does not.

This is the last phase in this file. Everything downstream of it — the honest
re-baseline, the K-fold protocol, and the replication of Phases 1-10 on clean
data — lives in **[EXPERIMENTS.md](EXPERIMENTS.md)**.

## Phase 11 — Duplicate/leakage audit (BLOCKING; audit already run)

`scripts/data_scraping.py` pulls every sprite off a Pokémon's pokemondb.net
sprite page. That page serves, per generation, both the **normal** and the
**shiny** sprite. Shiny differs from normal *only in palette* — so after the
binary threshold in [data.py](pokemon_training/data.py) (`x > 0.5`), a
Pokémon's shiny and normal sprite for a given generation are frequently the
**same silhouette, pixel for pixel**.

Measured by [scripts/duplicate_audit.py](scripts/duplicate_audit.py):

| quantity | value |
|---|---|
| total images | 2162 |
| exact-duplicate redundant images | **706 (32.7%)** |
| near-duplicate (IoU > 0.97) redundant | **845 (39.1%)** |
| distinct silhouettes after dedup | **1317** |

Every one of the 151 classes has at least 4 redundant images. Because the
split in `split_dataset_indices` is a plain stratified random split over image
indices, duplicates land on both sides of it:

| seed | val n | exact twin in train | near twin in train |
|---|---|---|---|
| 42 | 217 | 134 (**61.8%**) | 146 (67.3%) |
| 43 | 217 | 120 (**55.3%**) | 143 (65.9%) |
| 44 | 217 | 131 (**60.4%**) | 151 (69.6%) |

**Roughly two thirds of every validation set is a memorization test, not a
generalization test.** The reported 0.906 is therefore not a generalization
estimate. If the leaked ~67% is answered near-perfectly, the implied accuracy
on the non-leaked remainder is only ≈ **0.71**.

This also explains the seed stability that made the result look trustworthy.
The binomial standard error alone at p≈0.9, n=217 is `sqrt(.9×.1/217) ≈ 0.020`
— the observed 0.007 stdev is *below the noise floor*, which is not a sign of a
robust config. It is what you get when two thirds of the answers are fixed
across seeds regardless of what the model learned.

Two consequences for Phases 1-10 that must be carried forward:

- **The noise floor was underestimated.** Phase 10 declared >3-4pt a real
  signal based on the 0.007 seed spread. Against a ±2pt binomial SE the honest
  threshold is closer to ±8pt. Every OFAT verdict there except the
  `backbone_lr=4e-4` collapse (-49.8pt) is within noise and should be treated
  as unresolved, not settled.
- **`test_size=0.1` has never been evaluated.** `test_loader` is built in
  [scripts/training.py:41](scripts/training.py:41) and never used. ~40 configs
  were selected on the val split, so val is thoroughly overfit by selection —
  but the test split is genuinely untouched. It should be spent exactly once,
  at the very end, on a single final config.

- [x] Audit duplication and split leakage
  ```bash
  uv run python scripts/duplicate_audit.py
  ```


---

## Deliberately not reran

- **Exact duplicates.** Several configs were run 2–4× identically (e.g. the final
  config ×4, `blr=1e-4/ls=0.2/lastN=5` ×3). Rerun each once; the historical spread
  above already documents variance.
- **One transient failure.** The `blr=1e-3, clr=1e-3, wd=1e-3, ls=0.1, lastN=5`
  config scored **0.885** three times and **0.005** once (~1/151 = random). The
  0.005 was a one-off crash/mis-init, not the config — not worth reproducing.
