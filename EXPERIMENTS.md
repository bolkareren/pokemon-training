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

- [ ] **lastN=0** (head only) — hist acc **0.544** — floor; classifier alone is not enough
  ```bash
  make train ARGS="--run-name p1-lastN0 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.0 --epochs 15 --train-last-n-layers 0"
  ```
- [ ] **lastN=1** — hist acc **0.843**
  ```bash
  make train ARGS="--run-name p1-lastN1 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.0 --epochs 15 --train-last-n-layers 1"
  ```
- [ ] **lastN=2** — hist acc **0.880** (best of the baseline sweep)
  ```bash
  make train ARGS="--run-name p1-lastN2 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.0 --epochs 15 --train-last-n-layers 2"
  ```
- [ ] **lastN=3** — hist acc **0.853**
  ```bash
  make train ARGS="--run-name p1-lastN3 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.0 --epochs 15 --train-last-n-layers 3"
  ```
- [ ] **lastN=4** — hist acc **0.876**
  ```bash
  make train ARGS="--run-name p1-lastN4 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.0 --epochs 15 --train-last-n-layers 4"
  ```
- [ ] **lastN=5** — hist acc **0.857**
  ```bash
  make train ARGS="--run-name p1-lastN5 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.0 --epochs 15 --train-last-n-layers 5"
  ```
- [ ] **lastN=6** — hist acc **0.816** (too much unfrozen)
  ```bash
  make train ARGS="--run-name p1-lastN6 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.0 --epochs 15 --train-last-n-layers 6"
  ```

## Phase 2 — Add regularization (label smoothing + weight decay)

Fix `lastN=5`, single LR `1e-3`. Adds label smoothing, then bumps weight decay
`1e-4 → 1e-3`.

- [ ] **ls=0.1, wd=1e-4** — hist acc **0.876**
  ```bash
  make train ARGS="--run-name p2-ls0.1-wd1e-4 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-4 --label-smoothing 0.1 --epochs 15 --train-last-n-layers 5"
  ```
- [ ] **ls=0.1, wd=1e-3** — hist acc **0.885** (stronger weight decay helps)
  ```bash
  make train ARGS="--run-name p2-ls0.1-wd1e-3 --backbone-lr 1e-3 --classifier-lr 1e-3 --weight-decay 1e-3 --label-smoothing 0.1 --epochs 15 --train-last-n-layers 5"
  ```

## Phase 3 — Differential learning rate (slower backbone)

Drop the backbone LR to `1e-4` while keeping the classifier at `1e-3`.

- [ ] **blr=1e-4, clr=1e-3** — hist acc **0.889** (best of session 1)
  ```bash
  make train ARGS="--run-name p3-diff-lr --backbone-lr 1e-4 --classifier-lr 1e-3 --weight-decay 1e-3 --label-smoothing 0.1 --epochs 15 --train-last-n-layers 5"
  ```

## Phase 4 — Batch-norm handling ablation

Confirms the current default (BN in **train** mode, affine params **frozen**) is
right.

- [ ] **BN in eval mode** — hist acc **0.820** (clearly worse; don't freeze BN stats)
  ```bash
  make train ARGS="--run-name p4-bn-eval --batch-norm-mode eval --backbone-lr 1e-4 --classifier-lr 1e-3 --weight-decay 1e-3 --label-smoothing 0.1 --epochs 15 --train-last-n-layers 5"
  ```
- [ ] **BN affine trainable** (lastN=3) — hist acc **0.885** (no clear gain from unfreezing affine)
  ```bash
  make train ARGS="--run-name p4-bn-affine --train-batch-norm-affine --backbone-lr 1e-4 --classifier-lr 1e-3 --weight-decay 1e-3 --label-smoothing 0.1 --epochs 15 --train-last-n-layers 3"
  ```

## Phase 5 — Session 2: stronger smoothing, LR and depth around the optimum

Label smoothing to `0.2`; find the backbone-LR sweet spot and re-check depth.

- [ ] **blr=1e-4, ls=0.2** — hist acc **0.894**
  ```bash
  make train ARGS="--run-name p5-blr1e-4-ls0.2 --backbone-lr 1e-4 --classifier-lr 1e-3 --weight-decay 1e-3 --label-smoothing 0.2 --epochs 15 --train-last-n-layers 5"
  ```
- [ ] **blr=5e-4, ls=0.2** — hist acc **0.903** (top result; `5e-4` beats both `1e-4` and `1e-3`)
  ```bash
  make train ARGS="--run-name p5-blr5e-4-ls0.2 --backbone-lr 5e-4 --classifier-lr 1e-3 --weight-decay 1e-3 --label-smoothing 0.2 --epochs 15 --train-last-n-layers 5"
  ```
- [ ] **wd=3e-3** (weight-decay ablation) — hist acc **0.889** (more decay doesn't help)
  ```bash
  make train ARGS="--run-name p5-wd3e-3 --backbone-lr 1e-4 --classifier-lr 1e-3 --weight-decay 3e-3 --label-smoothing 0.2 --epochs 15 --train-last-n-layers 5"
  ```
- [ ] **lastN=4, ep=18** — hist acc **0.903** (depth robustness near the optimum)
  ```bash
  make train ARGS="--run-name p5-lastN4-ep18 --backbone-lr 5e-4 --classifier-lr 1e-3 --weight-decay 1e-3 --label-smoothing 0.2 --epochs 18 --train-last-n-layers 4"
  ```
- [ ] **lastN=6, ep=18** — hist acc **0.880** (deeper again regresses)
  ```bash
  make train ARGS="--run-name p5-lastN6-ep18 --backbone-lr 5e-4 --classifier-lr 1e-3 --weight-decay 1e-3 --label-smoothing 0.2 --epochs 18 --train-last-n-layers 6"
  ```

## Phase 6 — Final configuration (current `ExperimentConfig` default)

`blr=5e-4, clr=1e-3, wd=1e-3, ls=0.2, lastN=5, epochs=18, BN train/no-affine`.
This is exactly the default, so it runs with **no args**. Historically run 4×,
scoring **0.839 / 0.866 / 0.903 / 0.903** (top-3 ≈ 0.94, top-5 ≈ 0.95).

- [ ] **Final config** — hist acc **0.90** (best), ~0.86 typical
  ```bash
  make train ARGS="--run-name final-default"
  ```

---

## Deliberately not reran

- **Exact duplicates.** Several configs were run 2–4× identically (e.g. the final
  config ×4, `blr=1e-4/ls=0.2/lastN=5` ×3). Rerun each once; the historical spread
  above already documents variance.
- **One transient failure.** The `blr=1e-3, clr=1e-3, wd=1e-3, ls=0.1, lastN=5`
  config scored **0.885** three times and **0.005** once (~1/151 = random). The
  0.005 was a one-off crash/mis-init, not the config — not worth reproducing.
