# Experiment log

Everything here runs on the deduplicated dataset (1307 images, 151 classes,
shiny recolours excluded) and is scored by 5-fold grouped CV after a 15% test
split is carved out: **`oof_accuracy` pooled over 1110 out-of-fold predictions,
compared on `fold_accuracy_sem`**. Full per-phase logs live in git history;
the pre-dedup era is summarized in [LEAKY-EXPERIMENTS.md](LEAKY-EXPERIMENTS.md)
and none of its numbers are comparable to anything here.

## Protocol

- Every run: `uv run python scripts/training.py --run-name <name> --folds 5 ...`
  Runs are findable by name in MLflow (`make ui`, experiment
  `pokemon-classification-clean`); filter on `status == "FINISHED"`.
- **`--folds 5` is required.** The default (`folds=0`) runs a single split,
  reports `validation_accuracy` instead of `oof_accuracy`, and still succeeds —
  a missing flag produces a plausible, incomparable number.
- **Significance bar: 2× combined SEM** (binomial floor at n=1110 is ±1.5pt;
  `fold_accuracy_sem` is the number to use). Confirm anything load-bearing with
  a second `random_state` — measured reference variance on a seed change alone
  is ~±1pt.
- A no-flag `--folds 5` run reproduces the current best config
  (`p2-blr4e4-32`, seed 42, ~40 min). Changing
  `--model-name` with a checkpoint set requires `--weights-checkpoint None`
  (fails fast otherwise). `val_size` is unused in fold mode.
- Every fold run logs `oof_predictions.json`; `scripts/confusion_study.py`
  turns it into confusion, shape-similarity, and evolution-line metrics.
- **No metadata shortcuts**: preprocessing may only use information present in
  the image itself. Sprite index, generation, and source resolution are dataset
  properties, not properties of an arbitrary input silhouette.
- **No ensembling during exploration** — single model per experiment; ensembles
  belong to the final phase only (see Phase 6, including the OOF-ensemble
  leakage trap).

## Results so far

| question | answer | runs |
|---|---|---|
| is the historical 0.906 real? | no — ~62% of val had a pixel-identical shiny twin in train; honest score 0.617 | c0-leaky-reference |
| honest baseline | **0.653** (resnet50, ImageNet weights, lastN=3 — the config defaults) | c2-resnet50-standard |
| shape-biased checkpoint? | actively harmful: −5.7pt vs ImageNet at 4.0× SEM — the one large confirmed effect | c0-5fold vs c2-resnet50-standard |
| how deep to fine-tune? | plateau across lastN 2/3/4; lastN 5 and 7 are not distinct configs; deep unfreezing unstable without warmup | c1-lastN* |
| backbone size? | not the axis (50 > 18 > 34); weight origin was the confound | c2-* |
| label smoothing hurting? | no — removing it costs 3.5pt; 0.05–0.2 flat; the train/val gap responds to it but accuracy doesn't → **gap is not predictive** | c3-ls* |
| does added augmentation help? | no; elastic is significantly harmful (−4.9pt, 2.1× SEM) on contour-only input | c7-* |
| does removing augmentation help? | no; nothing clears the bar, scale jitter removal moves nothing → size cue not load-bearing; augmentation closed in both directions | n1-* |
| is the error irreducible? | no — silhouette collisions explain ~3% of errors (electrode/voltorb IoU 0.969 is the max); evolution-line confusions are 12.7% of errors at 15× chance | confusion study |
| mask polarity? | worth ~3pt despite being information-free; background = 1 wins (confirmed, 2 paired seeds, ~2.3× SEM) | n2-mask-inverted* |
| SDT input channel? | **confirmed**: +1.5pt over 4 paired seeds (+2.1/+0.4/+0.8/+2.6), 15/20 matched folds positive, t=2.86, p=0.01; now the default | n2-*sdt* |
| curvature proxy channel? | dead — too sparse (1-3px slivers) to survive the stem's 4× downsample; diagnosis motivates "edge" and the stem phase | n2-curv, n2-sdt-curv |
| edge channel? | retired — dilutes SDT (−2.6pt, 5/5 folds); alone +0.65pt at 0.4× SEM | p1-* |
| LR schedule? | **+5.1pt confirmed at 2 seeds**: cosine+warmup+restore at blr 4e-4, 32-epoch horizon; the historical 4e-4 collapse was a warmup artifact | p2-* |

**Current best: 0.716, two-seed mean of the config defaults** (`(mask, sdt,
mask)` input + cosine/warmup/restore at blr 4e-4 over 32 epochs: 0.725 at seed
42, 0.707 at seed 43). Per-seed pairing is the comparison standard: a Phase 3+
run at seed 42 compares against **0.725** (`p2-blr4e4-32`). Progression of
confirmed gains: 0.596 → 0.653 (ImageNet weights) → 0.677 (SDT channel) →
0.716 (schedule/LR/horizon).

What the failures collectively point at: the bottleneck is not capacity,
regularization, data variety, or task ambiguity — it is how much discriminating
information reaches (and survives) the network. That is what the roadmap
attacks: input representation first, then the stem that downsamples it away.

Also measured, and load-bearing:

- **Models are high-variance**: five ~equal configs agree on only 75-82% of
  predictions; 230 of 1110 images are solved by no config, 880 by at least one.
  Real ensemble headroom — deliberately deferred to Phase 6.
- **Sprite scale mixes artifact with signal** (generation canvas vs real size,
  same ~2× magnitude, so they cancel). Normalising it away via generation
  metadata was built and removed as a shortcut; size is near-useless as a cue
  in this framing, which N1 then confirmed from the augmentation side.

---

# Roadmap

Phases run in order. Every phase: single model, 5-fold grouped CV, 2× SEM bar,
second seed on anything that would change a default.

## Phase 1 — Third channel: edge filtering

The curv failure showed *sparse* boundary encodings die in the stem; `"edge"`
(implemented) is the dense retry: a Gaussian band `exp(−d²/2σ²)`, σ = 8px, on
the contour — wide enough that ~2px survive the 4× stem downsample. It
concentrates input contrast where all silhouette information lives, in the form
pretrained edge-sensitive stem filters respond to.

The gate resolved: SDT confirmed, so the base was `(mask, sdt, mask)`.

- [x] **p1-sdt-edge** — `(mask, sdt, edge)`: **0.648, −2.6pt vs the same-seed
      default (0.674), all five paired folds negative.** The informational-
      redundancy caveat played out — unlike polarity, this representational
      convenience did not pay.
- [x] **p1-edge** — `(mask, edge, mask)`: 0.660, +0.65pt vs the same-seed
      all-mask baseline (0.653) at ~0.4× SEM, 3/5 folds. Inconclusive; edge
      alone is not better than SDT alone (0.674).

**Verdict: the default stays `(mask, sdt, mask)`; edge is retired.** Weak
support only for the thin-feature hypothesis from the input side — Phase 3
remains motivated primarily by the curv diagnosis, not boosted by this.

A pattern worth recording: every two-derived-channel config has lost to its
one-derived-channel counterpart on the same folds (sdt+curv 0.641 < sdt 0.654;
sdt+edge 0.648 < sdt 0.674). The winning config keeps **two copies of the raw
mask**; whether the duplicate mask is load-bearing (channel-weighting under
RGB-correlated pretrained filters) is an open, cheap-to-test hypothesis —
e.g. `(sdt, mask, sdt)` or channel-position swaps — parked in Phase 5.

Backup third-channel candidate (untested): level-set curvature of the EDT
field (`div(∇φ/|∇φ|)` via two `numpy.gradient` calls — dense, smooth, no new
dependencies).

## Phase 2 — Schedule and checkpointing (done: **+5.1pt, the largest phase
gain in the project**)

Design (settled with the user): best epoch selected on val_loss; fixed budget
with restore-best, no early exit, so the cosine horizon stays deterministic;
per-step cosine to ~0 with 2-epoch linear warmup, each param group keeping its
own base LR.

Results at seed 42, paired against `n2-origmask-sdt` (0.674, identical folds):

| run | oof | delta | × SEM | fold best-epochs |
|---|---|---|---|---|
| **p2-blr4e4-32** | **0.725** | **+5.1** | **4.6** | — |
| p2-blr4e-4 (16ep) | 0.707 | +3.3 | 2.2 | 10-14 |
| p2-cosine-restore-32 | 0.705 | +3.2 | 1.7 | 16-28 |
| p2-cosine-restore-16 | 0.696 | +2.3 | 1.0 | 11-15 |
| p2-cosine-warmup | 0.689 | +1.5 | 0.7 | — |
| p2-best-epoch | 0.677 | +0.3 | 0.3 | 14-15 |

Second seed: p2-blr4e-4-seed43 +2.4pt, **p2-blr4e4-32-seed43 0.707, +4.1pt**
— the winner replicates decisively.

- **The schedule is the driver; restoration alone is nearly worthless**
  (best epochs under constant LR are 14-15 of 16 — the final epoch was already
  near-optimal). Restoration's value is enabling long horizons safely.
- **The LR ceiling doubled under warmup.** `blr=4e-4`, a −49.8pt collapse
  without warmup, is now the best setting — the historical collapse was a
  warmup artifact, exactly as hypothesized.
- **The horizon is genuinely used** (best epochs up to 28 of 32) and combines
  superadditively with the higher LR (+1.8 to +2.0 over each alone).
- **The gain is broad, not mechanism-specific**: errors 385 → 305 vs the
  pre-SDT era while the evolution-line share stays ~11% — unlike SDT, which
  targeted that mechanism.

**Defaults flipped** (evidence above): `scheduler="cosine"`,
`restore_best_epoch=True`, `backbone_lr=4e-4`, `epochs=32`. A default run now
takes ~40 min. Untested residue for Phase 5: warmup length, higher LRs (8e-4),
longer horizons (64).

## Phase 3 — Reduced-stride stem

The architecture-side fix for the same mechanism Phase 1 works around: the
ImageNet stem (stride-2 conv + stride-2 maxpool) discards thin contour detail
before the first residual block. Keep ResNet50 + ImageNet weights; surgery only
where our input provably differs from natural images.

- [ ] **p3-nomaxpool** — drop the stem maxpool (cheapest; 2× feature maps).
- [ ] **p3-stride1-conv** — also conv1 stride 1 if compute allows (4× maps).
- [ ] Interpret jointly with Phase 1: edge-channel gain and stem gain should
      be partially redundant if the thin-feature diagnosis is right.

## Phase 4 — Sketch-pretrained backbones (+ edge)

Checkpoint-swap protocol, exactly like the C2 comparison. Expectations
deliberately low: the shape-biased (Stylized-ImageNet) checkpoint was this
idea's precedent and lost by 5.7pt; sketches are strokes, silhouettes are
filled regions; community checkpoints trade ImageNet's scale for a partial
domain match.

- [ ] **p4-sketch-checkpoint** — one credible sketch/QuickDraw-pretrained
      backbone, best input channels from Phases 1-3, one run.
- [ ] **p4-sketch-edge** — same checkpoint with the edge channel: a stroke-
      pretrained network sees contours natively, so edge may interact.
- The stronger domain-matched alternative if this fails: pretrain on all-gen
  Pokémon silhouettes (~900 classes beyond Gen 1, scrapable with the existing
  pipeline) — gated on the leak-decomposition item in Phase 5 saying data
  quantity matters.

## Phase 5 — Backlog: everything never rigorously tested

In rough value order; each is cheap and uses whatever config Phases 1-4 settle:

- [ ] **leak decomposition** — train on the full unfiltered set, validate only
      on twin-free images; separates "measurement was wrong" from "we halved
      the data", and gates data expansion (the strongest untested lever).
- [ ] **depth re-sweep** (lastN ∈ {2,3,4,6}) — last swept on the discarded
      shape-biased checkpoint; skip 5/7 (not distinct configs).
- [ ] **optimizer** — AdamW was assumed, never swept; SGD+momentum, Adam.
- [ ] **weight decay, BN affine** — the unfinished regularization axes; low
      expected value (gap is not predictive), run for completeness.
- [ ] **single-channel stem** — sum pretrained RGB filters; "drop the
      redundant capacity" vs Phase 1's "fill it".
- [ ] **duplicate-mask hypothesis** — is the second raw mask copy in the
      winning `(mask, sdt, mask)` load-bearing? Channel-position swaps,
      e.g. `(sdt, mask, sdt)`. See the Phase 1 pattern note.
- [ ] **aspect-preserved crop** — bbox-crop + pad; resolution gain at a size
      cost N1 measured as ~zero.
- [ ] **backbone re-check** — resnet18 vs 50 was 1.9× SEM, just under the bar.
- [ ] **confusion-study leftovers** — body-plan-aware descriptor, error rate
      by generation (+ generation-held-out split), cross-class near-duplicate
      check, embedding visualization, hard-example contact sheet.

## Phase 6 — Final: ensemble, then the one-shot test evaluation

Ensembling stays out of exploration (it multiplies every experiment's cost and
judges later changes as ensembles, which is not how they'd ship). At the end,
on the settled config:

- [ ] **Seed ensemble within each fold** and/or **TTA** (identity + hflip +
      small rotations) — valid on OOF data.
- [ ] **Fold ensemble on the held-out test split only.** Averaging the K fold
      models against `oof_predictions.json` is leakage: each image is
      out-of-fold for exactly one model, in-training for K−1 — the shiny
      mistake from the opposite direction.
- [ ] **One-shot test evaluation** — never-touched 15% split, binomial CI,
      no tuning afterwards; anything learned becomes a new hypothesis.

---

## Known data issues

- 56 near-duplicate clusters remain in the normal series; grouped folds keep
  them from straddling splits.
- Raw sprite size is a near-perfect generation proxy (56×56 Gen 1 … 128×128
  Gen 6+) — a shortcut to keep out of preprocessing.
