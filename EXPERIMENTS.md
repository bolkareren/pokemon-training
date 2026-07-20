# Experiment log

Everything here runs on the deduplicated dataset (1307 images, 151 classes,
shiny recolours excluded) and is scored by 5-fold grouped CV after a 15% test
split is carved out: **`oof_accuracy` pooled over 1110 out-of-fold predictions,
compared on `fold_accuracy_sem`**. Full per-phase logs live in git history;
the pre-dedup era is summarized in [LEAKY-EXPERIMENTS.md](LEAKY-EXPERIMENTS.md)
and none of its numbers are comparable to anything here.

> **Fold correction, 2026-07-20 — read before comparing anything to an older
> number.** Two bugs in fold construction were fixed in one session, and both
> changed which images land in which fold:
>
> 1. **Mask polarity** in the near-duplicate grouping (IoU measured over the
>    background, not the creature). Worth **+3.4pt** on the *same config*:
>    `p2-blr4e4-32` scored 0.725 pre-fix and 0.7586 post-fix.
> 2. **IoU grouping replaced by index-based pairing** of shiny sprites to the
>    normals they recolour. Only ~21% of images land in the same fold under the
>    two schemes — barely above the 20% chance floor for 5 folds — so *any*
>    reference measured under the old grouping is unusable for pairing.
>
> Consequently: **every absolute number recorded before 2026-07-20 is on
> distorted folds and understates the config by roughly 3-4pt.** Paired deltas
> mostly survive, but two "confirmed" results have already weakened on
> corrected folds (mask polarity, the 32-epoch horizon), so any effect *smaller
> than the 3.4pt artifact* should be treated as unverified until re-measured.
> Rows below carry a ⚠ where this applies.

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
  - `fold_accuracy_sem` is `fold_accuracy_stdev / sqrt(5)`, and the `sqrt(n)`
    assumes the folds are independent. They are not: any two folds share ~3/4
    of their training data, so the models and their errors are correlated, the
    cross-covariance terms are positive, and **the logged SEM understates true
    uncertainty by an unknown amount**. There is no unbiased estimator of
    K-fold CV variance (Bengio & Grandvalet 2004), so this is not fixable in
    code — it means the 2× bar is more permissive than it looks and borderline
    results are weaker than their ratio suggests. It is also why the
    second-seed requirement does real work that more folds would not: reseeding
    gives a genuinely independent draw.
  - Prefer *paired* comparisons on identical folds. When two runs use different
    fold structures the comparison is still valid on pooled OOF over the same
    image set, but it loses pairing and carries more variance than the
    arithmetic shows — flag it when it happens.
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
| mask polarity? | ⚠ **weakened**: was ~3pt at 2 seeds; on corrected folds only −1.27pt at **0.78× SEM** (0.7586 vs 0.7459), i.e. not significant. Direction still favours background = 1, so the default stands, but the effect is not established. One seed vs the original two — needs a second seed to resolve | n2-mask-inverted*, p5-invert-mask-32 |
| SDT input channel? | **consistent, not independently confirmed**. Original: +1.5pt over 4 paired seeds (t=2.86, p=0.01). On corrected folds, one paired seed gives **+1.08pt at 0.64× SEM** (3/5 folds, paired t=0.789 p=0.47; McNemar 66 fixed / 54 broken, p=0.32). That sits inside the original per-seed range (+0.4…+2.6), so it replicates in direction and magnitude but cannot confirm alone — **a +1.5pt effect is not detectable in a single run** (2× SEM ≈ 3.4pt here). Default stands; full re-confirmation needs 3 more paired seeds (~3h) | n2-*sdt*, p6-ref-26 vs p6-allmask-26 |
| curvature proxy channel? | dead — too sparse (1-3px slivers) to survive the stem's 4× downsample; diagnosis motivates "edge" and the stem phase | n2-curv, n2-sdt-curv |
| edge channel? | retired — dilutes SDT (−2.6pt, 5/5 folds); alone +0.65pt at 0.4× SEM | p1-* |
| LR schedule? | **+5.1pt confirmed at 2 seeds**: cosine+warmup+restore at blr 4e-4; the historical 4e-4 collapse was a warmup artifact. Effect is larger than the fold artifact, so likely robust — but see the horizon row, which did not survive | p2-* |
| epoch horizon? | ⚠ **saturated, was overstated**: 26 and 32 epochs are within 0.1pt on corrected folds (0.7595 vs 0.7586). Phase 2 recorded the 32-epoch horizon as "genuinely used" (best epochs to 28); that was a fold artifact. 26 epochs is ~20% cheaper for the same result — default not yet changed, wants a second seed | p5-ref-26-stepmatched, p5-ref-32 |
| higher backbone LR? | no — blr 8e-4 is +0.4pt at 0.3× SEM; the LR ceiling does not extend past 4e-4. Phase 2 schedule residue closed on this axis | p5-blr8e4 |
| duplicate-mask hypothesis? | no — `(sdt, mask, sdt)` is −1.2pt at 0.6× SEM; the second raw mask copy in `(mask, sdt, mask)` is not load-bearing. Phase 1's open question resolved | p5-dupmask-sdt-mask-sdt |
| does more data help (leak decomposition)? | **plausible, unconfirmed**: training on the full unfiltered set scores 0.7802 on the normal-series subset vs 0.7595 step-matched control — **+2.1pt at 1.2× SEM**, below the bar, and *unpaired* (different fold structures). Was +2.9pt under IoU grouping; ~0.8pt of that was animation-frame leakage. Does not justify the all-gen scrape yet | p5-leak-decomposition, p6-leak-idxgroup |
| reduced-stride stem? | no — nomaxpool −0.8pt at 0.4× SEM for 2.8× compute; stride1 gated off; thin-feature hypothesis retired | p3-nomaxpool |

**Current reference: 0.7496** (`p6-ref-26`, the config defaults at seed 42 on
index-grouped folds — **pair every new experiment against this**). The best
number ever measured is 0.7586 (`p5-ref-32`) but it is on superseded folds.
The previously recorded 0.716/0.725 was the same config on distorted folds —
**no model change accounts for the difference**; see the fold-correction note.

Reference table, all seed 42, all `(mask, sdt, mask)` + cosine/warmup/restore
at blr 4e-4:

| run | grouping | epochs | oof | SEM |
|---|---|---|---|---|
| `p5-ref-16` | IoU (superseded) | 16 | 0.7450 | 0.0118 |
| `p5-ref-26-stepmatched` | IoU (superseded) | 26 | 0.7595 | 0.0121 |
| `p5-ref-32` | IoU (superseded) | 32 | 0.7586 | 0.0154 |
| **`p6-ref-26`** | **index (current)** | **26** | **0.7496** | **0.0155** |

**Pair only against `p6-ref-26`.** The three `p5-ref-*` runs are on IoU-grouped
folds, which share ~21% of fold membership with index-grouped folds —
effectively unrelated splits. They remain useful as evidence about the *size*
of the fold artifact, not as comparison baselines. (`p6-ref-26` vs
`p5-ref-26-stepmatched` is 0.7496 vs 0.7595: unpaired, ~1pt, noise.)

**Single-run resolution is ~3.4pt.** Combined SEM against this reference is
~0.017, so the 2× bar is ~3.4pt — larger than *most* effects in the table
above. Any hypothesis worth under ~3pt needs paired folds across multiple
seeds, not one run per arm; `p6-ref-26` fold accuracies alone span 0.7207 to
0.7973. Budget accordingly when planning an experiment, or it cannot answer
the question asked of it.

Progression of confirmed gains (mixed fold regimes, so read as trend not
ledger): 0.596 → 0.653 (ImageNet weights) → 0.677 (SDT channel) → 0.716
(schedule/LR) → 0.7586 (same config, folds corrected).

What the failures collectively point at: the bottleneck is not capacity,
regularization, data variety, or task ambiguity — it is how much discriminating
information reaches (and survives) the network. That is what the roadmap
attacks: input representation first, then the stem that downsamples it away.

A pattern across the confirmed gains: **every one so far is model-agnostic
tooling** — data hygiene (dedup/grouped CV), pretrained-weight origin, input
encoding and polarity, LR schedule and horizon. None required touching the
architecture, and all of them transfer to whatever backbone eventually wins.
The remaining roadmap is where that stops: Phases 3-4 are the first
model-*specific* interventions, which is also why they were sequenced after
the agnostic levers were exhausted.

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
second seed on anything that would change a default. **Exploration protocol:**
cheap single-factor tests run at `--epochs 26` paired against `p6-ref-26`
(0.7496 at seed 42) — 26 rather than the old 16, since 26 and 32 measure the
same and 16 loses ~1.4pt of real optimization headroom. Winners re-run at the
full default.

**Power check before you run anything.** Single-run resolution against the
reference is ~3.4pt (2× combined SEM). Most remaining hypotheses are worth
less than that, so a one-run-per-arm design will return a null regardless of
truth — the SDT re-verification demonstrated exactly this failure. For any
effect expected under ~3pt, budget paired folds across ≥2 seeds and report the
paired t-test and McNemar counts, not just the oof delta.

## Next session — start here

The 2026-07-20 session was spent almost entirely on fold correctness, not on
the roadmap. Two fold bugs were found and fixed, the leak decomposition ran
twice (once under each grouping), and four Phase 5 flag items were cleared.
**The roadmap did not advance; the measuring instrument did.**

Re-validation is **done**: the baseline is re-established (`p6-ref-26`, 0.7496)
and SDT replicated in direction and magnitude. The defaults are not disturbed,
so the roadmap resumes. In priority order:

1. **The 148-pose-variant run** — the sharpest version of the data question,
   and it did not exist before this session. `exclude_shiny=True` drops all 853
   shiny sprites, but **705 are exact duplicates and 148 are pose variants**
   (different animation frames of the gen-5 sprite: genuinely different
   silhouettes, same subject — see Known data issues). Training on normal + the
   148 isolates novel-pose value with no duplicate contamination and no
   step-count confound, unlike the leak decomposition which carries both. Needs
   a small data.py change to admit a manifest-driven subset. This settles the
   data lever more cheaply and far more cleanly than an all-gen scrape.
2. **Aspect-preserved crop** (Phase 5) — promoted, because this session
   measured the thing that motivates it: body occupancy averages **24.9%** of
   the canvas and ranges 7.2% (magnemite) to 48.5% (venusaur). Roughly 3/4 of
   every image is empty, and small Pokémon are effectively trained at far lower
   resolution than large ones. Bbox-crop + pad also removes absolute size,
   which is the near-perfect generation proxy the "no metadata shortcuts" rule
   exists to keep out — N1 measured size as ~useless as a cue, so this should
   be close to free. `scripts/confusion_study.py` already has the crop helper.
   Plausibly the largest untested effect on the list.
3. **Phase 4 sketch checkpoints** — gated on finding a credible ResNet-50
   sketch/quickdraw checkpoint; the loading-order fix in
   `load_pretrained_model` (~5 lines) is needed for non-1000-class heads.
   Expectations stay low (the shape-biased precedent lost 5.7pt).
4. **Remaining Phase 5 items**: depth re-sweep (lastN 2/4/6), optimizer
   (AdamW never swept against SGD/Adam), weight decay, single-channel stem.
   All are cheap, and all are **likely below single-run resolution** — plan
   them as multi-seed paired batteries or accept they will read as null.
5. **Second seeds on the two weakened results** — mask polarity (0.78× SEM)
   and the 26-vs-32 horizon. Recorded as unresolved rather than overturned,
   which is not a state to leave indefinitely, but neither blocks progress.
6. **Phase 6 last**, and not until the config stops moving: ensembling
   amplifies whatever config it is handed.

Cleared this session, no longer worth running: `--backbone-lr 8e-4` (null),
`(sdt, mask, sdt)` duplicate-mask hypothesis (null), the leak decomposition
itself (plausible but unconfirmed at 1.2× SEM), and the 64-epoch horizon
(pointless — 32 already buys nothing over 26).

State at session end: reference **0.7496** (`p6-ref-26`, defaults on
index-grouped folds — the number to pair against). Best ever measured is
0.7586 (`p5-ref-32`) but on superseded folds; the old 0.716 was the same
config mismeasured. Tree clean, nothing running. Branch
`fix/near-duplicate-mask-polarity` carries both fold fixes plus the log
rewrite and is **unpushed**.

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

Metric note: `fold_gap` for runs up to and including `p3-nomaxpool` was
measured at the *final* epoch; later runs measure it at the epoch actually
scored (the restored one). Accuracy metrics are unaffected.

## Phase 3 — Reduced-stride stem

The architecture-side fix for the same mechanism Phase 1 worked around: the
ImageNet stem (stride-2 conv + stride-2 maxpool) discards thin contour detail
before the first residual block. Keep ResNet50 + ImageNet weights; surgery only
where our input provably differs from natural images. Phase 2's warmup landing
first matters here: stem changes shift early-layer statistics, and constant-LR
training near the instability boundary would have confounded the comparison.
Compute note: larger early feature maps multiply the now-40-min default run.

- [x] **p3-nomaxpool** — `--stem nomaxpool --epochs 16`: **0.699, −0.8pt vs
      the paired 16-epoch reference (`p2-blr4e-4`, 0.707) at −0.4× SEM**, for
      2.8× the compute (56 min). Mixed fold signs (2 up, 3 down).
- [x] **p3-stride1-conv** — **not run**: gated on nomaxpool showing ≥1× SEM
      signal, which it did not. The `stem` config option remains for later use.

**Verdict: the stem is not the bottleneck.** Third strike for the thin-feature
hypothesis (sparse curv channel dead, edge channel retired, now doubled stem
resolution flat-negative): at this data scale the network is not starved of
contour detail. Whatever separates 0.72 from the ceiling lives elsewhere —
consistent with the diversity measurement (high variance in *what* gets
learned) pointing at data quantity, which the leak decomposition tests next.

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
- [ ] **aspect-preserved crop** — **promoted to the top of the queue**, see
      "Next session". Bbox-crop + pad; resolution gain at a size cost N1
      measured as ~zero. Occupancy was quantified 2026-07-20: mean 24.9% of
      canvas, range 7.2-48.5%, so ~3/4 of every image is empty and small
      Pokémon train at far lower effective resolution than large ones. Crop
      helper already exists in `scripts/confusion_study.py`.
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

- **The normal series is essentially duplicate-free: 1306 clusters over 1307
  images — one genuine near-duplicate pair.** The previously recorded "56
  clusters" was an artifact of a polarity bug in the old IoU-based grouping
  code (`near_duplicate_groups`, since replaced by `sprite_groups`), which
  computed IoU over the *background* rather than the creature. Background IoU is
  inflated by the empty-canvas fraction, and body occupancy ranges 7.2%
  (magnemite) to 48.5% (venusaur), mean 24.9% — so the 0.97 threshold was
  effectively testing "is this Pokémon small?". Mean within-class pairwise IoU:
  0.381 creature-mask vs 0.737 background-mask; corr(occupancy, gap) = −0.975.
  The bug over-merged 41 classes, all low-occupancy (voltorb 9→5, electrode
  9→6, metapod 8→6). Fixed 2026-07-20.
  - Consequence: grouped folds were not binding on the deduplicated dataset.
    Absolute numbers *do* need correcting (the fold shuffle was worth 3.4pt),
    but grouping itself could only ever have been conservative there. It *is*
    load-bearing with `--no-exclude-shiny`, where twins are real.
- **The gen-5 sprites are animated, and normal/shiny screenshots catch
  different frames.** This is why IoU grouping was replaced with index-based
  pairing (`sprite_groups`, 2026-07-20). Measured over all 853 shiny sprites,
  paired by index rather than overlap:
  - **705 are exact recolours** (IoU ≥ 0.999 — pixel-identical after
    thresholding), **148 are not**.
  - All 148 involve one sprite index: **145 at `image-5`, 3 at `image-6`**
    (golbat, slowpoke, gastly). Sprites at every other position are 100% exact
    across all 151 classes.
  - Severity tracks pose mobility, confirming the animation-frame cause:
    aerodactyl 0.138, golbat 0.238 (wings open vs closed), charizard 0.460 …
    ninetales 0.997, rhydon 0.996 (static poses).
  - **No IoU threshold can work.** The animated pairs run 0.14–0.997 and are
    interleaved with genuinely-distinct artwork by pose, not separated from it.
    Catching ninetales needs ~0.99, which still misses every winged case;
    catching aerodactyl would group most of the dataset. The old 0.97 cutoff let
    **39 pairs in the 0.90–0.97 window** straddle folds — near-identical images
    of the same sprite, which is ~3.5% of the 1110 scored images and enough to
    manufacture the leak decomposition's original +2.9pt on its own. Re-running
    under index grouping cost it 0.8pt of that.
  - The pairing rule is `normal = shiny - shiny_start + 1`, pairing from
    `image-1` because `image-0` is the gen-1 sprite and predates shinies. It
    resolves all 853 with no leftovers: 2160 images → 1307 groups, exactly the
    normal-series count.
  - **Supersedes an earlier claim in this file** that 134 shiny sprites were
    "unpaired novel silhouettes". They were never unpaired — they are the
    animated `image-5`/`image-6` partners, found only by index, not overlap.
- **`exclude_shiny=True` discards 148 genuine pose variants** along with the 705
  duplicates it is meant to remove. Different animation frames are different
  silhouettes of the same subject, which is plausibly useful training signal and
  is not a duplicate in any sense that matters to a silhouette classifier. This
  is the basis of the 148-pose-variant experiment in "Next session".
- Raw sprite size is a near-perfect generation proxy (56×56 Gen 1 … 128×128
  Gen 6+) — a shortcut to keep out of preprocessing.
