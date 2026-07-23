# Experiment log

> # PROJECT CLOSED — 2026-07-23
>
> **Final: 0.8325 top-1 / 0.9442 top-5** on the 197-image held-out test split
> (Wilson 95% CI [0.774, 0.878]) — a 15-model fold×seed ensemble, mean softmax,
> 6-view TTA. **The test split is spent**; nothing further may be tuned against
> it. See Phase 6 for the full report.
>
> **How it was won**, from the clean 0.596 baseline:
>
> | lever | gain | confirmed by |
> |---|---|---|
> | ImageNet weights | +5.7pt | — |
> | SDT input channel | +2.4pt | 4 paired seeds |
> | cosine + warmup + restore-best | **+5.1pt** | 2 seeds |
> | full backbone unfreeze (lastN 3→6) | **+1.68pt** | 3-seed battery, p=0.003 |
> | 6-view TTA | +1.38pt | 15 paired folds, p=0.0001 |
> | 15-model ensemble | **+4.74pt** | one-shot test |
>
> **What the project actually taught**, which is not in the accuracy number:
>
> 1. **The first number was a lie, and finding that out was the whole game.**
>    0.906 was duplicate leakage; the honest baseline was 0.596. Every
>    methodology rule in CLAUDE.md exists because of that 31-point correction.
> 2. **Single-run resolution is ~3.4pt, and most "wins" are noise.** Aspect crop
>    read +1.89pt at seed 42 and died at 3 seeds (+0.78pt, p=0.28). Weight decay
>    read +0.99pt and died (−0.18pt, p=0.74). The pose-variant data lever read
>    +2.1pt and died (+0.81pt, p=0.28). **The 3-seed paired battery caught every
>    one of them.** Without it this log would contain three phantom improvements.
> 3. **The nulls localized where the signal is.** A classical shape floor at
>    0.285 and a frozen DINOv2 probe at 0.618 bracket the problem: gross shape is
>    worth little, generic visual features get most of the way, and the last 13pt
>    is task-specific fine structure that only fine-tuning finds.
> 4. **Two "obvious" mechanisms were backwards.** A trainable stem does *not*
>    learn to replace hand-designed input channels (single-channel stem, −1.17pt).
>    And an out-of-distribution TTA view (hflip, which the model never trains on)
>    *helps*, because TTA pays for error decorrelation rather than per-view
>    quality — null alone at 50% weight, decisive when diluted across six views.
> 5. **Where the model still fails is structural, not perceptual.** Silhouette
>    collisions stopped mattering entirely under ensembling (1.02× lift, zero
>    errors above IoU 0.9, electrode/voltorb solved), while evolution-line
>    confusions survived untouched at 13× chance. The remaining errors are
>    Pokémon that genuinely look like their own evolutions.
>
> **If anyone picks this up again**, the live leads are: evolution-line
> confusions (the one unbeaten failure mode), and the two ensemble findings
> generated on spent test data — unanimous members are 100% accurate, and
> confidence is sharply calibrated — which are **hypotheses needing fresh
> measurement**, not results. Untouched checklist items (optimizer sweep, edge
> redux, channel-position, SDT re-confirmation) are all low-prior and must be
> measured against the post-split 0.7643 reference, not the older 0.7604/0.7586.

Everything here runs on the deduplicated dataset (1307 images, 151 classes,
shiny recolours excluded) and is scored by 5-fold grouped CV after a 15% test
split is carved out: **`oof_accuracy` pooled over 1110 out-of-fold predictions,
compared on `fold_accuracy_sem`**. Full per-phase logs live in git history;
the pre-dedup era is summarized in [LEAKY-EXPERIMENTS.md](LEAKY-EXPERIMENTS.md)
and none of its numbers are comparable to anything here.

> **Reference lineage.** Three reference numbers appear below and they are *not*
> interchangeable: **0.7496** (lastN 3, pre-depth-win), **0.7604**
> (`p10-lastn6-s*`, post-depth-win), and **0.7643** (`p14-ref-extsplit-s*`,
> post-on-disk-split — same config, different fold composition). Compare only
> within a lineage.

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
- A no-flag `--folds 5` run reproduces the current reference config
  (`p7-ref-26-s42`, the defaults at seed 42 and 26 epochs, ~32 min). Changing
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
| how deep to fine-tune? | ⚠ **superseded** — the "plateau across lastN 2/3/4" was a distorted-fold artifact. On corrected folds depth is monotonic; see the "depth re-sweep" row. (Original: lastN 5/7 not distinct — true, because lastN 6 = full unfreeze.) | c1-lastN* |
| depth re-sweep (lastN 2/4/6)? | **confirmed win — default raised 3 → 6 (full unfreeze).** 3-seed battery vs `p7-ref-26-s*` (lastN 3): lastN 2 **−2.91pt** (all 15 folds down), lastN 4 **+0.99pt** (borderline, t=1.72 p=0.11), **lastN 6 +1.68pt** (t=3.59 **p=0.003**, McNemar +56 **p=0.005**, all 3 seeds +, 13/15 folds +) — clears the 2× bar. Depth is monotonic 2≪3<4<6; lastN 6 unfreezes all 6 feature blocks (conv1,bn1,layer1-4), so it *is* full fine-tuning and lastN 7+ ≡ 6. Stable only because Phase 2's warmup landed first (deep unfreezing was unstable before). Biggest model-side gain since the LR schedule; new reference **0.7604** (3-seed mean) | p10-lastn{2,4,6}-s* |
| re-tune blr/wd for full unfreeze? | **no — the lr/wd region is validated-null; `blr 4e-4 / wd 2e-3` survive the depth regime change.** Roadmap's #1 open lever after the depth win, since the LR was tuned at lastN 3. Seed-42 centered cross + diagonal (5 arms) around the reference: **backbone_lr settled** — both neighbors worse (1.5e-4 −1.6pt, 1e-3 **−2.9pt / >2× SEM**), 4e-4 is a flat top (a higher LR damages the now-trainable stem/layer1). **weight_decay showed only a seed-42 mirage**: wd 5e-4 was +0.99pt (0.74× SEM) and wd 8e-3 flat, so wd 5e-4 went to a 3-seed paired confirmation vs `p10-lastn6-s{42,43,44}` — and washed out: **−0.18pt mean** (+0.99/−1.08/−0.45), 15-fold paired t=−0.34 **p=0.74**, McNemar 169 fixed / 175 broke net −6 **p=0.79**. The "full unfreeze overfits → wants more regularization" hypothesis is dead (higher wd flat, lower wd null). Defaults stand; region closed | p12-lrwd-{A,B,C,D,E}-s42, p12-lrwd-C-*-s{43,44} |
| backbone size? | not the axis (50 > 18 > 34); weight origin was the confound | c2-* |
| label smoothing hurting? | no — removing it costs 3.5pt; 0.05–0.2 flat; the train/val gap responds to it but accuracy doesn't → **gap is not predictive** | c3-ls* |
| does added augmentation help? | no; elastic is significantly harmful (−4.9pt, 2.1× SEM) on contour-only input | c7-* |
| does removing augmentation help? | no; nothing clears the bar, scale jitter removal moves nothing → size cue not load-bearing; augmentation closed in both directions | n1-* |
| is the error irreducible? | no — silhouette collisions explain ~3% of errors (electrode/voltorb IoU 0.969 is the max); evolution-line confusions are 12.7% of errors at 15× chance | confusion study |
| mask polarity? | **resolved — default confirmed.** 3-seed paired battery on index folds: inverting the mask is **−1.50pt mean** (−2.16/+1.26/−3.60), 15-fold paired t=−2.00 p=0.066, **McNemar net −50 (183 fixed / 233 broke), p=0.016 — significant at image level**. Under the conservative 2×-OOF-SEM bar (2.68pt) but real-signed and significant per McNemar, and firmer than the old single-seed −1.27pt. Default `invert_mask=False` (background=1) stands | p9-invmask-26-s*, was p5-invert-mask-32 |
| SDT input channel? | **consistent, not independently confirmed**. Original: +1.5pt over 4 paired seeds (t=2.86, p=0.01). On corrected folds, one paired seed gives **+1.08pt at 0.64× SEM** (3/5 folds, paired t=0.789 p=0.47; McNemar 66 fixed / 54 broken, p=0.32). That sits inside the original per-seed range (+0.4…+2.6), so it replicates in direction and magnitude but cannot confirm alone — **a +1.5pt effect is not detectable in a single run** (2× SEM ≈ 3.4pt here). Default stands; full re-confirmation needs 3 more paired seeds (~3h) | n2-*sdt*, p6-ref-26 vs p6-allmask-26 |
| curvature proxy channel? | dead — too sparse (1-3px slivers) to survive the stem's 4× downsample; diagnosis motivates "edge" and the stem phase | n2-curv, n2-sdt-curv |
| edge channel? | retired — dilutes SDT (−2.6pt, 5/5 folds); alone +0.65pt at 0.4× SEM | p1-* |
| single-channel stem (does a trainable stem replace the hand-designed channels)? | **no — the 3-channel encoding is load-bearing even at full unfreeze.** The sharpest test of the input-encoding reopen: with conv1/bn1/layer1 now trainable, can the stem learn from a raw mask what `(mask,sdt,mask)` supplies? `conv1` is rebuilt to 1 input channel, re-seeded by **summing the pretrained RGB filters** (`model.adapt_input_channels`; `input_channels` is now variable-length). 3-seed paired batteries vs `p10-lastn6-s*`: **mono-mask −1.17pt** (−1.08/−1.80/−0.63, 15-fold t=−2.12 p=0.052, McNemar 177 fixed / 216 broke net −39 p=0.055 — under the 2× bar but real-signed, all 3 seeds negative) and **mono-sdt −3.18pt** (t=−5.56 **p=0.0001**, McNemar net −106 **p<1e-5**, 13/15 folds negative — clears the bar). Ordering **3-channel > mono-mask > mono-sdt**: raw mask is the better single channel by ~2pt, so if forced to one, keep the crisp binary boundary. **SDT is a complement, not a substitute** — the confirmed-useful *added* channel is the *worst* one alone, carrying value only next to a sharp mask edge. Phase 1's "fill it" stands; "drop the redundant capacity" refuted. Default unchanged | p13-mono-{mask,sdt}-s* |
| LR schedule? | **+5.1pt confirmed at 2 seeds**: cosine+warmup+restore at blr 4e-4; the historical 4e-4 collapse was a warmup artifact. Effect is larger than the fold artifact, so likely robust — but see the horizon row, which did not survive | p2-* |
| epoch horizon? | **resolved — flat, default flipped 32 → 26.** 3-seed paired battery on index folds: 32ep is **+0.45pt over 26ep** (+1.71/+0.63/−0.99), 15-fold paired t=0.83 p=0.42, McNemar net +15 p=0.40 — indistinguishable. The seed-42 +1.71pt washed out across seeds (the single-seed mirage again). Per the pre-registered rule, **`epochs` default lowered to 26** for ~20% cheaper runs at the same accuracy | p9-ep32-s*, p7-ref-26-s* |
| higher backbone LR? | no — blr 8e-4 is +0.4pt at 0.3× SEM; the LR ceiling does not extend past 4e-4. Phase 2 schedule residue closed on this axis | p5-blr8e4 |
| duplicate-mask hypothesis? | no — `(sdt, mask, sdt)` is −1.2pt at 0.6× SEM; the second raw mask copy in `(mask, sdt, mask)` is not load-bearing. Phase 1's open question resolved | p5-dupmask-sdt-mask-sdt |
| does more data help (leak decomposition)? | **plausible, unconfirmed**: training on the full unfiltered set scores 0.7802 on the normal-series subset vs 0.7595 step-matched control — **+2.1pt at 1.2× SEM**, below the bar, and *unpaired* (different fold structures). Was +2.9pt under IoU grouping; ~0.8pt of that was animation-frame leakage. Does not justify the all-gen scrape yet | p5-leak-decomposition, p6-leak-idxgroup |
| does more data help (pose variants, clean test)? | **no — data lever closed.** Adding the 148 animated-frame pose variants to training (truly paired: same scored images, 0 leakage) gives **+0.81pt mean over 3 seeds** — all three positive (+1.35/+0.90/+0.18) but pooled 15-fold paired t=1.11 p=0.28, McNemar +27 net p=0.20, **well under the 2× bar**. This is the clean version of the leak-decomposition question (no leakage, no step-count confound), so the +2.1pt there was mostly non-reproducible: soft-leakage + budget, not novel-data value. Two independent angles now say scraping more data is a weak lever; `exclude_shiny=True` stands. `include_pose_variants` flag kept for the record | p7-ref-26-s*, p7-pose-26-s* |
| reduced-stride stem? | no — nomaxpool −0.8pt at 0.4× SEM for 2.8× compute; stride1 gated off; thin-feature hypothesis retired | p3-nomaxpool |
| frozen DINOv2 features (Phase 4)? | **transfers well but doesn't beat fine-tuning: 0.618 OOF, ~13pt under the CNN.** Frozen DINOv2 ViT-L/14 @ 518 (CLS+meanpatch, 2048-d) → shallow probe, same seed-42 split as `p7-ref-26-s42`. Best is standardized **logistic regression 0.6180** (top5 0.804, test 0.675); raw silhouette input beats the (mask,sdt,mask) encoding (0.618 vs 0.589 — SDT is OOD for DINOv2's ImageNet normalization, the channel that helps the trainable CNN hurts frozen features). **+33pt over the classical floor, −13pt vs the CNN's 0.7496** (~5× SEM, solid). **Classifier choice swings ~15pt**: RF (the floor's winner) is *worst* here (0.474), std+logreg best — dense 2048-d embeddings need a linear probe, not axis-aligned trees. Caveat: frozen+linear vs fine-tuned, so not a clean pretraining comparison; end-to-end DINOv2 fine-tuning is the unrun ceiling test | dinov2L518-{silhouette,msm}-logreg-s42 |
| fine-tuned CNN as a frozen extractor (cross-fit probe)? | **no — its accuracy doesn't survive decomposition into frozen features + fresh classifier: 0.534 OOF, −22pt under its own head, and *below* the DINOv2 frozen probe.** Cross-fit to stay leak-free: per fold, train the reference CNN and cache the 2048-d post-avgpool feature only for that fold's held-out val (`scripts/cnn_feature_probe.py`), so every feature comes from a backbone that never saw it. Shallow-classifier sweep on the pooled OOF features, same seed-42 split as `p10-lastn6-s42`. Anchor validates the pipeline: the CNN's own head reproduces the reference exactly (**0.7586**). Best probe is **logreg 0.5342** (top5 0.800), svm_rbf 0.459, RF 0.446 — all far under the 0.759 head, and under DINOv2's 0.618 single-frozen-backbone probe. The gap is **cross-backbone misalignment, not representation quality**: the CNN's head wins because it is co-trained with *its* backbone's coordinate frame, whereas pooling features from 5 independently-fine-tuned backbones and fitting one linear boundary eats ~22pt to frame mismatch. Unlike DINOv2 (one frozen backbone, one frame) a fine-tuned-CNN probe *cannot* be clean — any backbone memorizes whatever train set you'd fit on, so cross-fit's confound is inherent, not a design slip. top5 holding at 0.80 says the right class stays near in each frame; it's the shared linear boundary that fails. ~20× SEM, so seed-robust; a diagnostic, not a config change | cnnprobe-resnet50-lastn6-logreg-s42 |
| classical shape-descriptor floor? | **~0.285 OOF — the CNN wins by ~46pt.** Normalized elliptic Fourier (20 harmonics) + log Hu moments + 6 dimensionless ratios → shallow classifier (random forest best of logreg/SVM/RF), same seed-42 split as `p7-ref-26-s42` (byte-identical OOF set). Orientation-preserving EFD (canonical sprite pose kept as signal) beats fully rotation-invariant by +1.8pt OOF / +7.6pt test: **0.2847 OOF / 0.3807 test** vs 0.2667 / 0.3046. ~43× chance, so global shape carries real signal, but the CNN's 0.7496 comes overwhelmingly from learned local/fine structure, not gross silhouette form. Gap ~30× the fold SEM → one seed settles it. Sets the floor any silhouette-native architecture must clear decisively | sd-efd{inv,orient}-random_forest-s42 |
| does filling the canvas help (aspect crop)? | **no — the top model-side lever closed too.** Bbox-crop + pad to fill the 224 canvas (occupancy ~25% → near-full), truly paired 3-seed vs `p7-ref-26-s*`: **+0.78pt mean** (+1.89/+0.99/−0.54), 15-fold paired t=1.13 p=0.28, McNemar 250 fixed / 224 broke net +26 p=0.25 — **under the 2× bar (1.95pt)**. s42 alone was +1.89pt (reads as a clear win); the 3-seed battery caught it — the power-check working. Third lever to land ~+0.8pt sub-bar after pose variants and leak-decomposition, and the first that is purely architecture/framing rather than data. `aspect_crop` flag kept, off by default | p8-crop-26-s* |

**Current reference: 0.7586** (`p10-lastn6-s42`, the config defaults at seed 42
on index-grouped folds — **pair every new experiment against `p10-lastn6-s*`**).
The default became lastN 6 (full unfreeze) on 2026-07-22; the previous lastN-3
reference was 0.7496 (`p7-ref-26-s42`). 3-seed spread now **0.7586 / 0.7676 /
0.7550, mean 0.7604**. The old 0.716/0.725 was the same config on distorted
folds — **no model change accounts for that difference**; see the fold-correction
note. Every `p7-ref-26-s*` run is still valid as the *lastN-3* baseline the depth
battery was paired against, just no longer the default.

Reference table, all seed 42, all `(mask, sdt, mask)` + cosine/warmup/restore
at blr 4e-4:

| run | grouping | epochs | lastN | oof | SEM |
|---|---|---|---|---|---|
| `p5-ref-32` | IoU (superseded) | 32 | 3 | 0.7586 | 0.0154 |
| `p7-ref-26-s42` | index | 26 | 3 (old default) | 0.7496 | 0.0155 |
| `p7-ref-26-s43` | index | 26 | 3 | 0.7387 | — |
| `p7-ref-26-s44` | index | 26 | 3 | 0.7423 | — |
| **`p10-lastn6-s42`** | **index (current)** | **26** | **6 (default)** | **0.7586** | — |
| `p10-lastn6-s43` | index (current) | 26 | 6 | 0.7676 | — |
| `p10-lastn6-s44` | index (current) | 26 | 6 | 0.7550 | — |

**Pair new experiments against `p10-lastn6-s*`** (the current default). The
`p7-ref-26-s*` runs remain the correct paired baseline for anything measured
*before* the depth change, and the lastN-3 arm the depth battery used.

The three index-grouped default seeds give the honest baseline spread: **0.7586 /
0.7676 / 0.7550, seed range ~1.3pt** — consistent with the ~±1pt seed variance
noted in the protocol, and the reason single-seed comparisons at this effect
size cannot resolve anything under ~3pt.

**Single-run resolution is ~3.4pt.** Combined SEM against this reference is
~0.017, so the 2× bar is ~3.4pt — larger than *most* effects in the table
above. Any hypothesis worth under ~3pt needs paired folds across multiple
seeds, not one run per arm; `p6-ref-26` fold accuracies alone span 0.7207 to
0.7973. Budget accordingly when planning an experiment, or it cannot answer
the question asked of it.

Progression of confirmed gains (mixed fold regimes, so read as trend not
ledger): 0.596 → 0.653 (ImageNet weights) → 0.677 (SDT channel) → 0.716
(schedule/LR) → 0.7586 (same config, folds corrected) → **0.7604 (full unfreeze,
lastN 3 → 6; 3-seed mean, index folds)**.

What the *data/representation* failures collectively point at: the bottleneck
there is not data variety or task ambiguity — it is how much discriminating
information reaches (and survives) the network. **But the depth win (2026-07-22)
qualifies this**: full unfreeze at +1.68pt shows trainable capacity / optimization
depth *was* a real lever, once Phase 2's warmup made deep fine-tuning stable.
So the picture is now two-part — representation levers are largely exhausted, but
the optimization/fine-tuning axis has live headroom (depth, and its follow-ups:
LR/weight-decay re-tuning and BN affine for the full-fine-tune regime).

A pattern that held across the *early* confirmed gains: they were **model-agnostic
tooling** — data hygiene (dedup/grouped CV), pretrained-weight origin, input
encoding and polarity, LR schedule and horizon — none touching the architecture,
all transferable to whatever backbone wins. The depth win is the first that is
model-*specific* (how much of ResNet-50 to fine-tune), and it is the largest
single accuracy gain since the LR schedule — evidence the model-side region the
fold correction had frozen is worth mining.

Also measured, and load-bearing:

- **Models are high-variance**: five ~equal configs agree on only 75-82% of
  predictions; 230 of 1110 images are solved by no config, 880 by at least one.
  Real ensemble headroom — deliberately deferred to Phase 6. Refreshed on the
  three 26-epoch reference seeds (`scripts/seed_agreement_study.py`, over the 797
  images out-of-fold in all three — the seeds don't share a split): pairwise
  top-1 agreement 70-73% (when two agree, ~91% correct); all three agree on 63%
  (95% correct there); 60% solved by all three, 14.7% by none; **85.3% solved by
  ≥1 seed vs 74.2% best single** — the ~11pt ensemble headroom, still there.
- **Why silhouettes get confused** (same script, 631 pooled errors, each factor's
  rate vs its chance baseline): **same evolution line is the one sharp driver —
  10.9% of errors at 13.1× chance**; similar shape (IoU>0.8) 2.8×, shares a type
  2.3×, similar size only 1.4× (confused pairs' mean occupancy gap 1.27× closer
  than random). Family/type/shape tags together explain just 43.7% of errors; the
  other **56% are collisions between unrelated Pokemon** whose renderings happen
  to overlap — reinforcing the floor study's read that the residual is fine local
  structure, not gross shape/type/size categories. Size being the weakest factor
  matches N1 and the aspect-crop null.
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

## Roadmap as of 2026-07-22 — SUPERSEDED, kept for history

**This section is closed.** It was the live roadmap up to the day before the
project finished; everything it schedules has since been run and resolved (the
optimizer follow-ups came back null, the input-encoding re-exploration's sharpest
question came back negative, and Phase 6 was executed). Read the banner at the top
of this file for the final state, and Phase 6 for the result. Kept because the
reasoning about *which* levers looked promising, and why most of them were wrong,
is the more useful record.

The data and framing levers are closed, the two weakened ⚠ rows are resolved,
and a from-scratch floor is now on the board. Recent arc (2026-07-20 → -07-22):
the fold-correction session re-established the baseline (`p6-ref-26`, 0.7496) and
replicated SDT; the pose-variant test **closed the data lever** (+0.81pt, 3
seeds, sub-bar); the aspect-preserved crop — the roadmap's former #1 "most likely
real win" — **also came back null** (`p8-crop-26-s*` vs `p7-ref-26-s*`, +0.78pt
mean, paired t p=0.28 / McNemar p=0.25); **mask polarity confirmed** the default
(inverting −1.5pt, McNemar p=0.016) and the **epoch horizon is flat** (32 vs 26
+0.45pt p=0.42) so the `epochs` default is now 26. A **classical shape-descriptor
floor** was measured at **~0.285 OOF** (46pt below the CNN): global shape carries
real but limited signal, so the network's win is learned local/fine structure.
Phase 4 was then run as a **DINOv2 frozen-feature probe** (not the sketch swap)
and **closed**: 0.618 OOF, +33pt over the floor but −13pt vs the CNN — strong
frozen transfer, but fine-tuning still wins. Then the **depth re-sweep landed the
biggest model-side win since the LR schedule**: full unfreeze (**lastN 3 → 6**)
is **+1.68pt confirmed** (p10, p=0.003 / McNemar p=0.005, all 3 seeds), and the
old "depth plateau" was a distorted-fold artifact — the `train_last_n_layers`
default is now 6 and the reference is **0.7604** (`p10-lastn6-s*`). That reopens
the depth/optimizer region the fold correction had frozen. The two highest-prior
full-unfreeze follow-ups are now **both closed as null**: BN affine (`p11-bnaffine-s*`,
−0.30pt, p=0.61 — conv layers already trainable) and the **blr/wd re-tune**
(`p12-lrwd-*`, see the results row — LR settled at 4e-4 with both neighbors worse,
wd null at 3 seeds p=0.74; the LR schedule tuned at lastN 3 survived the regime
change). So the optimizer/regularization side of the reopened region is spent.
In priority order:

1. **Input-encoding re-exploration** — the conceptually strongest reopen, now
   **partly answered and looking weaker than hoped**. Full unfreeze moved the
   freeze boundary to the input: at lastN 3 the stem (conv1/bn1) and layer1 were
   frozen ImageNet filters, so *every* Phase-1/3 channel/stem finding was a
   statement about a fixed natural-image stem — a constraint now gone. The
   sharpest question is **closed negative**: the single-channel stem loses
   (mono-mask −1.17pt, mono-sdt −3.18pt), so a trainable stem does *not* learn to
   replace the hand-designed channels and `(mask,sdt,mask)` is load-bearing.
   That lowers the prior on the rest of the cluster, which now reads as
   "close the checklist, expect a null": **edge channel redux** and
   **channel-position/dup-mask** (`(sdt,mask,sdt)`). Down-weight curv (its
   failure was stem *downsampling*, not frozen weights — a trainable stem doesn't
   add resolution). Multi-seed paired vs `p10-lastn6-s*`.
   - Worth folding in here: **re-confirm SDT itself** (currently "consistent, not
     independently confirmed" on corrected folds; needs 3 paired seeds). The p13
     result raises its prior — SDT alone is the *worst* single channel, so its
     value is specifically as a complement, which is what the +1.5pt claims.
2. **Remaining Phase 5 items**: optimizer (AdamW never swept against SGD/Adam,
   though SGD needs its own LR), bigger/deeper backbone (the depth-of-fine-tuning
   monotonicity raises the prior that raw capacity now pays — resnet101, but run
   *with* awareness of overfitting at n=1110). Cheap, likely sub-resolution
   — multi-seed paired batteries or accept a null.
   - **Silhouette-native architecture** is the one genuinely orthogonal idea
     (contour-sequence / point model over the boundary, vs the raster CNN).
     It must clear the ~0.285 descriptor floor decisively *and* approach the
     0.7496 CNN to matter; treat it as a research probe, not an expected win.
   - **End-to-end DINOv2 fine-tuning** is the one Phase-4 follow-up left on the
     table (frozen probe got within 13pt): fine-tune last-N blocks of a ViT-S/B
     on the same split. Higher prior than the sketch swap but needs ViT backbone
     integration and overfits easily at n=1110 — parked, not scheduled.
3. **Phase 6 last**, and not until the config stops moving: ensembling
   amplifies whatever config it is handed. The config just moved (depth), so
   the gate is not open.

Cleared, no longer worth running: `--backbone-lr 8e-4` (null), `(sdt, mask,
sdt)` duplicate-mask hypothesis (null), the leak decomposition (superseded by
the clean pose-variant test), the 64-epoch horizon (pointless — 32 already
buys nothing over 26), **data expansion / all-gen scraping** (the data
lever is closed: pose variants gave +0.81pt at 3 seeds, and that is the clean
upper bound on novel-silhouette value at this scale), and the **aspect-preserved
crop** (`aspect_crop`, null at +0.78pt / 3 seeds — the top model-side lever, now
closed; flag kept off by default).

State at session end (2026-07-22): reference **0.7586** (`p10-lastn6-s42`,
defaults — now lastN 6 / full unfreeze — at seed 42; 3-seed spread
0.7586/0.7676/0.7550, mean **0.7604** — the number to pair against, via
`p10-lastn6-s*`). The lastN-3 reference was 0.7496 (`p7-ref-26-s42`), still the
valid baseline for pre-depth comparisons. **Default changes this session**:
`epochs` 32 → 26 (horizon flat), `train_last_n_layers` 3 → 6 (full unfreeze,
+1.68pt confirmed — the depth win). Mask polarity confirmed the default. The
pose-variant (`include_pose_variants`) and aspect-crop (`aspect_crop`) flags are
off by default (null results). Diagnostics on `main`: the classical shape floor
(`scripts/shape_descriptor_baseline.py`, ~0.285), the DINOv2 frozen probe
(`scripts/dinov2_probe.py`, 0.618), and the seed-agreement / confusion study
(`scripts/seed_agreement_study.py`). **Next movers**: the full-fine-tune
follow-ups the depth win reopened — BN affine on top of full unfreeze, and
re-tuning `backbone_lr` / `weight_decay` for the full-fine-tune regime (LR was
set at lastN 3). Budget as 3-seed paired batteries vs `p10-lastn6-s*`.

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
`restore_best_epoch=True`, `backbone_lr=4e-4`, `epochs=32`. Untested residue for
Phase 5: warmup length, higher LRs (8e-4), longer horizons (64).

> **Horizon superseded, 2026-07-21.** The 3-seed `p9-ep32-s*` battery on index
> folds found 32ep only +0.45pt over 26ep (paired t p=0.42, McNemar p=0.40) —
> flat. The `epochs` default is now **26** (~20% cheaper per run); the "horizon
> is genuinely used" bullet above was measured on the superseded IoU folds.

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

## Phase 4 — Non-ImageNet / domain-matched representations

Original plan was a sketch/QuickDraw checkpoint swap. That was **deprioritized**
in favour of a DINOv2 frozen-feature probe (2026-07-22): the sketch idea is
low-prior (its precedent, the shape-biased Stylized-ImageNet checkpoint, lost
5.7pt; sketches are strokes, silhouettes filled regions) *and* blocked on
sourcing a credible ResNet-50 checkpoint. DINOv2 answers the more general
"does a strong non-ImageNet representation transfer to silhouettes?" with no
sourcing problem.

- [x] **dinov2-frozen-probe** — `scripts/dinov2_probe.py`, frozen DINOv2
      ViT-L/14 @ 518 (CLS+meanpatch) → shallow probe on the seed-42 split.
      **0.618 OOF** (std+logreg), +33pt over the classical floor but −13pt vs
      the fine-tuned CNN (0.7496). Raw silhouette input > (mask,sdt,mask) for
      frozen features (SDT is OOD for DINOv2's normalization). RF is the wrong
      probe for 2048-d embeddings (0.474, worst) — a ~15pt classifier swing vs
      std+logreg. See the results-table row. A logreg scaler×C grid on both
      caches confirmed the probe choice: silhouette peaks at std+logreg C≈1
      (0.618), msm at none/std+logreg C≈1 (0.599, silhouette still wins); L2-norm
      is a trap for logreg (collapses to ~5% at low C, needs C≈100). Well-chosen
      linear probes all land ~0.59-0.62; the classifier, not more tuning, was the
      lever.
- [x] **cnn-crossfit-feature-probe** — `scripts/cnn_feature_probe.py`, the
      fine-tuned-CNN analog of the DINOv2 probe: is the ResNet-50 a good frozen
      extractor, and can a shallow head on its penultimate features beat its own
      softmax? **No.** Cross-fit (train the reference CNN per fold, cache the
      2048-d post-avgpool feature only for that fold's held-out val) keeps every
      feature leak-free; the shallow sweep on the pooled seed-42 OOF set peaks at
      **logreg 0.5342** (top5 0.800) vs the head's **0.7586** (which reproduces
      `p10-lastn6-s42` exactly — pipeline validated). The −22pt is **cross-backbone
      misalignment**, not representation quality: the head is co-trained with its
      own backbone's frame, while pooling 5 independently-fine-tuned backbones'
      features under one linear boundary pays for frame mismatch — hence the probe
      even falls *below* DINOv2's single-frozen-backbone 0.618. The asymmetry is
      the point: unlike a frozen ViT, a fine-tuned CNN has no clean, matched,
      large-n probe (any backbone memorizes whatever you'd fit on), so the confound
      is inherent. Takeaway: the CNN's accuracy lives in the jointly-trained
      backbone+head pairing and does not decompose into transferable frozen
      features. See the results-table row. Diagnostic, not a config change.
- [ ] **p4-sketch-checkpoint** — parked; only revisit if a credible ResNet-50
      sketch/QuickDraw checkpoint surfaces.
- **Open ceiling test**: the probe is frozen+linear, so it is *not* a clean
  pretraining comparison. End-to-end **fine-tuning DINOv2** (last-N blocks on a
  ViT-S/B to avoid overfitting 1110 images) is the unrun experiment that would
  say whether a stronger backbone can beat the ResNet-50 — heavier, real prior.
- The stronger domain-matched alternative if pursued: pretrain on all-gen
  Pokémon silhouettes — but the data lever is closed (pose variants +0.81pt),
  so this is low-prior now.

## Phase 5 — Backlog: everything never rigorously tested

In rough value order; each is cheap and uses whatever config Phases 1-4 settle:

- [ ] **leak decomposition** — train on the full unfiltered set, validate only
      on twin-free images; separates "measurement was wrong" from "we halved
      the data", and gates data expansion (the strongest untested lever).
- [x] **depth re-sweep** (lastN ∈ {2,4,6}) — **done, the win of the session:
      default raised 3 → 6 (full unfreeze), +1.68pt confirmed** (p10 battery; see
      the results row). Depth is monotonic on corrected folds; lastN 6 is the max
      (= full feature unfreeze). Follow-ups now: BN affine on top of full unfreeze,
      and re-tuning blr / weight_decay for the full-fine-tune regime (the LR was
      set at lastN 3).
- [ ] **optimizer** — AdamW was assumed, never swept; SGD+momentum, Adam.
- [x] **weight decay, BN affine** — both **done, null**. BN affine −0.30pt at 3
      seeds (`p11-bnaffine-s*`, p=0.61). Weight decay swept in the `p12-lrwd-*`
      cross: lower wd (5e-4) was a seed-42 mirage (+0.99pt) that washed to −0.18pt
      over 3 paired seeds (t p=0.74, McNemar p=0.79); higher wd (8e-3) flat.
      Paired with the LR result (4e-4 settled, both neighbors worse, 1e-3 −2.9pt
      >2× SEM), **the full-unfreeze lr/wd region is validated-null — defaults
      stand.** See the results-table row.
- [x] **single-channel stem** — **done, negative; "fill it" wins.** Implemented
      as designed (sum the pretrained RGB filters into a 1-channel conv1,
      `model.adapt_input_channels`; `input_channels` generalized to variable
      length, default path byte-identical). 3-seed paired batteries: mono-mask
      **−1.17pt** (p=0.052, all seeds negative, under the 2× bar but real-signed),
      mono-sdt **−3.18pt** (p=0.0001, clears the bar). A trainable stem does *not*
      learn to replace the hand-designed channels, and SDT alone is the worst
      option despite being the useful *added* channel — it complements a sharp
      mask edge rather than substituting for it. See the results-table row.
- [ ] **duplicate-mask hypothesis** — is the second raw mask copy in the
      winning `(mask, sdt, mask)` load-bearing? Channel-position swaps,
      e.g. `(sdt, mask, sdt)`. See the Phase 1 pattern note.
- [x] **aspect-preserved crop** — **null, closed.** Bbox-crop + pad to fill the
      224 canvas (`aspect_crop` in `data.py`, polarity-robust), applied at train
      and eval alike. Lifts occupancy from mean 24.9% (range 7.2-48.5%) to
      near-full, and removes absolute size (the generation proxy N1 measured as
      ~useless). 3-seed paired battery `p8-crop-26-s*` vs `p7-ref-26-s*`
      (fold-identical): **+0.78pt mean** (+1.89/+0.99/−0.54), paired t=1.13
      p=0.28, McNemar 250 fixed / 224 broke net +26 p=0.25 — under the 1.95pt
      bar. Same sub-bar ~+0.8pt as the pose-variant and leak-decomposition
      results. Flag kept, off by default.
- [ ] **backbone re-check** — resnet18 vs 50 was 1.9× SEM, just under the bar.
- [x] **classical shape-descriptor floor** — done, **~0.285 OOF vs the CNN's
      0.7496 (~46pt gap).** `scripts/shape_descriptor_baseline.py`: normalized
      elliptic Fourier + log Hu moments + dimensionless ratios → shallow
      classifier, reusing `fold_indices` so the seed-42 OOF set is byte-identical
      to `p7-ref-26-s42`. Orientation-preserving EFD (canonical pose kept) is the
      better floor (+1.8pt OOF / +7.6pt test over fully rotation-invariant). ~43×
      chance, so global shape has real signal, but the network's win is
      overwhelmingly learned local/fine structure. The gap is ~30× the fold SEM,
      so seed 42 alone settles it. This is the number a silhouette-native
      architecture (contour-Transformer, etc.) must clear decisively to matter.
- [ ] **confusion-study leftovers** — body-plan-aware descriptor, error rate
      by generation (+ generation-held-out split), cross-class near-duplicate
      check, embedding visualization, hard-example contact sheet.

## Phase 6 — Final: ensemble, then the one-shot test evaluation

Ensembling stays out of exploration (it multiplies every experiment's cost and
judges later changes as ensembles, which is not how they'd ship). At the end,
on the settled config:

**PHASE 6 COMPLETE. Final: 0.8325 top-1 / 0.9442 top-5 on the held-out split.**

- [x] **On-disk test split** — `scripts/create_test_split.py` moved the 197
      seed-42 test normals *and their 122 shiny recolours* (319 files) to
      `test_data/`, so the guarantee is physical rather than dependent on
      `exclude_shiny=True`. `config.test_dir` makes all of `data/` the CV pool;
      omitting it now raises rather than silently carving a second split.
      Fold composition changed, so the reference was re-baselined:
      **0.7643** (`p14-ref-extsplit-s*`, 0.7658/0.7577/0.7694), replacing 0.7604.
- [x] **TTA — done, the win.** `scripts/tta_selection.py`, OOF-valid (each fold's
      model on its *own* val fold only). **All six views (identity + hflip +
      rot±10 + rot±20): +1.38pt, 0.7643 → 0.7781, paired t=+5.60 p=0.0001,
      14/15 folds up.** The mechanism is **decorrelation, not per-view quality**:
      rotations are inside the training distribution (`affine_degrees=20`), so
      they correlate with identity and add little alone (4v rot-only +0.57pt,
      p=0.11 — the view-count control that rules out "more views is all that
      matters"). hflip is *out*-of-distribution and null on its own at 50% weight
      (−0.06pt, p=0.86), but its errors are decorrelated, so diluted across six
      views it is what makes the set win. Estimate is selection-optimistic over 7
      comparisons.
- [x] **Fold × seed ensemble on the held-out test split only** — 15 models
      (3 seeds × 5 folds), combined by **arithmetic mean of softmax** (not logits:
      logit-averaging is a geometric mean, so one confidently-wrong member can veto
      a class). **+4.74pt over the mean member on the same images (0.7851 →
      0.8325)** — the gain that was unmeasurable before the spend, and larger than
      TTA's. Per-fold checkpoints come from `--save-model` (`weights/<run>/foldN.pt`).
- [x] **One-shot test evaluation** — `scripts/final_test_evaluation.py`, 197
      never-trained images. **top-1 0.8325 [0.7741, 0.8782], top-3 0.9239,
      top-5 0.9442** (Wilson 95%). 33 errors. The CI is ±5pt: this is one split,
      not a precise number.

**Error studies on the ensemble** (both reused verbatim from the confusion study,
since they are class-level properties):
- **Evolution-line confusions survive ensembling and remain the dominant
  structured error**: 4/33 errors (12.1%) at **13× chance**, essentially unchanged
  from the single model's 12.7% / 15×. (`raichu→pikachu`, `pidgeot→pidgeotto`,
  `kadabra→alakazam`, `dewgong→seel`.)
- **Silhouette collisions have stopped mattering**: confused pairs average IoU
  0.710 vs 0.699 for a random class pair — **1.02× lift, i.e. no signal** — and
  **zero errors at IoU ≥ 0.9**. electrode/voltorb (IoU 0.969, the worst collision
  in the dataset) is now answered correctly. For a single model this explained ~3%
  of errors; the ensemble absorbed that class of mistake.

**Ensemble-only findings — these are NEW HYPOTHESES generated on spent test data,
not validated results. Acting on them requires measuring on OOF or fresh data.**
- **Unanimity is perfect.** All 15 members agree on 50.8% of images (100/197) and
  are **100.0% accurate there**; all 33 errors fall in the 97 disagreement cases
  (66.0% accurate). 
- **Confidence is sharply calibrated and monotonic**: 42.5% → 76.9% → 97.4% →
  100% → 100% across ascending max-probability bins. Above ~0.34 the ensemble is
  essentially never wrong. For the guessing game this is the practical payoff —
  the averaged softmax is a real reliability signal, so the app can tell when to
  commit versus hedge.

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
  silhouettes of the same subject, not duplicates in any sense that matters to a
  silhouette classifier. The `include_pose_variants` flag admits just those 148
  (`_pose_variant_pairs` in data.py). Adding them to training was **tested and
  gave +0.81pt at 3 seeds** — real-signed but under the bar, so the flag stays
  off by default; see the results table. It remains the cleanest measurement of
  novel-silhouette value at this scale.
- Raw sprite size is a near-perfect generation proxy (56×56 Gen 1 … 128×128
  Gen 6+) — a shortcut to keep out of preprocessing.
