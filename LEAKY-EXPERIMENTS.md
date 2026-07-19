# Leaky-era experiments (SUPERSEDED — tombstone)

> **Every accuracy this file ever recorded is invalid as a measure of
> generalization.** The dataset included shiny sprites — recolours whose
> silhouettes are pixel-identical to the normal series after thresholding —
> so 32.7% of images were exact duplicates and ~62% of every validation set
> had a twin in train. The headline 0.906 re-measures as **0.617** on
> deduplicated data. See [EXPERIMENTS.md](EXPERIMENTS.md) for everything
> current.

The full 50-run leaky log (Phases 1-11, commands and per-run numbers) lives in
git history for this file. What is kept here is the part that must not be
re-inherited:

## What this file got wrong

| claim | why it is wrong |
|---|---|
| "0.906 ± 0.007 is the strongest validated config" | ~62% of val was memorized. Honest score: **0.617**. |
| "tight seed variance means it isn't a lucky split" | Binomial SE at n=217 is ±0.020; an observed 0.007 is *below the noise floor*. Leaked images answer identically every seed. Stability was evidence *of* the leak. |
| "`lastN=3` has the smallest train/val gap (0.279), so it is well-regularized" | The gap was compressed mechanically — `val_loss` partly re-measured `train_loss`. Honest gap ~0.80. |
| "resnet50 + shape-biased beats resnet18" | Selected in a regime that rewarded memorization. Retested clean: shape-biased *loses* by 5.7pt at 4.0× SEM. |
| "`epochs` is flat past 16" | Measured on leaky val; clean `val_loss` was still falling at 16. |
| "OFAT: anything >3-4pt is real signal" | Derived from the artificially suppressed 0.007 spread; honest threshold ~±8pt at n=217, so nearly every OFAT verdict was noise. |
| "lastN=4 and lastN=5 are different depths" | Identical configurations (`bn1`'s affine params are re-frozen after unfreezing); their 1.9pt gap directly measures run-to-run noise. Same for lastN 6 vs 7. |

The one finding that survived: the `backbone_lr=4e-4` collapse (−49.8pt) —
an effect too large to be a sampling artifact, later tied to the absence of
warmup.
