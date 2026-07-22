from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
	"""Single source of truth for one training run. Every field is picked up by
	the tyro CLI and logged as an MLflow param; add a field to add a
	hyperparameter. Measured verdicts behind the defaults live in EXPERIMENTS.md.
	"""

	# Data
	data_dir: Path | None = None  # defaults to <project_root>/data when unset
	batch_size: int = 16
	# Both must stay >= 0.15: a smaller split has fewer images than the 151
	# classes and the stratified split raises.
	val_size: float = 0.15
	test_size: float = 0.15
	# Shiny sprites are recolours whose silhouettes duplicate the normal series;
	# keeping them leaks pixel-identical twins across splits.
	exclude_shiny: bool = True
	# Add the 148 animated-frame pose variants (not the 705 exact recolours) to
	# training only, paired to their normal partner's fold. Requires
	# exclude_shiny=True; scored images stay identical to the baseline. Takes
	# effect only in fold mode (folds > 0); the single-split path rejects it
	# rather than silently ignore it. See EXPERIMENTS.md "Known data issues".
	include_pose_variants: bool = False
	# 0 = single split (validation_accuracy). K > 0 = grouped CV pooled over
	# out-of-fold predictions (oof_accuracy); --folds 5 is the reporting standard.
	folds: int = 0
	# Pair each shiny sprite with the normal it recolours (by index) so the two
	# never straddle a fold.
	group_aware_folds: bool = True

	# Augmentations, independently togglable; the RandomAffine below is always on.
	augment_hflip: bool = False
	augment_morphological: bool = False
	augment_resolution_jitter: bool = False
	augment_elastic: bool = False

	# Always-on RandomAffine. Defaults measured locally optimal (N1).
	affine_degrees: float = 20.0
	# Symmetric max translation fraction, applied as (t, t).
	affine_translate: float = 0.2
	affine_scale: tuple[float, float] = (0.85, 1.15)

	# Per-channel input encoding, applied at train and eval alike:
	# "mask" (binary silhouette), "sdt" (signed distance transform),
	# "curv" (morphological curvature proxy), "edge" (thick boundary band).
	# SDT confirmed over four paired seeds (+1.5pt, p=0.01 on 20 matched
	# folds); all-mask was the pre-N2 default. See EXPERIMENTS.md.
	input_channels: tuple[str, str, str] = ("mask", "sdt", "mask")
	# Polarity of emitted "mask" channels: False = background 1 (the measured
	# winner); derived channels always treat the creature as inside.
	invert_mask: bool = False

	# Bbox-crop the creature and rescale to fill the canvas (aspect preserved via
	# padding), applied at train and eval alike. Body occupancy averages ~25% of
	# the canvas (7.2-48.5%), so this lifts small Pokemon to a comparable
	# effective resolution and drops absolute size, a near-perfect generation
	# proxy that N1 measured as ~useless as a cue. See EXPERIMENTS.md Phase 5.
	aspect_crop: bool = False

	# Single global seed: the split, all RNGs, and the train loader shuffle.
	random_state: int = 42

	# Model
	model_name: str = "resnet50"
	weights: str | None = "DEFAULT"  # torchvision weights enum name, or None
	# Local state-dict checkpoint; overrides `weights` when set. Standard
	# ImageNet weights beat the shape-biased checkpoint (C2).
	weights_checkpoint: Path | None = None
	# Unfreeze the last N of the 6 parameterized feature blocks (conv1, bn1,
	# layer1-4) + the classifier; 6 = full feature unfreeze (7+ is identical).
	# Raised 3 -> 6 by the p10 depth battery: full unfreeze is +1.68pt over lastN 3
	# across 3 seeds (paired t p=0.003, McNemar p=0.005, all seeds positive). The
	# old "2/3/4 plateau" was a distorted-fold artifact; depth is monotonic and
	# only stable this deep because Phase 2's warmup landed first. See EXPERIMENTS.md.
	train_last_n_layers: int = 6
	# Stem geometry: "default", "nomaxpool" (drop the stem maxpool; layer1 sees
	# 112x112, ~2x compute), or "stride1" (conv1 stride 1, maxpool kept - full-
	# resolution first conv, same layer1 size as nomaxpool). Kernels are
	# stride-agnostic, so pretrained weights load unchanged. See Phase 3.
	stem: str = "default"
	train_batch_norm_affine: bool = False
	# Must stay "train" - "eval" costs ~10pt of accuracy.
	batch_norm_mode: str = "train"

	# Optimizer. blr 4e-4 collapsed under constant LR but wins under warmup;
	# confirmed at two seeds (Phase 2).
	optimizer_type: str = "AdamW"
	backbone_lr: float = 4e-4
	classifier_lr: float = 1e-3
	weight_decay: float = 2e-3
	label_smoothing: float = 0.2

	# Training loop. `epochs` is a fixed budget and, under the cosine scheduler,
	# the schedule horizon - there is no early exit. The cosine + restore
	# schedule is the Phase 2 winner (confirmed at two seeds). Horizon lowered
	# 32 -> 26 after the p9 battery: 32ep is +0.45pt over 26 across 3 index-fold
	# seeds (paired t p=0.42, McNemar p=0.40) - flat, so 26 for ~20% cheaper runs.
	epochs: int = 26
	# Restore the best-val-loss epoch's weights before scoring; False scores
	# the final epoch (the historical behaviour).
	restore_best_epoch: bool = True
	# "none" (constant LR) or "cosine": per-step cosine decay to 0 over the
	# full budget after `warmup_epochs` of linear warmup.
	scheduler: str = "cosine"
	warmup_epochs: float = 2.0

	# Tracking. The leaky-split runs live in "pokemon-classification"; the two
	# experiments use the same metric names for different things - never mix.
	experiment_name: str = "pokemon-classification-clean"
	run_name: str | None = None
	save_model: bool = False

	@property
	def augmentations(self) -> dict[str, bool | float | tuple]:
		"""Transform options in the form `get_transforms` expects."""
		return {
			"hflip": self.augment_hflip,
			"morphological": self.augment_morphological,
			"resolution_jitter": self.augment_resolution_jitter,
			"elastic": self.augment_elastic,
			"affine_degrees": self.affine_degrees,
			"affine_translate": self.affine_translate,
			"affine_scale": self.affine_scale,
			"input_channels": self.input_channels,
			"invert_mask": self.invert_mask,
			"aspect_crop": self.aspect_crop,
		}
