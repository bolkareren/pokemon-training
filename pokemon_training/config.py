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
	# 0 = single split (validation_accuracy). K > 0 = grouped CV pooled over
	# out-of-fold predictions (oof_accuracy); --folds 5 is the reporting standard.
	folds: int = 0
	# Keep near-duplicate silhouettes (IoU > 0.97) within a single fold.
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
	input_channels: tuple[str, str, str] = ("mask", "mask", "mask")
	# Polarity of emitted "mask" channels: False = background 1 (the measured
	# winner); derived channels always treat the creature as inside.
	invert_mask: bool = False

	# Single global seed: the split, all RNGs, and the train loader shuffle.
	random_state: int = 42

	# Model
	model_name: str = "resnet50"
	weights: str | None = "DEFAULT"  # torchvision weights enum name, or None
	# Local state-dict checkpoint; overrides `weights` when set. Standard
	# ImageNet weights beat the shape-biased checkpoint (C2).
	weights_checkpoint: Path | None = None
	train_last_n_layers: int = 3
	train_batch_norm_affine: bool = False
	# Must stay "train" - "eval" costs ~10pt of accuracy.
	batch_norm_mode: str = "train"

	# Optimizer
	optimizer_type: str = "AdamW"
	backbone_lr: float = 2e-4
	classifier_lr: float = 1e-3
	weight_decay: float = 2e-3
	label_smoothing: float = 0.2

	# Training loop
	epochs: int = 16

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
		}
