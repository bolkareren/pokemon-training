from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Single source of truth for one training run.

    Every tunable lives here exactly once. Add a field to add a hyperparameter:
    it is picked up by the CLI, logged to MLflow, and saved with the run
    automatically. Nothing needs to be repeated anywhere else.
    """

    # Data
    data_dir: Path | None = None  # defaults to <project_root>/data when unset
    batch_size: int = 16
    # 0.15, not 0.1: with shiny sprites excluded the dataset is 1307 images over
    # 151 classes, and a 10% split (131) is smaller than the class count, which
    # makes a stratified split impossible. See EXPERIMENTS.md Phase 12.
    val_size: float = 0.15
    test_size: float = 0.15
    # Drop shiny sprites, which are recolours of the normal sprites and so have
    # duplicate silhouettes. Leaving them in put a pixel-identical twin in train
    # for ~62% of the validation set. See EXPERIMENTS.md Phase 11.
    exclude_shiny: bool = True
    # 0 runs a single train/val/test split. K > 0 cross-validates: the test split
    # is held out first, then K folds over the rest, reported as mean +/- SE over
    # pooled out-of-fold predictions. A single 196-image val split cannot resolve
    # the few-point differences these experiments compare.
    folds: int = 0
    # Keep near-duplicate silhouettes (IoU > 0.97) within a single fold.
    group_aware_folds: bool = True

    # Augmentation. Each is independently togglable so single effects can be
    # measured before combining. A RandomAffine (rotation/translation/scale) is
    # always applied and is not gated by these. See EXPERIMENTS.md Phase C7.
    # Mirror horizontally: sprite facing direction is arbitrary (later
    # generations flipped the convention) and a mirrored silhouette is still the
    # same Pokemon, so this roughly doubles the effective dataset.
    augment_hflip: bool = False
    # Dilate/erode the mask by 1-2px, perturbing contour thickness.
    augment_morphological: bool = False
    # Re-render at a random original sprite resolution (56-128px) before
    # upsampling, removing the generation-dependent edge artifact.
    augment_resolution_jitter: bool = False
    # Mild elastic warp - a pose change is roughly an elastic deformation.
    augment_elastic: bool = False

    # Rescale each sprite by a per-generation factor so the creature occupies a
    # comparable share of the frame regardless of which generation drew it,
    # while every Pokemon at a given generation is scaled identically so their
    # relative sizes survive. Sprite scale currently mixes a canvas artifact
    # (2.06x across generation medians) with real signal (1.83x within a
    # generation; size rises along the evolution line in 95% of cases), and the
    # two cancel. See EXPERIMENTS.md Phase C11.
    normalize_sprite_scale: bool = False
    # Single global seed for the run: the stratified split, all RNGs
    # (random/numpy/torch/cuda/mps), and the train DataLoader shuffle.
    random_state: int = 42

    # Model
    model_name: str = "resnet50"
    weights: str | None = "DEFAULT"  # torchvision weights enum name, or None
    # Local path to a raw state-dict checkpoint (relative paths resolve against
    # <project_root>). Overrides `weights` when set - the checkpoint's own
    # pretrained backbone is used instead of a torchvision weights enum.
    #
    # None (standard ImageNet weights) rather than the shape-biased checkpoint:
    # on clean data the same architecture and hyperparameters score 0.653 with
    # ImageNet weights vs 0.596 with shape-biased, a 5.7pt gap at 4.0x the
    # combined SEM. The shape-biased checkpoint looked better only under the
    # duplicate leak. To use it:
    #   --weights-checkpoint weights/resnet50_shape_biased.pth.tar
    # See EXPERIMENTS.md Phase C2.
    weights_checkpoint: Path | None = None
    # Depth is a plateau rather than a peak on clean data: lastN 2/3/4 land
    # within 1-2x their combined SEMs. Kept at 3 because that is where the
    # backbone comparison was run. Not re-swept since the checkpoint changed.
    # See EXPERIMENTS.md Phase C1.
    train_last_n_layers: int = 3
    train_batch_norm_affine: bool = False
    # Must stay "train" - "eval" costs ~10-13pts of validation accuracy.
    batch_norm_mode: str = "train"

    # Optimizer
    optimizer_type: str = "AdamW"
    backbone_lr: float = 2e-4
    classifier_lr: float = 1e-3
    weight_decay: float = 2e-3
    label_smoothing: float = 0.2

    # Training loop
    epochs: int = 16

    # Tracking
    # Separate from the leaky-split "pokemon-classification" experiment, whose
    # accuracies mean something different and are not comparable. See
    # EXPERIMENTS.md vs LEAKY-EXPERIMENTS.md.
    experiment_name: str = "pokemon-classification-clean"
    run_name: str | None = None
    save_model: bool = False

    @property
    def augmentations(self) -> dict[str, bool]:
        """Augmentation flags in the form `get_transforms` expects."""
        return {
            "hflip": self.augment_hflip,
            "morphological": self.augment_morphological,
            "resolution_jitter": self.augment_resolution_jitter,
            "elastic": self.augment_elastic,
        }
