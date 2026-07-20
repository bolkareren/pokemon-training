import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHINY_MANIFEST_PATH = PROJECT_ROOT / "shiny_index.json"


class RandomResolutionJitter:
	"""Downsample to a random sprite-era resolution and back up, so effective
	source resolution stops being a generation shortcut."""

	SPRITE_SIZES = (56, 64, 80, 96, 120, 128)

	def __init__(self, p=0.5):
		self.p = p

	def __call__(self, image):
		if torch.rand(1).item() >= self.p:
			return image
		index = torch.randint(len(self.SPRITE_SIZES), (1,)).item()
		size = self.SPRITE_SIZES[index]
		original = image.size
		return image.resize((size, size), Image.BILINEAR).resize(original, Image.BILINEAR)


class RandomMorphology:
	"""Randomly dilate or erode the binary mask by 1-2px, perturbing the contour."""

	def __init__(self, p=0.5):
		self.p = p

	def __call__(self, mask):
		if torch.rand(1).item() >= self.p:
			return mask

		kernel = 3 if torch.rand(1).item() < 0.5 else 5
		padding = kernel // 2
		batched = mask.unsqueeze(0)

		if torch.rand(1).item() < 0.5:
			out = torch.nn.functional.max_pool2d(batched, kernel, stride=1, padding=padding)
		else:
			out = -torch.nn.functional.max_pool2d(-batched, kernel, stride=1, padding=padding)

		return out.squeeze(0)


class SilhouetteChannels:
	"""Encode the binary mask into per-channel descriptors ("mask", "sdt",
	"curv", "edge") - the input representation, applied at train and eval alike.

	Derived channels always treat the creature as inside; `invert_mask` only
	sets the polarity of emitted "mask" channels. See EXPERIMENTS.md.
	"""

	VALID = ("mask", "sdt", "curv", "edge")
	# 64px spends the [0, 1] range on the occupied distance range without clipping.
	SDT_SCALE = 64.0
	# 7px: above RandomMorphology's 1-2px so augmentation can't erase the signal.
	CURV_KERNEL = 7
	# 8px: the boundary band is still ~2px wide after the stem's 4x downsample.
	EDGE_SIGMA = 8.0

	def __init__(self, channels=("mask", "mask", "mask"), invert_mask=False):
		unknown = set(channels) - set(self.VALID)
		if unknown:
			raise ValueError(f"unknown input channels {sorted(unknown)}; valid: {self.VALID}")
		self.channels = tuple(channels)
		self.invert_mask = invert_mask

	def _sdt(self, mask):
		"""Signed distance to the boundary, fixed global scale, 0.5 on the contour."""
		inside = mask.numpy() > 0.5
		signed = ndimage.distance_transform_edt(inside) - ndimage.distance_transform_edt(~inside)
		scaled = 0.5 + signed / (2.0 * self.SDT_SCALE)
		return torch.from_numpy(scaled).clamp(0.0, 1.0).float()

	def _curv(self, mask):
		"""Morphological curvature proxy: protrusions bright, concavities dark."""
		# scipy's separable filters: ~30x faster than a dense max_pool2d here.
		inside = mask.numpy() > 0.5
		kernel = self.CURV_KERNEL
		opening = ndimage.maximum_filter(ndimage.minimum_filter(inside, kernel), kernel)
		closing = ndimage.minimum_filter(ndimage.maximum_filter(inside, kernel), kernel)

		out = np.full(inside.shape, 0.5, dtype=np.float32)
		out[inside & ~opening] = 1.0
		out[closing & ~inside] = 0.0
		return torch.from_numpy(out)

	def _edge(self, mask):
		"""Gaussian band on the contour, thick enough to survive stem downsampling."""
		inside = mask.numpy() > 0.5
		distance = ndimage.distance_transform_edt(inside) + ndimage.distance_transform_edt(~inside)
		return torch.from_numpy(np.exp(-(distance**2) / (2.0 * self.EDGE_SIGMA**2))).float()

	def __call__(self, x):
		# Incoming tensor is creature = 1; polarity applies only on emission.
		if all(channel == "mask" for channel in self.channels):
			return x if self.invert_mask else 1.0 - x

		mask = x[0]
		cache = {"mask": mask if self.invert_mask else 1.0 - mask}
		built = []
		for channel in self.channels:
			if channel not in cache:
				cache[channel] = getattr(self, f"_{channel}")(mask)
			built.append(cache[channel])
		return torch.stack(built)


def get_transforms(
	hflip=False,
	morphological=False,
	resolution_jitter=False,
	elastic=False,
	affine_degrees=20.0,
	affine_translate=0.2,
	affine_scale=(0.85, 1.15),
	input_channels=("mask", "mask", "mask"),
	invert_mask=False,
):
	"""Build (train, eval) transforms. Geometric augmentations run on the PIL
	image; the threshold re-binarizes their interpolation; mask ops and channel
	encoding run on the binary mask."""
	geometric = []
	if resolution_jitter:
		geometric.append(RandomResolutionJitter())

	geometric.append(
		transforms.RandomAffine(
			degrees=affine_degrees,
			translate=(affine_translate, affine_translate),
			scale=tuple(affine_scale),
		)
	)

	if elastic:
		geometric.append(transforms.ElasticTransform(alpha=40.0, sigma=5.0))
	if hflip:
		geometric.append(transforms.RandomHorizontalFlip(p=0.5))

	mask_ops = [RandomMorphology()] if morphological else []
	# Internal representation is creature = 1; SilhouetteChannels emits mask
	# channels as background = 1 unless invert_mask is set.
	threshold = transforms.Lambda(lambda x: (x < 0.5).float())
	encode = SilhouetteChannels(input_channels, invert_mask=invert_mask)

	train_transform = transforms.Compose(
		[
			*geometric,
			transforms.ToTensor(),
			threshold,
			*mask_ops,
			encode,
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
		]
	)

	test_transform = transforms.Compose(
		[
			transforms.ToTensor(),
			threshold,
			encode,
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
		]
	)

	return train_transform, test_transform


# Shared decode cache: pure speedup, consumes no RNG, training stays byte-identical.
_IMAGE_CACHE = {}


def _cached_loader(path):
	image = _IMAGE_CACHE.get(path)
	if image is None:
		with open(path, "rb") as handle:
			image = Image.open(handle).convert("RGB")
		_IMAGE_CACHE[path] = image
	return image


class EvalTransformCache(torch.utils.data.Dataset):
	"""Cache (tensor, target) pairs of a deterministic-transform dataset.
	Only ever wrap eval datasets - caching train would freeze augmentation."""

	def __init__(self, dataset):
		self.dataset = dataset
		self._cache = {}

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		if index not in self._cache:
			self._cache[index] = self.dataset[index]
		return self._cache[index]


def load_dataset(data_dir="data", transform=None):
	return datasets.ImageFolder(data_dir, transform=transform, loader=_cached_loader)


def _normal_sprite_indices(dataset):
	"""Indices of normal (non-shiny) sprites; shiny silhouettes are duplicates.

	The per-class boundary index lives in shiny_index.json (regenerate with
	scripts/generate_shiny_manifest.py) - it varies by class and is only
	derivable from raw sprite dimensions.
	"""
	if not SHINY_MANIFEST_PATH.exists():
		raise FileNotFoundError(
			f"{SHINY_MANIFEST_PATH} not found - generate it with "
			"`uv run python scripts/generate_shiny_manifest.py`, or set "
			"exclude_shiny=False to train on the duplicated dataset."
		)

	shiny_start = json.loads(SHINY_MANIFEST_PATH.read_text())

	keep = []
	for i, (path, target) in enumerate(dataset.samples):
		# Exact canonical-name match: stray files from older conventions would
		# otherwise readmit the duplicates being removed.
		match = re.fullmatch(r"image-(\d+)\.png", Path(path).name)
		if match is None:
			continue

		class_name = dataset.classes[target]
		if int(match.group(1)) < shiny_start[class_name]:
			keep.append(i)

	return np.array(keep)


def _pose_variant_pairs(dataset, min_diff_px=10):
	"""(variant_idx, partner_normal_idx) for shiny sprites that are animation-frame
	pose variants of their index-paired normal rather than exact recolours.

	Pairing is by index (normal number = shiny - shiny_start + 1, from image-1
	since image-0 predates shinies). A pair is a variant when the two silhouettes
	differ by more than `min_diff_px` pixels: exact recolours differ by <=2px
	(threshold/antialias residue), real variants by >=22px, so any cut in the gap
	selects the same 148 (all at the animated gen-5 image-5, three at image-6).
	See EXPERIMENTS.md "Known data issues".
	"""
	shiny_start = json.loads(SHINY_MANIFEST_PATH.read_text())
	targets = np.array(dataset.targets)

	# class name -> {sprite number: dataset index}
	by_class = {}
	for i in range(len(dataset.samples)):
		match = re.fullmatch(r"image-(\d+)\.png", Path(dataset.samples[i][0]).name)
		if match is None:
			continue
		by_class.setdefault(dataset.classes[targets[i]], {})[int(match.group(1))] = i

	def mask(i):
		with Image.open(dataset.samples[i][0]) as image:
			return np.array(image.convert("L")) <= 127

	pairs = []
	for name, sprites in by_class.items():
		start = shiny_start[name]
		for number, idx in sprites.items():
			if number < start:
				continue  # a normal sprite, not a shiny
			partner = sprites.get(number - start + 1)
			if partner is None:
				continue
			if int(np.count_nonzero(mask(idx) ^ mask(partner))) > min_diff_px:
				pairs.append((idx, partner))
	return pairs


def sprite_groups(dataset, indices):
	"""Group each shiny sprite with the normal sprite it recolours.

	Sprites are numbered per class as `image-<n>.png`. `image-0` is the gen-1
	sprite, which predates shinies, so the shiny run - starting at the class
	boundary in shiny_index.json - pairs with the normal series from `image-1`
	onward: `normal = shiny - shiny_start + 1`. That rule resolves all 853
	shiny sprites with no leftovers.

	Pairing by index rather than by silhouette overlap is exact, and overlap
	provably cannot do the job. The gen-5 sprites (`image-5`, and `image-6` for
	three classes) are animated, so a shiny screenshot can catch a different
	animation frame than its normal counterpart: golbat's wings open vs closed
	lands at IoU 0.24, while a static pokemon like ninetales lands at 0.997.
	Those two populations are interleaved by pose rather than separated, so no
	threshold splits recolours from genuinely distinct artwork - 148 of the 853
	pairs escape any usable cutoff.

	Cross-class pairs are never grouped: a shared shape there is task
	difficulty, not redundancy.
	"""
	if not SHINY_MANIFEST_PATH.exists():
		raise FileNotFoundError(
			f"{SHINY_MANIFEST_PATH} not found - grouped folds pair shiny sprites "
			"with the normals they recolour and cannot be built without it. "
			"Generate it with `uv run python scripts/generate_shiny_manifest.py`, "
			"or set group_aware_folds=False."
		)

	shiny_start = json.loads(SHINY_MANIFEST_PATH.read_text())
	targets = np.array(dataset.targets)

	groups = np.empty(len(indices), dtype=int)
	group_ids = {}

	for position, i in enumerate(indices):
		class_name = dataset.classes[targets[i]]
		match = re.fullmatch(r"image-(\d+)\.png", Path(dataset.samples[i][0]).name)
		if match is None:
			# Stray file from an older naming convention: give it its own group
			# rather than silently pairing it with an unrelated sprite.
			key = (class_name, "unmatched", i)
		else:
			number = int(match.group(1))
			start = shiny_start[class_name]
			if number >= start:
				number = number - start + 1
			key = (class_name, number)

		if key not in group_ids:
			group_ids[key] = len(group_ids)
		groups[position] = group_ids[key]

	return groups


def fold_indices(dataset, indices, folds, random_state=42, group_aware=True):
	"""Yield (train_idx, val_idx) per fold, stratified by class and grouped so a
	shiny sprite never lands in a different fold from the normal it recolours."""
	indices = np.asarray(indices)
	targets = np.array(dataset.targets)[indices]
	groups = sprite_groups(dataset, indices) if group_aware else np.arange(len(indices))

	splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state)
	for train_positions, val_positions in splitter.split(indices, targets, groups):
		yield indices[train_positions], indices[val_positions]


def split_dataset_indices(dataset, val_size, test_size, random_state=42, indices=None):
	"""Stratified train/val/test split over `indices` (or the whole dataset);
	returned indices address the full dataset."""
	targets = np.array(dataset.targets)
	indices = np.arange(len(dataset)) if indices is None else np.asarray(indices)
	labels = targets[indices]

	temp_idx, test_idx = train_test_split(
		indices,
		test_size=test_size,
		stratify=labels,
		random_state=random_state,
	)

	train_idx, val_idx = train_test_split(
		temp_idx,
		test_size=val_size / (1 - test_size),
		# temp_idx holds dataset-level ids: index the full target array here.
		stratify=targets[temp_idx],
		random_state=random_state,
	)

	return train_idx, val_idx, test_idx


def create_datasets(
	data_dir,
	val_size,
	test_size,
	random_state=42,
	exclude_shiny=True,
	augmentations=None,
):
	train_transform, test_transform = get_transforms(**(augmentations or {}))
	base_dataset = load_dataset(data_dir)
	indices = _normal_sprite_indices(base_dataset) if exclude_shiny else None
	train_idx, val_idx, test_idx = split_dataset_indices(
		base_dataset,
		val_size=val_size,
		test_size=test_size,
		random_state=random_state,
		indices=indices,
	)

	train_dataset = Subset(load_dataset(data_dir, transform=train_transform), train_idx)
	val_dataset = EvalTransformCache(
		Subset(load_dataset(data_dir, transform=test_transform), val_idx)
	)
	test_dataset = EvalTransformCache(
		Subset(load_dataset(data_dir, transform=test_transform), test_idx)
	)

	return train_dataset, val_dataset, test_dataset, base_dataset.classes


def create_fold_data_loaders(
	data_dir,
	folds,
	test_size=0.15,
	batch_size=16,
	random_state=42,
	exclude_shiny=True,
	group_aware=True,
	include_pose_variants=False,
	augmentations=None,
):
	"""Cross-validation loaders; the test split is carved out first and stays
	untouched. Each fold is (train_loader, val_loader, val_idx), val_idx
	addressing the base dataset so out-of-fold predictions trace to images.

	`include_pose_variants` adds the 148 animated-frame shiny sprites to the
	*training* side of each fold, but only where their normal partner is already
	in that fold's train split - so the normal folds (and thus every scored
	image) stay byte-identical to the exclude_shiny baseline, the comparison is
	truly paired, and no variant sits opposite its near-identical partner.
	"""
	if include_pose_variants and not exclude_shiny:
		raise ValueError("include_pose_variants requires exclude_shiny=True")

	train_transform, test_transform = get_transforms(**(augmentations or {}))
	base_dataset = load_dataset(data_dir)
	indices = (
		_normal_sprite_indices(base_dataset) if exclude_shiny else np.arange(len(base_dataset))
	)
	targets = np.array(base_dataset.targets)
	variant_pairs = _pose_variant_pairs(base_dataset) if include_pose_variants else []

	pool_idx, test_idx = train_test_split(
		indices,
		test_size=test_size,
		stratify=targets[indices],
		random_state=random_state,
	)

	test_loader = DataLoader(
		EvalTransformCache(Subset(load_dataset(data_dir, transform=test_transform), test_idx)),
		batch_size=batch_size,
		shuffle=False,
	)

	fold_loaders = []
	for train_idx, val_idx in fold_indices(
		base_dataset, pool_idx, folds, random_state=random_state, group_aware=group_aware
	):
		if variant_pairs:
			train_members = set(train_idx.tolist())
			extra = [variant for variant, partner in variant_pairs if partner in train_members]
			if extra:
				train_idx = np.concatenate([train_idx, np.array(extra, dtype=train_idx.dtype)])

		shuffle_generator = torch.Generator().manual_seed(random_state)
		train_loader = DataLoader(
			Subset(load_dataset(data_dir, transform=train_transform), train_idx),
			batch_size=batch_size,
			shuffle=True,
			generator=shuffle_generator,
		)
		val_loader = DataLoader(
			EvalTransformCache(Subset(load_dataset(data_dir, transform=test_transform), val_idx)),
			batch_size=batch_size,
			shuffle=False,
		)
		fold_loaders.append((train_loader, val_loader, val_idx))

	return fold_loaders, test_loader, base_dataset.classes


def create_data_loaders(
	data_dir,
	# >= 0.15: a smaller split has fewer images than the 151 classes and the
	# stratified split raises.
	val_size=0.15,
	test_size=0.15,
	batch_size=16,
	random_state=42,
	exclude_shiny=True,
	augmentations=None,
):
	train_dataset, val_dataset, test_dataset, classes = create_datasets(
		data_dir=data_dir,
		val_size=val_size,
		test_size=test_size,
		random_state=random_state,
		exclude_shiny=exclude_shiny,
		augmentations=augmentations,
	)

	# Own generator so epoch ordering is independent of global-RNG consumption.
	shuffle_generator = torch.Generator().manual_seed(random_state)
	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		generator=shuffle_generator,
	)
	val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

	return train_loader, val_loader, test_loader, classes
