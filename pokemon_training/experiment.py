import random

import numpy as np
import torch


def set_random_seed(seed):
	"""Seed every RNG a run touches so results are reproducible across devices.

	`torch.manual_seed` covers the CPU (and, on recent torch, the default MPS)
	generator; the device-specific calls below make CUDA/MPS explicit and, on
	CUDA, force cuDNN into deterministic mode. Note this seeds global generators
	only — the training DataLoader is shuffled with its own generator pinned to
	the same seed (see `create_data_loaders`).
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	if torch.backends.mps.is_available():
		torch.mps.manual_seed(seed)


def get_device():
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")
