import torch
from torchvision import models

_WEIGHTS_REGISTRY = {
	"resnet18": models.ResNet18_Weights,
	"resnet34": models.ResNet34_Weights,
	"resnet50": models.ResNet50_Weights,
}

_MODEL_BUILDERS = {
	"resnet18": models.resnet18,
	"resnet34": models.resnet34,
	"resnet50": models.resnet50,
}


def resolve_weights(model_name, weights="DEFAULT"):
	"""Map a weights enum name (e.g. "DEFAULT", "IMAGENET1K_V1") to the enum.

	Returns None for randomly initialized weights.
	"""
	if weights is None:
		return None
	enum = _WEIGHTS_REGISTRY[model_name]
	return getattr(enum, weights)


def load_checkpoint_weights(model, checkpoint_path, map_location="cpu"):
	"""Load a raw state-dict checkpoint (e.g. a research-repo release) into model.

	Handles the two conventions such checkpoints commonly use: wrapping the
	state dict under a "state_dict" key, and a "module." prefix left over from
	training with nn.DataParallel.
	"""
	checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
	if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
		checkpoint = checkpoint["state_dict"]
	state_dict = {key.removeprefix("module."): value for key, value in checkpoint.items()}
	model.load_state_dict(state_dict)
	return model


def load_pretrained_model(
	num_classes, weights=None, model_name="resnet18", weights_checkpoint=None
):
	"""Build model_name with either torchvision weights or a custom checkpoint.

	weights_checkpoint (if given) is loaded into the full 1000-class backbone
	before the classifier head is replaced, since such checkpoints assume the
	original ImageNet class count.
	"""
	model = _MODEL_BUILDERS[model_name](weights=weights)
	if weights_checkpoint is not None:
		load_checkpoint_weights(model, weights_checkpoint)
	model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
	return model


def apply_stem(model, stem="default"):
	"""Reduce the stem's 4x downsampling so thin contour detail survives."""
	if stem == "default":
		return model
	if stem == "nomaxpool":
		model.maxpool = torch.nn.Identity()
	elif stem == "stride1":
		model.conv1.stride = (1, 1)
	else:
		raise ValueError(f"unknown stem: {stem!r}; valid: 'default', 'nomaxpool', 'stride1'")
	return model


def adapt_input_channels(model, in_channels):
	"""Rebuild conv1 to accept `in_channels`, re-seeded from the pretrained stem.

	Only acts when the count differs from the current stem (so the default
	3-channel model is untouched). The pretrained RGB filters are summed over the
	input dimension and split equally across the new channels: for a 1-channel
	stem this is exactly "sum the RGB filters", which reproduces the pretrained
	response for a replicated-grayscale input and keeps activations in the scale
	BN and the later layers expect. Must run before set_trainable_weights so the
	new conv1's requires_grad is set, and after apply_stem so a stride/geometry
	tweak is inherited (stride/padding are copied from the existing conv1)."""
	old = model.conv1
	if in_channels == old.in_channels:
		return model
	new = torch.nn.Conv2d(
		in_channels,
		old.out_channels,
		kernel_size=old.kernel_size,
		stride=old.stride,
		padding=old.padding,
		bias=old.bias is not None,
	)
	summed = old.weight.data.sum(dim=1, keepdim=True)  # [out, 1, k, k]
	new.weight.data = summed.repeat(1, in_channels, 1, 1) / in_channels
	model.conv1 = new
	return model


def set_trainable_weights(model, train_last_n_layers=0):
	for parameter in model.parameters():
		parameter.requires_grad = False

	children = list(model.children())

	classifier = children[-1]
	feature_layers = children[:-1]

	parameterized_feature_layers = [
		layer for layer in feature_layers if sum(p.numel() for p in layer.parameters()) > 0
	]

	if train_last_n_layers > 0:
		for layer in parameterized_feature_layers[-train_last_n_layers:]:
			for parameter in layer.parameters():
				parameter.requires_grad = True

	for parameter in classifier.parameters():
		parameter.requires_grad = True

	return model


def set_batch_norm_trainable(model, trainable=True):
	for module in model.modules():
		if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
			if module.weight is not None:
				module.weight.requires_grad = trainable
			if module.bias is not None:
				module.bias.requires_grad = trainable

	return model


def create_optimizer(
	model,
	type="Adam",
	lr=1e-3,
	weight_decay=0.0,
	classifier_lr=None,
):
	if classifier_lr is None:
		parameters = filter(lambda p: p.requires_grad, model.parameters())
	else:
		classifier_parameters = list(model.fc.parameters())
		classifier_parameter_ids = {id(parameter) for parameter in classifier_parameters}
		backbone_parameters = [
			parameter
			for parameter in model.parameters()
			if parameter.requires_grad and id(parameter) not in classifier_parameter_ids
		]

		parameters = [
			{"params": backbone_parameters, "lr": lr},
			{"params": classifier_parameters, "lr": classifier_lr},
		]

	if type == "AdamW":
		return torch.optim.AdamW(
			parameters,
			lr=lr,
			weight_decay=weight_decay,
		)

	return torch.optim.Adam(parameters, lr=lr)
