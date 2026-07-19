import copy
import math

import torch


def set_batch_norm_eval(model):
	for module in model.modules():
		if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
			module.eval()


def _resolve_device(device):
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def build_scheduler(optimizer, scheduler_type, warmup_epochs, epochs, steps_per_epoch):
	"""Per-step LR scheduler over the full fixed budget, or None for constant LR."""
	if scheduler_type == "none":
		return None
	if scheduler_type != "cosine":
		raise ValueError(f"unknown scheduler: {scheduler_type!r}; valid: 'none', 'cosine'")

	total_steps = epochs * steps_per_epoch
	# int so a fractional warmup can't push the ramp's last factor above 1.0.
	warmup_steps = int(round(warmup_epochs * steps_per_epoch))

	def factor(step):
		if step < warmup_steps:
			return (step + 1) / warmup_steps
		progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
		return 0.5 * (1.0 + math.cos(math.pi * progress))

	return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def training_single_epoch(
	model,
	train_loader,
	optimizer,
	criterion,
	device=None,
	batch_norm_mode="train",
	scheduler=None,
):
	device = _resolve_device(device)
	if batch_norm_mode not in {"train", "eval"}:
		raise ValueError(f"Unknown batch_norm_mode: {batch_norm_mode}")

	model.to(device)
	model.train()
	if batch_norm_mode == "eval":
		set_batch_norm_eval(model)

	total_loss = 0.0
	total_examples = 0

	for inputs, labels in train_loader:
		inputs = inputs.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		if scheduler is not None:
			scheduler.step()

		batch_size = inputs.size(0)
		total_loss += loss.item() * batch_size
		total_examples += batch_size

	return total_loss / total_examples


def _validation_loss(model, val_loader, criterion, device):
	model.eval()
	total_loss = 0.0
	total_examples = 0

	with torch.no_grad():
		for inputs, labels in val_loader:
			inputs = inputs.to(device)
			labels = labels.to(device)

			outputs = model(inputs)
			loss = criterion(outputs, labels)

			batch_size = inputs.size(0)
			total_loss += loss.item() * batch_size
			total_examples += batch_size

	return total_loss / total_examples


def train_outer_loop(
	model,
	train_loader,
	val_loader,
	optimizer,
	criterion,
	epochs=50,
	device=None,
	batch_norm_mode="train",
	on_epoch_end=None,
	scheduler_type="none",
	warmup_epochs=2.0,
	restore_best_epoch=False,
):
	device = _resolve_device(device)
	model.to(device)
	scheduler = build_scheduler(optimizer, scheduler_type, warmup_epochs, epochs, len(train_loader))
	history = {"train_loss": [], "val_loss": []}

	best_val_loss = math.inf
	best_state = None
	best_epoch = None

	for epoch in range(epochs):
		train_loss = training_single_epoch(
			model,
			train_loader,
			optimizer,
			criterion,
			device=device,
			batch_norm_mode=batch_norm_mode,
			scheduler=scheduler,
		)
		val_loss = _validation_loss(model, val_loader, criterion, device)
		print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

		history["train_loss"].append(train_loss)
		history["val_loss"].append(val_loss)

		if restore_best_epoch and val_loss < best_val_loss:
			best_val_loss = val_loss
			best_epoch = epoch
			# deepcopy consumes no RNG, so tracking the best state cannot
			# perturb an otherwise-identical run.
			best_state = copy.deepcopy(model.state_dict())

		if on_epoch_end is not None:
			on_epoch_end(epoch, {"train_loss": train_loss, "val_loss": val_loss})

	if restore_best_epoch and best_state is not None:
		model.load_state_dict(best_state)
		history["best_epoch"] = best_epoch
		print(f"restored epoch {best_epoch + 1} (val_loss {best_val_loss:.4f})")

	return history
