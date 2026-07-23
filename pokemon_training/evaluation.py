import torch


def _resolve_device(device):
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def predict_top_k(model, data_loader, k=5, device=None):
	"""Return (top_k_predictions, labels) as lists, in data_loader order -
	the raw predictions the confusion study consumes."""
	device = _resolve_device(device)
	model.to(device)
	model.eval()

	predictions = []
	labels = []

	with torch.no_grad():
		for inputs, batch_labels in data_loader:
			outputs = model(inputs.to(device))
			predictions.extend(outputs.topk(k, dim=1).indices.cpu().tolist())
			labels.extend(batch_labels.tolist())

	return predictions, labels


def predict_probabilities(model, data_loader, device=None):
	"""Full per-class softmax for every image, in data_loader order.

	`predict_top_k` returns only the top-k *indices*, which cannot be combined
	across models - averaging an ensemble needs the whole distribution. Returns
	(probabilities [N, num_classes], labels).
	"""
	device = _resolve_device(device)
	model.to(device)
	model.eval()

	probabilities = []
	labels = []

	with torch.no_grad():
		for inputs, batch_labels in data_loader:
			outputs = model(inputs.to(device))
			probabilities.append(torch.softmax(outputs, dim=1).cpu())
			labels.extend(batch_labels.tolist())

	return torch.cat(probabilities), labels


def top_k_from_probabilities(probabilities, k=5):
	"""Ranked top-k indices from a probability matrix, in `predict_top_k`'s exact
	format so the existing metrics and the confusion study consume it unchanged."""
	return probabilities.topk(k, dim=1).indices.tolist()


def top_k_accuracy_from_predictions(predictions, labels, k=1):
	"""Top-k accuracy from `predict_top_k` output, without re-running the model."""
	correct = sum(label in prediction[:k] for prediction, label in zip(predictions, labels))
	return correct / len(labels)


def accuracy(model, data_loader, k=1, device=None):
	device = _resolve_device(device)
	model.to(device)
	model.eval()

	correct = 0
	total = 0

	with torch.no_grad():
		for inputs, labels in data_loader:
			inputs = inputs.to(device)
			labels = labels.to(device)

			outputs = model(inputs)
			predictions = outputs.topk(k, dim=1).indices

			correct += predictions.eq(labels.view(-1, 1)).any(dim=1).sum().item()
			total += labels.size(0)

	return correct / total
