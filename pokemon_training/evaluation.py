import torch


def _resolve_device(device):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_top_k(model, data_loader, k=5, device=None):
    """Return (top_k_predictions, labels) as lists, in data_loader order.

    Kept separate from `accuracy` because the Phase 15 data study needs the
    predictions themselves - a confusion matrix, and which classes a wrong
    answer was confused with - not just a scalar score.
    """
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
