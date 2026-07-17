import torch


def _resolve_device(device):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
