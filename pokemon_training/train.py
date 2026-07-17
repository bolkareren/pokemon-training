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


def training_single_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device=None,
    batch_norm_mode="train",
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
):
    device = _resolve_device(device)
    model.to(device)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        train_loss = training_single_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            batch_norm_mode=batch_norm_mode,
        )
        val_loss = _validation_loss(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if on_epoch_end is not None:
            on_epoch_end(epoch, {"train_loss": train_loss, "val_loss": val_loss})

    return history
