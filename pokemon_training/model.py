import torch
from torchvision import models

_WEIGHTS_REGISTRY = {
    "resnet18": models.ResNet18_Weights,
    "resnet34": models.ResNet34_Weights,
}


def resolve_weights(model_name, weights="DEFAULT"):
    """Map a weights enum name (e.g. "DEFAULT", "IMAGENET1K_V1") to the enum.

    Returns None for randomly initialized weights.
    """
    if weights is None:
        return None
    enum = _WEIGHTS_REGISTRY[model_name]
    return getattr(enum, weights)


def load_pretrained_model(num_classes, weights=None, model_name="resnet18"):
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
    elif model_name == "resnet34":
        model = models.resnet34(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
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
