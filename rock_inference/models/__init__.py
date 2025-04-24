from .resnet import get_resnet18, get_resnet34, get_resnet50, get_resnet101


def get_model(name, **kwargs):
    # resnet
    if name == "resnet18":
        return get_resnet18(**kwargs)
    elif name == "resnet34":
        return get_resnet34(**kwargs)
    elif name == "resnet50":
        return get_resnet50(**kwargs)
    elif name == "resnet101":
        return get_resnet101(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")