"""Four ImageNet-pretrained CNNs with a 2-logit head (paper Sec. IV-E)."""
import torch.nn as nn
import torchvision


def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    m = torchvision.models.get_model(arch, weights=weights)
    if arch == "resnet50":
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif arch == "densenet121":
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    elif arch in ("efficientnet_b0", "mobilenet_v2"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return m


def gradcam_layer(model: nn.Module, arch: str) -> nn.Module:
    if arch == "resnet50":
        return model.layer4[-1]
    if arch == "densenet121":
        # denseblock4 (not features[-1]): DenseNet applies an in-place ReLU right after
        # norm5, which breaks tensor hooks on the final feature map.
        return model.features.denseblock4
    if arch in ("efficientnet_b0", "mobilenet_v2"):
        return model.features[-1]
    raise ValueError(arch)
