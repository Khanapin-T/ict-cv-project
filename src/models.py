"""Model factory for every architecture compared in SmartWaste AI (contract: AGENTS.md)."""

from torch import nn
from torchvision import models

MODEL_NAMES = ("baseline_cnn", "modified_cnn", "mobilenet_v2")
BASELINE_CHANNELS = (32, 64, 128)
MODIFIED_CHANNELS = (32, 64, 128, 256)  # the baseline plus one extra conv block


def _simple_cnn(channels: tuple[int, ...], num_classes: int, dropout: float) -> nn.Sequential:
    """Conv(3x3)-BN-ReLU-MaxPool blocks, then AdaptiveAvgPool(4) -> Linear(256) -> Dropout -> Linear."""
    layers, in_ch = [], 3
    for out_ch in channels:
        layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                   nn.MaxPool2d(2)]
        in_ch = out_ch
    layers += [nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(in_ch * 16, 256), nn.ReLU(inplace=True),
               nn.Dropout(dropout), nn.Linear(256, num_classes)]
    return nn.Sequential(*layers)


def get_model(name: str, num_classes: int = 6, dropout: float = 0.0, *, pretrained: bool = True) -> nn.Module:
    """Build a model by name.

    pretrained=False skips the ImageNet download (the app loads models/best.pth on top anyway); the CNNs ignore it.
    dropout never changes the state_dict keys: the Dropout layer always exists, p=0 is a no-op.
    """
    if name == "baseline_cnn":
        return _simple_cnn(BASELINE_CHANNELS, num_classes, dropout)
    if name == "modified_cnn":
        return _simple_cnn(MODIFIED_CHANNELS, num_classes, dropout)
    if name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.last_channel, num_classes))
        return model
    raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
