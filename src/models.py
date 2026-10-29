"""Model factory for every architecture compared in SmartWaste AI (contract: AGENTS.md).

Stub: only "mobilenet_v2" is implemented. Participant 2 adds "baseline_cnn" and "modified_cnn".
"""

from torch import nn
from torchvision import models

MODEL_NAMES = ("baseline_cnn", "modified_cnn", "mobilenet_v2")


def get_model(name: str, num_classes: int = 6, dropout: float = 0.0, *, pretrained: bool = True) -> nn.Module:
    """Build a model by name.

    pretrained=False skips the ImageNet download (the app loads models/best.pth on top anyway).
    dropout never changes the state_dict keys: the Dropout layer always exists, p=0 is a no-op.
    """
    if name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.last_channel, num_classes))
        return model
    if name in MODEL_NAMES:
        raise NotImplementedError(f"get_model({name!r}) is not implemented yet (Participant 2 adds it)")
    raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")
