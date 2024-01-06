import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(EfficientNetB0, self).__init__()
        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        self.base = efficientnet
        self.classifier = nn.Linear(
            efficientnet.classifier[-1].in_features, num_classes
        )
        self.base.classifier[-1] = nn.Identity()

    def forward(self, x):
        return self.base(x)
