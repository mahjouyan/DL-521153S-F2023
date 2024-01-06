import torch
import torch.nn as nn

from torchvision import models

class ResNet18(nn.Module):
        def __init__(self, num_classes, pretrained=True):
            super(ResNet18, self).__init__()
            resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.features = nn.Sequential(*list(resnet18.children())[:-1])
            self.fc = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNet50, self).__init__()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        modules = list(resnet50.children())[:-2]
        self.features = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(resnet50.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

