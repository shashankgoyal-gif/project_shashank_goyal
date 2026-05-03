# model.py
import torch.nn as nn
from torchvision import models

class MushroomResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(MushroomResNet18, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.base_model.fc.in_features
        
        # Modify the final layer
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.base_model(x)