import os
import torch
from torchvision.models import resnet50

class ResNet_50_regressor(torch.nn.Module):
    def __init__(self):
        super(ResNet_50_regressor, self).__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(1, bias=True)

    def forward(self, input):
        x = self.model(input)
        return x