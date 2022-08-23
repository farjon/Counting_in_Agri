import os
import torch
import torchvision.models
import torchvision.models as models

class RegressionNetwork(torch.nn.Module):
    def __init__(self):
        super(RegressionNetwork, self).__init__()

        self.model = resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(2048, 1, bias=True)

    def forward(self, input):
        x = self.model(input)
        return x.view(-1)
