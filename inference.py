import torch
from torchvision import models
from torch import nn


class ResNet34(nn.Module):
    def __init__(self, fp16, pretrained=True, num_features=512, dropout=0.0):
        super(ResNet34, self).__init__()
        self.fp16 = fp16
        self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.model.fc = nn.Sequential(
            nn.Linear(512, num_features),
            nn.BatchNorm1d(num_features),
        )

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=self.fp16):
            x = self.model(x)
        return x.float() if self.fp16 else x


def get_resnet34(pretrained=True, num_features=512, dropout=0.0, **kwargs):
    fp16 = kwargs.get('fp16', False)
    return ResNet34(fp16, pretrained, num_features, dropout)


