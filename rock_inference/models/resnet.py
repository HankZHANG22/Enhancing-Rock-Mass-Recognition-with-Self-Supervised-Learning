import torch
import torchvision.models as models
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, fp16, pretrained=True, num_features=512, dropout=0.0):
        super(ResNet18, self).__init__()
        self.fp16 = fp16
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.model.fc = nn.Linear(512, num_features)

    def forward(self, x):
        return self.model(x)
    

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
    

class ResNet50(nn.Module):
    def __init__(self, fp16, pretrained=True, num_features=512, dropout=0.0):
        super(ResNet50, self).__init__()
        self.fp16 = fp16
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.model.fc = nn.Linear(2048, num_features)

    def forward(self, x):
        return self.model(x)
    

class ResNet101(nn.Module):
    def __init__(self, fp16, pretrained=True, num_features=512, dropout=0.0):
        super(ResNet101, self).__init__()
        self.fp16 = fp16
        self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        self.model.fc = nn.Linear(2048, num_features)

    def forward(self, x):
        return self.model(x)
    

def get_resnet18(pretrained=True, num_features=512, dropout=0.0, **kwargs):
    fp16 = kwargs.get('fp16', False)
    return ResNet18(fp16, pretrained, num_features, dropout)


def get_resnet34(pretrained=True, num_features=512, dropout=0.0, **kwargs):
    fp16 = kwargs.get('fp16', False)
    return ResNet34(fp16, pretrained, num_features, dropout)


def get_resnet50(pretrained=True, num_features=512, dropout=0.0, **kwargs):
    fp16 = kwargs.get('fp16', False)
    return ResNet50(fp16, pretrained, num_features, dropout)


def get_resnet101(pretrained=True, num_features=512, dropout=0.0, **kwargs):
    fp16 = kwargs.get('fp16', False)
    return ResNet101(fp16, pretrained, num_features, dropout)
