import numpy as np
from torch import nn
import torch.nn.functional as F
import os

class Feature_Generator(nn.Module):

    def __init__(self, noise_shape):
        super(Feature_Generator, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(noise_shape, 128 * 4 * 4),
            nn.BatchNorm1d(128 * 4 * 4)
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(128, 48, 3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(48, 12, 3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(12, 6, 4, stride=1, padding=0, bias=False),
            nn.Tanh(),
            nn.BatchNorm2d(6)
        )

    def forward(self, x, einsum):
        x = self.dense(x)
        x = x.view(x.size(0), 512, 4, 4)
        x = self.model(x)
        x = torch.einsum('aijk, ai -> aijk', x, einsum)  # (batch, 256, 56, 56) * (batch, 256) -> (batch, 256, 56, 56)
        return x


class Feature_Discriminator(nn.Module):

    def __init__(self, in_channels):
        super(Feature_Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(True),

            nn.Conv2d(32, 64, 5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(True),

            nn.Conv2d(64, 128, 5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(True)
        )

        self.dense = nn.Linear(128 * 2 * 2, 1)

    def forward(self, x):
        x = self.conv(x)  # (batch, 128, 2, 2)
        x = x.view(x.size(0), -1)  # (batch, 128 * 2 * 2)
        x = self.dense(x)  # (batch, 1)
        return x


class Feature_Extractor(nn.Module):

    def __init__(self, pretrained_weight=None, num_classes=10):
        super(Feature_Extractor, self).__init__()

        if pretrained_weight is None:
            lenet5_list = list(Lenet5(num_classes=num_classes).children())
        else:
            lenet5_ = Lenet5(num_classes=num_classes)
            lenet5_.load_state_dict(torch.load(pretrained_weight, map_location='cpu'))
            lenet5_list = list(lenet5_.children())

        self.model = nn.Sequential(
            # stop after first conv layer
            *list(lenet5_list[0])[:3]
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Feature_Classifier(nn.Module):
    def __init__(self, pretrained_weight=None, num_classes=10):
        super(Feature_Classifier, self).__init__()
        if pretrained_weight is None:
            lenet5_list = list(Lenet5(num_classes=num_classes).children())
        else:
            lenet5_ = Lenet5(num_classes=num_classes)
            lenet5_.load_state_dict(torch.load(pretrained_weight, map_location='cpu'))
            lenet5_list = list(lenet5_.children())

        self.feature = nn.Sequential(
            *list(lenet5_list[0])[3:]
        )
        self.classifier = nn.Sequential(
            *list(lenet5_list[1])[:]
        )

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Lenet5(nn.Module):

    def __init__(self, num_classes=10):
        super(Lenet5, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dense = nn.Sequential(
            nn.Linear(16 * 5 * 5, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        return x
