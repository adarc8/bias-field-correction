import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.3))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False, ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False, ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.3))
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=2, bias=False, ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.3))
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x
