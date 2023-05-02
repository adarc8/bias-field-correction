import torch
import torch.nn as nn


class STN(nn.Module):
    def __init__(self, n_channels):
        super(STN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2 * n_channels, out_channels=8, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flat = nn.Flatten()

        self.fc = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=6),
        )

        torch.nn.init.zeros_(self.fc[-1].weight)
        torch.nn.init.zeros_(self.fc[-1].bias)

    def forward(self, img, features=None):
        x = self.conv(img)
        x = self.flat(x)
        theta_ADD = self.fc(x).view(-1, 2, 3)

        theta_0 = torch.Tensor([[[1, 0, 0], [0, 1, 0]]]).to(img, non_blocking=True)

        theta = theta_0 + theta_ADD

        grid = torch.nn.functional.affine_grid(theta, img[:, 0:1].size(), align_corners=False)
        img = torch.nn.functional.grid_sample(img[:, 0:1].float(), grid.float(),
                                              align_corners=False)  # add: <,mode='bicubic'>

        return grid, img

    def warp(self, img, grid, interp=False):
        warped = torch.nn.functional.grid_sample(img.float(), grid.float(), align_corners=False)
        if interp and (warped.shape != img.shape):
            warped = torch.nn.functional.interpolate( \
                warped, size=img.shape[2:])
        return warped
