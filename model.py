import torch
import torch.nn as nn
from torch.nn.functional import relu


class ResidualTwoConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3) -> None:
        super(ResidualTwoConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel, padding=kernel//2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel, padding=kernel//2)
        self.resconv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        y = self.resconv(x)
        y = relu(y)

        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)

        return x + y


class UNet(nn.Module):
    def __init__(
            self, classes
    ):
        super(UNet, self).__init__()

        self.ups = nn.ModuleDict()
        self.downs = nn.ModuleDict()

        self.pool = nn.MaxPool2d(2)

        self.downs.add_module('conv1', ResidualTwoConv(3, 64))
        self.downs.add_module('conv2', ResidualTwoConv(64, 128))
        self.downs.add_module('conv3', ResidualTwoConv(128, 256))
        self.downs.add_module('conv4', ResidualTwoConv(256, 512))
        self.downs.add_module('conv5', ResidualTwoConv(512, 1024))
        self.downs.add_module('pool', self.pool)

        self.ups.add_module('convtrans4', nn.ConvTranspose2d(1024, 512, 2, stride=2))
        self.ups.add_module('conv4', ResidualTwoConv(1024, 512))
        self.ups.add_module('convtrans3', nn.ConvTranspose2d(512, 256, 2, 2))
        self.ups.add_module('conv3', ResidualTwoConv(512, 256))
        self.ups.add_module('convtrans2', nn.ConvTranspose2d(256, 128, 2, 2))
        self.ups.add_module('conv2', ResidualTwoConv(256, 128))
        self.ups.add_module('convtrans1', nn.ConvTranspose2d(128, 64, 2, 2))
        self.ups.add_module('conv1', ResidualTwoConv(128, 64))
        self.ups.add_module('outconv', nn.Conv2d(64, classes, 1))

    def forward(self, x):
        y1 = self.downs['conv1'](x)
        x = self.downs['pool'](y1)

        y2 = self.downs['conv2'](x)
        x = self.downs['pool'](y2)

        y3 = self.downs['conv3'](x)
        x = self.downs['pool'](y3)

        y4 = self.downs['conv4'](x)
        x = self.downs['pool'](y4)

        x = self.downs['conv5'](x)

        x = self.ups['convtrans4'](x)
        x = torch.concat((x, y4), 1)
        x = self.ups['conv4'](x)

        x = self.ups['convtrans3'](x)
        x = torch.concat((x, y3), 1)
        x = self.ups['conv3'](x)

        x = self.ups['convtrans2'](x)
        x = torch.concat((x, y2), 1)
        x = self.ups['conv2'](x)

        x = self.ups['convtrans1'](x)
        x = torch.concat((x, y1), 1)
        x = self.ups['conv1'](x)

        x = self.ups['outconv'](x)

        return x
