import torch
import torch.nn as nn
import torchvision
from config import DefaultConfig

config = DefaultConfig()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, has_relu=True, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.has_relu = has_relu

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


# keep the H and W of the input and output the same
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # 1x1 at the first and end to reduce channels
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1, 1, 0)
        self.conv2 = ConvBlock(hidden_channels, hidden_channels, kernel_size, stride, padding)
        self.conv3 = ConvBlock(hidden_channels, out_channels, 1, 1, 0, has_relu=False)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + residual
        # x = self.relu(x + residual)
        return x


# use pixel shuffle at the end
# LR: (3, ?, 1020); HR: (3, 2*?, 2040)
class Generator(nn.Module):
    def __init__(self, block_nums):
        super().__init__()
        self.block_nums = block_nums
        self.conv_input = ConvBlock(3, 64, 3, 1, 1)
        self.residual_blocks = self._make_res_blocks()
        self.conv_output = nn.Sequential(
            ConvBlock(64, 3 * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(3, 3, 3, 1, 1)
        )

    def _make_res_blocks(self):
        blocks = []
        for i in range(self.block_nums):
            blocks.append(ResidualBlock(64, 32, 64, 3, 1, 1))
        return nn.Sequential(*blocks)

    # lr -> hr: only learn the residual
    def forward(self, lr):
        x = self.conv_input(lr)
        res = self.residual_blocks(x)
        x = res + x
        hr = self.conv_output(x)
        return hr
