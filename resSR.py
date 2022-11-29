import torch
import torch.nn as nn
import torchvision
from config import DefaultConfig

config = DefaultConfig()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.relu(x)
        return x


# keep the H and W of the input and output the same
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # 1x1 at the first and end to reduce channels
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1, 1, 0)
        self.conv2 = ConvBlock(hidden_channels, hidden_channels, kernel_size, stride, padding)
        self.conv3 = ConvBlock(hidden_channels, out_channels, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        res = self.conv1(res)
        res = self.conv2(res)
        res = self.conv3(res)
        x = x + res
        # res = self.relu(res + residual)
        return x


# use pixel shuffle at the end
# LR: (3, 678, 1020); HR: (3, 1356, 2040)
class Generator(nn.Module):
    def __init__(self, block_nums):
        super().__init__()
        self.block_nums = block_nums
        self.conv_input = ConvBlock(3, 64, 3, 1, 1)
        self.residual_blocks = self._make_res_blocks()
        self.conv_output = nn.Sequential(
            # test conv or not
            ConvBlock(64, 3 * 2, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 3, 3, 1, 1)  # 3
        )

    def _make_res_blocks(self):
        blocks = []
        for i in range(self.block_nums):
            blocks.append(ResidualBlock(64, 32, 64, 3, 1, 1))
        return nn.Sequential(*blocks)

    # lr -> hr: only learn the residual
    def forward(self, lr):
        x = self.conv_input(lr)
        x = self.residual_blocks(x)
        x = self.conv_output(x)
        hr = x + lr
        return hr
