import torch
import torch.nn as nn
import torchvision
from config import DefaultConfig

config = DefaultConfig()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, has_act=True, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.l_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.relu = nn.ReLU(inplace=True)
        self.has_act = has_act

    def forward(self, x):
        x = self.conv(x)
        if self.has_act:
            x = self.l_relu(x)
            # x = self.relu(x)
        return x


# keep the H and W of the input and output the same

class DenseBlock(nn.Module):
    def __init__(self, feature_num=64, grow_num=16, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = ConvBlock(feature_num, grow_num, kernel_size, stride, padding)
        self.conv2 = ConvBlock(feature_num + grow_num, grow_num, kernel_size, stride, padding)
        # self.conv3 = ConvBlock(feature_num + grow_num * 2, feature_num, kernel_size, stride, padding, has_act=False)
        self.conv3 = ConvBlock(feature_num + 2 * grow_num, grow_num, kernel_size, stride, padding)
        self.conv4 = ConvBlock(feature_num + 3 * grow_num, grow_num, kernel_size, stride, padding)
        self.conv5 = ConvBlock(feature_num + 4 * grow_num, feature_num, kernel_size, stride, padding, has_act=False)

    def forward(self, x):
        res = x
        res1 = self.conv1(res)
        res2 = self.conv2(torch.cat([res, res1], dim=1))
        res3 = self.conv3(torch.cat([res, res1, res2], dim=1))
        res4 = self.conv4(torch.cat([res, res1, res2, res3], dim=1))
        res5 = self.conv5(torch.cat([res, res1, res2, res3, res4], dim=1))
        # ESRGAN: Empirically, we use 0.2 to scale the residual for better performance
        x = x + res5 * 0.2
        # x = x + res3 * 0.2
        return x


class ResidualDenseBlock(nn.Module):
    def __init__(self, feature_num=64, grow_num=16, kernel_size=3, stride=1, padding=1, block_num=3):
        super().__init__()
        self.block_num = block_num
        self.dense_blocks = \
            nn.ModuleList([DenseBlock(feature_num, grow_num, kernel_size, stride, padding)
                           for _ in range(block_num)])

    def forward(self, x):
        res = x
        for i in range(self.block_num):
            res = self.dense_blocks[i](res)
        return x + res * 0.2


# use pixel shuffle at the end
class RDSR(nn.Module):
    def __init__(self, block_num, feature_num=64, grow_num=16, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block_nums = block_num
        self.feature_num = feature_num
        self.conv_input = ConvBlock(3, feature_num, kernel_size, stride, padding, has_act=False)
        self.res_dense_blocks = self._make_res_dense_blocks(block_num, feature_num, grow_num,
                                                            kernel_size, stride, padding)
        self.res_dense_output = ConvBlock(feature_num, feature_num, kernel_size, stride, padding, has_act=False)
        self.conv_output = nn.Sequential(
            # (feature_num, H, W) -> (feature_num // 4, H * 2, W * 2)
            nn.PixelShuffle(2),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ConvBlock(feature_num // 4, feature_num // 4, kernel_size, stride, padding),
            ConvBlock(feature_num // 4, 3, kernel_size, stride, padding, has_act=False)
        )

    @staticmethod
    def _make_res_dense_blocks(block_num, feature_num=64, grow_num=16, kernel_size=3, stride=1, padding=1):
        blocks = []
        for i in range(block_num):
            blocks.append(ResidualDenseBlock(feature_num, grow_num, kernel_size, stride, padding))
        return nn.Sequential(*blocks)

    # lr -> hr: only learn the residual
    def forward(self, lr):
        x = self.conv_input(lr)
        res = self.res_dense_blocks(x)
        res = self.res_dense_output(res)
        x = x + res
        hr = self.conv_output(x)

        return hr
