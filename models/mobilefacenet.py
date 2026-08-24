
from typing import OrderedDict

import torch.nn.functional as F
import torch
from torch import nn

from models import Model, RegressionHead


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, padding=padding, groups=in_planes, bias=bias)
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class GDConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=in_planes, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileFaceNet(Model):
    first_channel = 64
    hidden_features = 64
    out_features = 128
    input_channel = 64
    last_channel = 512

    def __init__(self, num_outputs: int = 1, num_channels: int = 1, width_mult: float = 1.0, inverted_residual_setting=None, round_nearest: int = 8, dropout_rate: float = 0.3, freezed: bool = False):
        self.width_mult = width_mult
        self.inverted_residual_setting = inverted_residual_setting
        self.round_nearest = round_nearest

        super().__init__(model_name="mobilefacenet", num_channels=num_channels, num_outputs=num_outputs, dropout_rate=dropout_rate, freezed=freezed)

    @staticmethod
    def _make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def initialize_backbone(self):
        # Initialize MobileFaceNet backbone
        block = InvertedResidual
        last_channel = 512
        input_channel = self.__first_channel

        if self.inverted_residual_setting is None:
            # Configuration originale
            self.inverted_residual_setting = [
                [2, 64, 5, 2],
                [4, 128, 1, 2],
                [2, 128, 6, 1],
                [4, 128, 1, 2],
                [2, 128, 2, 1],
            ]

        self.last_channel = self._make_divisible(last_channel * max(1.0, self.width_mult), self.round_nearest)

        feature_list = list()
        for t, c, n, s in self.inverted_residual_setting:
            output_channel = self._make_divisible(c * self.width_mult, self.round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1

                feature_list.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.gdconv_kernel_size = 6   

        backbone = nn.Sequential(
            OrderedDict([
                ("conv1", ConvBNReLU(self.num_channels, self.first_channel, stride=1)),
                ("dw_conv", DepthwiseSeparableConv(in_planes=self.first_channel, out_planes=self.first_channel, kernel_size=3, padding=1)),
                ("features", nn.Sequential(*feature_list)  ),
                ("conv2", ConvBNReLU(input_channel, self.last_channel, kernel_size=1)),
                ("gdconv", GDConv(in_planes=self.last_channel, out_planes=self.last_channel, kernel_size=self.gdconv_kernel_size, padding=0)),
                ("conv3", nn.Conv2d(self.last_channel, self.out_features, kernel_size=1)),
                ("bn", nn.BatchNorm2d(self.out_features))
            ])
        )
        return backbone

    def set_classifier(self):
        self.head = RegressionHead(self.out_features, self.hidden_features, self.num_outputs, self.dropout_rate)


    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.dw_conv(x)
        x = self.backbone.features(x)
        x = self.backbone.conv2(x)
        
        h, w = x.shape[2], x.shape[3]
        if h != self.gdconv_kernel_size:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        else:
            x = self.backbone.gdconv(x)
            
        x = self.backbone.conv3(x)
        x = self.backbone.bn(x)
        x = x.view(x.size(0), -1) # Flatten (Batch, 128)
        
        return self.head(x)
        


if __name__ == "__main__":
    print(f"Testing MobileFaceNet...")
    model = MobileFaceNet(num_outputs=1, num_channels=1, dropout_rate=0.3, freezed=True)
    dummy_input = torch.randn(4, 1, 48, 48)  # Batch of size 4, 1 channel, 48x48 images
    
    try:
        output = model(dummy_input)
        print(f"MobileNet Input: {dummy_input.shape} -> Output: {output.shape}")
        print("Success!")
    except Exception as e:
        print(f"Error with MobileNet: {e}")