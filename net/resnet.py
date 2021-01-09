from typing import Union, Tuple, Any

from torch import nn
import math

__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1):
    """
    3 by 3 layer with padding
    @param in_planes:
    @param out_planes:
    @param stride:
    @return:
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


Block = Union[BasicBlock, Bottleneck]


class ResNet(nn.Module):
    def __init__(self, depth, num_classes, block_name='BasicBlock'):
        super().__init__()
        n, block = _depth_and_block(block_name, depth)

        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, planes=self.inplanes, blocks=n, stride=1)
        self.layer2 = self._make_layer(block, planes=32, blocks=n, stride=2)
        self.layer3 = self._make_layer(block, planes=64, blocks=n, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(in_features=64 * block.expansion, out_features=num_classes)

        _initialize(self.modules())

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = [
            block(in_planes=self.inplanes, planes=planes, stride=stride, downsample=downsample)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(in_planes=self.inplanes, planes=planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _initialize(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def _depth_and_block(block_name, depth) -> Tuple[int, Any]:
    block_name = block_name.lower()
    if block_name == 'basicblock':
        assert (depth - 2) % 6 == 0, "when using basicblock depth should be 6n+2"
        return (depth - 2) // 6, BasicBlock

    if block_name == 'bottleneck':
        assert (depth - 2) % 9 == 0, "when using bottleneck depth should be 9n+2"
        return (depth - 2) // 9, Bottleneck

    raise ValueError(f"Unsupported block name: {block_name}")


def test_resnet():
    import torch
    for num_classes in [5, 10]:
        model = ResNet(depth=20, num_classes=num_classes)
        result = model.forward(torch.zeros((4, 3, 32, 32)))
        assert result.size() == (4, num_classes)
