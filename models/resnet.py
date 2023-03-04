'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

This is a standard ResNet setup. For numerical stability we add a soft_lrelu. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_lrelu(x):
    # Reducing to ReLU when a=0.5 and e=0
    # Here, we set a-->0.5 from left and e-->0 from right,
    # where adding eps is to make the derivatives have better rounding behavior around 0.
    a = 0.49
    e = torch.finfo(torch.float32).eps
    return (1-a)*x + a*torch.sqrt(x*x + e*e) - a*e

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,shortcut_connection=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut_connection = shortcut_connection

        if self.shortcut_connection:
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = soft_lrelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut_connection:
            out += self.shortcut(x)
        out = soft_lrelu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,shortcut_connection=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut_connection = shortcut_connection

        if self.shortcut_connection:
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = soft_lrelu(self.bn1(self.conv1(x)))
        out = soft_lrelu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.shortcut_connection:
            out += self.shortcut(x)
        out = soft_lrelu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,shortcut_connection=True):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,shortcut_connection=shortcut_connection)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,shortcut_connection=shortcut_connection)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,shortcut_connection=shortcut_connection)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,shortcut_connection=shortcut_connection)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride,shortcut_connection=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,shortcut_connection=shortcut_connection))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = soft_lrelu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(shortcut_connection):
    return ResNet(BasicBlock, [2, 2, 2, 2],shortcut_connection=shortcut_connection)


def ResNet34(shortcut_connection):
    return ResNet(BasicBlock, [3, 4, 6, 3],shortcut_connection=shortcut_connection)


def ResNet50(shortcut_connection):
    return ResNet(Bottleneck, [3, 4, 6, 3],shortcut_connection=shortcut_connection)


def ResNet101(shortcut_connection):
    return ResNet(Bottleneck, [3, 4, 23, 3],shortcut_connection=shortcut_connection)


def ResNet152(shortcut_connection):
    return ResNet(Bottleneck, [3, 8, 36, 3],shortcut_connection=shortcut_connection)

def test():
    net = ResNet18(shortcut_connection=True)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
