'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

This is a standard ResNet setup. 
For numerical stability we add a soft_lrelu. 
For compatibility with PSGD Affine we wrapped the layers into Affine structures
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

class AffineConv2d(torch.nn.Module):
    """
    Let's wrap function
        torch.nn.functional.conv2d
    as a class. The affine transform is
        [vectorized(image patch), 1] @ W
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super(AffineConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        self.out_in_height_width = (out_channels, in_channels, kernel_size, kernel_size)

        std = (in_channels * kernel_size**2) ** (-0.5)
        w = torch.empty(out_channels, in_channels * kernel_size ** 2).normal_(std=std)
        if bias:
            b = torch.zeros(out_channels, 1)
            self.weight = torch.nn.Parameter(torch.cat([w, b], dim=1))
        else:
            self.weight = torch.nn.Parameter(w)

    def forward(self, x):
        if self.has_bias:
            return F.conv2d(
                x,
                self.weight[:, :-1].view(self.out_in_height_width),
                bias=self.weight[:, -1],
                stride=self.stride,
                padding=self.padding,
            )
        else:
            return F.conv2d(
                x,
                self.weight.view(self.out_in_height_width),
                stride=self.stride,
                padding=self.padding,
            )



class AffineLinear(torch.nn.Module):
    """
    A linear layer clearly is an affine transform
    """
    def __init__(self, in_features, out_features, has_bias=True):
        super(AffineLinear, self).__init__()
        self.has_bias = has_bias
        w = torch.empty(in_features, out_features).normal_(std=in_features ** (-0.5))
        if has_bias:
            b = torch.zeros(1, out_features)
            self.weight = torch.nn.Parameter(torch.cat([w, b]))
        else:
            self.weight = torch.nn.Parameter(w)

    def forward(self, x):
        if self.has_bias:
            return x @ self.weight[:-1] + self.weight[-1]
        else:
            return x @ self.weight


import torch
import torch.nn as nn
import math

class AffineBatchNorm2d(nn.Module):
    def __init__(self, N, momentum=0.1, eps=1e-5, track_running_stats=True):
        super(AffineBatchNorm2d, self).__init__()
        self.N = N
        self.momentum = momentum
        self.eps = eps
        self.track_running_stats = track_running_stats

        # Xavier initialization for the scale parameter (gamma)
        gamma = torch.Tensor(N).uniform_(-1, 1)
        gamma = gamma * math.sqrt(2.0 / (N + N))

        # Zero initialization for the shift parameter (beta)
        beta = torch.zeros(N)

        self.p = nn.Parameter(torch.cat([gamma.unsqueeze(0), beta.unsqueeze(0)]))

        if self.track_running_stats:
            self.running_mean = nn.Parameter(torch.zeros(N), requires_grad=False)
            self.running_var = nn.Parameter(torch.ones(N), requires_grad=False)

    def forward(self, x):
        if self.training:
            mean = torch.mean(x, dim=[0, 2, 3])
            var = torch.var(x, dim=[0, 2, 3], unbiased=False)

            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                    self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)

            x = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        else:
            x = (x - self.running_mean[None, :, None, None]) / torch.sqrt(self.running_var[None, :, None, None] + self.eps)

        x = self.p[:1, :, None, None] * x + self.p[1:, :, None, None]
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,shortcut_connection=True):
        super(BasicBlock, self).__init__()
        self.conv1 = AffineConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = AffineBatchNorm2d(planes)
        self.conv2 = AffineConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = AffineBatchNorm2d(planes)
        self.shortcut_connection = shortcut_connection

        if shortcut_connection:
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    AffineConv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    AffineBatchNorm2d(self.expansion * planes),
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
        self.conv1 = AffineConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = AffineBatchNorm2d(planes)
        self.conv2 = AffineConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = AffineBatchNorm2d(planes)
        self.conv3 = AffineConv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = AffineBatchNorm2d(self.expansion * planes)
        self.shortcut_connection = shortcut_connection
        
        if shortcut_connection:
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    AffineConv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    AffineBatchNorm2d(self.expansion * planes),
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

        self.conv1 = AffineConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = AffineBatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = AffineLinear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
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


def ResNet18Affine(shortcut_connection):
    return ResNet(BasicBlock, [2, 2, 2, 2],shortcut_connection=shortcut_connection)


def ResNet34Affine(shortcut_connection):
    return ResNet(BasicBlock, [3, 4, 6, 3],shortcut_connection=shortcut_connection)


def ResNet50Affine(shortcut_connection):
    return ResNet(Bottleneck, [3, 4, 6, 3],shortcut_connection=shortcut_connection)


def ResNet101Affine(shortcut_connection):
    return ResNet(Bottleneck, [3, 4, 23, 3],shortcut_connection=shortcut_connection)


def ResNet152Affine(shortcut_connection):
    return ResNet(Bottleneck, [3, 8, 36, 3],shortcut_connection=shortcut_connection)
