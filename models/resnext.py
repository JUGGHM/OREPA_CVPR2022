import torch.nn as nn
import torch.nn.functional as F
from convnet_utils import conv_bn, conv_bn_relu


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, cardinality=32, width_per_group=4):
        super(Bottleneck, self).__init__()

        D = cardinality * int(planes*width_per_group/64)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = conv_bn(in_planes, self.expansion*planes, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

        self.conv1 = conv_bn_relu(in_planes, D, kernel_size=1)
        self.conv2 = conv_bn_relu(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality)
        self.conv3 = conv_bn(D, self.expansion*planes, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNeXt(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, width_multiplier=1, cardinality=32, width_per_group=4):
        super(ResNeXt, self).__init__()

        self.in_planes = int(64 * width_multiplier)
        self.stage0 = nn.Sequential()
        self.stage0.add_module('conv1', conv_bn_relu(in_channels=3, out_channels=self.in_planes, kernel_size=7, stride=2, padding=3))
        self.stage0.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stage1 = self._make_stage(block, int(64 * width_multiplier), num_blocks[0], stride=1, cardinality=cardinality, width_per_group=width_per_group)
        self.stage2 = self._make_stage(block, int(128 * width_multiplier), num_blocks[1], stride=2, cardinality=cardinality, width_per_group=width_per_group)
        self.stage3 = self._make_stage(block, int(256 * width_multiplier), num_blocks[2], stride=2, cardinality=cardinality, width_per_group=width_per_group)
        self.stage4 = self._make_stage(block, int(512 * width_multiplier), num_blocks[3], stride=2, cardinality=cardinality, width_per_group=width_per_group)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512*block.expansion*width_multiplier), num_classes)

        output_channels = int(512*block.expansion*width_multiplier)

    def _make_stage(self, block, planes, num_blocks, stride, cardinality, width_per_group):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(block(in_planes=self.in_planes, planes=int(planes), stride=stride, cardinality=cardinality, width_per_group=width_per_group))
            self.in_planes = int(planes * block.expansion)
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def create_Res50_32x4d():
    return ResNeXt(Bottleneck, [3,4,6,3], num_classes=1000, width_multiplier=1, cardinality=32, width_per_group=4)