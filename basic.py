import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import math
from dbb_transforms import transI_fusebn

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                             stride=1, padding=0, dilation=1, groups=1, deploy=False, nonlinear=None):
        super().__init__()
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        if deploy:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if hasattr(self, 'bn'):
            return self.nonlinear(self.bn(self.conv(x)))
        else:
            return self.nonlinear(self.conv(x))

    def switch_to_deploy(self):
        kernel, bias = transI_fusebn(self.conv.weight, self.bn)
        conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                                      stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True)
        conv.weight.data = kernel
        conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.conv = conv

class IdentityBasedConv1x1(nn.Conv2d):

    def __init__(self, channels, groups=1):
        super(IdentityBasedConv1x1, self).__init__(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)

        assert channels % groups == 0
        input_dim = channels // groups
        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value).type_as(self.weight)
        nn.init.zeros_(self.weight)

    def forward(self, input):
        kernel = self.weight + self.id_tensor.to(self.weight.device)
        result = F.conv2d(input, kernel, None, stride=1, padding=0, dilation=self.dilation, groups=self.groups)
        return result

    def get_actual_kernel(self):
        return self.weight + self.id_tensor.to(self.weight.device)


class BNAndPadLayer(nn.Module):
    def __init__(self,
                 pad_pixels,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(self.bn.running_var + self.bn.eps)
            else:
                pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

class PriorFilter(nn.Module):

    def __init__(self, channels, stride, padding, width=4):
        super(PriorFilter, self).__init__()
        self.stride = stride
        self.width = width
        self.group = int(channels/width)
        self.padding = padding
        self.prior_tensor = torch.Tensor(self.group, 1, 3, 3)

        half_g = int(self.group/2)
        for i in range(self.group):
            for h in range(3):
                for w in range(3):
                    if i < half_g:
                        self.prior_tensor[i, 0, h, w] = math.cos(math.pi*(h+0.5)*(i+1)/3)
                    else:
                        self.prior_tensor[i, 0, h, w] = math.cos(math.pi*(w+0.5)*(i+1-half_g)/3)

        self.register_buffer('prior', self.prior_tensor)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b*self.width, self.group, h, w)
        return F.conv2d(input=x, weight=self.prior, bias=None, stride=self.stride, padding=self.padding, groups=self.group).view(b, c, int(h/self.stride), int(w/self.stride)) 

class PriorConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, width=4):
        super(PriorConv, self).__init__()
        self.stride = stride
        self.width = width
        self.group = int(in_channels/width)
        self.padding = padding
        self.prior_tensor = torch.Tensor(self.group, 3, 3)

        half_g = int(self.group/2)
        for i in range(self.group):
            for h in range(3):
                for w in range(3):
                    if i < half_g:
                        self.prior_tensor[i, h, w] = math.cos(math.pi*(h+0.5)*(i+1)/3)
                    else:
                        self.prior_tensor[i, h, w] = math.cos(math.pi*(w+0.5)*(i+1-half_g)/3)

        self.register_buffer('prior', self.prior_tensor)

        self.weight  = torch.nn.Parameter(torch.FloatTensor(out_channels, width, self.group, kernel_size, kernel_size))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        c_out, wi, g, kh, kw = self.weight.size()
        weight = torch.einsum('gmn,cwgmn->cwgmn', self.prior, self.weight).view(c_out, int(wi*g), kh, kw)
        return F.conv2d(input=x, weight=weight, bias=None, stride=self.stride, padding=self.padding)