
from basic import *
import numpy as np
from dbb_transforms import transI_fusebn, transII_addbranch, transIII_1x1_kxk, transV_avg, transVI_multiscale

class DBB(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, nonlinear=None, single_init=False):
        super(DBB, self).__init__()
        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.dbb_origin = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module('conv',
                                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                  stride=1, padding=0, groups=groups, bias=False))
                self.dbb_avg.add_module('bn', BNAndPadLayer(pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=0, groups=groups)
            else:
                self.dbb_avg.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))

            self.dbb_avg.add_module('avgbn', nn.BatchNorm2d(out_channels))


            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels   # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_module('idconv1', IdentityBasedConv1x1(channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=internal_channels_1x1_3x3,
                                                            kernel_size=1, stride=1, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn1', BNAndPadLayer(pad_pixels=padding, num_features=internal_channels_1x1_3x3, affine=True))
            self.dbb_1x1_kxk.add_module('conv2', nn.Conv2d(in_channels=internal_channels_1x1_3x3, out_channels=out_channels,
                                                            kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=False))
            self.dbb_1x1_kxk.add_module('bn2', nn.BatchNorm2d(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight, self.dbb_origin.bn)

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight, self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first, self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(k_1x1_kxk_first, b_1x1_kxk_first, k_1x1_kxk_second, b_1x1_kxk_second, groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg.to(self.dbb_avg.avgbn.weight.device), self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(k_1x1_avg_first, b_1x1_avg_first, k_1x1_avg_second, b_1x1_avg_second, groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch((k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged), (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def switch_to_deploy(self):
        if hasattr(self, 'dbb_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(in_channels=self.dbb_origin.conv.in_channels, out_channels=self.dbb_origin.conv.out_channels,
                                     kernel_size=self.dbb_origin.conv.kernel_size, stride=self.dbb_origin.conv.stride,
                                     padding=self.dbb_origin.conv.padding, dilation=self.dbb_origin.conv.dilation, groups=self.dbb_origin.conv.groups, bias=True)
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')

    def forward(self, inputs):

        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs))

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            torch.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            torch.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)

class OREPA_1x1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding=0, dilation=1, groups=1,
                 deploy=False, nonlinear=None, single_init=False):
        super(OREPA_1x1, self).__init__()
        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        assert groups == 1
        assert kernel_size == 1
        assert padding == kernel_size // 2

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if deploy:
            self.or1x1_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)

        else: 
            self.branch_counter = 0

            self.weight_or1x1_origin = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
            init.kaiming_uniform_(self.weight_or1x1_origin, a=math.sqrt(1.0))
            self.branch_counter += 1
    
            if out_channels > in_channels:
                self.weight_or1x1_l2i_conv1 = nn.Parameter(torch.eye(in_channels).unsqueeze(2).unsqueeze(3))
                self.weight_or1x1_l2i_conv2 = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
                init.kaiming_uniform_(self.weight_or1x1_l2i_conv2, a=math.sqrt(1.0))
            else:
                self.weight_or1x1_l2i_conv1 = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
                init.kaiming_uniform_(self.weight_or1x1_l2i_conv1, a=math.sqrt(1.0))
                self.weight_or1x1_l2i_conv2 = nn.Parameter(torch.eye(out_channels).unsqueeze(2).unsqueeze(3))
            self.branch_counter += 1

            self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
            self.bn = nn.BatchNorm2d(self.out_channels)

            init.constant_(self.vector[0, :], 1.0)
            init.constant_(self.vector[1, :], 0.5)      

            if single_init:
                #   Initialize the vector.weight of origin as 1 and others as 0. This is not the default setting.
                self.single_init()  

    def weight_gen(self):

        weight_or1x1_origin = torch.einsum('oihw,o->oihw', self.weight_or1x1_origin, self.vector[0, :])

        weight_or1x1_l2i = torch.einsum('tihw,othw->oihw', self.weight_or1x1_l2i_conv1, self.weight_or1x1_l2i_conv2)
        weight_or1x1_l2i = torch.einsum('oihw,o->oihw', weight_or1x1_l2i, self.vector[1, :])

        return weight_or1x1_origin + weight_or1x1_l2i

    def forward(self, inputs):
        if hasattr(self, 'or1x1_reparam'):
            return self.nonlinear(self.or1x1_reparam(inputs))

        weight = self.weight_gen()
        out = F.conv2d(inputs, weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return self.nonlinear(self.bn(out))

    def get_equivalent_kernel_bias(self):
        return transI_fusebn(self.weight_gen(), self.bn)

    def switch_to_deploy(self):
        if hasattr(self, 'or1x1_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.or1x1_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                     kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.or1x1_reparam.weight.data = kernel
        self.or1x1_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('weight_or1x1_origin')
        self.__delattr__('weight_or1x1_l2i_conv1')
        self.__delattr__('weight_or1x1_l2i_conv2')
        self.__delattr__('vector')
        self.__delattr__('bn')

    def init_gamma(self, gamma_value):
        init.constant_(self.vector, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        init.constant_(self.vector[0, :], 1.0)

class OREPA(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False,
                 nonlinear=None,
                 single_init=False, 
                 weight_only=False,
                 init_hyper_para=1.0, init_hyper_gamma=1.0):
        super(OREPA, self).__init__()
        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        self.weight_only = weight_only

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if deploy:
            self.orepa_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.branch_counter = 0

            self.weight_orepa_origin = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups),
                            kernel_size, kernel_size))
            init.kaiming_uniform_(self.weight_orepa_origin, a=math.sqrt(0.0))
            self.branch_counter += 1

            self.weight_orepa_avg_conv = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1,
                            1))
            self.weight_orepa_pfir_conv = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1,
                            1))
            init.kaiming_uniform_(self.weight_orepa_avg_conv, a=0.0)
            init.kaiming_uniform_(self.weight_orepa_pfir_conv, a=0.0)
            self.register_buffer(
                'weight_orepa_avg_avg',
                torch.ones(kernel_size,
                        kernel_size).mul(1.0 / kernel_size / kernel_size))
            self.branch_counter += 1
            self.branch_counter += 1

            self.weight_orepa_1x1 = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1,
                            1))
            init.kaiming_uniform_(self.weight_orepa_1x1, a=0.0)
            self.branch_counter += 1

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups <= 4 else 2 * in_channels

            if internal_channels_1x1_3x3 == in_channels:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(
                    torch.zeros(in_channels, int(in_channels / self.groups), 1, 1))
                id_value = np.zeros(
                    (in_channels, int(in_channels / self.groups), 1, 1))
                for i in range(in_channels):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(
                    self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer('id_tensor', id_tensor)

            else:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(
                    torch.zeros(internal_channels_1x1_3x3,
                                int(in_channels / self.groups), 1, 1))
                id_value = np.zeros(
                    (internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1))
                for i in range(internal_channels_1x1_3x3):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(
                    self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer('id_tensor', id_tensor)
                #init.kaiming_uniform_(
                    #self.weight_orepa_1x1_kxk_conv1, a=math.sqrt(0.0))
            self.weight_orepa_1x1_kxk_conv2 = nn.Parameter(
                torch.Tensor(out_channels,
                            int(internal_channels_1x1_3x3 / self.groups),
                            kernel_size, kernel_size))
            init.kaiming_uniform_(self.weight_orepa_1x1_kxk_conv2, a=math.sqrt(0.0))
            self.branch_counter += 1

            expand_ratio = 8
            self.weight_orepa_gconv_dw = nn.Parameter(
                torch.Tensor(in_channels * expand_ratio, 1, kernel_size,
                            kernel_size))
            self.weight_orepa_gconv_pw = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels * expand_ratio / self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_orepa_gconv_dw, a=math.sqrt(0.0))
            init.kaiming_uniform_(self.weight_orepa_gconv_pw, a=math.sqrt(0.0))
            self.branch_counter += 1

            self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
            if weight_only is False:
                self.bn = nn.BatchNorm2d(self.out_channels)

            self.fre_init()

            init.constant_(self.vector[0, :], 0.25 * math.sqrt(init_hyper_gamma))  #origin
            init.constant_(self.vector[1, :], 0.25 * math.sqrt(init_hyper_gamma))  #avg
            init.constant_(self.vector[2, :], 0.0 * math.sqrt(init_hyper_gamma))  #prior
            init.constant_(self.vector[3, :], 0.5 * math.sqrt(init_hyper_gamma))  #1x1_kxk
            init.constant_(self.vector[4, :], 1.0 * math.sqrt(init_hyper_gamma))  #1x1
            init.constant_(self.vector[5, :], 0.5 * math.sqrt(init_hyper_gamma))  #dws_conv

            self.weight_orepa_1x1.data = self.weight_orepa_1x1.mul(init_hyper_para)
            self.weight_orepa_origin.data = self.weight_orepa_origin.mul(init_hyper_para)
            self.weight_orepa_1x1_kxk_conv2.data = self.weight_orepa_1x1_kxk_conv2.mul(init_hyper_para)
            self.weight_orepa_avg_conv.data = self.weight_orepa_avg_conv.mul(init_hyper_para)
            self.weight_orepa_pfir_conv.data = self.weight_orepa_pfir_conv.mul(init_hyper_para)

            self.weight_orepa_gconv_dw.data = self.weight_orepa_gconv_dw.mul(math.sqrt(init_hyper_para))
            self.weight_orepa_gconv_pw.data = self.weight_orepa_gconv_pw.mul(math.sqrt(init_hyper_para))

            if single_init:
                #   Initialize the vector.weight of origin as 1 and others as 0. This is not the default setting.
                self.single_init()  

    def fre_init(self):
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size,
                                    self.kernel_size)
        half_fg = self.out_channels / 2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi * (h + 0.5) *
                                                         (i + 1) / 3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi * (w + 0.5) *
                                                         (i + 1 - half_fg) / 3)

        self.register_buffer('weight_orepa_prior', prior_tensor)

    def weight_gen(self):
        weight_orepa_origin = torch.einsum('oihw,o->oihw',
                                          self.weight_orepa_origin,
                                          self.vector[0, :])

        weight_orepa_avg = torch.einsum('oihw,hw->oihw', self.weight_orepa_avg_conv, self.weight_orepa_avg_avg)
        weight_orepa_avg = torch.einsum(
             'oihw,o->oihw',
             torch.einsum('oi,hw->oihw', self.weight_orepa_avg_conv.squeeze(3).squeeze(2),
                          self.weight_orepa_avg_avg), self.vector[1, :])


        weight_orepa_pfir = torch.einsum(
            'oihw,o->oihw',
            torch.einsum('oi,ohw->oihw', self.weight_orepa_pfir_conv.squeeze(3).squeeze(2),
                          self.weight_orepa_prior), self.vector[2, :])

        weight_orepa_1x1_kxk_conv1 = None
        if hasattr(self, 'weight_orepa_1x1_kxk_idconv1'):
            weight_orepa_1x1_kxk_conv1 = (self.weight_orepa_1x1_kxk_idconv1 +
                                        self.id_tensor).squeeze(3).squeeze(2)
        elif hasattr(self, 'weight_orepa_1x1_kxk_conv1'):
            weight_orepa_1x1_kxk_conv1 = self.weight_orepa_1x1_kxk_conv1.squeeze(3).squeeze(2)
        else:
            raise NotImplementedError
        weight_orepa_1x1_kxk_conv2 = self.weight_orepa_1x1_kxk_conv2

        if self.groups > 1:
            g = self.groups
            t, ig = weight_orepa_1x1_kxk_conv1.size()
            o, tg, h, w = weight_orepa_1x1_kxk_conv2.size()
            weight_orepa_1x1_kxk_conv1 = weight_orepa_1x1_kxk_conv1.view(
                g, int(t / g), ig)
            weight_orepa_1x1_kxk_conv2 = weight_orepa_1x1_kxk_conv2.view(
                g, int(o / g), tg, h, w)
            weight_orepa_1x1_kxk = torch.einsum('gti,gothw->goihw',
                                              weight_orepa_1x1_kxk_conv1,
                                              weight_orepa_1x1_kxk_conv2).reshape(
                                                  o, ig, h, w)
        else:
            weight_orepa_1x1_kxk = torch.einsum('ti,othw->oihw',
                                              weight_orepa_1x1_kxk_conv1,
                                              weight_orepa_1x1_kxk_conv2)
        weight_orepa_1x1_kxk = torch.einsum('oihw,o->oihw', weight_orepa_1x1_kxk, self.vector[3, :])

        weight_orepa_1x1 = 0
        if hasattr(self, 'weight_orepa_1x1'):
            weight_orepa_1x1 = transVI_multiscale(self.weight_orepa_1x1,
                                                self.kernel_size)
            weight_orepa_1x1 = torch.einsum('oihw,o->oihw', weight_orepa_1x1,
                                           self.vector[4, :])

        weight_orepa_gconv = self.dwsc2full(self.weight_orepa_gconv_dw,
                                          self.weight_orepa_gconv_pw,
                                          self.in_channels, self.groups)
        weight_orepa_gconv = torch.einsum('oihw,o->oihw', weight_orepa_gconv,
                                        self.vector[5, :])

        weight = weight_orepa_origin + weight_orepa_avg + weight_orepa_1x1 + weight_orepa_1x1_kxk + weight_orepa_pfir + weight_orepa_gconv

        return weight

    def dwsc2full(self, weight_dw, weight_pw, groups, groups_conv=1):

        t, ig, h, w = weight_dw.size()
        o, _, _, _ = weight_pw.size()
        tg = int(t / groups)
        i = int(ig * groups)
        ogc = int(o / groups_conv)
        groups_gc = int(groups / groups_conv)
        weight_dw = weight_dw.view(groups_conv, groups_gc, tg, ig, h, w)
        weight_pw = weight_pw.squeeze().view(ogc, groups_conv, groups_gc, tg)

        weight_dsc = torch.einsum('cgtihw,ocgt->cogihw', weight_dw, weight_pw)
        return weight_dsc.reshape(o, int(i/groups_conv), h, w)

    def forward(self, inputs=None):
        if hasattr(self, 'orepa_reparam'):
            return self.nonlinear(self.orepa_reparam(inputs))
        
        weight = self.weight_gen()

        if self.weight_only is True:
            return weight

        out = F.conv2d(
            inputs,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)
        return self.nonlinear(self.bn(out))

    def get_equivalent_kernel_bias(self):
        return transI_fusebn(self.weight_gen(), self.bn)

    def switch_to_deploy(self):
        if hasattr(self, 'or1x1_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.orepa_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                     kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.orepa_reparam.weight.data = kernel
        self.orepa_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('weight_orepa_origin')
        self.__delattr__('weight_orepa_1x1')
        self.__delattr__('weight_orepa_1x1_kxk_conv2')
        if hasattr(self, 'weight_orepa_1x1_kxk_idconv1'):
            self.__delattr__('id_tensor')
            self.__delattr__('weight_orepa_1x1_kxk_idconv1')
        elif hasattr(self, 'weight_orepa_1x1_kxk_conv1'):
            self.__delattr__('weight_orepa_1x1_kxk_conv1')
        else:
            raise NotImplementedError
        self.__delattr__('weight_orepa_avg_avg')  
        self.__delattr__('weight_orepa_avg_conv')
        self.__delattr__('weight_orepa_pfir_conv')
        self.__delattr__('weight_orepa_prior')
        self.__delattr__('weight_orepa_gconv_dw')
        self.__delattr__('weight_orepa_gconv_pw')

        self.__delattr__('bn')
        self.__delattr__('vector')

    def init_gamma(self, gamma_value):
        init.constant_(self.vector, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        init.constant_(self.vector[0, :], 1.0)

class OREPA_LargeConvBase(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,  dilation=1, groups=1, deploy=False, nonlinear=None):
        super(OREPA_LargeConvBase, self).__init__()
        assert kernel_size % 2 == 1 and kernel_size > 3
        
        self.stride = stride
        self.padding = padding
        self.layers = int((kernel_size - 1) / 2)
        self.groups = groups
        self.dilation = dilation

        internal_channels = out_channels

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        if deploy:
            self.or_large_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)

        else:
            for i in range(self.layers):
                if i == 0:
                    self.__setattr__('weight'+str(i), nn.Parameter(torch.Tensor(internal_channels, int(in_channels/self.groups), 3, 3)))
                elif i == self.layers - 1:
                    self.__setattr__('weight'+str(i), nn.Parameter(torch.Tensor(out_channels, int(internal_channels/self.groups), 3, 3)))
                else:
                    self.__setattr__('weight'+str(i), nn.Parameter(torch.Tensor(internal_channels, int(internal_channels/self.groups), 3, 3)))
                init.kaiming_uniform_(getattr(self, 'weight'+str(i)), a=math.sqrt(5))

            self.bn = nn.BatchNorm2d(out_channels)
            #self.unfold = torch.nn.Unfold(kernel_size=3, dilation=1, padding=2, stride=1)

    def weight_gen(self):
        weight = getattr(self, 'weight'+str(0)).transpose(0, 1)
        for i in range(self.layers - 1):
            weight2 = getattr(self, 'weight'+str(i+1))
            weight = F.conv2d(weight, weight2, groups=self.groups, padding=2)
        
        return weight.transpose(0, 1)
        '''
        weight = getattr(self, 'weight'+str(0)).transpose(0, 1)
        for i in range(self.layers - 1):
            weight = self.unfold(weight)
            weight2 = getattr(self, 'weight'+str(i+1))

            weight = torch.einsum('akl,bk->abl', weight, weight2.view(weight2.size(0), -1))
            k = i * 2 + 5
            weight = weight.view(weight.size(0), weight.size(1), k, k)
        
        return weight.transpose(0, 1)
        '''

    def forward(self, inputs):
        if hasattr(self, 'or_large_reparam'):
            return self.nonlinear(self.or_large_reparam(inputs))
        
        weight = self.weight_gen()
        out = F.conv2d(inputs, weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return self.nonlinear(self.bn(out))

    def get_equivalent_kernel_bias(self):
        return transI_fusebn(self.weight_gen(), self.bn)

    def switch_to_deploy(self):
        if hasattr(self, 'or_large_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.or_large_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                     kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.or_large_reparam.weight.data = kernel
        self.or_large_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        for i in range(self.layers):
            self.__delattr__('weight'+str(i))
        self.__delattr__('bn')


class OREPA_LargeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, deploy=False, nonlinear=None):
        super(OREPA_LargeConv, self).__init__()
        assert kernel_size % 2 == 1 and kernel_size > 3
        
        self.stride = stride
        self.padding = padding
        self.layers = int((kernel_size - 1) / 2)
        self.groups = groups
        self.dilation = dilation

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        internal_channels = out_channels

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        if deploy:
            self.or_large_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)

        else:
            for i in range(self.layers):
                if i == 0:
                    self.__setattr__('weight'+str(i), OREPA(in_channels, internal_channels, kernel_size=3, stride=1, padding=1, groups=groups, weight_only=True))
                elif i == self.layers - 1:
                    self.__setattr__('weight'+str(i), OREPA(internal_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, weight_only=True))
                else:
                    self.__setattr__('weight'+str(i), OREPA(internal_channels, internal_channels, kernel_size=3, stride=1, padding=1, weight_only=True))

            self.bn = nn.BatchNorm2d(out_channels)
            #self.unfold = torch.nn.Unfold(kernel_size=3, dilation=1, padding=2, stride=1)

    def weight_gen(self):
        weight = getattr(self, 'weight'+str(0)).weight_gen().transpose(0, 1)
        for i in range(self.layers - 1):
            weight2 = getattr(self, 'weight'+str(i+1)).weight_gen()
            weight = F.conv2d(weight, weight2, groups=self.groups, padding=2)
        
        return weight.transpose(0, 1)
        '''
        weight = getattr(self, 'weight'+str(0))(inputs=None).transpose(0, 1)
        for i in range(self.layers - 1):
            weight = self.unfold(weight)
            weight2 = getattr(self, 'weight'+str(i+1))(inputs=None)

            weight = torch.einsum('akl,bk->abl', weight, weight2.view(weight2.size(0), -1))
            k = i * 2 + 5
            weight = weight.view(weight.size(0), weight.size(1), k, k)
        
        return weight.transpose(0, 1)
        '''

    def forward(self, inputs):
        if hasattr(self, 'or_large_reparam'):
            return self.nonlinear(self.or_large_reparam(inputs))

        weight = self.weight_gen()
        out = F.conv2d(inputs, weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return self.nonlinear(self.bn(out))

    def get_equivalent_kernel_bias(self):
        return transI_fusebn(self.weight_gen(), self.bn)

    def switch_to_deploy(self):
        if hasattr(self, 'or_large_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.or_large_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                     kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.or_large_reparam.weight.data = kernel
        self.or_large_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        for i in range(self.layers):
            self.__delattr__('weight'+str(i))
        self.__delattr__('bn')
