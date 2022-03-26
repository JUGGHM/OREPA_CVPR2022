import torch.nn as nn

from basic import ConvBN
from blocks import DBB, OREPA_1x1, OREPA, OREPA_LargeConvBase, OREPA_LargeConv
from blocks_repvgg import RepVGGBlock, RepVGGBlock_OREPA

CONV_BN_IMPL = 'base'

DEPLOY_FLAG = False

def choose_blk(kernel_size):
    if CONV_BN_IMPL == 'OREPA':
        if kernel_size == 1:
            blk_type = OREPA_1x1
        elif kernel_size >= 7:
            blk_type = OREPA_LargeConv
        else:
            blk_type = OREPA
    elif CONV_BN_IMPL == 'base' or kernel_size == 1 or kernel_size >= 7:
        blk_type = ConvBN
    elif CONV_BN_IMPL == 'DBB':
        blk_type = DBB
    elif CONV_BN_IMPL == 'RepVGG':
        blk_type = RepVGGBlock
    elif CONV_BN_IMPL == 'OREPA_VGG':
        blk_type = RepVGGBlock_OREPA
    else:
        raise NotImplementedError
    
    return blk_type

def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, assign_type=None):
    if assign_type is not None:
        blk_type = assign_type
    else:
        blk_type = choose_blk(kernel_size)
    return blk_type(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG)

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, assign_type=None):
    if assign_type is not None:
        blk_type = assign_type
    else:
        blk_type = choose_blk(kernel_size)
    return blk_type(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, groups=groups, deploy=DEPLOY_FLAG, nonlinear=nn.ReLU())

def switch_conv_bn_impl(block_type):
    global CONV_BN_IMPL
    CONV_BN_IMPL = block_type

def switch_deploy_flag(deploy):
    global DEPLOY_FLAG
    DEPLOY_FLAG = deploy
    print('deploy flag: ', DEPLOY_FLAG)

def build_model(arch):
    if arch == 'ResNet-18':
        from models.resnet import create_Res18
        model = create_Res18()
    elif arch == 'ResNet-34':
        from models.resnet import create_Res34
        model = create_Res34()
    elif arch == 'ResNet-50':
        from models.resnet import create_Res50
        model = create_Res50()
    elif arch == 'ResNet-101':
        from models.resnet import create_Res101
        model = create_Res101()
    elif arch == 'RepVGG-A0':
        from models.repvgg import create_RepVGG_A0
        model = create_RepVGG_A0()
    elif arch == 'RepVGG-A1':
        from models.repvgg import create_RepVGG_A1
        model = create_RepVGG_A1()
    elif arch == 'RepVGG-A2':
        from models.repvgg import create_RepVGG_A2
        model = create_RepVGG_A2()
    elif arch == 'RepVGG-B1':
        from models.repvgg import create_RepVGG_B1
        model = create_RepVGG_B1()
    elif arch == 'ResNet-18-1.5x':
        from models.resnet import create_Res18_1d5x
        model = create_Res18_1d5x()
    elif arch == 'ResNet-18-2x':
        from models.resnet import create_Res18_2x
        model = create_Res18_2x()
    elif arch == 'ResNeXt-50':
        from models.resnext import create_Res50_32x4d
        model = create_Res50_32x4d()
    #elif arch == 'RegNet-800MF':
        #from models.regnet import create_Reg800MF
        #model = create_Reg800MF()
    #elif arch == 'ConvNext-T-0.5x':
        #from models.convnext import convnext_tiny_0d5x
        #model = convnext_tiny_0d5x()
    else:
        raise ValueError('TODO')
    return model