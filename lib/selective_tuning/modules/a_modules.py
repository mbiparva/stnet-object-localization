# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------

import torch
from torch.nn import Module
from selective_tuning.functions import aconv, apool, alinear


class AConv(Module):
    def __init__(self, ff_conv):
        super(AConv, self).__init__()
        self.kernel_size = (ff_conv.in_channels, ) + ff_conv.kernel_size
        self.stride = (1,) + ff_conv.stride
        self.padding = (1,) + ff_conv.padding
        self.dilation = (1,) + ff_conv.dilation
        self.group = ff_conv.groups

    def forward(self, ff_h, ff_conv, a):
        return aconv(ff_h, ff_conv.weight, a,
                     self.kernel_size,
                     self.stride,
                     self.padding,
                     self.dilation,
                     self.group)


class APool(Module):
    def __init__(self, ff_pool):
        super(APool, self).__init__()
        self.kernel_size = (0, ) + (ff_pool.kernel_size, )*2
        self.stride = (1,) + (ff_pool.stride, )*2
        self.padding = (1,) + (ff_pool.padding, )*2
        self.dilation = (1,) + (ff_pool.dilation, )*2
        self.ptable = torch.FloatTensor([[1, 3],
                                         [1, 2],
                                         [1, 1],
                                         [1, 0],
                                         [1/2., 0],
                                         [1/5., 0]])
        self.ptable_offset = 0
        self.second_stage = 0

    def forward(self, ff_h, a):
        return apool(ff_h, a, self.ptable,
                     self.ptable_offset, self.second_stage,
                     self.kernel_size,
                     self.stride,
                     self.padding,
                     self.dilation)


class ALinear(Module):
    def __init__(self, second_stage_in, ptable_offset_in):
        super(ALinear, self).__init__()
        self.kernel_size = (0, 0, 0)  # Not used in the CUDA code
        self.stride = (0, 0, 0)
        self.padding = (0, 0, 0)
        self.dilation = (0, 0, 0)
        self.ptable = torch.FloatTensor([[1, 3],
                                         [1, 2],
                                         [1, 1],
                                         [1, 0],
                                         [1 / 2., 0],
                                         [1 / 5., 0]])
        self.ptable_offset = ptable_offset_in
        self.second_stage = {'1': 0, 'ALL': 1, '699': 2}[second_stage_in]

    def forward(self, ff_h, ff_linear, a):
        return alinear(ff_h, ff_linear.weight, a, self.ptable,
                       self.ptable_offset, self.second_stage,
                       self.kernel_size,
                       self.stride,
                       self.padding,
                       self.dilation)
