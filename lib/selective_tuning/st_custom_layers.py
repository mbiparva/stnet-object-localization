# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------

import torch.nn as nn
from selective_tuning.modules import AConv, APool, ALinear


class AttentiveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, relu=True):
        super(AttentiveConv, self).__init__()
        self.h = None   # Bottom Up Input Hidden Activities
        self.g = None   # Top Down Output Gating Activities

        self.bu_conv = nn.Conv2d(in_channels, out_channels,                            # Bottom Up Pass
                                 kernel_size=kernel_size, stride=stride, padding=padding)
        self.bu_relu = nn.ReLU(inplace=True) if relu else None

        self.td_conv = AConv(self.bu_conv)                                          # Top Down Pass

    def forward(self, h):
        self.h = h  # TODO: I can instead keep a copy of the data rather than the entire variable, check memory leakage
        if self.bu_relu is not None:
            return self.bu_relu(self.bu_conv(h))
        else:
            return self.bu_conv(h)

    def attend(self, g):
        self.g = self.td_conv(self.h, self.bu_conv, g)
        return self.g


class AttentivePool(nn.Module):
    def __init__(self, kernel_size, stride):
        super(AttentivePool, self).__init__()
        self.h = None   # Bottom Up Input Hidden Activities
        self.g = None   # Top Down Output Gating Activities

        self.bu_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)        # Bottom Up Pass

        self.td_pool = APool(self.bu_pool)                                      # Top Down Pass

    def forward(self, h):
        self.h = h
        return self.bu_pool(h)

    def attend(self, g):
        self.g = self.td_pool(self.h, g)
        return self.g


class AttentiveLinear(nn.Module):
    def __init__(self, in_features, out_features, second_stage, ptable_offset, relu=True):
        super(AttentiveLinear, self).__init__()
        self.h = None   # Bottom Up Input Hidden Activities
        self.g = None   # Top Down Output Gating Activities

        self.bu_linear = nn.Linear(in_features, out_features)                         # Bottom Up Pass
        self.bu_relu = nn.ReLU(inplace=True) if relu else None

        self.td_linear = ALinear(second_stage, ptable_offset)                      # Top Down Pass

    def forward(self, h):
        self.h = h
        if self.bu_relu is not None:
            return self.bu_relu(self.bu_linear(h))
        else:
            return self.bu_linear(h)

    def attend(self, g):
        self.g = self.td_linear(self.h, self.bu_linear, g)
        return self.g


class AttentiveBridge:
    def __init__(self, in_features, second_stage, ptable_offset):
        self.g = None   # Top Down Output Gating Activities
        self.second_stage = second_stage

        self.bu_linear = nn.Linear(in_features, 1, bias=False)                     # Fake weights
        self.bu_linear.weight.data.fill_(1)
        self.bu_linear.weight.requires_grad = False

        self.td_linear = ALinear(second_stage, ptable_offset)                      # Top Down Pass

    def forward(self, h):
        raise Exception('bridge layer does not have forward pass.')

    def attend(self, g_top, h):
        if self.second_stage == 'ALL':
            self.g = h
        else:
            if not self.bu_linear.weight.device == h.device:    # since the bridge is not module, manually set to device
                self.bu_linear = self.bu_linear.to(h.device)
            g = g_top[g_top != 0]
            g = g.unsqueeze(dim=-1)
            self.g = self.td_linear(h, self.bu_linear, g)
        return self.g
