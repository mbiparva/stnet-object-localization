# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------

import torch
from torch.autograd import Function
from selective_tuning.functions.attentive_layers import attentive_conv, attentive_pool, attentive_linear


class AConvFunction(Function):
    @staticmethod
    def forward(ctx, *args):
        ff_h, ff_conv_w, a, kernel_size, stride, padding, dilation, group = args
        out = torch.zeros_like(ff_h)
        theta = torch.zeros_like(a)
        out_k = torch.zeros_like(ff_conv_w)
        out_k_hit = torch.zeros_like(ff_conv_w)
        if ff_h.is_cuda:
            attentive_conv(
                ff_h,
                ff_conv_w,
                a,
                out,
                theta,
                out_k,
                out_k_hit,
                *kernel_size,
                *stride,
                *padding,
                *dilation,
                group)
        else:
            raise NotImplementedError
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError


class APoolFunction(Function):
    @staticmethod
    def forward(ctx, *args):
        ff_h, a, ptable, ptable_off, second_stage, kernel_size, stride, padding, dilation = args
        out = torch.zeros_like(ff_h)
        ptable = ptable.to(ff_h.device)
        if ff_h.is_cuda:
            attentive_pool(
                ff_h,
                a,
                out,
                ptable,
                ptable_off, second_stage,
                *kernel_size,
                *stride,
                *padding,
                *dilation)

        else:
            raise NotImplementedError
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError


class ALinearFunction(Function):
    @staticmethod
    def forward(ctx, *args):
        ff_h, ff_conv_w, a, ptable, ptable_off, second_stage, kernel_size, stride, padding, dilation = args
        ff_h_ndim = ff_h.ndimension()  # CUDA implementation expect spatial bot and top hidden planes
        if ff_h_ndim == 2:
            ff_h = ff_h.unsqueeze(dim=-1).unsqueeze(dim=-1)
            a = a.unsqueeze(dim=-1).unsqueeze(dim=-1)
        elif ff_h_ndim == 4:
            pass
        else:
            raise NotImplementedError

        out = torch.zeros_like(ff_h)
        theta = torch.zeros_like(a)
        out_k = torch.zeros_like(ff_conv_w)
        out_k_hit = torch.zeros_like(ff_conv_w)
        ptable = ptable.to(ff_h.device)
        if ff_h.is_cuda:
            attentive_linear(
                ff_h,
                ff_conv_w,
                a,
                out,
                theta,
                out_k,
                out_k_hit,
                ptable,
                ptable_off, second_stage,
                *kernel_size,
                *stride,
                *padding,
                *dilation)
        else:
            raise NotImplementedError

        if ff_h_ndim == 2:
            out = out.squeeze(dim=-1).squeeze(dim=-1)
        elif ff_h_ndim == 4:
            pass
        else:
            raise NotImplementedError

        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError


# Naturally we call the apply function rather than
# directly instantiate from the Function sub-classes
aconv = AConvFunction.apply
apool = APoolFunction.apply
alinear = ALinearFunction.apply
