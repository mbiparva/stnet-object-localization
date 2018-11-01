# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------

import glob
import torch
from os import path as osp
from torch.utils.ffi import create_extension
# import shutil

abs_path = osp.dirname(osp.realpath(__file__))
extra_objects = [osp.join(abs_path, 'build/attentive_cuda_kernel.o'), osp.join(abs_path, 'build/attentive_cuda_kernel_dlink.o')]
extra_objects += glob.glob('/usr/local/cuda/lib64/*.a')

ffi = create_extension(
    'attentive_layers',
    headers=['include/attentive_cuda.h'],
    sources=['src/attentive_cuda.c'],
    define_macros=[('WITH_CUDA', None)],
    relative_to=__file__,
    with_cuda=True,
    extra_objects=extra_objects,
    include_dirs=[osp.join(abs_path, 'include')]
)


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'Please install CUDA for GPU support.'
    ffi.build()
    # shutil.move('attentive_layers', 'functions')
