# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------

""" This holds miscellaneous functions such as showing images with bboxes"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from utils.box_proposal import ind_retrieval
from scipy.ndimage.filters import gaussian_filter
import math


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val, self.avg, self.sum, self.count = (0,)*4
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def show_image_bboxes(boxes, image, path, label, proposal=None):
    """
    This function draw the BB on the image and save/show it
    """
    image_size = image.shape[1:]
    img_show = image.astype(np.uint8, copy=False)
    img_show = np.moveaxis(img_show, 0, -1)
    my_dpi = 96
    plt.figure(figsize=(3 * image_size[0] / my_dpi, 3 * image_size[1] / my_dpi), dpi=my_dpi)
    plt.imshow(img_show, interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    fig = plt.gcf()
    fig.tight_layout(pad=0)
    current_axis = plt.gca()
    col = 'b'   # gt boxes are blue
    for i, annoIter in enumerate(boxes):
        box = annoIter['bbox'] if isinstance(annoIter, dict) else annoIter
        current_axis.add_patch(Rectangle(box[:2],
                                         box[2] - box[0], box[3] - box[1],
                                         fill=False, facecolor='none', edgecolor=col, linewidth=3, alpha=1.0))
    if proposal is not None:
        col = 'r'  # predicted boxes are blue
        box = proposal
        current_axis.add_patch(Rectangle(box[:2],
                                         box[2] - box[0], box[3] - box[1],
                                         fill=False, facecolor='none', edgecolor=col, linewidth=3, alpha=1.0))
        current_axis.text(box[0], box[1] - 5, '{}'.format(label), fontsize=13, color=col)

    plt.savefig(path, format='jpg')
    plt.close('all')

    return None


# TODO: Add the Class Hypothesis Visualizer
def class_hypo_map(g, cfg, accum_spec, image_size):
    """
    This function generates the class hypothesis map from the gating activities.
    """
    ch_map = np.zeros(image_size)

    if not (g != 0).any():
        return ch_map

    g_nz_ind, g_nz_value = ind_retrieval(g, cfg)

    if len(g_nz_ind) == 0:
        return ch_map

    rf_tl = np.array([accum_spec.kernel]*2, dtype=np.int) // 2
    rf_br = np.array([accum_spec.kernel]*2, dtype=np.int) - rf_tl
    stride = np.array([accum_spec.stride]*2)
    start = np.array([accum_spec.start]*2)

    top_left = np.maximum(stride * g_nz_ind - rf_tl + start, [0, 0])
    bottom_right = np.minimum(stride * g_nz_ind + rf_br + start, image_size[1:])

    # Populate the class hypothesis map
    for i in range(len(g_nz_ind)):
        ch_map[:, top_left[i][0]:bottom_right[i][0], top_left[i][1]:bottom_right[i][1]] += g_nz_value[i]
    return ch_map


def class_hypo_visualize(g_bot, cfg, accum_spec, img_size, out_path):
    ch_map = class_hypo_map(g_bot, cfg, accum_spec, img_size)
    ch_map = gaussian_filter(ch_map, img_size[1] / 40., mode='reflect', cval=0)

    my_dpi = 96
    plt.figure(figsize=(3 * img_size[1] / my_dpi, 3 * img_size[2] / my_dpi), dpi=my_dpi)
    plt.xticks([]), plt.yticks([])
    fig = plt.gcf()
    fig.tight_layout(pad=0)
    plt.imshow(ch_map[0], interpolation='spline16', cmap='jet', vmin=ch_map.min() * 0.40, vmax=ch_map.max() * 0.80)

    plt.savefig(out_path, format='png')
    plt.close('all')


# TODO: Add the function that calculates total stride, RF, padding, and center.
class Spec:
    def __init__(self, *args, **kwargs):
        """
        A class that holds common specifications of layer in neural networks.
        It assumes all are expressed with one value, basically square sizes.
        Args:
             size: input size
             stride: stride size
             kernel: kernel size
             padding: padding amount
             start: starting point of the first node in the input space
        """
        self.size = 1
        self.stride = 1
        self.kernel = 1
        self.padding = 0
        self.start = 0

        if len(args):
            self.set_fields_list(*args)
        elif len(kwargs):
            self.set_fields_dict(**kwargs)

    def set_fields_dict(self, **kwargs):
        for u, v in kwargs.items():
            self.__setattr__(u, v)

    def set_fields_list(self, *args):
        self.size, self.kernel, self.stride, self.padding, self.start = args

    def get_fields_dict(self):
        fields = {}
        for u, v in self.__dict__.items():
            fields[u] = v
        return fields

    def get_fields_list(self):
        return [self.size, self.kernel, self.stride, self.padding, self.start]

    def print_fields(self, l_type, l_num):
        print('{1:2d} | {0:3s} | Size ({spec.size:3d}, {spec.size:3d}), '
              'Stride {spec.stride:2d}, Kernel {spec.kernel:3d}, '
              'Start {spec.start:3d}, Padding {spec.padding:3d}'.format(l_type, l_num, spec=self))


class LayerSpec:
    def __init__(self, **kwargs):
        self.current_spec = Spec(**kwargs)
        self.accumulative_spec = Spec()

    def calculate_accumulations(self, previous_accum):
        """
        It is inspired from the following link. Check it for details:
        https://medium.com/@nikasa1889/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
        It sets the result to it's accumulative_spec attribute.

        Args:
            previous_accum (Spec): accumulation of the previous layer
        """
        set_int = True
        n_in, r_in, j_in, p_in, start_in = previous_accum.get_fields_list()
        k, s, p = self.current_spec.kernel, self.current_spec.stride, self.current_spec.padding

        n_out = int(math.floor((n_in - k + 2*p)/s) + 1)
        actual_p = (n_out-1)*s - n_in + k
        p_r = int(math.ceil(actual_p/2.))   # maximally set the center
        p_l = int(math.floor(actual_p/2.))  # minimally set the center

        r_out = r_in + (k - 1)*j_in
        j_out = j_in * s
        p_out = p_in + p * j_in
        start_out = start_in + ((k-1)/2 - p_r)*j_in

        accum_out = [n_out, r_out, j_out, p_out, start_out]
        if set_int:
            accum_out = [int(i) for i in accum_out]

        self.accumulative_spec.set_fields_list(*accum_out)


def calculate_net_specs(net, input_size, verbose=False):
    """
    This function iterate over layers and calculate all accumulative params such as RF etc.

    Args:
        net: the list that defines parameters such as RF, stride, padding, etc of all layers.
        input_size: size of the input layer
        verbose: whether to print the calculated accumulative

    Returns:
          the list that holds all the calculations.
    """
    net_specs = []
    l_type, l_param = net[0]
    assert l_type == 'd', 'first layer must be data type'
    layer_spec = LayerSpec(**l_param)
    size, stride, kernel, padding, start = input_size, 1, 1, 0, 0
    layer_spec.accumulative_spec.set_fields_dict(
        **{'size': size,
           'stride': stride,
           'kernel': kernel,
           'padding': padding,
           'start': start}
    )
    if verbose:
        layer_spec.accumulative_spec.print_fields(l_type, 0)
    net_specs.append(layer_spec)

    for i in range(1, len(net)):
        l_type, l_param = net[i]
        assert l_type == 'c' or l_type == 'p' or l_type == 'l', 'only c, p, l types are implemented'
        layer_spec = LayerSpec(**l_param)
        p_l_accum = net_specs[i - 1].accumulative_spec

        layer_spec.calculate_accumulations(p_l_accum)
        if verbose:
            layer_spec.accumulative_spec.print_fields(l_type, i)
        net_specs.append(layer_spec)
    return net_specs
