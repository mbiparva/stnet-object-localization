# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------

""" Defines routines to handle bbox proposal from gating activities """

import numpy as np


def iou_function(s_boxes, d_boxes):
    """Intersection-Over-Union function

    Args:
        s_boxes (no_s, xyxy): Source boxes to use.
        d_boxes (no_k, xyxy): Destination boxes to measure the IoU with.
    Returns:
        iou_value (no_s, no_k): iou values are returned
    """
    no_s = len(s_boxes)
    no_k = len(d_boxes)

    iou = np.zeros((no_s, no_k))

    for n in range(no_s):
        for k in range(no_k):
            do_overlap = s_boxes[n][0] < d_boxes[k][2] and \
                         s_boxes[n][2] > d_boxes[k][0] and \
                         s_boxes[n][1] < d_boxes[k][3] and \
                         s_boxes[n][3] > d_boxes[k][1]
            if do_overlap:
                i_box = [
                    max(s_boxes[n][0], d_boxes[k][0]),
                    max(s_boxes[n][1], d_boxes[k][1]),
                    min(s_boxes[n][2], d_boxes[k][2]),
                    min(s_boxes[n][3], d_boxes[k][3])
                ]
                i_box_area = (i_box[2] - i_box[0] + 1) * (i_box[3] - i_box[1] + 1)

                s_box_area = (s_boxes[n][2] - s_boxes[n][0] + 1) * (s_boxes[n][3] - s_boxes[n][1] + 1)
                d_box_area = (d_boxes[k][2] - d_boxes[k][0] + 1) * (d_boxes[k][3] - d_boxes[k][1] + 1)

                u_box_area = s_box_area + d_box_area - i_box_area

                iou[n, k] = i_box_area / u_box_area
    return iou


def lb_find(g_nz, cfg):
    """It finds the lower bound given the non zero values and config params.

    Args:
        g_nz (np.ndarray): Source non-zero gating array (1D).
        cfg (dict): Dictionary containing config params.
    Returns:
        lb (scalar): Lower bound for thresholding
    """
    lb = None
    assert g_nz.ndim == 1, 'the input array must be one dimensional.'

    if cfg.ST.LB_MODE in ['NEW_MEAN', 'OLD_MEAN']:
        mean = np.mean(g_nz)
        std = np.std(g_nz)
        lb = cfg.ST.MEAN_MULTIPLIER * mean + cfg.ST.STD_MULTIPLIER * std

    elif cfg.ST.LB_MODE == 'PERCENTILE':
        lb = np.sort(g_nz)[int(g_nz.size * cfg.ST.PERCENTAGE)]

    elif cfg.ST.LB_MODE == 'ENERGY':
        eb = cfg.ST.ENERGY_LB * g_nz.sum()
        g_nz_s = np.sort(g_nz)
        ea = 0
        for i in range(len(g_nz)):
            ea += g_nz_s[i]
            if ea > eb:
                lb = g_nz_s[i]
                break
    else:
        raise NotImplementedError
    assert lb is not None, 'lb is not found!'

    return lb


def ind_retrieval(g, cfg):
    """It calculates the indices of non-zero gating nodes after some accumulation and thresholding.

    Args:
        g (np.ndarray): Source gating tenser (3D).
        cfg (dict): Dictionary containing config params.
    Returns:
        nz_ind (np.ndarray): Indices of the non-zero units
    """
    if cfg.ST.COLLAPSE_MODE == 'SUM':
        g_c = np.sum(g, axis=0)
    elif cfg.ST.COLLAPSE_MODE == 'MAX':
        g_c = np.max(g, axis=0)
    else:
        g_c = g

    if not cfg.ST.PRUNE:
        nz_ind = np.argwhere(g_c != 0)
        nz_ind = nz_ind[:, 1] if nz_ind.shape[1] == 3 else nz_ind   # if collapse not applied, forget the 1st dimension
        nz_value = g_c[g_c != 0]
    else:
        nz_val = g_c[g_c != 0]
        lb = lb_find(g_c.ravel() if cfg.ST.LB_MODE == 'OLD_MEAN' else nz_val.ravel(), cfg)
        nz_ind = np.argwhere(g_c >= lb)
        nz_value = g_c[g_c >= lb]

    return nz_ind, nz_value


def propose_box(g, cfg, image_size):
    """It queries the active gating nodes and fit the best box to them.

    Args:
        g (np.ndarray): Source gating tenser (3D).
        cfg (dict): Dictionary containing config params.
        image_size (tuple): size of the input image for which box is proposed.
    Returns:
        bbox (list, xyxy): best fitted bounding-box
    """
    if not (g != 0).any():
        # print('the gating tensor is all zero, hence empty box of length one is proposed.')
        return [0, 0, 1, 1]

    g_nz_ind, _ = ind_retrieval(g, cfg)

    if len(g_nz_ind) == 0:
        print('Due to over thresholding, empty box is returned.')
        return [0, 0, 1, 1]

    rf = np.array([0, 0])
    stride = np.array([1, 1])

    top_left = np.maximum(stride * np.min(g_nz_ind, axis=0) - rf/2, [0, 0])
    bottom_right = np.minimum(stride * np.max(g_nz_ind, axis=0) + rf/2, image_size)

    return [top_left[1], top_left[0], bottom_right[1], bottom_right[0]]   # output bbox format is [xyxy]
