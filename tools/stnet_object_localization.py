# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------

""" This is the main entry point to this implementation """

# noinspection PyUnresolvedReferences
import _init_lib_path
import argparse
import pprint
from utils.config import cfg, cfg_from_file, cfg_from_list
from stnet.stnet import STNet


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Weakly-supervised object localization using selective tuning')

    parser.add_argument('-x', '--exe-mode', dest='exe_mode',
                        help='execution mode', type=str, choices=['bbox_eval', 'bbox_viz', 'ch_viz'], required=False)
    parser.add_argument('-d', '--dataset-dir', dest='dataset_dir',
                        help='dataset directory', type=str, required=False)
    parser.add_argument('-m', '--model-dir', dest='model_dir',
                        help='folder that contains model modules', type=str, required=False)
    parser.add_argument('-e', '--experiment-dir', dest='experiment_dir',
                        help='a directory used to write experiment results', type=str, required=False)
    parser.add_argument('-i', '--iou', dest='iou',
                        help='Intersection-over-Union', type=float, required=False)
    parser.add_argument('-g', '--gpu-id', dest='gpu_id',
                        help='Intersection-over-Union', type=int, required=False)
    parser.add_argument('-c', '--cfg', dest='cfg_file',
                        help='optional config file to override the defaults', default=None, type=str)
    parser.add_argument('-s', '--set', dest='set_cfgs',
                        help='set config arg parameters', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def set_positional_cfg(args_in):
    args_list = []
    for n, a in args_in.__dict__.items():
        if a is not None and n not in ['cfg_file', 'set_cfgs']:
            args_list += [n, a]
    return args_list


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg_from_list(set_positional_cfg(args))     # input arguments override cfg files and defaults

    print('Using config:')
    pprint.pprint(cfg)

    stnet_wsol = STNet()

    stnet_wsol.run()
