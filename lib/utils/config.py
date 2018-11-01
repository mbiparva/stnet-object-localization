# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------

"""
Default configurations for weakly-supervised object localization using STNet

Do not change the default setting in this file, rather create a custom yaml
with preferred values to override the defaults and then pass it as input argument
to the main python file at the time of execution in the shell.
"""

import numpy as np
from easydict import EasyDict as edict
import os
import datetime
import yaml

__C = edict()
cfg = __C   # from config.py import cfg


# ================
# GENERAL
# ================
# Execution mode
__C.EXE_MODE = ['bbox_eval', 'bbox_viz', 'ch_viz'][0]

# For reproducibility
__C.RNG_SEED = 110

# Root directory of project
__C.ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

# Data directory
__C.DATASET_DIR = os.path.abspath(os.path.join(__C.ROOT_DIR, '..', 'dataset'))

# Model directory
__C.MODEL_DIR = os.path.abspath(os.path.join(__C.ROOT_DIR, 'models'))

# Experiment directory
__C.EXPERIMENT_DIR = os.path.abspath(os.path.join(__C.ROOT_DIR, '..', 'results'))
if not os.path.exists(__C.EXPERIMENT_DIR):
    os.mkdir(__C.EXPERIMENT_DIR)

# Set meters to use for experimental evaluation
__C.METERS = ['label_accuracy', 'local_accuracy']

# Use GPU
__C.USE_GPU = True

# Default GPU device id
__C.GPU_ID = 0

# IoU threshold for localization evaluation
__C.IOU = 0.5

# Dataset name
__C.DATASET_NAME = 'ILSVRC2012'

# Number of categories
__C.NUM_CLASSES = 1000

# Set parameters for snapshot and verbose routines
__C.MODEL_ID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')


# ================
# ST options
# ================

__C.ST = edict()

# Set bottom layer
__C.ST.BOTTOM = 0

# Set 2nd stage mode and offset for linear layers
__C.ST.LINEAR_S_MODE = ['1', 'ALL', '699'][2]
__C.ST.LINEAR_S_OFFSET = 3

# Set 2nd stage mode and offset for the bridge layer
__C.ST.LINEAR_B_MODE = ['1', 'ALL', '699'][2]
__C.ST.LINEAR_B_OFFSET = 1

# Set the way the 3D attention tensor collapses to a 2D tensor
__C.ST.COLLAPSE_MODE = ['INTACT', 'SUM', 'MAX'][1]

# Prune response based on some threshold
__C.ST.PRUNE = True

# Set the strategy to determine the threshold
__C.ST.LB_MODE = ['NEW_MEAN', 'OLD_MEAN', 'PERCENTILE', 'ENERGY'][0]
__C.ST.MEAN_MULTIPLIER = 1.0
__C.ST.STD_MULTIPLIER = 0
__C.ST.PERCENTAGE = 0.30
__C.ST.ENERGY_LB = 0.05


# ================
# Validation options
# ================

__C.VALID = edict()

# Input data size
__C.VALID.INPUT_SIZE = (3, 224, 224)

# Images to use per minibatch
__C.VALID.BATCH_SIZE = 128

# Shuffle the dataset
__C.VALID.SHUFFLE = False


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.upper().split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except Exception:
            # handle the case when v is a string literal
            value = v
        assert isinstance(value, type(d[subkey])), 'type {} does not match original type {}'.format(
            type(value), type(d[subkey])
        )
        d[subkey] = value


def cfg_to_file(cfg_in, path_in, name_in):
    """Write cfg to a yaml file"""
    with open(os.path.join(path_in, '{}.yml'.format(name_in)), 'w') as output_file:
        yaml.dump(cfg_in, output_file, default_flow_style=False)
