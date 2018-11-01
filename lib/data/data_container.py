# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------

""" This is the data container. It contains both the dataset
    and the dataloder instances."""

from utils.config import cfg
import torch.utils.data
from data import data_set
from torch.utils.data import DataLoader
from data import transformations as transform
from utils.miscellaneous import AverageMeter

# noinspection PyUnresolvedReferences
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


# adapted and modified from pytorch default collator function
# noinspection PyProtectedMember,PyUnresolvedReferences
def custom_collator(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    import re
    from torch._six import string_classes, int_classes
    import collections
    _use_shared_memory = True
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        # return {key: custom_collator([d[key] for d in batch]) for key in batch[0]}
        return batch
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [custom_collator(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class DataContainer:
    def __init__(self, mode):
        self.mode = mode
        self.mode_cfg = cfg.get(self.mode.upper())
        self.dataset, self.dataloader = None, None
        self.meters = {m: AverageMeter() for m in cfg.METERS}

        self.create()

    def create(self):
        self.create_dataset()
        self.create_dataloader()

    def create_transform(self):
        _, h, w = self.mode_cfg.INPUT_SIZE
        assert h == w, 'NotImplemented: Only square input sizes are addressed for now'

        transformations_tail = [
            transform.ToTensor(norm_value=255),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]
        transformations_head = [
            transform.Resize(h),
            transform.CenterCrop(h),
        ]

        return transform.Compose(
            transformations_head + transformations_tail
        )

    def create_dataset(self):
        spatial_transform = self.create_transform()
        self.dataset = data_set.ILSVRC12(self.mode, spatial_transform)

    def create_dataloader(self):
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.mode_cfg.BATCH_SIZE,
                                     shuffle=self.mode_cfg.SHUFFLE,
                                     num_workers=4,
                                     collate_fn=custom_collator,
                                     pin_memory=False,
                                     drop_last=True,
                                     )

    def reset_meters(self):
        for m in self.meters.values():
            m.reset()

    def update_meters(self, **kwargs):
        for k, m in self.meters.items():
            try:
                m.update(kwargs[k])
            except KeyError:
                raise KeyError('Key {} is not defined in the dictionary'.format(k))

    def get_avg(self):
        for k, m in self.meters.items():
            yield k, m.avg
