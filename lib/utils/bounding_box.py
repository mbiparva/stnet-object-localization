# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------

""" This is entirely taken from torchvision, layer branch, structure folder.
    For any issue, please check to
    vision/torchvision/structures/bounding_box.py on github repository of pytorch.
"""

import torch
import collections

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


# noinspection PyUnresolvedReferences
class BBox(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order ot uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    # noinspection PyCallingNonCallable
    def __init__(self, bbox, image_size, mode='xyxy'):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device('cpu')
        bbox = torch.tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                    "bbox should have 2 dimensions, got {}".format(bbox.ndimension()))
        if bbox.size(-1) != 4:
            raise ValueError(
                    "last dimension of bbox should have a "
                    "size of 4, got {}".format(bbox.size(-1)))
        if mode not in ('xyxy', 'xywh'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = (int(image_size[0]), int(image_size[1]))  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ('xyxy', 'xywh'):
            raise ValueError(
                    "mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split()
        if mode == 'xyxy':
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BBox(bbox, self.size, mode=mode)
        else:
            bbox = torch.cat((xmin, ymin, xmax - xmin, ymax - ymin), dim=-1)
            bbox = BBox(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split(self):
        if self.mode == 'xyxy':
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == 'xywh':
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmin + w, ymin + h
        else:
            raise RuntimeError('Should not be here')

    def resize(self, size):
        """
        Returns a resized copy of this bounding box

        Args:
             size: The requested size in pixels, as a tuple: (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BBox(scaled_box, size, mode=self.mode)
            bbox._copy_extra_fields(self)
            return bbox

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = self._split()
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat((scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1)
        bbox = BBox(scaled_box, size, mode='xyxy')
        bbox._copy_extra_fields(self)
        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        Args:
             method: One of {PIL.Image.FLIP_LEFT_RIGHT, PIL.Image.FLIP_TOP_BOTTOM,
             PIL.Image.ROTATE_90, PIL.Image.ROTATE_180, PIL.Image.ROTATE_270,
             PIL.Image.TRANSPOSE or PIL.Image.TRANSVERSE}.
        """
        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split()
        if method == FLIP_LEFT_RIGHT:
            transposed_xmin = image_width - xmax
            transposed_xmax = image_width - xmin
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin
        else:
            raise NotImplementedError("Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented")

        transposed_boxes = torch.cat((transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1)
        bbox = BBox(transposed_boxes, self.size, mode='xyxy')
        bbox._copy_extra_fields(self)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Crops a rectangular region from this bounding box. The box is a
        4-tuple (xyxy) defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        # is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

        cropped_box = torch.cat((cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1)
        bbox = BBox(cropped_box, (w, h), mode='xyxy')
        bbox._copy_extra_fields(self)
        return bbox.convert(self.mode)

    def center_crop(self, output_size):
        """
        It crops the center box of the output_size. It already knows the image size.
        Args:
            output_size: it is (width, height) of the center crop.
        Returns:
            a bbox instance with the cropped boxes
        """
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        elif isinstance(output_size, collections.Iterable) and len(output_size) == 2:
            pass
        else:
            raise NotImplementedError('Only size of (int)|(width, height) is expected')

        w, h = self.size
        ch, cw = output_size
        cx = int(round((w - cw) / 2.))
        cy = int(round((h - ch) / 2.))
        return self.crop((cx, cy, cx+cw, cy+ch))

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_boxes={}, '.format(self.bbox.size(0))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={}, '.format(self.size[1])
        s += 'mode={})'.format(self.mode)
        return s


if __name__ == '__main__':
    bbox_test = BBox([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    bbox_test_r = bbox_test.resize((5, 5))
    print(bbox_test_r)
    print(bbox_test_r.bbox)

    bbox_test_t = bbox_test.transpose(0)
    print(bbox_test_t)
    print(bbox_test_t.bbox)
