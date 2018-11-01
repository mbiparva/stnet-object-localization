# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------

""" This is a sub-class of Dataset in torchvision package.
    It defines Imagenet ILSVRC 2012 dataset class."""

from utils.config import cfg
import sys
import os
import xml.etree.ElementTree as xe2
from torch.utils.data import Dataset
from PIL import Image
from utils.bounding_box import BBox


def allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir_in):
    classes = [d for d in os.listdir(dir_in) if os.path.isdir(os.path.join(dir_in, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def find_class_names(dir_in, file_in):
    with open(os.path.join(dir_in, file_in), 'r') as file_handle:
        file_class_labels = file_handle.read().split('\n')[:-1]
    class_labels = {}
    for c in file_class_labels:
        synset_id, _, label_name = c.split()
        class_labels[synset_id] = label_name

    return class_labels


def make_dataset(dir_in, class_to_idx, class_to_name, extensions):
    images = []
    dir_in = os.path.expanduser(dir_in)
    for target in sorted(os.listdir(dir_in)):
        d = os.path.join(dir_in, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target], target, class_to_name[target])
                    images.append(item)

    return images


def parse_xml_anno(xml_file):
    """
    Process a single XML file containing bounding boxes.

    Args:
        xml_file: The xml file to parse and extract annotation from: (string path).
    Returns:
        A dictionary containing image annotation and BBox object, containing bboxes coordinates and labels: (dict).
    """
    try:
        tree = xe2.parse(xml_file)
    except Exception:
        print('Failed to parse: ' + xml_file, file=sys.stderr)
        return None

    image_anno = {
        'image_width': tree.find('.//width').text,
        'image_height': tree.find('.//height').text,
        'image_depth': tree.find('.//depth').text,
        'image_name': tree.find('.//filename').text,
        'image_folder': tree.find('.//folder').text,
    }

    object_boxes = []
    object_labels = []
    objects = tree.findall('object')
    for o in objects:
        label_synset = o.find('.//name').text
        xmin = int(o.find('.//xmin').text)
        ymin = int(o.find('.//ymin').text)
        xmax = int(o.find('.//xmax').text)
        ymax = int(o.find('.//ymax').text)
        object_boxes.append([xmin, ymin, xmax, ymax])
        object_labels.append(label_synset)

    # We start using __BBox__ class implemented in the layer branch of torchvision
    boxes_anno = BBox(object_boxes, (image_anno['image_width'], image_anno['image_height']))
    boxes_anno.add_field('label_synset', object_labels)

    return image_anno, boxes_anno


class ILSVRC12(Dataset):
    def __init__(self, mode, transform):
        self.mode = mode
        self.annotations_path = os.path.join(cfg.DATASET_DIR, cfg.DATASET_NAME, 'annotations', self.mode)
        self.images_path = os.path.join(cfg.DATASET_DIR, cfg.DATASET_NAME, 'images', self.mode)
        self.transform = transform

        print('start loading ILSVRC 2012 dataset ...')
        extensions = ['xml']
        classes, class_to_idx = find_classes(self.images_path)
        class_to_name = find_class_names(
            os.path.join(cfg.DATASET_DIR, cfg.DATASET_NAME, 'annotations'), 'class_labels.txt'
        )
        samples = make_dataset(self.annotations_path, class_to_idx, class_to_name, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + self.images_path + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        print('finished loading the dataset.')

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.class_to_name = class_to_name
        self.samples = samples

    def __getitem__(self, index):
        sample = self.samples[index]

        image = self.load_image(sample)     # Load image
        anno = self.load_anno(sample)       # Load annotation

        if self.transform is not None:
            image, anno['boxes_anno'] = self.transform((image, anno['boxes_anno']))
            anno['image_anno']['image_depth_t'] = image.shape[0]
            anno['image_anno']['image_height_t'] = image.shape[1]
            anno['image_anno']['image_width_t'] = image.shape[2]

        sample = (image, anno)

        return sample

    def __len__(self):
        return len(self.samples)

    def load_image(self, sample):
        sample_path, _, label_synset, _ = sample
        sample_name = os.path.splitext(os.path.basename(sample_path))[0]
        image_path = os.path.join(cfg.DATASET_DIR,
                                  cfg.DATASET_NAME,
                                  'images',
                                  self.mode,
                                  label_synset,
                                  '{}.JPEG'.format(sample_name))

        image = Image.open(image_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')

        return image

    @staticmethod
    def load_anno(sample):
        sample_path, label_idx, label_synset, label_name = sample

        image_anno, boxes_anno = parse_xml_anno(sample_path)

        sample_anno = {'samlpe_path': sample_path,
                       'label_idx': label_idx,
                       'label_synset': label_synset,
                       'label_name': label_name}

        file_anno = {'sample_anno': sample_anno,
                     'image_anno': image_anno,
                     'boxes_anno': boxes_anno}

        return file_anno
