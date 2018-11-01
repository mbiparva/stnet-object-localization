# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------

""" The STNet class performs data loading, core net creation, and other tasks """

from utils.config import cfg
import os
import datetime
import time
import torch
from utils.miscellaneous import AverageMeter, show_image_bboxes, class_hypo_visualize
from data.data_container import DataContainer
from models.st_alexnet import st_alexnet_create
from utils.box_proposal import propose_box, iou_function
from torchvision.transforms import ToPILImage
from data.transformations import UnNormalize

# noinspection PyUnresolvedReferences
class STNet:
    def __init__(self):
        self.s_time = time.time()

        self.valid_set = None
        self.stnet = None

        assert cfg.USE_GPU and torch.cuda.is_available(), 'STNet is only implemented for gpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU_ID)
        self.device = torch.device('cuda')

    def net_setup(self):
        self.stnet = st_alexnet_create().to(self.device)
        self.stnet.eval()

    def dataset_create(self):
        self.valid_set = DataContainer('valid')
        print('len dataset is', len(self.valid_set.dataset))
        print('num mini-batches is', len(self.valid_set.dataloader))

    def gt_generate(self, anno):
        g_top = torch.zeros(len(anno), cfg.NUM_CLASSES, dtype=torch.float32, device=self.device)
        for i, a in enumerate(anno):
            g_top[i][a['sample_anno']['label_idx']] = 1     # set attention signal to initiate the top down pass

        return g_top

    # noinspection PyCallingNonCallable
    def bbox_predict(self, g_bot_in, image_size):
        g_bot_in = g_bot_in.cpu().detach().numpy()
        batch_size = len(g_bot_in)
        p_box = torch.zeros(batch_size, 4).numpy()
        for i in range(batch_size):
            p_box[i] = propose_box(g_bot_in[i], cfg, g_bot_in.shape[2:])
        if not cfg.ST.BOTTOM == 0:
            accum_spec = self.stnet.net_specs[cfg.ST.BOTTOM].accumulative_spec
            p_box = (accum_spec.stride * p_box + accum_spec.start)
            rf_2 = accum_spec.kernel // 2
            p_box = (p_box + torch.tensor([-rf_2, -rf_2, +rf_2, +rf_2])).clamp(min=0, max=image_size[0])

        return p_box, g_bot_in

    def evaluate_label(self, p, anno):
        a_label = torch.zeros(len(anno), device=self.device, dtype=torch.long)
        for i, a in enumerate(anno):
            a_label[i] = a['sample_anno']['label_idx']
        return (p.argmax(dim=1) == a_label).sum().item() / len(a_label)

    @staticmethod
    def evaluate_box(p_box, anno):
        a_box = []
        no_valid_anno = 0
        correct_localization = 0
        for a in anno:
            a_box.append(a['boxes_anno'].bbox.tolist())
            if not len(a_box[-1]):
                print('anno empty box is detected, due to out-of-box cropping.')

        for i, a in enumerate(a_box):
            if not len(a):
                continue
            no_valid_anno += 1
            iou_mat = iou_function(a, [p_box[i]])
            correct_localization += (1 if iou_mat.max() >= cfg.IOU else 0)

        return correct_localization / no_valid_anno

    def bbox_eval(self, batch_time, end, i, results):
        self.valid_set.update_meters(**results)
        batch_time.update(time.time() - end)
        end = time.time()
        print('{3} {0} [{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Label Acc. {label.val:.4f} ({label.avg:.4f})\tLocal Acc. {local.val:.3f} ({local.avg:.3f})'.format(
               'VALIDATION', i, len(self.valid_set.dataloader),
               str(datetime.timedelta(seconds=int(time.time() - self.s_time))), batch_time=batch_time,
               label=self.valid_set.meters['label_accuracy'], local=self.valid_set.meters['local_accuracy']))
        return end

    @staticmethod
    def bbox_viz(image, annotation, p_bbox):
        to_unorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale=255)
        image = [to_unorm(i).numpy() for i in image.cpu()]
        out_path = os.path.join(cfg.EXPERIMENT_DIR, cfg.MODEL_ID)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        for k in range(len(image)):
            if len(annotation[k]['boxes_anno'].bbox):
                show_image_bboxes(annotation[k]['boxes_anno'].bbox, image[k],
                                  os.path.join(out_path, '{}.jpg'.format(annotation[k]['image_anno']['image_name'])),
                                  annotation[k]['sample_anno']['label_name'], proposal=p_bbox[k])

    def ch_viz(self, g_bot, annotation):
        """
        This function generates the Class Hypothesis Visualization of STNet.
        """
        assert cfg.ST.BOTTOM != 0, 'set cfg.ST.BOTTOM = 2 for Class-Hypothesis Visualization mode'
        img_size = cfg.VALID.INPUT_SIZE
        accum_spec = self.stnet.net_specs[cfg.ST.BOTTOM].accumulative_spec
        out_path = os.path.join(cfg.EXPERIMENT_DIR, cfg.MODEL_ID)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        for i in range(len(g_bot)):
            out_path_chv = os.path.join(out_path, '{}.png'.format(annotation[i]['image_anno']['image_name']))
            class_hypo_visualize(g_bot[i], cfg, accum_spec, img_size, out_path_chv)

    def batch_main(self, image, anno):
        p_score = self.stnet(image)

        g_top = self.gt_generate(anno)

        g_bot = self.stnet.attend(g_top)

        p_bbox, g_bot = self.bbox_predict(g_bot, image.shape[2:])

        acc_lab = self.evaluate_label(p_score, anno)

        acc_loc = self.evaluate_box(p_bbox, anno)

        return {'label_accuracy': acc_lab,
                'local_accuracy': acc_loc}, p_bbox, g_bot

    def batch_loop(self):
        batch_time = AverageMeter()
        end = time.time()
        for i, (image, annotation) in enumerate(self.valid_set.dataloader):
            image = image.to(self.device)
            image.requires_grad = False
            results, p_bbox, g_bot = self.batch_main(image, annotation)
            # TODO: continuing debugging all the modes, then we are done, you can push everything up
            if cfg.EXE_MODE == 'bbox_eval':
                end = self.bbox_eval(batch_time, end, i, results)
            elif cfg.EXE_MODE == 'bbox_viz':
                self.bbox_viz(image, annotation, p_bbox)
                print('{:5d} | {:<5d} done'.format(i, len(self.valid_set.dataloader)))
            elif cfg.EXE_MODE == 'ch_viz':
                self.ch_viz(g_bot, annotation)
                print('{:5d} | {:<5d} done'.format(i, len(self.valid_set.dataloader)))
            else:
                raise NotImplementedError

    def run(self):
        if cfg.EXE_MODE == 'ch_viz':
            cfg.ST.BOTTOM = 2
        self.dataset_create()        # 1- Create datasets
        self.net_setup()             # 2- Setup networks
        self.batch_loop()            # 3- Enter batch loop
