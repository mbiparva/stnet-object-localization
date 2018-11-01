# -------------------------------------------------------------------------
# STNet: Selective Tuning of Convolutional Networks for Object Localization
#
# Licensed under The GNU GPL v3.0 License [see LICENSE for details]
# Written by Mahdi Biparva
# -------------------------------------------------------------------------

""" This is adapted from alexnet.py provided by pytorch library.
    In STNet, We need to leverage the new attention wrapper modules. """

from utils.config import cfg
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import selective_tuning as st
from utils.miscellaneous import calculate_net_specs


__all__ = ['st_alexnet_create']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class STAlexNet(nn.Module):
    net_params = [
        ['d', {'kernel': 1, 'stride': 1, 'padding': 0}],
        ['c', {'kernel': 11, 'stride': 4, 'padding': 2}],
        ['p', {'kernel': 3, 'stride': 2, 'padding': 0}],
        ['c', {'kernel': 5, 'stride': 1, 'padding': 2}],
        ['p', {'kernel': 3, 'stride': 2, 'padding': 0}],
        ['c', {'kernel': 3, 'stride': 1, 'padding': 1}],
        ['c', {'kernel': 3, 'stride': 1, 'padding': 1}],
        ['c', {'kernel': 3, 'stride': 1, 'padding': 1}],
        ['p', {'kernel': 3, 'stride': 2, 'padding': 0}],
        ['c', {'kernel': 6, 'stride': 1, 'padding': 0}],
        ['l', {'kernel': 1, 'stride': 1, 'padding': 0}],
        ['l', {'kernel': 1, 'stride': 1, 'padding': 0}],
    ]

    def __init__(self, num_classes=1000):
        super(STAlexNet, self).__init__()
        self.g_top = None
        self.net_specs = calculate_net_specs(self.net_params, cfg.VALID.INPUT_SIZE[-1], verbose=True)

        # Bottom Up Hierarchy
        #                                                                        - Feature Block
        self.a_conv_01 = st.AttentiveConv(3, 64, kernel_size=11, stride=4, padding=2)     # 01
        self.a_pool_02 = st.AttentivePool(kernel_size=3, stride=2)                        # 02
        self.a_conv_03 = st.AttentiveConv(64, 192, kernel_size=5, padding=2)              # 03
        self.a_pool_04 = st.AttentivePool(kernel_size=3, stride=2)                        # 04
        self.a_conv_05 = st.AttentiveConv(192, 384, kernel_size=3, padding=1)             # 05
        self.a_conv_06 = st.AttentiveConv(384, 256, kernel_size=3, padding=1)             # 06
        self.a_conv_07 = st.AttentiveConv(256, 256, kernel_size=3, padding=1)             # 07
        self.a_pool_08 = st.AttentivePool(kernel_size=3, stride=2)                        # 08

        #                                                                       - Classification Block
        self.a_dout_09 = nn.Dropout()                                                     # 09
        self.a_conv_09 = st.AttentiveConv(256, 4096, kernel_size=6, padding=0)
        self.a_dout_10 = nn.Dropout()                                                     # 10
        self.a_bridge_10 = st.AttentiveBridge(4096,
                                              cfg.ST.LINEAR_B_MODE,
                                              cfg.ST.LINEAR_B_OFFSET)
        self.a_linear_10 = st.AttentiveLinear(4096, 4096,
                                              cfg.ST.LINEAR_S_MODE,
                                              cfg.ST.LINEAR_S_OFFSET)
        self.a_linear_11 = st.AttentiveLinear(4096, num_classes,
                                              cfg.ST.LINEAR_S_MODE,
                                              cfg.ST.LINEAR_S_OFFSET)                     # 11

    def forward(self, x):

        x = self.a_conv_01(x)
        x = self.a_pool_02(x)
        x = self.a_conv_03(x)
        x = self.a_pool_04(x)
        x = self.a_conv_05(x)
        x = self.a_conv_06(x)
        x = self.a_conv_07(x)
        x = self.a_pool_08(x)

        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.a_dout_09(x)
        x = x.view(x.size(0), 256, 6, 6)
        x = self.a_conv_09(x)
        x = x.view(x.size(0), 4096)

        x = self.a_dout_10(x)
        x = self.a_linear_10(x)
        x = self.a_linear_11(x)

        return x

    def attend(self, g):

        self.g_top = g
        g = self.a_linear_11.attend(g)
        g = self.a_linear_10.attend(g)
        g = self.a_bridge_10.attend(self.g_top, g)
        g = g.view(len(g), 4096, 1, 1)
        g = self.a_conv_09.attend(g)

        g = self.a_pool_08.attend(g)
        g = self.a_conv_07.attend(g)
        g = self.a_conv_06.attend(g)
        g = self.a_conv_05.attend(g)
        g = self.a_pool_04.attend(g)
        g = self.a_conv_03.attend(g)    # 2
        if cfg.ST.BOTTOM == 2:
            return g
        g = self.a_pool_02.attend(g)
        g = self.a_conv_01.attend(g)

        return g

    def load_state_dict(self, state_dict_pt, strict=True):
        skip_params = []
        state_dict_self = self.state_dict()
        state_dict_self_keys = iter(state_dict_self.keys())
        for i, (name, param) in enumerate(state_dict_pt.items()):
            if name in skip_params:
                continue
            state_dict_self_key = next(state_dict_self_keys)
            if name == 'classifier.1.weight':
                param = param.view(state_dict_self[state_dict_self_key].shape)
            while not state_dict_self[state_dict_self_key].shape == param.shape:
                state_dict_self_key = next(state_dict_self_keys)
            state_dict_self[state_dict_self_key].copy_(param)


def st_alexnet_create(pretrained=True):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = STAlexNet()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

