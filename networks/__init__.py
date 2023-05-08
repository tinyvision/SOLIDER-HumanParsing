from __future__ import absolute_import
from networks.swin_transformer import swin_base_patch4_window7_224 as swin_base
from networks.swin_transformer import swin_small_patch4_window7_224 as swin_small
from networks.swin_transformer import swin_tiny_patch4_window7_224 as swin_tiny

from networks.AugmentCE2P import resnet101
from networks.AugmentCE2P import resnet50
__factory = {
    'resnet101': resnet101,
    'resnet50': resnet50,
    'swin_base': swin_base,
    'swin_tiny': swin_tiny,
    'swin_small': swin_small,
}


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model arch: {}".format(name))
    return __factory[name](*args, **kwargs)
