#  -*-  coding  =  utf-8  -*-
#  @Time  :2023/3/27  15:29
#  @Author:Su
#  @File  :  14.VGG.py
#  @Software:  PyCharm
import torch
from torch import nn
from d2l import torch as d2l


def vgg_block(num_convs, in_channel, out_channel):
    layer = []
    for _ in range(num_convs):
        layer.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
        layer.append(nn.ReLU())
        in_channel = out_channel
    layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layer)


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(*conv_blks, nn.Flatten(),
                         nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
                         nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
                         nn.Dropout(0.5), nn.Linear(4096, 10))


net = vgg(conv_arch)
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape\t', X.shape)