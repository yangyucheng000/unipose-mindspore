from collections import namedtuple

import mindspore.nn as nn
import mindspore.ops as ops
import numpy

from src.config import config as cfg

import numpy as np


class MaxPool2d(nn.Cell):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.max_pool = ops.MaxPool(kernel_size, stride)
        self.use_pad = padding != 0
        if isinstance(padding, tuple):
            assert len(padding) == 2
            paddings = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
        elif isinstance(padding, int):
            paddings = ((0, 0),) * 2 + ((padding, padding),) * 2
        else:
            raise ValueError('padding should be a tuple include 2 numbers or a int number')
        self.pad = ops.Pad(paddings)

    def construct(self, x):
        if self.use_pad:
            x = self.pad(x)
        return self.max_pool(x)


class AverageMeter(object):
    """ Computes ans stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(base_lr=cfg.lr, gamma=cfg.gamma, step_size=cfg.step_size):
    lr_list = []
    for step in range(int(cfg.epochs * cfg.dataset_size / cfg.batch_size)):
        lr = base_lr * (gamma ** (step / step_size))
        lr_list.append(lr)
    return numpy.array(lr_list)


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


