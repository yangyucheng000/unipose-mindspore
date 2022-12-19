from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import yaml
from easydict import EasyDict as edict


config = edict()
config.IS_DISTRIBUTE = False
config.IS_MODELART = True
config.CACHE_INPUT = '/cache/dataset'
config.CACHE_TRAIN = '/LSP/TRAIN'
config.CACHE_VAL = '/LSP/VAL'

config.train_dir = 'data/LSP/TRAIN'
config.val_dir = 'data/LSP/VAL'
config.starter_epoch = 0
config.epochs = 110 #训练的epoch数
config.pretrained = None #"model/resnet101.ckpt"


config.workers = 2
config.weight_decay = 0.0005 # Adam优化器
config.momentum = 0.9
config.dataset_size = 1000
config.batch_size = 1  # 训练的batch_size
config.TEST_batch_size = 1 # 测试的batch_size
config.lr = 0.00005 # 学习率
config.gamma = 0.333 # 学习率指数衰减的指数
config.step_size = 30000 # epoch_size= step / (1000/batch)
config.sigma = 3 # 生成heatmap所用的sigma
config.stride = 1 # 生成heatmap所用的stride步长
config.numClasses = 14  # 14个关键点


config.height = 184 # 图片resize的高度
config.width = 92 # 图片resize的宽度

config.heatmap_h = round(config.height / config.stride)
config.heatmap_w = round(config.width / config.stride)

config.best_epoch = 0
config.bestPCK = 0
config.bestPCKh = 0
config.save_freq = 20