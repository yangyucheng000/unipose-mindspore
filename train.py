import argparse
import time
import mindspore
import mindspore.nn as nn
from mindspore.train.model import Model
import mindspore.context as context
from mindspore.train.callback import TimeMonitor,LossMonitor
import numpy as np
import math
import cv2

from model.unipose import unipose

from src.config import config as cfg
cfg.IS_MODELART = False
cfg.IS_DISTRIBUTE = False

from src.dataset import CreateDatasetLSP
from src.eval import EvaluateCallBack
from src.utils import adjust_learning_rate
from src.loss import PoseNetWithLoss,JointsMSELoss
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Train keypoints network')

parser.add_argument('--train_url', required=False,
                    default=None, help='Location of training outputs.')

parser.add_argument('--data_url', required=False,
                    default=cfg.CACHE_INPUT, help='Location of data.')
args = parser.parse_args()

if __name__ =='__main__':

    context.set_context(mode=context.GRAPH_MODE, device_id=0, device_target='CPU')

    train_dataset,val_dataset = CreateDatasetLSP(cfg)

    net = unipose(dataset='lsp',num_classes=cfg.numClasses, backbone='resnet', output_stride=16,
                    sync_bn=True,freeze_bn=False, stride=cfg.stride)
    loss = JointsMSELoss()
    lr_schedule = adjust_learning_rate()
    optimizer = nn.Adam(params=net.get_parameters(),learning_rate=lr_schedule)

    net_with_loss = PoseNetWithLoss(network=net, loss=loss)


    model = Model(network=net_with_loss, optimizer=optimizer)

    dataset_size = train_dataset.get_dataset_size()
    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossMonitor()
    eval_cb = EvaluateCallBack(model=net,eval_dataset=val_dataset, loss_fn=loss)
    callback_list = [time_cb, loss_cb,eval_cb]



    model.train(cfg.epochs,train_dataset,callbacks=callback_list)
    print("train success")

