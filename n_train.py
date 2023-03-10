import argparse
import os
import mindspore
import mindspore.nn as nn
from mindspore.train.model import Model
import mindspore.context as context
from mindspore.train.callback import TimeMonitor,LossMonitor
from mindspore.communication.management import get_rank, init
from mindspore.context import ParallelMode

import numpy as np
import math
import cv2

from model.unipose import unipose
from src.config import config as cfg
cfg.IS_DISTRIBUTE = True

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
                    default=None, help='Location of data.')
parser.add_argument(
    '--device_target',
    type=str,
    default="Ascend",
    choices=['Ascend'],
    help='若要在启智平台上使用NPU，需要在启智平台训练界面上加上运行参数device_target=Ascend')
args = parser.parse_args()
if cfg.IS_MODELART:
    import moxing as mox


if __name__ =='__main__':
    context.set_context(mode=context.GRAPH_MODE, device_id=int(os.getenv("DEVICE_ID")), device_target='Ascend')
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL,gradients_mean=True, parameter_broadcast=True,
                                      auto_parallel_search_mode="recursive_programming")
    init()

    if not os.path.exists(cfg.CACHE_INPUT):
        os.mkdir(cfg.CACHE_INPUT)
    mox.file.copy_parallel(args.data_url, cfg.CACHE_INPUT)

    train_dataset,val_dataset = CreateDatasetLSP(cfg)

    net = unipose(dataset='lsp',num_classes=cfg.numClasses, backbone='resnet', output_stride=16,
                    sync_bn=True,freeze_bn=False, stride=cfg.stride)
    loss = JointsMSELoss()
    lr_schedule = adjust_learning_rate()
    optimizer = nn.Adam(params=net.get_parameters(),learning_rate=cfg.lr,
                        weight_decay=cfg.weight_decay
                        )

    net_with_loss = PoseNetWithLoss(network=net, loss=loss)


    model = Model(network=net_with_loss, optimizer=optimizer,amp_level="O0")

    dataset_size = train_dataset.get_dataset_size()
    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossMonitor()
    eval_cb = EvaluateCallBack(model=net,eval_dataset=val_dataset,loss_fn=loss)
    callback_list = [time_cb, loss_cb,eval_cb]



    model.train(cfg.epochs,train_dataset,callbacks=callback_list)
    print("train success")

