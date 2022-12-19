import mindspore
import mindspore.nn as nn
#import mindspore.nn.functional as F

from model.wasp import build_wasp
from mindspore import load_checkpoint, load_param_into_net
from model.decoder import build_decoder
from model.resnet import *
from model.wasp import *
from mindspore.common.initializer import HeNormal
from src.config import config as cfg


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return ResNet101(output_stride, BatchNorm)
    else:
        raise NotImplementedError

class unipose(nn.Cell):
    def __init__(self, dataset, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, stride=8):
        super(unipose, self).__init__()
        self.stride = stride
        self.heatmap_h = cfg.heatmap_h
        self.heatmap_w = cfg.heatmap_w
        BatchNorm = nn.BatchNorm2d

        self.num_classes = num_classes

        self.pool_center   = nn.AvgPool2d(kernel_size=9, stride=8)

        self.backbone      = build_backbone(backbone, output_stride, BatchNorm)
        # if cfg.pretrained is not None:
        #     ckpt_file_name = cfg.pretrained
        #     param_dict = load_checkpoint(ckpt_file_name)
        #     load_param_into_net(self.backbone, param_dict)
        self.wasp          = build_wasp(output_stride, BatchNorm)
        self.decoder       = build_decoder(dataset, num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()
            pass

    def construct(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.wasp(x)
        x = self.decoder(x, low_level_feat)
        # print(x.shape)
        if self.stride != 8:
        #     # x = nn.ResizeBilinear()(x, size=(x.size()[2:]), align_corners=True)
             x = nn.ResizeBilinear()(x, size=(self.heatmap_h,self.heatmap_w), align_corners=True)
        return x



if __name__ == "__main__":
    #model = build_wasp(backbone='resnet', output_stride=16,BatchNorm=nn.BatchNorm2d)
    model = unipose(backbone='resnet',dataset='lsp',stride=cfg.stride)
    #model.eval()
    input = mindspore.numpy.rand(1, 3, 184, 92)
    output = model(input)
    print(output.shape)# (1, 22, 46, 23)


