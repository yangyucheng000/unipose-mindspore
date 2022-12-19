import math
import mindspore
import mindspore.nn as nn
import mindspore.context as context
from src.utils import MaxPool2d
from mindspore.common.initializer import HeNormal

class Decoder(nn.Cell):
    def __init__(self, dataset, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet':
            low_level_inplanes = 256

        if dataset == "NTID":
            limbsNum = 18
        else:
            limbsNum = 13

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, has_bias=False,
                                                 pad_mode='pad',weight_init=HeNormal())
        self.bn1 = BatchNorm(48,)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(2048, 256, 1, has_bias=False,
                                                 pad_mode='pad',weight_init=HeNormal())
        self.bn2 = BatchNorm(256, )
        self.last_conv = nn.SequentialCell(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, has_bias=False,
                                                 pad_mode='pad',weight_init=HeNormal()),
                                       BatchNorm(256,),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, has_bias=False,
                                                 pad_mode='pad',weight_init=HeNormal()),
                                       BatchNorm(256,),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes+1, kernel_size=1, stride=1,
                                                 pad_mode='pad',weight_init=HeNormal()))
#                                        nn.Conv2d(256, num_classes+5+1, kernel_size=1, stride=1)) # Use in case of extacting the bounding box

        self.maxpool = MaxPool2d(kernel_size=3, stride=2,padding=1)

        #self._init_weight()


    def construct(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        #x = self.conv2(x)
        #x = self.bn2(x)
        #x = self.relu(x)

        low_level_feat = self.maxpool(low_level_feat)

        x = nn.ResizeBilinear()(x, size=low_level_feat.shape[2:], align_corners=True)

        x = mindspore.ops.Concat(axis=1)((x, low_level_feat))
        x = self.last_conv(x)

        #x = self.maxpool(x)

        return x
    '''
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()'''

def build_decoder(dataset, num_classes, backbone, BatchNorm):
    return Decoder(dataset ,num_classes, backbone, BatchNorm)


if __name__ =='__main__':

    context.set_context(mode=context.PYNATIVE_MODE)
    decoder = build_decoder(dataset='lsp',backbone='resnet',num_classes=21, BatchNorm=nn.BatchNorm2d)

    input = mindspore.numpy.rand((1, 2048, 64, 64))
    low_level_feat = mindspore.numpy.rand((1, 256, 127, 127))
    output = decoder(input,low_level_feat)
    print('decoder output.shape : ',output.shape)
    print("OK !")