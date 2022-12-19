import math
import mindspore
import mindspore.nn as nn
import mindspore.context as context
# import mindspore.nn.functional as F
import mindspore.ops as ops
import mindspore.common.initializer as initializer
from mindspore.common.initializer import HeNormal

import mindspore.numpy as np
from mindspore import dtype as mstype

'''
class AdaptiveAvgPool2d(nn.Cell):
    def __init__(self, output_size):
        """Initialize AdaptiveAvgPool2d."""
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def adaptive_avgpool2d(self, inputs):
        """ NCHW """
        H = self.output_size[0]
        W = self.output_size[1]

        H_start = ops.Cast()(np.arange(start=0, stop=H, dtype=mstype.float32) * (inputs.shape[-2] / H), mstype.int64)
        H_end = ops.Cast()(np.ceil(((np.arange(start=0, stop=H, dtype=mstype.float32)+1) * (inputs.shape[-2] / H))), mstype.int64)

        W_start = ops.Cast()(np.arange(start=0, stop=W, dtype=mstype.float32) * (inputs.shape[-1] / W), mstype.int64)
        W_end = ops.Cast()(np.ceil(((np.arange(start=0, stop=W, dtype=mstype.float32)+1) * (inputs.shape[-1] / W))), mstype.int64)

        pooled2 = []
        for idx_H in range(H):
            pooled1 = []
            for idx_W in range(W):
                h_s = int(H_start[idx_H].asnumpy())
                h_e = int(H_end[idx_H].asnumpy())
                w_s = int(W_start[idx_W].asnumpy())
                w_e = int(W_end[idx_W].asnumpy())
                res = inputs[:, :, h_s:h_e, w_s:w_e]
                # res = inputs[:, :, H_start[idx_H]:H_end[idx_H], W_start[idx_W]:W_end[idx_W]]  # 这样写mindspore tensor切片报类型错误，不知道为啥
                pooled1.append(ops.ReduceMean(keep_dims=True)(res, (-2,-1)))
            pooled1 = ops.Concat(-1)(pooled1)
            pooled2.append(pooled1)
        pooled2 = ops.Concat(-2)(pooled2)

        return pooled2

    def construct(self, x):
        x = self.adaptive_avgpool2d(x)
        return x
'''

class AdaptiveAvgPool2D(nn.Cell):
    def __init__(self):
        """Initialize AdaptiveAvgPool2D."""
        super(AdaptiveAvgPool2D, self).__init__()

    def construct(self, input):
        H = input.shape[-2]
        W = input.shape[-1]
        x = ops.AvgPool(kernel_size=(H,W))(input)
        return x

class _AtrousModule(nn.Cell):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_AtrousModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, pad_mode='pad', weight_init=HeNormal(),
                                     stride=1, padding=padding, dilation=dilation, has_bias=False)
        self.bn = BatchNorm(planes,)
        self.relu = nn.ReLU()

        # self._init_weight()

    def construct(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    '''
    def _init_weight(self):
        for m in self.get_parameters():
            if isinstance(m, nn.Conv2d):
                m.trainable_params()
                initializer.HeNormal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()'''


class wasp(nn.Cell):
    def __init__(self, output_stride, BatchNorm):
        super(wasp, self).__init__()
        inplanes = 2048
        if output_stride == 16:
            dilations = [48, 24, 12, 6]
            # dilations = [ 6, 12, 18, 24]
            #dilations = [24, 18, 12, 6]
            # dilations = [6, 6, 6, 6]
        elif output_stride == 8:
            dilations = [48, 36, 24, 12]
        else:
            raise NotImplementedError

        self.aspp1 = _AtrousModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _AtrousModule(256, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _AtrousModule(256, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _AtrousModule(256, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.SequentialCell(AdaptiveAvgPool2D(),
                                                 nn.Conv2d(inplanes, 256, 1, stride=1, has_bias=False, pad_mode='pad',
                                                           weight_init=HeNormal()),
                                                 nn.BatchNorm2d(256),
                                                 nn.ReLU())

        # self.global_avg_pool = nn.Sequential(nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
        #                                      nn.BatchNorm2d(256),
        #                                      nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, has_bias=False, pad_mode='pad', weight_init=HeNormal())
        self.conv2 = nn.Conv2d(256, 256, 1, has_bias=False, pad_mode='pad', weight_init=HeNormal())
        self.bn1 = BatchNorm(256, )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        # self._init_weight()

    def construct(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x1)
        x3 = self.aspp3(x2)
        x4 = self.aspp4(x3)

        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)

        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)

        x5 = self.global_avg_pool(x)
        # x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x5 = nn.ResizeBilinear()(x5, size=x4.shape[2:], align_corners=True)
        x = ops.Concat(axis=1)((x1, x2, x3, x4, x5))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    '''
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()'''


def build_wasp(output_stride, BatchNorm):
    return wasp(output_stride, BatchNorm)


if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE)
    # net = build_wasp(output_stride=16, BatchNorm=nn.BatchNorm2d)
    input = mindspore.numpy.rand((1, 2048, 64, 64))
    # output, low_level_feat = net(input)
    # output = net(input)
    # print('wasp output.shape : ', output.shape)

    my_pool = AdaptiveAvgPool2D()
    # ms_pool = AdaptiveAvgPool2d(output_size=(1,1))

    output1 = my_pool(input)
    # output2 = ms_pool(input)
    print("OK!")
