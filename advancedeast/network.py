# coding=utf-8
from keras import Input
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Concatenate, Conv2D, UpSampling2D, BatchNormalization

import advancedeast.cfg as cfg

"""
input_shape=(img.height, img.width, 3), height and width must scaled by 32.
So images's height and width need to be pre-processed to the nearest num that
scaled by 32.And the annotations xy need to be scaled by the same ratio 
as height and width respectively.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class East:
    def __init__(self):
        self.input_img = Input(name='input_img',
                               shape=(None, None, cfg.num_channels),  # num_channels = 3
                               dtype='float32')

        vgg16 = VGG16(input_tensor=self.input_img,
                      weights='imagenet',
                      include_top=False)
        if cfg.locked_layers:
            # locked_layers = False
            # locked first two conv layers
            locked_layers = [vgg16.get_layer('block1_conv1'),
                             # Tensor("block1_conv1/Relu:0", shape=(?, ?, ?, 64), dtype=float32)
                             vgg16.get_layer('block1_conv2')]
                             # Tensor("block1_conv2/Relu:0", shape=(?, ?, ?, 64), dtype=float32)
            for layer in locked_layers:
                layer.trainable = False

        self.f = [vgg16.get_layer('block%d_pool' % i).output
                  for i in cfg.feature_layers_range]  # feature_layers_range = range(5, 1, -1) -- 5,4,3,2
        self.f.insert(0, None)
        self.diff = cfg.feature_layers_range[0] - cfg.feature_layers_num  # 1

    def g(self, i):
        # i+diff in cfg.feature_layers_range
        assert i + self.diff in cfg.feature_layers_range, ('i=%d+diff=%d not in ' % (i, self.diff)) + \
            str(cfg.feature_layers_range)
        if i == cfg.feature_layers_num:
            # BatchNormalization规范层 就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布，该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1
            # BN层的作用:（1）加速收敛 （2）控制过拟合，可以少用或不用Dropout和正则 （3）降低网络对初始化权重不敏感 （4）允许使用较大的学习率
            bn = BatchNormalization()(self.h(i))

            return Conv2D(32, 3, activation='relu', padding='same')(bn)
        else:
            return UpSampling2D((2, 2))(self.h(i))

    def h(self, i):
        # i+diff in cfg.feature_layers_range
        assert i + self.diff in cfg.feature_layers_range,('i=%d+diff=%d not in ' % (i, self.diff)) + \
            str(cfg.feature_layers_range)
        if i == 1:
            return self.f[i]
        else:
            concat = Concatenate(axis=-1)([self.g(i - 1), self.f[i]])
            bn1 = BatchNormalization()(concat)
            conv_1 = Conv2D(128 // 2 ** (i - 2), 1,
                            activation='relu', padding='same',)(bn1)
            bn2 = BatchNormalization()(conv_1)
            conv_3 = Conv2D(128 // 2 ** (i - 2), 3,
                            activation='relu', padding='same',)(bn2)
            return conv_3

    def east_network(self):
        # print('inside_score:')
        # 1位score map, 是否在文本框内
        inside_score = Conv2D(1, 1, padding='same', name='inside_score')(self.g(cfg.feature_layers_num))

        # print('side_v_code:')
        # 2位vertex code，是否属于文本框边界像素以及是头还是尾
        side_v_code = Conv2D(2, 1, padding='same', name='side_vertex_code')(self.g(cfg.feature_layers_num))

        # print('side_v_coord:')
        # 4位geo，是边界像素可以预测的2个顶点坐标
        side_v_coord = Conv2D(4, 1, padding='same', name='side_vertex_coord')(self.g(cfg.feature_layers_num))

        east_detect = Concatenate(axis=-1,name='east_detect')([inside_score,
                                                               side_v_code,
                                                               side_v_coord])
        return Model(inputs=self.input_img, outputs=east_detect)


if __name__ == '__main__':
    print('East():')
    east = East()
    # print('east_network():')
    # east_network = east.east_network()

