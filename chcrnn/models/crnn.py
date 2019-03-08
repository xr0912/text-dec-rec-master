#!/usr/bin/python
# encoding: utf-8
import torch.nn as nn


# 双向LSTM
class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        # bidirectional:是否是双向RNN，默认为false，
        # 若为true，则num_directions = 2，否则为1
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size() # 26,1,512
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


# 循环神经网络
class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):  # 32, 1, 37, 256
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        # ‘k’, ‘s’ and ‘p’ stand for kernel size, stride and padding size
        # 'nm' maps
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))

            # CNN+RNN的训练比较困难，所以加入了BatchNorm，有助于模型收敛
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))

            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        # 在第3和第4个maxpooling层，我们采用1 X 2大小的矩形池窗，而不是传统的平方池。
        # 此调整产生具有更大宽度的特征映射，因此具有更长的特征序列。
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn

        # 通过SquentialS快速搭建神经网络将双向LSTM结合起来，输出激活后的网络节点。
        # 这里有两层双向LSTM
        # 'nh':size of the lstm hidden state, default = 256
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),

            #  (0): BidirectionalLSTM(
            #     (rnn): LSTM(512, nHidden=256, bidirectional=True)
            #     (embedding): Linear(nHidden=512, out_features=256, bias=True)
            #   )
            # (1): BidirectionalLSTM(
            #     (rnn): LSTM(256, nHidden=256, bidirectional=True)
            #     (embedding): Linear(nHidden=512, out_features=37, bias=True)
            #   )
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()  # 1,512,1,26
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output


if __name__ == '__main__':
    pass
    # print('CRNN():')
    # # imgH, nc, nclass, nh,
    # # imgH = 32
    # # nc = 1
    # # nclass = len(opt.alphabet) + 1
    # # nh = 256
    # CRNN(32, 1, 37, 256)

