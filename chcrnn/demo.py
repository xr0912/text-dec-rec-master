#!/usr/bin/python
# encoding: utf-8
# -*- coding:utf-8 -*-
import utils
import os
import csv
import dataset
import keys
import numpy as np
from PIL import Image
import models.crnn as crnn
import torch
from torch.autograd import Variable


# model_path = './data/crnn.pth'
# img_path = './data/demo.png'
# alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model_path = './samples/model_acc97.pth'
img_path = './data/demo_ch.png'
alphabet = keys.alphabet
# 加载文本框位置信息
def load_annoataion(p):
    boxes = []
    index = 0
    if not os.path.exists(p):
        return False
    with open(p, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            x1, y1, x2, y2, x3, y3, x4, y4 = (list(map(float, line[:8])))
            boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])

    text_recs = np.zeros((len(boxes), 8), np.float)
    for box in boxes:
        text_recs[index, 0] = box[0]
        text_recs[index, 1] = box[1]
        text_recs[index, 2] = box[2]
        text_recs[index, 3] = box[3]
        text_recs[index, 4] = box[4]
        text_recs[index, 5] = box[5]
        text_recs[index, 6] = box[6]
        text_recs[index, 7] = box[7]
        index += 1
    boxes_sort = sorted(text_recs, key=lambda x: sum([x[1], x[3], x[5], x[7]]))

    return boxes_sort

# 初始化网络模型
# imgH, nc, nclass, nh,
# imgH = 32
# nc = 1
# nclass = 36 + 1
# nh = 256
model = crnn.CRNN(32, 1, len(alphabet)+1, 256)

if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)
# 0123456789abcdefghijklmnopqrstuvwxyz-
# {'0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10,
# 'a': 11, 'b': 12, 'c': 13, 'd': 14, 'e': 15, 'f': 16, 'g': 17, 'h': 18, 'i': 19, 'j': 20,
# 'k': 21, 'l': 22, 'm': 23, 'n': 24, 'o': 25, 'p': 26, 'q': 27, 'r': 28, 's': 29, 't': 30,
# 'u': 31, 'v': 32, 'w': 33, 'x': 34, 'y': 35, 'z': 36}
# 这里的字符加字母一共有36个，加上“-”一共37个nclass

image = Image.open(img_path).convert('L')
# 调整标准化，样本的尺寸不一样。大部分的神经网络希望一个固定大小的图像。返回标准化后的图像
scale = image.size[1]*1.0 / 32
w = image.size[0] / scale
w = int(w)
# 生成一个转换器对象以便后续使用
transformer = dataset.resizeNormalize((w, 32))

# 使用转换器对象进行图像标准化
image = transformer(image)

if torch.cuda.is_available():
    image = image.cuda()

image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)

# 生成特征序列26帧
# transpose（）函数主要用来转换矩阵的维度。
# 调用view之前最好先contiguous
# x.contiguous().view()
# 因为view需要tensor的内存是整块的
# view()函数作用是将一个多行的Tensor,拼接成一行。
preds = preds.transpose(1, 0).contiguous().view(-1)
# tensor([11,  0,  0,  0,  0,  0, 32,  0,  0, 11,  0, 19,  0, 22,  0, 11,  0, 12,
#         12,  0, 22,  0, 15,  0,  0,  0], device='cuda:0')

preds_size = Variable(torch.IntTensor([preds.size(0)]))
# tensor([26], dtype=torch.int32)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
if len(sim_pred) > 0:
    if sim_pred[0] == u'-':
        sim_pred = sim_pred[1:]

# a-----v--a-i-l-a-bb-l-e--- => available
result = open('.{}.txt'.format(img_path.split('.')[1]),'w')
result.write(sim_pred)
result.close()
print('%-20s => %-20s' % (raw_pred, sim_pred))