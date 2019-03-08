#!/usr/bin/python
# encoding: utf-8

import torch
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import chcrnn.utils as utils
import chcrnn.dataset as dataset
import chcrnn.keys as keys
from chcrnn.models import crnn
from PIL import Image


def crnnSource():
    alphabet = keys.alphabet
    converter = utils.strLabelConverter(alphabet)
    if torch.cuda.is_available():
        model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cuda()
    else:
        model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cpu()
    path = './chcrnn/samples/model_acc97.pth'
    # path = './chcrnn/samples/mixed_second_finetune_acc97p7.pth'
    model.eval()
    model.load_state_dict(torch.load(path))
    return model, converter


model, converter = crnnSource()  # 加载模型


def crnnOcr(img_path, box):
    img = Image.open(img_path).convert('L')

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = np.int(min(Xs))
    x2 = np.int(max(Xs))
    y1 = np.int(min(Ys))
    y2 = np.int(max(Ys))

    image = img.crop((x1, y1, x2, y2))

    scale = image.size[1] * 1.0 / 32

    w = int(image.size[0] / scale)

    transformer = dataset.resizeNormalize((w, 32))
    if torch.cuda.is_available():
        image = transformer(image).cuda()
    else:
        image = transformer(image).cpu()

    image = image.view(1, *image.size())
    image = Variable(image)
    model.eval()
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    return sim_pred
