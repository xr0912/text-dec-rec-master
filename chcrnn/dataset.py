#!/usr/bin/python
# encoding: utf-8
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np


# LMDB和SQLite/MySQL等关系型数据库不同，属于key-value数据库（把LMDB想成dict会比较容易理解），键key与值value都是字符串。
class lmdbDataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None):
        # 通过env = lmdb.open()打开环境
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        # 建立事务
        with self.env.begin(write=False) as txn:
            # 进行查询
            nSamples = int(txn.get('num-samples'))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    # 返回数据集大小
    def __len__(self):
        return self.nSamples

    # 实现数据集的下标索引，返回对应的图像和标记tensor
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            # 进行查询
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key))

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


# 调整标准化，样本的尺寸不一样。大部分的神经网络希望一个固定大小的图像。返回标准化后的图像
class resizeNormalize(object):
    # size = 32x100, h = 32 w = 100
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    # 我们将把它们写成一个可调用的类而不是函数，所以变换所需的参数不必在每次调用时都传递。为此，我们需实现 __call__ 方法
    def __call__(self, img):
        # 重新处理输入的图像大小
        img = img.resize(self.size, self.interpolation)
        # 把 numpy 图像转换为 PyTorch 图像（我们需要交换轴）
        img = self.toTensor(img)
        # 将图像进行标准化调整
        img.sub_(0.5).div_(0.5)
        return img


# 随机顺序采样器，进行迭代
class randomSequentialSampler(sampler.Sampler):
    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    # 使用生成器函数实现可迭代对象 __iter__
    def __iter__(self):
        # 将图像进行batch分支处理
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    # 返回数据集大小
    def __len__(self):
        return self.num_samples


# 对齐调整，返回图像和标签
class alignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)
        # 32x100
        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        # 调整标准化，生成transform对象
        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        # 将图像进行联合
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
