#!/usr/bin/python
# encoding: utf-8

from  predict import textPredict
import time


im = './demo/005.png'

t = time.time()
box_items = textPredict(im)
with open(im[:-4] + '.txt', 'w') as f_txt:
    f_txt.writelines(box_items)

print("It takes time:{}s".format(time.time()-t))
print("---------------------------------------")