#!/usr/bin/python
# encoding: utf-8
import os
import time
from glob import glob
from advancedeast.predict import predict_txt
from chcrnn.crnn import crnnOcr

img_paths = glob('./test/data/*.*')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':

    print("----------------------------------------")
    print("Being detected and recognized...")
    t = time.time()

    for path in img_paths:
        print(path)
        t1 = time.time()
        box_items = predict_txt(path)
        print("TextBoxesPredict takes time:{}s".format(time.time() - t1))

        t2 = time.time()
        with open('./test/result/' + path.split('/')[3] + '.txt', 'w') as f:
            for box in box_items:
                result = crnnOcr(path, box)
                f.write('{},{}\r\n'.format(box, result))
        f.close()
        print("TextPredict takes time:{}s".format(time.time() - t2))

    print("It takes time:{}s".format(time.time() - t))
    print("----------------------------------------")
