# -*- coding:utf-8 -*-
import os
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import advancedeast.cfg as cfg

# batch_size = 1
def gen(batch_size=cfg.batch_size, is_val=False):
    # max_train_img_size = 736
    img_h, img_w = cfg.max_train_img_size, cfg.max_train_img_size  # 736x736
    # num_channels = 3
    # 这里的x是用来存图片的数组也就是736x736，3通道的图片
    x = np.zeros((batch_size, img_h, img_w, cfg.num_channels), dtype=np.float32)  # [1,736,736,3]
    # pixel_size = 4
    pixel_num_h = img_h // cfg.pixel_size  # 184
    pixel_num_w = img_w // cfg.pixel_size  # 184
    y = np.zeros((batch_size, pixel_num_h, pixel_num_w, 7), dtype=np.float32)  # [1,184,184,7]
    if is_val:  # False
        with open(os.path.join(cfg.data_dir, cfg.val_fname), 'r') as f_val:
            f_list = f_val.readlines()
    else:
        # train_fname = 'train_2T736.txt'
        with open(os.path.join(cfg.data_dir, cfg.train_fname), 'r') as f_train:
            # 每一行读入图片和图片的尺寸 736x736
            f_list = f_train.readlines()
    while True:
        for i in range(batch_size):
            # random gen an image name
            random_img = np.random.choice(f_list)
            img_filename = str(random_img).strip().split(',')[0]
            # load img and img anno
            img_path = os.path.join(cfg.data_dir,
                                    cfg.train_image_dir_name,
                                    img_filename)
            img = image.load_img(img_path)
            img = image.img_to_array(img)
            x[i] = preprocess_input(img, mode='tf')

            gt_file = os.path.join(cfg.data_dir,
                                   cfg.train_label_dir_name,
                                   img_filename[:-4] + '_gt.npy')
            y[i] = np.load(gt_file)

        # python中有一个非常有用的语法叫做生成器, 所利用到的关键字就是yield。
        # 有效利用生成器这个工具可以有效地节约系统资源, 避免不必要的内存占用。
        yield x, y
