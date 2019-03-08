# -*- coding:utf-8 -*-
import numpy as np
from PIL import Image, ImageDraw
import os
import random
from tqdm import tqdm
import advancedeast.cfg as cfg
from advancedeast.label import shrink


def batch_reorder_vertexes(xy_list_array):
    reorder_xy_list_array = np.zeros_like(xy_list_array)
    for xy_list, i in zip(xy_list_array, range(len(xy_list_array))):
        reorder_xy_list_array[i] = reorder_vertexes(xy_list)
    return reorder_xy_list_array


# 重新排序顶点
def reorder_vertexes(xy_list):
    # determine the first point with the smallest x,if two has same x, choose that with smallest y,
    # 确定具有最小x的第一个点，如果两个具有相同的x，则选择具有最小y的那个，
    # xy_list = [[30.91, 20.4], [26.91, 44.2], [343.85, 45.2], [345.85, 21.4]]
    reorder_xy_list = np.zeros_like(xy_list)  # array([[0,0],[0,0],[0,0],[0,0]])
    ordered = np.argsort(xy_list, axis=0)  # array([[1,0],[0,3],[2,1],[3,2]])
    xmin1_index = ordered[0, 0]  # 1
    xmin2_index = ordered[1, 0]  # 0
    if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:  # 26.91  30.91
        if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index
        else:
            reorder_xy_list[0] = xy_list[xmin2_index]
            first_v = xmin2_index
    else:
        reorder_xy_list[0] = xy_list[xmin1_index]  # [26,91,44.2]
        first_v = xmin1_index  # 1
    # connect the first point to others, the third point on the other side of the line with the middle slope
    # 将第一个点连接到其他点，第三个点位于具有中间斜率的线的另一侧
    others = list(range(4))  # [0,1,2,3]
    others.remove(first_v)  # [0,2,3]
    k = np.zeros((len(others),))  # array([0,0,0])
    for index, i in zip(others, range(len(others))):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) / (xy_list[index, 0] - xy_list[first_v, 0] + cfg.epsilon)
    # k = array([-5.94985125e+00,  3.15517033e-03, -7.14867776e-02])
    k_mid = np.argsort(k)[1]  # 2
    third_v = others[k_mid]  # 3
    reorder_xy_list[2] = xy_list[third_v]  # array([345.85,  21.4 ])
    # determine the second point which on the bigger side of the middle line
    # 确定中线较大一侧的第二个点
    others.remove(third_v) # [0,2]
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]  # 46.12370918551791
    second_v, fourth_v = 0, 0
    for index, i in zip(others, range(len(others))):
        # delta = y - (k * x + b)
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index # 2
        else:
            fourth_v = index # 0
    reorder_xy_list[1] = xy_list[second_v]  # array([343.85,  45.2 ])
    reorder_xy_list[3] = xy_list[fourth_v]  # array([30.91, 20.4 ])
    # reorder_xy_list =[[ 26.91,  44.2 ],[343.85,  45.2 ],[345.85,  21.4 ],[ 30.91,  20.4 ]]

    # compare slope of 13 and 24, determine the final order
    # 比较13和24的斜率，确定最终的顺序
    k13 = k[k_mid]  # -0.07148677761121917
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (xy_list[second_v, 0] - xy_list[fourth_v, 0] + cfg.epsilon)
    # 0.0792483929033
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]  # 30.91  20.4
        for i in range(2, -1, -1):  #2,1,0
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list  # reorder_xy_list = [[30.91, 20.4], [26.91, 44.2], [343.85, 45.2], [345.85, 21.4]]


def resize_image(im, max_img_size=cfg.max_train_img_size):
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height


def preprocess():
    data_dir = cfg.data_dir
    origin_image_dir = os.path.join(data_dir, cfg.origin_image_dir_name)
    origin_txt_dir = os.path.join(data_dir, cfg.origin_txt_dir_name)
    train_image_dir = os.path.join(data_dir, cfg.train_image_dir_name)
    train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)
    if not os.path.exists(train_image_dir):
        os.mkdir(train_image_dir)
    if not os.path.exists(train_label_dir):
        os.mkdir(train_label_dir)
    draw_gt_quad = cfg.draw_gt_quad
    show_gt_image_dir = os.path.join(data_dir, cfg.show_gt_image_dir_name)
    if not os.path.exists(show_gt_image_dir):
        os.mkdir(show_gt_image_dir)
    show_act_image_dir = os.path.join(cfg.data_dir, cfg.show_act_image_dir_name)
    if not os.path.exists(show_act_image_dir):
        os.mkdir(show_act_image_dir)

    o_img_list = os.listdir(origin_image_dir)
    print('found %d origin images.' % len(o_img_list))
    train_val_set = []
    for o_img_fname, _ in zip(o_img_list, tqdm(range(len(o_img_list)))):
        with Image.open(os.path.join(origin_image_dir, o_img_fname)) as im:
            # d_wight, d_height = resize_image(im)
            # w, h = 736x736
            d_wight, d_height = cfg.max_train_img_size, cfg.max_train_img_size
            scale_ratio_w = d_wight / im.width
            scale_ratio_h = d_height / im.height
            im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
            show_gt_im = im.copy()
            # draw on the img
            draw = ImageDraw.Draw(show_gt_im)
            with open(os.path.join(origin_txt_dir,
                                   o_img_fname[:-4] + '.txt'), 'r') as f:
                anno_list = f.readlines()
            xy_list_array = np.zeros((len(anno_list), 4, 2)) # xy = [[[x0,y0],[x1,y1],[x2,y2],[x3,y3]],...]
            for anno, i in zip(anno_list, range(len(anno_list))):
                # anno = 88.82, 98.33, 98.82, 191.42, 456.95, 184.42, 447.95, 99.33,薄利汽配
                anno_colums = anno.strip().split(',')
                # anno_colums = ['30.91', ' 20.4', ' 26.91', ' 44.2', ' 343.85', ' 45.2', ' 345.85', ' 21.4', '123123']
                anno_array = np.array(anno_colums)
                # anno_array = np.array(['30.91', ' 20.4', ' 26.91', ' 44.2', ' 343.85', ' 45.2', ' 345.85',' 21.4', '123123'], dtype='<U7')
                xy_list = np.reshape(anno_array[:8].astype(float), (4, 2))
                # xy_list = np.array([[ 30.91,  20.4 ],[ 26.91,  44.2 ],[343.85,  45.2 ],[345.85,  21.4 ]])
                # 将x坐标进行缩放
                xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w
                # 将y坐标进行缩放
                xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
                # 重新排序顶点
                xy_list = reorder_vertexes(xy_list)
                # xy_list = np.array([[30.91, 20.4], [26.91, 44.2], [343.85, 45.2], [345.85, 21.4]])
                xy_list_array[i] = xy_list
                # 将长短边进行一次放缩得到第一个小矩形框
                _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)
                # 将边进行一次放缩得到两个小矩形框，也就是文本开始和结束的位置
                shrink_1, _, long_edge = shrink(xy_list, cfg.shrink_side_ratio)
                if draw_gt_quad:
                    # 画出人工标注的大矩阵框
                    draw.line([tuple(xy_list[0]),
                               tuple(xy_list[1]),
                               tuple(xy_list[2]),
                               tuple(xy_list[3]),
                               tuple(xy_list[0])
                               ],
                              width=2, fill='green')
                    # 画出放缩一次后的小矩形框
                    draw.line([tuple(shrink_xy_list[0]),
                               tuple(shrink_xy_list[1]),
                               tuple(shrink_xy_list[2]),
                               tuple(shrink_xy_list[3]),
                               tuple(shrink_xy_list[0])
                               ],
                              width=2, fill='blue')
                    # 注意这里数组设置的顺序和数，后面还会用到这个数组
                    vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
                          [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
                    # 画出后一次放缩的两个小矩形框
                    for q_th in range(2):
                        draw.line([tuple(xy_list[vs[long_edge][q_th][0]]),
                                   tuple(shrink_1[vs[long_edge][q_th][1]]),
                                   tuple(shrink_1[vs[long_edge][q_th][2]]),
                                   tuple(xy_list[vs[long_edge][q_th][3]]),
                                   tuple(xy_list[vs[long_edge][q_th][4]])],
                                  width=3, fill='yellow')

            if cfg.gen_origin_img:
                im.save(os.path.join(train_image_dir, o_img_fname))
            np.save(os.path.join(
                train_label_dir,
                o_img_fname[:-4] + '.npy'),
                xy_list_array)
            if draw_gt_quad:
                show_gt_im.save(os.path.join(show_gt_image_dir, o_img_fname))
            train_val_set.append('{},{},{}\n'.format(o_img_fname,
                                                     d_wight,
                                                     d_height))

    train_img_list = os.listdir(train_image_dir)
    print('found %d train images.' % len(train_img_list))
    train_label_list = os.listdir(train_label_dir)
    print('found %d train labels.' % len(train_label_list))

    # shuffle() 方法将序列的所有元素随机排序
    random.shuffle(train_val_set)
    val_count = int(cfg.validation_split_ratio * len(train_val_set))  # 1000
    # 将图片写入测试集
    with open(os.path.join(data_dir, cfg.val_fname), 'w') as f_val:
        f_val.writelines(train_val_set[:val_count])  # 0 -- 1000
    # 将图片写入训练集
    with open(os.path.join(data_dir, cfg.train_fname), 'w') as f_train:
        f_train.writelines(train_val_set[val_count:])  # 1000 -- 10000


if __name__ == '__main__':
    preprocess()
