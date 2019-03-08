# -*- coding:utf-8 -*-
import numpy as np
import os
from PIL import Image, ImageDraw
from tqdm import tqdm
import advancedeast.cfg as cfg

def point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
    # 判断px的位置是否在框中，也就是是否在shrink后的框的范围内
    # px = 42.0 py = 54.0
    # [[52.50499833,59.99110933][49.18473059,94.04287662][458.71980924,96.26063708][460.31711259,62.28612669]]
    # p_min = [49.18473059,59.99110933] p_max = [460.3171125996.26063708]
    if (p_min[0] <= px <= p_max[0]) and (p_min[1] <= py <= p_max[1]):
        xy_list = np.zeros((4, 2))
        xy_list[:3, :] = quad_xy_list[1:4, :] - quad_xy_list[:3, :]
        xy_list[3] = quad_xy_list[0, :] - quad_xy_list[3, :]
        yx_list = np.zeros((4, 2))
        yx_list[:, :] = quad_xy_list[:, -1:-3:-1]
        a = xy_list * ([py, px] - yx_list)
        b = a[:, 0] - a[:, 1]
        if np.amin(b) >= 0 or np.amax(b) <= 0:
            return True
        else:
            return False
    else:
        return False


def point_inside_of_nth_quad(px, py, xy_list, shrink_1, long_edge):
    nth = -1
    # 注意这里第二次用到了这个数组，因为需要处理两个框所以这么写
    vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
          [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
    for ith in range(2):
        quad_xy_list = np.concatenate((
            np.reshape(xy_list[vs[long_edge][ith][0]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][1]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][2]], (1, 2)),
            np.reshape(xy_list[vs[long_edge][ith][3]], (1, 2))), axis=0)
        p_min = np.amin(quad_xy_list, axis=0)
        p_max = np.amax(quad_xy_list, axis=0)
        if point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
            if nth == -1:
                nth = ith
            else:
                nth = -1
                break
    return nth


def shrink(xy_list, ratio = cfg.shrink_ratio):
    # shrink_ratio = 0.2
    if ratio == 0.0:
        return xy_list, xy_list

    # xy_list = np.array([[30.91, 20.4], [26.91, 44.2], [343.85, 45.2], [345.85, 21.4]])
    diff_1to3 = xy_list[:3, :] - xy_list[1:4, :]  # [[4., -23.8],[-316.94, -1.],[-2., 23.8]]
    diff_4 = xy_list[3:4, :] - xy_list[0:1, :]  # array([[314.94,1.]])
    diff = np.concatenate((diff_1to3, diff_4), axis=0)  # [[4., -23.8],[-316.94, -1.],[-2., 23.8 ],[314.94, 1.]]
    # square()计算组各元素的平方，这个axis的取值就是这个精确定位某个元素需要经过多少数组的长度
    dis = np.sqrt(np.sum(np.square(diff), axis=-1))  # array([ 24.13379373, 316.94157758, 23.88388578, 314.9415876 ])

    # determine which are long or short edges 确定哪边是长边或者短边
    # np.reshape(dis, (2,2)) = [[24.13379373, 316.94157758],[23.88388578, 314.9415876]]
    # np.sum = [48.01767952, 631.88316518]
    # 数组中最小最大元素的索引：np.argmin(a)，np.argmax(a)
    long_edge = int(np.argmax(np.sum(np.reshape(dis, (2, 2)), axis=0)))  # 1
    short_edge = 1 - long_edge  # 0
    # cal r length array
    r = [np.minimum(dis[i], dis[(i + 1) % 4]) for i in range(4)]
    # [24.133793734098255, 23.88388578100306, 23.88388578100306, 24.133793734098255]

    # cal theta array
    diff_abs = np.abs(diff)  # [[4., 23.8],[316.94, 1.],[2., 23.8],[314.94, 1.]]
    diff_abs[:, 0] += cfg.epsilon  # [[4.0001, 23.8],[316.9401, 1.],[2.0001, 23.8],[314.9401, 1.]]
    # 对矩阵diff_ads中每个y/x取反正切,
    theta = np.arctan(diff_abs[:, 1] / diff_abs[:, 0])  # [1.40428117, 0.00315516, 1.48695551, 0.0031752 ]

    # shrink two long edges 缩小两条长边，也就是13两边1--3
    temp_new_xy_list = np.copy(xy_list)
    shrink_edge(xy_list, temp_new_xy_list, long_edge, r, theta, ratio)
    shrink_edge(xy_list, temp_new_xy_list, long_edge + 2, r, theta, ratio)

    # shrink two short edges 缩小两条短边，也就是24两边0--2
    new_xy_list = np.copy(temp_new_xy_list)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge, r, theta, ratio)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge + 2, r, theta, ratio)
    return temp_new_xy_list, new_xy_list, long_edge


def shrink_edge(xy_list, new_xy_list, edge, r, theta, ratio=cfg.shrink_ratio):
    # shrink_ratio = 0.2
    if ratio == 0.0:
        return

    start_point = edge  # 1
    end_point = (edge + 1) % 4  # 2
    # 数组元素的符号：np.sign(a)，返回数组中各元素的正负符号，用1和-1表示
    long_start_sign_x = np.sign(xy_list[end_point, 0] - xy_list[start_point, 0])  # 1.0
    new_xy_list[start_point, 0] = xy_list[start_point, 0] + long_start_sign_x * ratio * r[start_point] * np.cos(theta[start_point])  # 31.686753379716443
    long_start_sign_y = np.sign(xy_list[end_point, 1] - xy_list[start_point, 1])  # 1.0
    new_xy_list[start_point, 1] = xy_list[start_point, 1] + long_start_sign_y * ratio * r[start_point] * np.sin(theta[start_point])  # 44.21507147529412

    # long edge one, end point
    long_end_sign_x = -1 * long_start_sign_x  # -1
    new_xy_list[end_point, 0] = xy_list[end_point, 0] + long_end_sign_x * ratio * r[end_point] * np.cos(theta[start_point])  #339.07324662028356
    long_end_sign_y = -1 * long_start_sign_y
    new_xy_list[end_point, 1] = xy_list[end_point, 1] + long_end_sign_y * ratio * r[end_point] * np.sin(theta[start_point])  #45.18492852470589


def process_label(data_dir=cfg.data_dir):
    with open(os.path.join(data_dir, cfg.val_fname), 'r') as f_val:
        f_list = f_val.readlines()
    with open(os.path.join(data_dir, cfg.train_fname), 'r') as f_train:
        f_list.extend(f_train.readlines())

    for line, _ in zip(f_list, tqdm(range(len(f_list)))):
        line_cols = str(line).strip().split(',')
        img_name, width, height = line_cols[0].strip(), int(line_cols[1].strip()), int(line_cols[2].strip())

        # pixel_size = 4,height = width = 736
        gt = np.zeros((height // cfg.pixel_size, width // cfg.pixel_size, 7))  # 184,184,7
        # train_label_dir_name = 'labels_%s/' % train_task_id -- 'labels_2T736/'
        train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)
        # 'labels_2T736/TB1OKBxLXXXXXcuXVXXunYpLFXX.npy'
        xy_list_array = np.load(os.path.join(train_label_dir,img_name[:-4] + '.npy'))
        # train_image_dir_name = 'images_%s/' % train_task_id -- 'images_2T736/'
        train_image_dir = os.path.join(data_dir, cfg.train_image_dir_name)

        with Image.open(os.path.join(train_image_dir, img_name)) as im:
            # 在736x736的原图上画框
            draw = ImageDraw.Draw(im)
            for xy_list in xy_list_array:
                _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)
                shrink_1, _, long_edge = shrink(xy_list, cfg.shrink_side_ratio)
                # 在放缩后的顶点中找到最小的x,y
                p_min = np.amin(shrink_xy_list, axis=0)
                # 在放缩后的顶点中找到最大的x,y
                p_max = np.amax(shrink_xy_list, axis=0)
                # floor of the float
                ji_min = (p_min / cfg.pixel_size - 0.5).astype(int) - 1
                # +1 for ceil of the float and +1 for include the end
                ji_max = (p_max / cfg.pixel_size - 0.5).astype(int) + 3

                # 确定要画框的范围
                imin = np.maximum(0, ji_min[1])
                imax = np.minimum(height // cfg.pixel_size, ji_max[1])
                jmin = np.maximum(0, ji_min[0])
                jmax = np.minimum(width // cfg.pixel_size, ji_max[0])

                for i in range(imin, imax):
                    for j in range(jmin, jmax):
                        # 在大框的范围中，每隔4个坐标画一次顶点
                        px = (j + 0.5) * cfg.pixel_size
                        py = (i + 0.5) * cfg.pixel_size
                        # 判断顶点是否在框中
                        if point_inside_of_quad(px, py, shrink_xy_list, p_min, p_max):

                            gt[i, j, 0] = 1
                            line_width, line_color = 1, 'red'
                            # 判断框是否到了起始位置
                            ith = point_inside_of_nth_quad(px, py,
                                                           xy_list,
                                                           shrink_1,
                                                           long_edge)
                            vs = [[[3, 0], [1, 2]], [[0, 1], [2, 3]]]

                            if ith in range(2):
                                gt[i, j, 1] = 1
                                # 开始的位置
                                if ith == 0:
                                    line_width, line_color = 2, 'yellow'
                                # 结束的位置
                                else:
                                    line_width, line_color = 2, 'green'
                                gt[i, j, 2:3] = ith
                                gt[i, j, 3:5] = xy_list[vs[long_edge][ith][0]] - [px, py]
                                gt[i, j, 5:] = xy_list[vs[long_edge][ith][1]] - [px, py]

                            # 在放缩后的大框中每隔4个坐标的顶点再画一个上下左右各为2的小框框
                            draw.line([(px - 0.5 * cfg.pixel_size,
                                        py - 0.5 * cfg.pixel_size),
                                       (px + 0.5 * cfg.pixel_size,
                                        py - 0.5 * cfg.pixel_size),
                                       (px + 0.5 * cfg.pixel_size,
                                        py + 0.5 * cfg.pixel_size),
                                       (px - 0.5 * cfg.pixel_size,
                                        py + 0.5 * cfg.pixel_size),
                                       (px - 0.5 * cfg.pixel_size,
                                        py - 0.5 * cfg.pixel_size)],
                                      width=line_width, fill=line_color)

            act_image_dir = os.path.join(cfg.data_dir,
                                         cfg.show_act_image_dir_name)
            if cfg.draw_act_quad:
                im.save(os.path.join(act_image_dir, img_name))
        train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)
        np.save(os.path.join(train_label_dir,
                             img_name[:-4] + '_gt.npy'), gt)


if __name__ == '__main__':
    process_label()
