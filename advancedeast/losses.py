# -*- coding:utf-8 -*-
import tensorflow as tf
import advancedeast.cfg as cfg

def quad_loss(y_true, y_pred):
    # loss for inside_score
    # 1位，是否在文本框中
    print('y_true:{}  y_pred:{}'.format(y_true,y_pred))

    logits = y_pred[:, :, :, :1]
    labels = y_true[:, :, :, :1]
    print('logits: {} labels:{} '.format(logits,labels))
    # balance positive and negative samples in an image 平衡图像中的正负样本
    # reduce_mean()求均值的一个函数，如果不设置axis，所有维度上的元素都会被求平均值，并且只会返回一个只有一个元素的张量。
    beta = 1 - tf.reduce_mean(labels)  # 求labels上所有元素的和的平均值
    # first apply sigmoid activation 首先应用sigmoid激活
    predicts = tf.nn.sigmoid(logits)  # 将预测值进行激活，也就是输入的预测值
    # log +epsilon for stable cal
    # 这里算出来的损失函数必定为负值，按照算法的意思是为了平衡图像中的正负样本
    inside_score_loss = tf.reduce_mean(
        -1 * (beta * labels * tf.log(predicts + cfg.epsilon) +
              (1 - beta) * (1 - labels) * tf.log(1 - predicts + cfg.epsilon)))
    # lambda_inside_score_loss = 4.0 这里为什么乘以4我不知道
    inside_score_loss *= cfg.lambda_inside_score_loss


    # loss for side_vertex_code
    # 2位，是否在文本框边界像素，以及是头还是尾
    vertex_logits = y_pred[:, :, :, 1:3]
    vertex_labels = y_true[:, :, :, 1:3]
    print('vertex_logits: {} vertex_labels:{} '.format(vertex_logits, vertex_labels))
    # 这是2位中的第一位，是否在文本框边界像素
    # 这是真值中的是否在文本框边界像素，注意这里用到1位中的labels
    vertex_beta = 1 - (tf.reduce_mean(y_true[:, :, :, 1:2]) / (tf.reduce_mean(labels) + cfg.epsilon))

    vertex_predicts = tf.nn.sigmoid(vertex_logits)  # 将预测值先进行激活

    pos = -1 * vertex_beta * vertex_labels * tf.log(vertex_predicts +cfg.epsilon)
    neg = -1 * (1 - vertex_beta) * (1 - vertex_labels) * tf.log(1 - vertex_predicts + cfg.epsilon)
    # tf.cast() 将x的数据格式转化成 dtype = tf.float32
    # tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素,如果是相等的那就返回True,反正返回False,返回的值的矩阵维度和A是一样的
    positive_weights = tf.cast(tf.equal(y_true[:, :, :, 0], 1), tf.float32)
    # reduce_sum(
    #     input_tensor,
    #     axis=None,
    #     keep_dims=False,
    #     name=None,
    # )
    # input_tensor:表示输入
    # axis:表示在那个维度进行sum操作。
    # keep_dims:表示是否保留原始数据的维度，False相当于执行完后原始数据就会少一个维度。
    side_vertex_code_loss = tf.reduce_sum(tf.reduce_sum(pos + neg, axis=-1) * positive_weights) / \
                            (tf.reduce_sum(positive_weights) + cfg.epsilon)
    # lambda_side_vertex_code_loss = 1.0 乘以不乘以都一样,与第1位的计算有所区别
    side_vertex_code_loss *= cfg.lambda_side_vertex_code_loss

    # loss for side_vertex_coord delta
    # 4位，是边界像素可以预测的2个顶点坐标，头和尾部分边界像素分别预测2个顶点，最后得到4个顶点坐标。
    g_hat = y_pred[:, :, :, 3:]
    g_true = y_true[:, :, :, 3:]
    print('g_hat: {} g_true:{} '.format(g_hat, g_true))
    # 注意这里与第二个损失函数之间的计算区别
    vertex_weights = tf.cast(tf.equal(y_true[:, :, :, 1], 1), tf.float32)

    pixel_wise_smooth_l1norm = smooth_l1_loss(g_hat, g_true, vertex_weights)

    side_vertex_coord_loss = tf.reduce_sum(pixel_wise_smooth_l1norm) / (
            tf.reduce_sum(vertex_weights) + cfg.epsilon)
    # lambda_side_vertex_coord_loss = 1.0
    side_vertex_coord_loss *= cfg.lambda_side_vertex_coord_loss
    return inside_score_loss + side_vertex_code_loss + side_vertex_coord_loss

# smooth_l1_loss损失函数是F-RCNN里面计算距离的函数
def smooth_l1_loss(prediction_tensor, target_tensor, weights):
    n_q = tf.reshape(quad_norm(target_tensor), tf.shape(weights))
    diff = prediction_tensor - target_tensor
    abs_diff = tf.abs(diff)
    abs_diff_lt_1 = tf.less(abs_diff, 1)
    pixel_wise_smooth_l1norm = (tf.reduce_sum(
        tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5),
        axis=-1) / n_q) * weights
    return pixel_wise_smooth_l1norm


def quad_norm(g_true):
    shape = tf.shape(g_true)
    delta_xy_matrix = tf.reshape(g_true, [-1, 2, 2])
    diff = delta_xy_matrix[:, 0:1, :] - delta_xy_matrix[:, 1:2, :]
    square = tf.square(diff)
    distance = tf.sqrt(tf.reduce_sum(square, axis=-1))
    distance *= 4.0
    distance += cfg.epsilon
    return tf.reshape(distance, shape[:-1])

