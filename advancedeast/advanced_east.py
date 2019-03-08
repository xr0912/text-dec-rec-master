# -*- coding:utf-8 -*-
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

import advancedeast.cfg as cfg
from advancedeast.network import East
from advancedeast.losses import quad_loss
from advancedeast.data_generator import gen

east = East()
east_network = east.east_network()
east_network.summary()

# keras.model.compile() 配置学习过程
# lr = 1e-3
# decay = 5e-4
print('begin training: ')
east_network.compile(loss=quad_loss,
                     optimizer=Adam(lr=cfg.lr,
                                    # 用于对梯度进行裁剪
                                    # clipvalue=cfg.clipvalue, 0.5
                                    decay=cfg.decay))

# load_weights = False
if cfg.load_weights and os.path.exists(cfg.saved_model_weights_file_path):
    east_network.load_weights(cfg.saved_model_weights_file_path)

# epoch_num = 24
# patience = 2
# initial_epoch = 0
# steps_per_epoch = 9000
# validation_steps = 1000
# 这里用到的fit_generator（）函数，是每次产生一个batch样本的生成器函数
# 生成器与模型将并行执行以提高效率。例如，该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练
east_network.fit_generator(
                           # 生成器函数
                           generator = gen(),
                           # 生成训练集，当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch
                           steps_per_epoch = cfg.steps_per_epoch,
                           # 数据迭代的轮数
                           epochs = cfg.epoch_num,
                           # 生成验证集，当validation_data为生成器时，本参数指定验证集的生成器返回次数
                           validation_data = gen(is_val=True),
                           validation_steps = cfg.validation_steps,
                           # 日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
                           verbose = 1,
                           # initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。
                           initial_epoch = cfg.initial_epoch,
                           # 回调函数，召回以调整模型性能
                           callbacks = [
                               # 当监测值不再改善时，该回调函数将中止训练
                               # 当early stop被激活（如发现loss相比上一个epoch训练没有下降），则经过patience个epoch后停止训练。
                               EarlyStopping(patience=cfg.patience, verbose=1),
                               ModelCheckpoint(filepath=cfg.model_weights_path,
                                               # 当设置为True时，将只保存在验证集上性能最好的模型
                                               save_best_only=True,
                                               save_weights_only=True,
                                               verbose=1)])

east_network.save(cfg.saved_model_file_path)

east_network.save(cfg.saved_model_weights_file_path)


