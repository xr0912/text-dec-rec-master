train_task_id = '2T736'
# 从该参数指定的epoch开始训练，在继续之前的训练时有用。
initial_epoch = 0
epoch_num = 24
lr = 1e-3
decay = 5e-4
clipvalue = 0.5  # default 0.5, 0 means no clip
patience = 2
load_weights = False
lambda_inside_score_loss = 4.0
lambda_side_vertex_code_loss = 1.0
lambda_side_vertex_coord_loss = 1.0

total_img = 10000
# validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。
validation_split_ratio = 0.1
max_train_img_size = int(train_task_id[-3:])  # 736
max_predict_img_size = int(train_task_id[-3:])  # 2400
assert max_train_img_size in [256, 384, 512, 640, 736], \
    'max_train_img_size must in [256, 384, 512, 640, 736]'
if max_train_img_size == 256:
    batch_size = 8
elif max_train_img_size == 384:
    batch_size = 4
elif max_train_img_size == 512:
    batch_size = 2
else:
    batch_size = 1
steps_per_epoch = total_img * (1 - validation_split_ratio) // batch_size  # 9000
validation_steps = total_img * validation_split_ratio // batch_size  # 1000

data_dir = 'icpr/'
origin_image_dir_name = 'image_10000/'
origin_txt_dir_name = 'txt_10000/'
train_image_dir_name = 'images_%s/' % train_task_id  # 'images_2T736/'
train_label_dir_name = 'labels_%s/' % train_task_id  # 'labels_2T736/'
show_gt_image_dir_name = 'show_gt_images_%s/' % train_task_id
show_act_image_dir_name = 'show_act_images_%s/' % train_task_id
gen_origin_img = True
draw_gt_quad = True
draw_act_quad = True
val_fname = 'val_%s.txt' % train_task_id
train_fname = 'train_%s.txt' % train_task_id  # 'train_2T736.txt'
# in paper it's 0.3, maybe too large to this problem
shrink_ratio = 0.2
# pixels between 0.1 and 0.3 are side pixels
shrink_side_ratio = 0.6
epsilon = 1e-4

num_channels = 3
feature_layers_range = range(5, 1, -1)  # 左闭右开5到2倒着取 -- 5,4,3,2
# feature_layers_range = range(3, 0, -1)
feature_layers_num = len(feature_layers_range)  # 4
# pixel_size = 4
pixel_size = 2 ** feature_layers_range[-1]  # 2的平方
locked_layers = False

model_weights_path = './advancedeast/model/weights_%s.{epoch:03d}-{val_loss:.3f}.h5' \
                     % train_task_id
saved_model_file_path = './advancedeast/saved_model/east_model_%s.h5' \
                        % train_task_id
saved_model_weights_file_path = './advancedeast/saved_model/east_model_weights_%s.h5'\
                                % train_task_id

pixel_threshold = 0.9
side_vertex_pixel_threshold = 0.9
trunc_threshold = 0.1
predict_cut_text_line = False
predict_write2txt = True
