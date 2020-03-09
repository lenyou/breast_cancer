import os
#for generate_data
train_cubic_size = 64
train_cubic_spacing = [1.0,1.0,1.0]
expansion_factor = 0.7
fisrt_step_target_path = '/media/liu/data2/bm_project_clean_data/train_data_1_21_vesion2'
second_input_path = fisrt_step_target_path
second_step_target_path = './train_data/cache'
rec_name = 'bm_nodule_train'
gpus='0,1,2'

#for iterater setting
iterater_setting = dict()
iterater_setting['data_shape']=[int(train_cubic_size/expansion_factor)]*3
iterater_setting['seqlen']=int(train_cubic_size/expansion_factor)
iterater_setting['rand_mirror_prob'] =0.5
iterater_setting['rand_rotate_prob']=0.5
iterater_setting['max_rotate']=180
iterater_setting['preprocess_threads']=24
iterater_setting['has_crop']=False
iterater_setting['seg_seqlen']=0
iterater_setting['seg_inter_method']=0

# for train 

train_path = "/data1/advance_machine_learning/breast_cancer/train"
lr_rate = 1e-3
momentum = 0.9
print_frequent = 2
relation_num = 10
num_example = 100
batch_size = 15