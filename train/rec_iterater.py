#Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
import private_config
import random
import matplotlib.pyplot as plt
import numpy as np



def random_crop(img,label,image_shape=int(private_config.train_cubic_size/private_config.expansion_factor),target_shape=private_config.train_cubic_size):
    center_point = label[:3]
    z_pixel_range = label[5]*image_shape
    random_num = min(z_pixel_range,image_shape-target_shape)
    moving_center = [random.randint(min(-1,int(-random_num/2)),max(int(random_num/2),1)) for _ in center_point]
    center_index = [int(i*image_shape) for i in center_point]
    crop_offset = int(target_shape/2)
    new_center_index = [max(crop_offset,center_index[i]+moving_center[i]) for i in range(len(center_point))]
    if target_shape%2!=0:
        raise RuntimeError('target_shape is wrong')
    random_crop_img = img[new_center_index[0]-crop_offset:new_center_index[0]+crop_offset,
                          new_center_index[1]-crop_offset:new_center_index[1]+crop_offset,
                          new_center_index[2]-crop_offset:new_center_index[2]+crop_offset]
    regulization_moving_center = [-1*float(i)/target_shape for i in moving_center]
    label[:3] = np.array(regulization_moving_center)
    return random_crop_img,label

def append_label(label,ignore_index=3): #hard code for processing yanzheng tag
    bm_detail_label = label[:,-2:-1]
    type_label = label[:,-1:]
    bm_total_label = (bm_detail_label/private_config.bm_interval).astype(np.int)
    density_label = (type_label/private_config.density_interval).astype(np.int)
    density_label = np.where(density_label==3,-1,density_label)
    # print label[0,:]
    label = np.concatenate((label,bm_total_label,density_label),axis=1)
    # print label[:,-1]
    return label


class Rec_Iterator(mx.io.DataIter):
    """
    The new detection iterator wrapper for mx.io.ImageDetRecordIter which is
    written in C++, it takes record file as input and runs faster.
    Supports various augment operations for object detection.

    Parameters:
    -----------
    path_imgrec : str
        path to the record file
    path_imglist : str
        path to the list file to replace the labels in record
    batch_size : int
        batch size
    data_shape : tuple
        (3, height, width)
    label_width : int
        specify the label width, use -1 for variable length
    label_pad_width : int
        labels must have same shape in batches, use -1 for automatic estimation
        in each record, otherwise force padding to width in case you want t
        rain/validation to match the same width
    label_pad_value : float
        label padding value
    resize_mode : str
        force - resize to data_shape regardless of aspect ratio
        fit - try fit to data_shape preserving aspect ratio
        shrink - shrink to data_shape only, preserving aspect ratio
    mean_pixels : list or tuple
        mean values for red/green/blue
    kwargs : dict
        see mx.io.ImageDetRecordIter

    Returns:
    ----------

    """
    def __init__(self, data_path, batch_size,iterater_setting=private_config.iterater_setting):
        super(Rec_Iterator, self).__init__()
        self.rec = mx.io.ImageDetRecordIter(path_imgrec = data_path,batch_size = batch_size,**iterater_setting)
        self.provide_label = None
        self.draw_idx = 0
        self._get_batch()
        if not self.provide_label:
            raise RuntimeError("Invalid ImageDetRecordIter: " + data_path)
        self.reset()

    @property
    def provide_data(self):
        return [('data',(self.batch_size,1,private_config.train_cubic_size,private_config.train_cubic_size,private_config.train_cubic_size))]

    def reset(self):
        self.rec.reset()

    def iter_next(self):
        return self._get_batch()

    def next(self):
        if self.iter_next():
            return self._batch
        else:
            raise StopIteration

    def _get_batch(self):
        self._batch = self.rec.next()
        if not self._batch:
            return False
        if self.provide_label is None:
            # estimate the label shape for the first batch, always reshape to n*5
            first_label = self._batch.label[0][0].asnumpy()
            self.batch_size = self._batch.label[0].shape[0]
            self.label_header_width = int(first_label[4])
            self.label_object_width = int(first_label[5])
            self.label_start = 4 + self.label_header_width
            self.max_objects = (first_label.size - self.label_start) // self.label_object_width
            print self.max_objects
            self.label_shape = (self.batch_size, self.label_object_width)
            self.label_end = self.label_start + self.max_objects * self.label_object_width
            self.provide_label = [('label', (self.batch_size,10))]
        # modify label
        data = self._batch.data[0].asnumpy()
        label = self._batch.label[0].asnumpy()
        tmp = label[:, self.label_start:]
        label = tmp.reshape((self.batch_size,8))
        origin_shape=int(private_config.train_cubic_size/private_config.expansion_factor)
        target_shape = int(private_config.train_cubic_size)
        label[:,3:-2] = label[:,3:-2]*origin_shape/target_shape
        output_label_list=[]
        output_data_list = []
        for item_id,data_item in enumerate(data):
            tmp_data,tmp_label = random_crop(data_item,label[item_id])
            output_data_list.append(tmp_data)
            # print tmp_data.shape
            output_label_list.append(tmp_label)
        # print np.unique(label[:,-1])
        output_data = np.stack(output_data_list,axis=0)
        output_data = output_data.reshape(output_data.shape[0],1,output_data.shape[1],output_data.shape[2],output_data.shape[3])
        output_label = np.stack(output_label_list,axis=0)
        output_label = append_label(output_label)
        # print np.unique(output_label[:,-1])
        self._batch.data = [mx.nd.array(output_data)]
        self._batch.label = [mx.nd.array(output_label)]
        return True


if __name__=='__main__':
    reciterator = Rec_Iterator(private_config.train_path,1,private_config.iterater_setting)
    while  True:
        data_batch = reciterator.next()
        import matplotlib.pyplot as plt
        import numpy as np
        show_array = data_batch.data[0].asnumpy()
        label_array = data_batch.label[0].asnumpy()
        print show_array.shape
        print label_array
        plt.imshow(show_array[0,32,:,:],cmap='gray')
        plt.show()
    print data_batch.data[0].shape,data_batch.label[0].shape
