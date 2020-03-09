import mxnet as mx
import numpy as np
import cv2
import random
import private_config
import os
import pandas as pd
import cv2
import imutils
import matplotlib.pyplot as plt
import random


# def rotate_3D():

    
    
#     return 


def aug(imgs):
    for im_id,im in enumerate(imgs):
        im_ = im[0,:,:,:]
        a = random.randint(0,360)
        for id,item in enumerate(im_):
            im_[id,:,:] = imutils.rotate(item,a)
        imgs[im_id,0,:,:,:]=im_
    cubic_size = private_config.train_cubic_size/2
    center_point = imgs.shape[2]/2
    imgs=imgs[:,:,center_point-cubic_size:center_point+cubic_size,center_point-cubic_size:center_point+cubic_size,center_point-cubic_size:center_point+cubic_size]
    return imgs


class npy_file_Iter(mx.io.DataIter):
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
    def __init__(self,train_path,label_path,batch_size,**kwargs):
        super(npy_file_Iter, self).__init__()
        self.train_array = np.load(train_path)
        self.bm_label_array = np.load(label_path)
        self.bm_label_array = self.bm_label_array.reshape((self.bm_label_array.shape[0],1))
        self.random_index_list = range(self.train_array.shape[0])
        random.shuffle(self.random_index_list)
        self.border_length = len(self.random_index_list)
        self._batch_size =batch_size
        self.provide_data = [('data',(batch_size,1,self.train_array.shape[1],self.train_array.shape[2],self.train_array.shape[2]))]
        self.provide_label = [('label',(batch_size,1))]
        self._current_id = 0
        self._get_batch()
        self.reset()

    # @property
    # def provide_data(self):
    #     return self.provide_data
    
    # @property
    # def provide_laebl(self):
    #     return self.provide_label

    def reset(self):
        print 'starting'
        self._current_id =0
        random.shuffle(self.random_index_list)

    def iter_next(self):
        return self._get_batch()

    def next(self):
        if self.iter_next():
            return self._batch
        else:
            raise StopIteration

    def _get_batch(self):
        batch_array = None
        batch_label_array = None
        for i in range(self._current_id,self._current_id+self._batch_size):
            if i >= self.border_length:
                id = i%self.border_length
            else:
                id = i
            id_in_shuffle = self.random_index_list[id]
            item_array = self.train_array[id_in_shuffle,:,:,:]
            label_array = self.bm_label_array[id_in_shuffle]
            if batch_array is None:
                batch_array = item_array[np.newaxis,:,:,:]
            else:
                batch_array = np.concatenate((batch_array,item_array[np.newaxis,:,:,:]),axis=0)
            if batch_label_array is None:
                batch_label_array=label_array[np.newaxis,:]
            else:
                batch_label_array =  np.concatenate((batch_label_array,label_array[np.newaxis,:]),axis=0)
            self._current_id = i
        
        if self._current_id>=self.border_length:
            self.reset()
            return False 
        batch_array = batch_array.reshape(batch_array.shape[0],1,batch_array.shape[1],batch_array.shape[2],batch_array.shape[3])
        print np.sum(batch_label_array==0)
        batch_array = aug(batch_array)
        batch_train_array_mx = mx.nd.array(batch_array)
        batch_label_array_mx = mx.nd.array(batch_label_array)
        self._batch = mx.io.DataBatch(data=[batch_train_array_mx])
        self._batch.label = [batch_label_array_mx]
        return True

if __name__ == "__main__":
    data_iter = npy_file_Iter('/data1/new_start_project_mirror/bm_project/code/bm_multi_class_train_1_21/train_total_array.npy','/data1/new_start_project_mirror/bm_project/code/bm_multi_class_train_1_21/bm_label_array.npy',3)
    while True:
        data_iter.next()
    print 'finish'
