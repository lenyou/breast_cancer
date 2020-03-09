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
import pandas as pd
import cv2


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


class jpg_file_Iter(mx.io.DataIter):
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
    def __init__(self,train_path,batch_size,**kwargs):
        super(jpg_file_Iter, self).__init__()
        self.train_path=train_path
        csv_path = os.path.join(self.train_path,"feats.csv")
        self.csv_pd = pd.read_csv(csv_path)
        self.csv_pd = self.csv_pd[self.csv_pd['id']!='2a83e7c8345b3e893bc4bcc3e761ae731e1eec62f5dec43accab1207af6ca0f7']
        self.batch_size = batch_size
        # self.provide_data = [('data',(batch_size,3,256,256)),("age_data",(self.batch_si
        # e,1)),('her_data',(self.batch_size,1)),('ps3_data',(self.batch_size,1))]  
         
        self.provide_data = [('data',(batch_size,1,int(256*1.5),256)),('attach',(batch_size,3))]     
        # self.provide_data = [('data',(batch_size,1,int(256*1.5),256))] 
        self.provide_label = [('label',(batch_size,1))]
        self.random_index_list = range(self.csv_pd.shape[0])
        random.shuffle(self.random_index_list)
        self._current_id=0
        print 'before getting'
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
        image_array_list = []
        image_label_list = []
        attached_data_list = []
        image_root_folder = os.path.join(self.train_path,'images')
        for _ in range(self.batch_size):
            if self._current_id>=len(self.random_index_list):
                self.reset()
                return False
            csv_index = self.random_index_list[self._current_id]
            row = self.csv_pd.iloc[csv_index]
            age = int(row['age'])
            her = int(row['HER2'])
            if row['P53']=='True':
                P53 = 1
            else:
                P53 = 0
            type_label = int(row['molecular_subtype'])-1
            id = row['id']
            image_folder = os.path.join(image_root_folder,id)
            select_index = random.randint(0,max(0,int(len(os.listdir(image_folder))-1)))
            file_path = os.path.join(image_folder,os.listdir(image_folder)[select_index])
            array = cv2.imread(file_path)
            # print array.sha
            height = 256
            width = int(256*1.5)
            attached_data_list.append([age,her,P53])
            img_resize=cv2.resize(array,(width,height),interpolation = cv2.INTER_AREA)
            img_gray = cv2.cvtColor(img_resize,cv2.COLOR_RGB2GRAY)
            # cv2.imwrite('tmp.jpg',img_gray)
            image_array_list.append(img_gray)
            image_label_list.append(np.array([type_label]))
            # print img_gray.shape
            self._current_id +=1
        image_array = np.stack(image_array_list,axis=0)
        image_array = image_array[:,np.newaxis,:,:]
        label_array = np.stack(image_label_list,axis=0)
        image_array_mx = mx.nd.array(image_array)
        label_array_mx = mx.nd.array(label_array)
        attached_data_array = np.array(attached_data_list)
        attached_data_array_mx = mx.nd.array(attached_data_array)
        self._batch = mx.io.DataBatch(data=[image_array_mx,attached_data_array_mx])
        # self._batch = mx.io.DataBatch(data=[image_array_mx])
        self._batch.label = [label_array_mx]




      
        # batch_train_array_mx = mx.nd.array()
        # self._batch = mx.io.DataBatch(data=[batch_train_array_mx])
        # self._batch.label = [batch_label_array_mx]
        return True

if __name__ == "__main__":
    # data_iter = npy_file_Iter('/data1/new_start_project_mirror/bm_project/code/bm_multi_class_train_1_21/train_total_array.npy','/data1/new_start_project_mirror/bm_project/code/bm_multi_class_train_1_21/bm_label_array.npy',3)
    data_iter = jpg_file_Iter(train_path="/data1/advance_machine_learning/breast_cancer/train",batch_size=3)
    label_list = []
    while True:
        try:
            data = data_iter.next()
            label = data.label[0].asnumpy()
            label_list.append(label)
        except:
            break
    label_array = np.concatenate(label_list,axis=0)
    for i in range(4):
        print np.sum((label_array==i))
    print label_array.shape
    print 'finish'
