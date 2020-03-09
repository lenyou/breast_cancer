import mxnet as mx
import numpy as np
import cv2
import random
import private_config
import os



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
    def __init__(self, file_path_root,batch_size,relation_num,**kwargs):
        super(npy_file_Iter, self).__init__()
        self._patient_list = [os.path.join(file_path_root,i) for i in os.listdir(file_path_root)]
        self._current_id = 0
        self.provide_label = None
        self.draw_idx = 0
        self._batch_size = batch_size
        self.relation_num = relation_num
        self.provide_data = [('data',(batch_size,relation_num,216))]
        self.provide_label = [('softmax_label',(batch_size,relation_num,1))]
        self._get_batch()
        self.reset()

    # @property
    # def provide_data(self):
    #     return self.provide_data
    
    # @property
    # def provide_laebl(self):
    #     return self.provide_label

    def reset(self):
        self._current_id =0

    def iter_next(self):
        return self._get_batch()

    def next(self):
        if self.iter_next():
            return self._batch
        else:
            raise StopIteration

    def _get_batch(self):
        batch_array = None
        for i in range(self._current_id,self._current_id+self._batch_size):
            if i >= len(self._patient_list):
                id = i%len(self._patient_list)
            else:
                id = i
            current_train_path = self._patient_list[id]
            train_array_path = os.path.join(current_train_path,'train_total_array.npy')
            label_array_path = os.path.join(current_train_path,'bm_target_array.npy')
            train_array = np.load(train_array_path)
            label_array = np.load(label_array_path)
            # print train_array_path
            # print train_array.shape
            # print train_array_path
            # print label_array.shape
            assert train_array.shape[0]==label_array.shape[0],'train,label must be equal'
            num_nodule_per_patient = train_array.shape[0]
            total_array = np.concatenate((train_array,label_array[:,np.newaxis]),axis=1)
            if num_nodule_per_patient>=self.relation_num:
                np.random.shuffle(total_array)
                total_array = total_array[:self.relation_num]
            else:
                comple_num = self.relation_num-num_nodule_per_patient
                comple_array = np.zeros((comple_num,total_array.shape[1]))
                np.random.shuffle(total_array)
                total_array = np.concatenate((total_array,comple_array),axis=0)
            
            if batch_array is None:
                batch_array = total_array[np.newaxis,:,:]
            else:
                batch_array = np.concatenate((batch_array,total_array[np.newaxis,:,:]),axis=0)
            self._current_id = i

        if self._current_id>=len(self._patient_list):
            self.reset()
            return False 
        batch_train_array = batch_array[:,:,:-1]
        batch_label_array = batch_array[:,:,-1][:,:,np.newaxis]
        batch_train_array_mx = mx.nd.array(batch_train_array)
        batch_label_array_mx = mx.nd.array(batch_label_array)
        self._batch = mx.io.DataBatch(data=[batch_train_array_mx])
        self._batch.label = [batch_label_array_mx]
        return True

