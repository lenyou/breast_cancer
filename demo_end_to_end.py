# ------ coding: utf-8 --------
import mxnet as mx
from scipy.ndimage.interpolation import zoom
import numpy as np
import dicom
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import cv2
import argparse
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import time


#
CLASS_DICT = {'calcific nodule': 'calcific_nodule',
              'pleural nodule': 'pleural_nodule',
              '0-3nodule': '0-3nodule', '3-6nodule': '3-6nodule', '6-10nodule': '6-10nodule',
              'pleural calcific nodule': 'pleural_nodule', '10-30nodule': '10-30nodule',
              'mass': 'mass',
              '0-5GGN': '0-5GGN', '5GGN': '5GGN', '5pGGN': '5GGN', '0-5pGGN': '0-5GGN', '5mGGN': '5GGN',
              '0-5mGGN': '0-5GGN', "unknown": "unknown"}
detailed_class = CLASS_DICT.keys()+["fibrosis","ignore",'pathological']
modelB_reference_list =  ['benign',
                          'adeno',
                          'squamous',
                          'small cell',
                          'other cancer',
                          'carcinoma in situ']
# benign_list = ['doctor_benign', 'quasi-benign', 'AAH', 'CGL', 'JH', 'YZ', 'XWJJ', 'RYZXY', 'LBJ', 'MXBL',
#                'NYXJJ', 'CX', 'PHJL', 'TMCJ', 'FPXBL', 'LXQT']
# non_benign_list = ['benign','adeno', 'squamous', 'small cell', 'other cancer', 'carcinoma in situ', 'quasi-malignant', 'XA',
#                        'LINA', 'XLA', 'XXBA', 'FHXXBA', 'DXBA', 'LA', 'WFHL', 'ZYL', 'YWXA', 'WJRXA', 'JRXXA', 'TB',
#                        'XP', 'RT', 'WRT', 'SX', 'BY', 'EXQT', 'malignant']



high_risk_list = ['HighRisk']
mid_risk_list = ['MidRisk']
low_risk_list = ['LowRisk','FpLowRisk']

class Predictor():
    def __init__(self, symbol_file, param_file, gpu_id, input_desc, out_name, out_shape):
        self.out_name = out_name
        self.out_shape = out_shape
        if type(symbol_file)==str:
            self.mod = get_mod(symbol_file, input_desc, param_file, gpu_id)
        else:
            self.mod = get_mod_from_tarExFileObject(symbol_file, input_desc, param_file, gpu_id)


    def predict(self, data):
        tic = time.time()
        self.mod.forward(data, is_train=False)
        out = dict(zip(self.mod.output_names, self.mod.get_outputs()))
        result = out[self.out_name]
        # seg_result = out["seg_result_output"]
        result = mx.nd.reshape(data=result, shape=self.out_shape)
        result = result.asnumpy()
        # seg_result = mx.ndarray.argmax(data= seg_result,axis=1)
        # seg_result = seg_result.asnumpy()
        toc = time.time()
        print "predict_time:%f"%(toc-tic)
        # drawing
        # seg_3d_mask = np.argmax(seg_result,axis=1)
        # origin_data = data.data[0].asnumpy()[0,0,:,:,:]
        # seg_3d_mask = seg_result.reshape((64,64,64))
        # for im_id,im in enumerate(seg_3d_mask):
        #     plt.subplot(121)
        #     plt.imshow(im)

        #     plt.subplot(122)
        #     plt.imshow(origin_data[im_id,:,:])
        #     plt.show()


        return result

    def make_data(self, win_img, target_spacing, spacing_zyx, dl_side_len, bbox, z_start_end):
        xmin, ymin, xmax, ymax = bbox
        z_start, z_end = z_start_end
        print spacing_zyx

        # computer new start,end
        num_dcms = np.ceil(dl_side_len * target_spacing[0] / spacing_zyx[0])
        nodule_z_center = z_start * 0.5 + z_end * 0.5
        z_upper = np.ceil(nodule_z_center + num_dcms * 0.5)
        z_floor = np.int(nodule_z_center - num_dcms * 0.5)
        need_slices = np.array(range(z_floor, int(z_upper + 1)))
        need_slices[need_slices < 0] = 0
        need_slices[need_slices > (win_img.shape[0] - 1)] = win_img.shape[0] - 1
        # cut from start to end
        need_img = win_img[need_slices, :, :]
        # np.save('data.npy', need_img)
        # compute x,y side
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        scale_wh = spacing_zyx[1] / target_spacing[1]
        scale_d = spacing_zyx[0] / target_spacing[0]
        center_x = scale_wh * center_x
        center_y = scale_wh * center_y
        len_wh = int(scale_wh * need_img.shape[1])

        xmin = max(0, int(center_x - dl_side_len / 2.0))
        xmax = xmin + dl_side_len
        if xmax > len_wh:
            xmax = len_wh
            xmin = len_wh - dl_side_len

        ymin = max(0, int(center_y - dl_side_len / 2.0))
        ymax = ymin + dl_side_len
        if ymax > len_wh:
            ymax = len_wh
            ymin = len_wh - dl_side_len

        data = zoom(need_img, (scale_d, scale_wh, scale_wh), mode='nearest', order=1)

        center_z = data.shape[0] / 2
        zmin = max(0, int(center_z - dl_side_len / 2.0))
        zmax = zmin + dl_side_len

        data = data[np.newaxis, np.newaxis, zmin:zmax, ymin:ymax, xmin:xmax]
        np.save('./image_cubic',data)
        tmp_data = data[0,0,:,:,:]
        # data = data.transpose(0,1,3,4,2)
        data = mx.nd.array(data)
        data_batch = mx.io.DataBatch(data=[data])
        return data_batch,tmp_data




def load_checkpoint(param_file):
    """
    Load model checkpoint from file.
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    save_dict = mx.nd.load(param_file)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params

def load_checkpoint_from_tarExFileObject(param_ExFileObject):
    """
    Load model checkpoint from file.
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    save_dict = mx.nd.load_frombuffer(param_ExFileObject.read())
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def get_mod(json_symbol_file, input_desc, param_file, gpu_id):
    args, auxs = load_checkpoint(param_file)
    symbol = mx.symbol.load(json_symbol_file)
    ctx = mx.gpu(gpu_id)
    # ctx = mx.cpu()
    data_names = [desc[0] for desc in input_desc]
    mod = mx.mod.Module(symbol, data_names=['data','attach'], label_names=None, context=ctx)
    mod.bind(data_shapes=input_desc, for_training=False)
    mod.set_params(args, auxs)
    return mod

def get_mod_from_tarExFileObject(symbol_ExFileObject, input_desc, param_ExFileObject, gpu_id):
    args, auxs = load_checkpoint_from_tarExFileObject(param_ExFileObject)
    symbol = mx.symbol.load_json(symbol_ExFileObject.read())
    ctx = mx.gpu(gpu_id)
    data_names = [desc[0] for desc in input_desc]
    mod = mx.mod.Module(symbol, data_names=data_names, label_names=None, context=ctx)
    mod.bind(data_shapes=input_desc, for_training=False)
    mod.set_params(args, auxs)
    return mod


def parse_args():
    parser = argparse.ArgumentParser(description='Single-shot detection network demo')
    parser.add_argument('--dir', dest='dir', nargs='?',
                        help='demo image directory, optional', type=str)
    parser.add_argument('--save_name', dest='save_name', nargs='?',
                        help='demo image directory, optional', type=str)
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    # dcm_file_path = ""
    # dicom_names = get_dicom_names(dcm_file_path)
    # hu_origin = load_dicom(dicom_names)
    args = parse_args()
    overall_folder = args.dir
    save_name = args.save_name
    overall_folder = "/data1/advance_machine_learning/breast_cancer/test"
    csv_path = os.path.join(overall_folder,'feats.csv')
    image_folder = os.path.join(overall_folder,'images')
    csv_pd = pd.read_csv(csv_path)
    input_shape = [('data',(1,1,int(256*1.5),256)),('attach',(1,3))]
    out_name = 'multi_output'
    out_shape = (1,4)
    predictor = Predictor(
                          "inference.json",
                          "./model/breast_model-0130.params",
                          1,
                          input_shape,
                          out_name = out_name,
                          out_shape = out_shape,
                          )
    result_pd = pd.DataFrame()
    for rid,r in csv_pd.iterrows():
        age = int(r['age'])
        her = int(r['HER2'])
        if r['P53']=='True':
            P53 = 1
        else:
            P53 = 0
        id = r['id']
        patient_folder = os.path.join(image_folder,id)
        vote_list = []
        for file in os.listdir(patient_folder):
            file_path = os.path.join(patient_folder,file)
            array = cv2.imread(file_path)
            width = int(256*1.5)
            height =  256
            img_resize=cv2.resize(array,(width,height),interpolation = cv2.INTER_AREA)
            img_gray = cv2.cvtColor(img_resize,cv2.COLOR_RGB2GRAY)
            img_gray = img_gray[np.newaxis,np.newaxis,:,:]
            img_gray_mx = mx.nd.array(img_gray)
            attach_array = np.array([age,her,P53])
            attach_array = attach_array[np.newaxis,:]
            attach_array_mx = mx.nd.array(attach_array)
            data_batch = mx.io.DataBatch(data=[img_gray_mx,attach_array_mx])
            result  = predictor.predict(data_batch)
            result_index = np.argmax(result,axis=1)[0]
            vote_list.append(result_index)
        print vote_list
        vote_array = np.array(vote_list)
        bin_array = np.bincount(vote_array)
        max_index = np.argmax(bin_array)
        tmp_dict ={}
        tmp_dict['id'] = id
        tmp_dict['class'] = max_index+1
        result_pd = result_pd.append(tmp_dict,ignore_index=True)
    result_pd.to_csv('submission.csv')
