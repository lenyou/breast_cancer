import pandas as pd
import mxnet as mx
import numpy as np
import private_config
import logging
from metric import *
from symbol.get_train_symbol import get_train_symbol,get_inference_symbol

def convert_pretrained(name, args):
    """
    Special operations need to be made due to name inconsistance, etc

    Parameters:
    ---------
    name : str
        pretrained model name
    args : dict
        loaded arguments

    Returns:
    ---------
    processed arguments as dict
    """
    return args




def get_lr_scheduler(learning_rate, lr_refactor_step, lr_refactor_ratio,
                     num_example, batch_size, begin_epoch):
    """
    Compute learning rate and refactor scheduler

    Parameters:
    ---------
    learning_rate : float
        original learning rate
    lr_refactor_step : comma separated str
        epochs to change learning rate
    lr_refactor_ratio : float
        lr *= ratio at certain steps
    num_example : int
        number of training images, used to estimate the iterations given epochs
    batch_size : int
        training batch size
    begin_epoch : int
        starting epoch

    Returns:
    ---------
    (learning_rate, mx.lr_scheduler) as tuple
    """
    assert lr_refactor_ratio > 0
    iter_refactor = [int(r) for r in lr_refactor_step.split(',') if r.strip()]
    if lr_refactor_ratio >= 1:
        return (learning_rate, None)
    else:
        lr = learning_rate
        epoch_size = num_example // batch_size
        for s in iter_refactor:
            if begin_epoch >= s:
                lr *= lr_refactor_ratio
        if lr != learning_rate:
            logging.getLogger().info("Adjusted learning rate to {} for epoch {}".format(lr, begin_epoch))
        steps = [epoch_size * (x - begin_epoch) for x in iter_refactor if x > begin_epoch]
        if not steps:
            return (lr, None)
        lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_refactor_ratio)
        return (lr, lr_scheduler)

def load_checkpoint(prefix, epoch):
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
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params

def use_pretrain_params(pretrained_path,pretrained_epoch,train_iter,net):
    args, auxs = load_checkpoint(pretrained_path, pretrained_epoch)
    args = convert_pretrained(pretrained_path, args)
    data_shape_dict = dict(train_iter.provide_data + train_iter.provide_label)
    for k, v in data_shape_dict.items():
        v = list(v)
        v[0] = int(private_config.batch_size / len(private_config.gpus.split(',')))
        data_shape_dict[k] = tuple(v)
    print data_shape_dict
    arg_shape, out_shape, aux_shape = net.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(net.list_arguments(), arg_shape))
    out_shape_dict = zip(net.list_outputs(), out_shape)
    aux_shape_dict = dict(zip(net.list_auxiliary_states(), aux_shape))
    try:
        conv0_weight = args['conv0_weight'].asnumpy()
        if conv0_weight.shape[1] != data_cfg.seqlen:
            stride = data_cfg.seqlen / 3
            make_conv0_weight = []  # [conv0_weight[:,0,np.newaxis,:,:]]*seqlen
            for i in range(3):
                make_conv0_weight.extend([conv0_weight[:, i, np.newaxis, :, :]] * stride)

            args['conv0_weight'] = mx.nd.array(np.concatenate(tuple(make_conv0_weight), axis=1) / 3.0)
            # import test
            # test.test(make_conv0_weight, arg_params['conv0_weight'])
            bn_data_gamma = args['bn_data_gamma'].asnumpy()
            bn_data_beta = args['bn_data_beta'].asnumpy()
            bn_data_moving_mean = auxs['bn_data_moving_mean'].asnumpy()
            bn_data_moving_var = auxs['bn_data_moving_var'].asnumpy()
            gamma = []  # [bn_data_gamma[0]]*seqlen
            beta = []  # [bn_data_beta[0]]*seqlen
            mean = []  # [bn_data_moving_mean[0]]*seqlen
            var = []  # [bn_data_moving_var[0]]*seqlen
            for i in range(3):
                gamma.extend([bn_data_gamma[i]] * stride)
                beta.extend([bn_data_beta[i]] * stride)
                mean.extend([bn_data_moving_mean[i]] * stride)
                var.extend([bn_data_moving_var[i]] * stride)
            ret = [np.array(elem) for elem in [gamma, beta, mean, var]]

            args['bn_data_gamma'], args['bn_data_beta'], auxs['bn_data_moving_mean'], auxs[
                'bn_data_moving_var'] = (mx.nd.array(elem) for elem in ret)
    except:
        pass
    init_internal = mx.init.Normal(sigma=0.01)
    init = mx.init.Xavier(factor_type="in", rnd_type='gaussian', magnitude=2)
    for k in net.list_arguments():
        if k in (data_shape_dict):
            continue
        if k in args:
            if args[k].shape != arg_shape_dict[k]:
                orgin_shape = list(args[k].shape)
                new_shape = list(arg_shape_dict[k])
                print new_shape
                if new_shape[0] > orgin_shape[0] or (len(new_shape) >= 2 and new_shape[1] > orgin_shape[1]):
                    print 'init', k
                    args[k] = mx.nd.zeros(shape=arg_shape_dict[k])
                    init(k, args[k])
                    continue
                if len(new_shape) == 4:
                    new_arg = args[k].asnumpy()

                    if len(orgin_shape) != 5:
                        new_arg = new_arg[:new_shape[0], :new_shape[1], :, :, np.newaxis]
                        new_arg = np.concatenate([new_arg] * new_shape[-1], axis=4)
                        args[k] = mx.nd.array(
                            new_arg * float(orgin_shape[1]) / (float(new_shape[1]) * float(new_shape[-1])))
                    else:
                        new_arg = new_arg[:new_shape[0], :new_shape[1], :, :, :]
                        args[k] = mx.nd.array(
                            new_arg * float(orgin_shape[1]) / (float(new_shape[1])))
                elif len(new_shape) == 1:
                    args[k] = args[k][:new_shape[0]]
                elif len(new_shape) == 2:
                    new_arg = args[k].asnumpy()

                    new_arg = new_arg[:new_shape[0], :new_shape[1]]
                    args[k] = mx.nd.array(
                        new_arg * float(orgin_shape[1]) / (float(new_shape[1])))
                else:
                    raise NotImplementedError
                if args[k].shape != arg_shape_dict[k]:
                    print 'init', k
                    args[k] = mx.nd.zeros(shape=arg_shape_dict[k])
                    init(k, args[k])
        else:
            print 'init', k

    for k in net.list_auxiliary_states():
        if k in (data_shape_dict):
            continue
        if k in auxs:
            if auxs[k].shape != aux_shape_dict[k]:
                orgin_shape = list(auxs[k].shape)
                new_shape = list(aux_shape_dict[k])
                if new_shape[0] > orgin_shape[0]:
                    print 'init', k
                    auxs[k] = mx.nd.zeros(shape=aux_shape_dict[k])
                    init(k, auxs[k])
                    continue
                if len(new_shape) == 1:
                    auxs[k] = auxs[k][:new_shape[0]]
                else:
                    raise NotImplementedError
        else:
            print 'init', k
            auxs[k] = mx.nd.zeros(shape=aux_shape_dict[k])
            init(k, auxs[k])
    return args,auxs



def train(train_path,batch_size,pretrained_path,pretrained_epoch,prefix='./model/breast_model',log_file=None):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if log_file:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)
    train_symbol = get_train_symbol()
    inference_symbol = get_inference_symbol()
    inference_symbol.save('inference.json')
    gpu_list = private_config.gpus.split(',')
    context_list = [mx.gpu(int(i)) for i in gpu_list]
    from .jpg_iterater import jpg_file_Iter
    train_iter  = jpg_file_Iter(train_path,batch_size)
    val_iter = None
    batch_end_callback = mx.callback.Speedometer(train_iter.batch_size, frequent=private_config.print_frequent)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    mod =mx.mod.Module(train_symbol,data_names=['data','attach'],label_names=['label'],context=context_list)


    learning_rate, lr_scheduler = get_lr_scheduler(private_config.lr_rate, '250,400',
            0.1, private_config.num_example, 32, 0)
    optimizer_params={'learning_rate':learning_rate,
                      'momentum':0.9,
                      'wd':5e-3,
                      'lr_scheduler':lr_scheduler,
                      'clip_gradient':5,
                      'rescale_grad': 1.0 / 1 if 1> 0 else 1.0 }
    eval_metric = bm_Metric()

    pretrain_args = None
    pretrain_auxs = None
    if pretrained_path!='no_pretrain':
        pretrain_args,pretrain_auxs = use_pretrain_params(pretrained_path,pretrained_epoch,train_iter,train_symbol)
    print "starting training"
    mod.fit(train_iter,
            val_iter,
            eval_metric=eval_metric,
            validation_metric=eval_metric,
            batch_end_callback=batch_end_callback,
            epoch_end_callback=epoch_end_callback,
            kvstore='local',
            optimizer='sgd',
            optimizer_params=optimizer_params,
            begin_epoch=0,
            num_epoch=500,
            initializer=mx.init.Xavier(),
            arg_params = pretrain_args,
            aux_params = pretrain_auxs
           )
    return 
