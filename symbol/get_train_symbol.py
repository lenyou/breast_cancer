
import mxnet as mx
from symbol.resnet_50_2d import get_backbone
import private_config


def binary_branch(feature,label,name='binary_branch'):

    available=(label!=-1)
    label  = mx.sym.reshape(data=label,shape=(-3))
    feature = mx.symbol.FullyConnected(data=feature,num_hidden=4,name='%s_fullycon'%name)
    feature_act = mx.symbol.SoftmaxActivation(data=feature)
    feature = mx.sym.pick(data=feature_act,index=label,keepdims=True,axis=1)
    loss = (-1)*((1-feature)**2)*(available)*mx.sym.log10(data=feature+1e-24) #focal loss
    loss = mx.symbol.MakeLoss(data=loss)
    #metric_result

    result_arg = mx.sym.argmax(data=feature_act,axis=1)
    label_for_metric  =label
    positive_sum = mx.sym.sum((label_for_metric==0))
    negative_sum = mx.sym.sum((label_for_metric==1))
    negative_recall = positive_sum*mx.sym.sum((label_for_metric == result_arg)*(label_for_metric==0))/(positive_sum*positive_sum+1e-24)
    positive_recall =  negative_sum*mx.sym.sum((label_for_metric == result_arg)*(label_for_metric==1))/(negative_sum*negative_sum+1e-24)
    # acc = mx.sym.sum((label == result_connect_arg)/mx.sym.sum((label>-1))
    acc_sum =mx.sym.sum((label_for_metric>-1))
    acc = acc_sum*mx.sym.sum((label_for_metric == result_arg))/(acc_sum*acc_sum+1e-24)
    return loss,positive_recall,negative_recall,acc
    

def get_train_symbol():

    data = mx.symbol.Variable('data')
    attach = mx.symbol.Variable('attach')
    age,her,p53 = mx.sym.SliceChannel(data=attach,num_outputs=3,axis=1,squeeze_axis=False)
    label = mx.symbol.Variable('label')
    backbone_output = get_backbone(data) #
    backbone_vector = mx.symbol.Pooling(data=backbone_output, kernel=(3,3), stride=(2,2), pad=(1,1), pool_type='avg',global_pool=True)
    age_vector =mx.symbol.FullyConnected(data=age,num_hidden=8,name='age')
    her_vector =mx.symbol.FullyConnected(data=her,num_hidden=8,name = 'her')
    p53_vector =mx.symbol.FullyConnected(data=p53,num_hidden=8,name='p')
    backbone_vector = mx.symbol.reshape(data=backbone_vector,shape=(0,-1))
    backbone_vector = mx.symbol.concat(*[backbone_vector,age_vector,her_vector,p53_vector],dim=1)
    binary_loss,binary_positive_recall,binary_negative_recall,binary_acc=binary_branch(backbone_vector,label)    
    return mx.sym.Group([binary_loss,mx.sym.BlockGrad(binary_positive_recall),mx.sym.BlockGrad(binary_negative_recall),mx.sym.BlockGrad(binary_acc)])

def get_inference_symbol():
    data = mx.symbol.Variable('data')
    attach = mx.symbol.Variable('attach')
    age,her,p53 = mx.sym.SliceChannel(data=attach,num_outputs=3,axis=1,squeeze_axis=False)
    backbone_output = get_backbone(data) #
    backbone_vector = mx.symbol.Pooling(data=backbone_output, kernel=(3,3), stride=(2,2), pad=(1,1), pool_type='avg',global_pool=True)
    age_vector =mx.symbol.FullyConnected(data=age,num_hidden=8,name='age')
    her_vector =mx.symbol.FullyConnected(data=her,num_hidden=8,name = 'her')
    p53_vector =mx.symbol.FullyConnected(data=p53,num_hidden=8,name='p')
    backbone_vector = mx.symbol.reshape(data=backbone_vector,shape=(0,-1))
    backbone_vector = mx.symbol.concat(*[backbone_vector,age_vector,her_vector,p53_vector],dim=1)
    backbone_vector = mx.symbol.FullyConnected(data=backbone_vector,num_hidden=4,name='%s_fullycon'%'binary_branch')
    binary_act = mx.symbol.SoftmaxActivation(data=backbone_vector,name='multi')
    return binary_act

if __name__ == '__main__':
    
    infer_symbol = get_inference_symbol()
    infer_symbol.save('inference_symbol.json')
    print 'finish'
    