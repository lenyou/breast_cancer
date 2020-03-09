import mxnet as mx
import numpy as np
import os
import cv2
class bm_Metric(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self, eps=1e-8):
        super(bm_Metric, self).__init__('MultiBox')
        self.eps = eps
        self.name = ['binary_loss','binary_positive_recall','binary_negative_recall','binary_acc']
        self.num=len(self.name)
        self.draw_idx = 0
        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def update(self, labels, preds):
        """
        Implementation of updating metrics
        """
        # get generated multi label from network
        preds=(p.asnumpy() for p in preds )
        binary_loss,binary_positive_recall,binary_negative_recall,binary_acc = preds
        pred_list = [ binary_loss,binary_positive_recall,binary_negative_recall,binary_acc]
        for i in range(len(self.sum_metric)):
            self.sum_metric[i]+=pred_list[i][0]
            self.num_inst[i]+=1


        # loc_target,cls_target,anchor_boxes,imgs=preds
        # loc_target,cls_target,anchor_boxes,imgs=loc_target[0,:],cls_target[0,:],anchor_boxes[0,:],imgs[0,0,:]
        # save_dir='/media/A/workspace/ssd_3d/validate_pic'
        # save_dir=os.path.join(save_dir,'%03d'%self.draw_idx)
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        # d,h,w=imgs.shape
        # draw_boxes=[[] for _ in range(d)]
        # vx,vy,vz,vw,vh,vd=(0.1, 0.1,0.01,0.2, 0.2, 0.02)
        # for cls_id,loc_t,anchor_boxe in zip(cls_target,loc_target,anchor_boxes):
        #     cls_id=int(cls_id)
        #     if cls_id==-1:
        #         continue
        #     al,at,adl,ar,ab,adh=anchor_boxe
        #     aw = ar - al;
        #     ah = ab - at;
        #     ad = adh - adl;
        #     ax = (al + ar) / 2.
        #     ay = (at + ab) / 2.
        #     az = (adl + adh) / 2.
        #     ox = 0 * vx * aw + ax;
        #     oy = 0 * vy * ah + ay;
        #     oz = 0 * vz * ad + az;
        #     ow = exp(0 * vw) * aw / 2;
        #     oh = exp(0 * vh) * ah / 2;
        #     od = exp(0 * vd) * ad / 2;
        #     # ox = loc_t[0] * vx * aw + ax;
        #     # oy = loc_t[ 1] * vy * ah + ay;
        #     # oz = loc_t[ 2] * vz * ad + az;
        #     # ow = exp(loc_t[3] * vw) * aw / 2;
        #     # oh = exp(loc_t[ 4] * vh) * ah / 2;
        #     # od = exp(loc_t[5] * vd) * ad / 2;
        #     xmin = max(int((ox - ow)*w),0);
        #     ymin = max(int((oy - oh)*h),0);
        #     zmin = max(int((oz - od)*d),0);
        #     xmax = min(int((ox + ow)*w),w);
        #     ymax = min(int((oy + oh)*h),h);
        #     zmax = min(int((oz + od)*d),d);
        #     for z_idx in range(zmin,zmax):
        #         draw_boxes[z_idx].append([xmin,ymin,xmax,ymax,cls_id])
        # for idx_img,(img,boxes) in enumerate(zip(imgs,draw_boxes)):
        #     img=np.concatenate([img[:,:,np.newaxis]]*3,axis=2).astype(np.uint8)
        #     for draw_label in boxes:
        #         # print self.draw_idx,'22222222'
        #         cv2.rectangle(img, (draw_label[0], draw_label[1]), (draw_label[2], draw_label[3]), color=[0,0,255],
        #                   thickness=2)
        #         cv2.putText(img, '%s %.3f' % (private_config.classes[int(draw_label[4])][:4], 1.0), (draw_label[0], draw_label[1] - 1),
        #                     color=[0,0,255], fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
        #     cv2.imwrite(os.path.join(save_dir,'%03d'%idx_img+'.jpg'),img)
        # self.draw_idx+=1
        # self.sum_metric[0] += multiloss[0]
        # self.num_inst[0] += 1
        # self.sum_metric[1] += binaryloss[0]
        # self.num_inst[1] +=1
        # self.sum_metric[2] += positive_recall[0]
        # self.num_inst[2] +=1
        # self.sum_metric[3] += negetive_recall[0]
        # self.num_inst[3] +=1
        # self.sum_metric[4] += acc[0]
        # self.num_inst[4] +=1
        # self.sum_metric[5] += multi_acc[0]
        # self.num_inst[5] +=1

        # print 'label:',label_valid,' pos:',print_recall_sum[0]," neg:",print_fp_sum[0]
        # pos_label_idx = np.where(label > 0)
        # pos_label = label[pos_label_idx]
        # pos_pred = pred_idx_prob[pos_label_idx]
        #
        # self.sum_metric[4] += np.sum(pos_label != pos_pred)
        # self.num_inst[4] += len(pos_label_idx[0])

    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)

