# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from yolo.loss.utils import adjust_pred_tensor, adjust_true_tensor
from yolo.loss.utils import conf_delta_tensor
from yolo.loss.utils import loss_class_tensor, loss_conf_tensor, loss_coord_tensor, wh_scale_tensor

def sum_loss(losses):
    return tf.sqrt(tf.reduce_sum(losses))

def loss_fn(list_y_trues, list_y_preds,
            anchors=[23,121, 30,241, 40,174, 42,273, 53,316, 56,230, 66,303, 81,318, 104,337],
            image_size=[288, 288], 
            ignore_thresh=0.5, 
            grid_scale=1,
            obj_scale=5,
            noobj_scale=1,
            xywh_scale=1,
            class_scale=1):
    
    calculator = LossTensorCalculator(image_size=image_size,
                                        ignore_thresh=ignore_thresh, 
                                        grid_scale=grid_scale,
                                        obj_scale=obj_scale,
                                        noobj_scale=noobj_scale,
                                        xywh_scale=xywh_scale,
                                        class_scale=class_scale)
    loss_yolo_1, list_loss_1 = calculator.run(list_y_trues[0], list_y_preds[0], anchors=anchors[12:]) # y_true (1, 7, 7, 3, 15)
    loss_yolo_2, list_loss_2 = calculator.run(list_y_trues[1], list_y_preds[1], anchors=anchors[6:12]) # y_true (1, 14, 14, 3, 15)
    loss_yolo_3, list_loss_3 = calculator.run(list_y_trues[2], list_y_preds[2], anchors=anchors[:6]) # y_true (1, 28, 28, 3, 15)

    list_3_losses = [ list_loss_1, list_loss_2, list_loss_3 ]
    loss_box = [loss[0] for loss in list_3_losses]
    loss_conf = [loss[1] for loss in list_3_losses]
    loss_class = [loss[2] for loss in list_3_losses]

    return sum_loss([loss_yolo_1, loss_yolo_2, loss_yolo_3]), sum_loss(loss_box), sum_loss(loss_conf), tf.redude_sum(loss_class)


class LossTensorCalculator(object):
    def __init__(self,
                 image_size=[288, 288], 
                 ignore_thresh=0.5, 
                 grid_scale=1,
                 obj_scale=5,
                 noobj_scale=1,
                 xywh_scale=1,
                 class_scale=1):
        self.ignore_thresh  = ignore_thresh
        self.grid_scale     = grid_scale
        self.obj_scale      = obj_scale
        self.noobj_scale    = noobj_scale
        self.xywh_scale     = xywh_scale
        self.class_scale    = class_scale        
        self.image_size     = image_size        # (h, w)-ordered

    def run(self, y_true, y_pred, anchors=[66,303, 81,318, 104,337]):
        # 1. setup
        y_pred = tf.reshape(y_pred, y_true.shape)
        object_mask = tf.expand_dims(y_true[..., 4], 4)

        # 2. Adjust prediction (bxy, twh)
        preds = adjust_pred_tensor(y_pred)

        # 3. Adjust ground truth (bxy, twh)
        trues = adjust_true_tensor(y_true)

        # 4. conf_delta tensor
        conf_delta = conf_delta_tensor(y_true, preds, anchors, self.ignore_thresh)

        # 5. loss tensor
        wh_scale =  wh_scale_tensor(trues[..., 2:4], anchors, self.image_size)
        
        loss_box = loss_coord_tensor(object_mask, preds[..., :4], trues[..., :4], wh_scale, self.xywh_scale)
        loss_conf = loss_conf_tensor(object_mask, preds[..., 4], trues[..., 4], self.obj_scale, self.noobj_scale, conf_delta)
        loss_class = loss_class_tensor(object_mask, preds[..., 5:], trues[..., 5], self.class_scale)
        loss = loss_box + loss_conf + loss_class
        return loss * self.grid_scale, [loss_box * self.grid_scale, loss_conf * self.grid_scale, loss_class * self.grid_scale]


if __name__ == '__main__':
    import os
    from yolo import PROJECT_ROOT
    def test():
        yolo_1 = np.load(os.path.join(PROJECT_ROOT, "yolo_1.npy")).astype(np.float32)
        pred_yolo_1 = np.load(os.path.join(PROJECT_ROOT, "pred_yolo_1.npy")).astype(np.float32)

        calculator = LossTensorCalculator()
        loss_tensor = calculator.run(tf.constant(yolo_1), pred_yolo_1)
        loss_value = loss_tensor.numpy()[0]
        
        if np.allclose(loss_value, 63.16674):
            print("Test Passed")
        else:
            print("Test Failed")
            print(loss_value)

    test()