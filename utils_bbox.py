import numpy as np
import torch
from torch import nn
from torchvision.ops import nms


class BBoxUtility(object):
    def __init__(self, num_classes):
        self.num_classes    = num_classes

    def ssd_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #-----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def decode_boxes(self, mbox_loc, anchors, variances):
        # 获得先验框的宽与高
        anchor_width     = anchors[:, 2] - anchors[:, 0]
        anchor_height    = anchors[:, 3] - anchors[:, 1]
        # 获得先验框的中心点
        anchor_center_x  = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchor_center_y  = 0.5 * (anchors[:, 3] + anchors[:, 1])

        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center_x = mbox_loc[:, 0] * anchor_width * variances[0]
        decode_bbox_center_x += anchor_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * anchor_height * variances[0]
        decode_bbox_center_y += anchor_center_y
        
        # 真实框的宽与高的求取
        decode_bbox_width   = torch.exp(mbox_loc[:, 2] * variances[1])
        decode_bbox_width   *= anchor_width
        decode_bbox_height  = torch.exp(mbox_loc[:, 3] * variances[1])
        decode_bbox_height  *= anchor_height

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = torch.cat((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), dim=-1)
        # 防止超出0与1
        decode_bbox = torch.min(torch.max(decode_bbox, torch.zeros_like(decode_bbox)), torch.ones_like(decode_bbox))
        return decode_bbox

    def decode_box(self, predictions, anchors, image_shape, input_shape, letterbox_image, variances = [0.1, 0.2], nms_iou = 0.3, confidence = 0.5):
        all_boxes = [[[] for _ in range(1)] for _ in range(21)]
        target = []
        points = []


        for i in range(1):
            for j in range(1, predictions.size(1)):
                dets = predictions[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                # print(w)
                # print(h)
                boxes[:, 0] *= image_shape[1]
                boxes[:, 2] *= image_shape[1]
                boxes[:, 1] *= image_shape[0]
                boxes[:, 3] *= image_shape[0]
                scores = dets[:, 0].cpu().numpy()
                points.append(scores)
                cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32,copy=False)
                all_boxes[j][i] = cls_dets
                for i in range(len(cls_dets)):
                    target.append(j)
            #     if cls_dets != []:
            #         print(j)
            #         print(cls_dets)
            # input()
            print(points)
            bbs = []
            final_bbs = []
            for i in range(len(all_boxes)):
                for j in range(len(all_boxes[0])):
                    if all_boxes[i][j] != []:
                        if len(all_boxes[i][j][0]):
                            for data in all_boxes[i][j]:
                                bbs.append(data)
                        else:
                            bbs.append(all_boxes[i][j])
            #print(bbs)
            target = [i-1 for i in target]
            target = np.array(target)
            target = target.astype(np.float32)
            print(target)
            # for j in range(len(all_boxes)):
            #     for i in range(len(all_boxes[0])):
            #         if all_boxes[j][i] != []:
            #             print(j)
            if points:
                for point in points:
                    for p in point:
                        for bounding_box in bbs:
                            if bounding_box[-1] == p:
                                final_bbs.append(bounding_box)
            #print(final_bbs)
            for i in range(len(target)):
                final_bbs[i] = np.insert(final_bbs[i],4,target[i])
            results = np.ones((len(target), 6), dtype=np.float32)
            
            for i in range(len(final_bbs)):
                results[i,:] = final_bbs[i]
        #print([results]) 

        return [results]
