import cv2
import numpy as np
import matplotlib
import torch
import copy
matplotlib.use('agg')
import matplotlib.pyplot as plt

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

batch_size = 4

img_tensor = torch.load("vis/img.pt")

img_query_pos_heatmap = torch.load('vis/img_query_pos_heatmap.pt', map_location='cpu')
img_query_label = torch.load('vis/img_query_label.pt', map_location='cpu')
img_query_cls = torch.load('vis/img_query_cls.pt', map_location='cpu')
img_query_cls = torch.max(img_query_cls, dim=1)[1]

unnormal = UnNormalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
view_num = img_tensor.shape[1]

for i in range(batch_size):
    for j in range(view_num):
        img_tensor[i, j] = unnormal(img_tensor[i,j])

img_tensor = img_tensor.permute(0, 1, 3, 4, 2)
imgs = img_tensor.detach().cpu().numpy()

for i in range(batch_size):
    for view_id in range(view_num):
        view_img = imgs[i, view_id]
        view_img = cv2.cvtColor(view_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("vis/images/img_b%d_v%d.png"%(i, view_id), view_img)

    h, w = imgs.shape[2], imgs.shape[3]

    img_query_label_sample = img_query_label[i]
    img_query_cls_sample = img_query_cls[i]

    img_query_pos_heatmap_sample = torch.sigmoid(img_query_pos_heatmap[i])

    img_query_pos_heatmap_sample[..., 0] = img_query_pos_heatmap_sample[..., 0] * w
    img_query_pos_heatmap_sample[..., 1] = img_query_pos_heatmap_sample[..., 1] * h

    query_view = torch.load('vis/img_view_%d.pt' % i, map_location='cpu')

    for view_id in range(view_num):
        view_img = copy.deepcopy(imgs[i, view_id])
        view_img = cv2.cvtColor(view_img, cv2.COLOR_RGB2BGR)

        view_mask = query_view == view_id
        img_query_pos_heatmap_view = img_query_pos_heatmap_sample[view_mask]
        img_query_label_view = img_query_label_sample[view_mask]
        img_query_cls_view = img_query_cls_sample[view_mask]

        for id in range(img_query_pos_heatmap_view.shape[0]):
            # cv2.circle(view_img, (int(img_query_pos_heatmap_view[id, 0]), int(img_query_pos_heatmap_view[id, 1])), radius=1, color=(0, 0, 255))
            cls_id = img_query_label_view[id].item()
            cv2.putText(view_img, class_names[cls_id][:2], (int(img_query_pos_heatmap_view[id, 0]), int(img_query_pos_heatmap_view[id, 1])),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=(0, 0, 255))

        for id in range(img_query_pos_heatmap_view.shape[0]):
            # cv2.circle(view_img, (int(img_query_pos_heatmap_view[id, 0]), int(img_query_pos_heatmap_view[id, 1])), radius=1, color=(0, 0, 255))
            cls_id = img_query_cls_view[id].item()
            cv2.putText(view_img, class_names[cls_id][:2], (int(img_query_pos_heatmap_view[id, 0]), int(img_query_pos_heatmap_view[id, 1])),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=(0, 255, 0))

        cv2.imwrite("vis/images/img_b%d_v%d_heatmap.png"%(i, view_id), view_img)




