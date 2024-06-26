import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import time

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from .. import builder
from .mvx_two_stage import MVXTwoStageDetector
from mmdet3d.ops import Voxelization

@DETECTORS.register_module()
class SparseFusionDetector(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self, **kwargs):
        super(SparseFusionDetector, self).__init__(**kwargs)
        

        self.freeze_img = kwargs.get('freeze_img', True)
        self.freeze_img_head = kwargs.get('freeze_img_head', False)

        self.init_weights(pretrained=kwargs.get('pretrained', None))
        
        # mk: freeze backbone
        if True: # when wh training 
            self._freeze_backbone()


    def _freeze_backbone(self):
        
        for modules in [self.img_backbone, self.img_neck, self.pts_backbone, \
                        self.pts_middle_encoder, self.pts_neck]: 
            if modules is not None:
                modules.eval()
                for param in modules.parameters():
                    param.requires_grad = False
                    

        for modules in [self.pts_bbox_head]:
            if modules is not None:
                for name, param in modules.named_parameters():
                    if "center_loss" in name or "feature_aligner" in name:
                        param.requires_grad = True # True if not od-wh-finetune (wh training: True)
                    else:
                        param.requires_grad = (
                            False  # True if od-wh-finetune (wh training: False)
                        )


    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super(SparseFusionDetector, self).init_weights(pretrained)

        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
            if self.freeze_img_head:
                for param in self.pts_bbox_head.img_transformer.parameters():
                    param.requires_grad = False
                for param in self.pts_bbox_head.shared_conv_img.parameters():
                    param.requires_grad = False
                for param in self.pts_bbox_head.img_heatmap_head.parameters():
                    param.requires_grad = False

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)

            img_feats = self.img_backbone(img.float())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_voxel_heights(self, voxels, coors):
        batch_size = coors[-1, 0].item() + 1
        grid_size = self.test_cfg['pts']['grid_size']
        out_size_factor = self.test_cfg['pts']['out_size_factor']

        height_num = grid_size[2]
        x_num = grid_size[0] // out_size_factor
        y_num = grid_size[1] // out_size_factor

        voxels_ = voxels[:, :, 2].clone()
        voxels_[voxels_==0] = 100
        min_voxel = torch.min(voxels_, dim=-1)[0]
        voxels_[voxels_==100] = -200
        max_voxel = torch.max(voxels_, dim=-1)[0]

        min_voxel_height = torch.zeros((batch_size, y_num, x_num, out_size_factor*out_size_factor)).to(voxels.device) + 100
        max_voxel_height = torch.zeros((batch_size, y_num, x_num, out_size_factor*out_size_factor)).to(voxels.device) - 200

        batch_ids = coors[:, 0].long()
        height_ids = coors[:, 1].long()
        y_ids = (coors[:, 2] // out_size_factor).long()
        x_ids = (coors[:, 3] // out_size_factor).long()
        y_offsets = (coors[:, 2] % out_size_factor).long()
        x_offsets = (coors[:, 3] % out_size_factor).long()

        for hid in range(height_num):
            height_mask = height_ids == hid
            batch_mask = batch_ids[height_mask]
            y_ids_mask = y_ids[height_mask]
            x_ids_mask = x_ids[height_mask]
            y_offsets_mask = y_offsets[height_mask]
            x_offsets_mask = x_offsets[height_mask]

            min_voxel_height[batch_mask, y_ids_mask, x_ids_mask, y_offsets_mask * out_size_factor + x_offsets_mask] = torch.minimum(min_voxel_height[batch_mask, y_ids_mask, x_ids_mask, y_offsets_mask * out_size_factor + x_offsets_mask], min_voxel[height_mask])
            max_voxel_height[batch_mask, y_ids_mask, x_ids_mask, y_offsets_mask * out_size_factor + x_offsets_mask] = torch.maximum(max_voxel_height[batch_mask, y_ids_mask, x_ids_mask, y_offsets_mask * out_size_factor + x_offsets_mask], max_voxel[height_mask])

        min_voxel_height = torch.min(min_voxel_height, dim=-1)[0]
        max_voxel_height = torch.max(max_voxel_height, dim=-1)[0]

        return min_voxel_height, max_voxel_height

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors, min_voxel_height, max_voxel_height = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)

        min_voxel_height = min_voxel_height[:, None]
        max_voxel_height = max_voxel_height[:, None]

        x[0] = torch.cat([x[0], min_voxel_height, max_voxel_height], dim=1)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)

        min_voxel_height, max_voxel_height = self.extract_voxel_heights(voxels, coors_batch)

        return voxels, num_points, coors_batch, min_voxel_height, max_voxel_height

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_pts_centers_view=None,
                      gt_img_centers_view=None,
                      gt_bboxes_cam_view=None,
                      img=None,
                      sparse_depth=None,
                      gt_visible_3d=None,
                      gt_bboxes_lidar_view=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(
                pts_feats, img_feats, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_pts_centers_view, gt_img_centers_view, gt_bboxes_cam_view, img_metas, gt_bboxes_ignore, sparse_depth, gt_visible_3d, gt_bboxes_lidar_view
            )
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          gt_pts_centers_view,
                          gt_img_centers_view,
                          gt_bboxes_cam_view,
                          img_metas,
                          gt_bboxes_ignore=None,
                          sparse_depth=None,
                          gt_visible_3d=None,
                          gt_bboxes_lidar_view=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs, pts_query_feat, img_query_feat = self.pts_bbox_head(pts_feats, img_feats, img_metas, sparse_depth)

        # print (len(pts_query_feat))
        # print (len(img_query_feat))
        # print (f"detector/ pts_query_feat: {pts_query_feat[0].shape} {pts_query_feat[0][0][0][:5]}")
        # print (f"detector/ img_query_feat: {img_query_feat[0].shape} {img_query_feat[0][0][0][:5]}")
        # # print (outs)
        # print ("detector/", outs[0][0].keys())
        
        
        
        # original sparsefusion code
        # loss_inputs = [gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_pts_centers_view, gt_img_centers_view, gt_bboxes_cam_view, gt_visible_3d, gt_bboxes_lidar_view, img_metas, outs]
        # losses = self.pts_bbox_head.loss(*loss_inputs)
       
       
        # 2D image label: gt_labels[0][:,0] (shape: torch.Size([22]))
        # 3D labels:gt_labels_3d[0] (shape: torch.Size([19]))

        loss_inputs = [gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, gt_pts_centers_view, gt_img_centers_view, gt_bboxes_cam_view, gt_visible_3d, gt_bboxes_lidar_view, img_metas, outs]
        losses = self.pts_bbox_head.auxiliary_loss(  # for decoupled training and calibration phase
            *loss_inputs,
            pts_query_feat=pts_query_feat, # not aligned
            img_query_feat=img_query_feat, # not aligned 
            gt_bboxes_ignore=gt_bboxes_ignore,
            img_pts_output_dict=None,
        )

        return losses

    def simple_test_pts(self, x, x_img, img_metas, rescale=False, sparse_depth=None):
        """Test function of point cloud branch."""

        outs, pts_query_feat, img_query_feat = self.pts_bbox_head(x, x_img, img_metas, sparse_depth)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, sparse_depth=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale, sparse_depth=sparse_depth)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox

        return bbox_list

    def forward_test(self, points, img_metas, img=None, sparse_depth=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(points)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(points), len(img_metas)))

        if num_augs == 1:
            img = [img] if img is None else img
            return self.simple_test(points[0], img_metas[0], img[0], sparse_depth[0], **kwargs)
        else: # True
            return self.aug_test(points, img_metas, img, **kwargs)