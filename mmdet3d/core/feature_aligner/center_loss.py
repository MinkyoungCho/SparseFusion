import torch
import torch.nn as nn


class CrossModalCenterLoss(nn.Module):
    """Center loss.
    This class is basically derived from https://github.com/LongLong-Jing/Cross-Modal-Center-Loss/tree/main.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=128, use_gpu=True, mode=None):
        super(CrossModalCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
            # self.centers1 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
            # self.centers2 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
            # self.centers3 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
            # self.centers4 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
            # self.centers5 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
            # self.centers1 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
            # self.centers2 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
            # self.centers3 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
            # self.centers4 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
            # self.centers5 = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

        assert mode in ["train", "test"]
        # if mode == "test":
        #     self.load_model()

    def forward(self, gt_labels_2d_list, gt_labels_3d_list, img_feats, pts_feats): # For SparseFusion (=/=  FUTR3D's center_loss)
        """
        Args:
            X_feats: feature matrix with shape (num_queries, feat_dim).
            labels: ground truth labels with shape (num_queries).
        """
        # print("[Current Loaded Model] center", self.centers)

        num_queries = img_feats.size(1)

        # img_feats: [900, 128] --> [22, 128]
        # pts_feats: [900, 128] --> [19, 128]
        # selc.centers: [10, 128] --> [22, 128] or [19, 128]
        img_feats = img_feats.squeeze() # SparseFusion: [200, 128]
        pts_feats = pts_feats.squeeze()
                
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
            
        self_centers = self.centers # torch.Size([10, 128])
        
        
        mask_2d = gt_labels_2d_list != 10 # V
        img_feats_masked = img_feats[mask_2d[0]] # [22, 128] V

        masked_labels_2d = gt_labels_2d_list[0][mask_2d[0]] # V
        centers_masked_2d = self_centers[masked_labels_2d] # [22, 128] V
        
        gt_labels_3d_list = gt_labels_3d_list[0][:200].unsqueeze(0) # V
        mask_3d = gt_labels_3d_list != 10 # V
        pts_feats_masked = pts_feats[mask_3d[0]] # goal: [19, 128] V

        masked_labels_3d = gt_labels_3d_list[0][mask_3d[0]] # V
        centers_masked_3d = self_centers[masked_labels_3d] # V
        
        
        ######
        
    
        loss_center = self._calculate_center_loss(
            img_feats_masked, pts_feats_masked, centers_masked_2d, centers_masked_3d, num_queries=200
        )

        # loss_geomed
        loss_geomed = self._calculate_geomed_loss(
            img_feats_masked, pts_feats_masked, centers_masked_2d, centers_masked_3d, num_queries
        )
        
        loss_sep = self._calculate_separate_loss(self_centers, num_queries)

        return loss_center, loss_geomed, loss_sep

    def _calculate_center_loss(
        self, img_feats_masked, pts_feats_masked, centers_masked_2d, centers_masked_3d, num_queries=200
    ):
        # Compute the L2 norms
        norms_img = torch.norm(img_feats_masked - centers_masked_2d, p=2, dim=1)  # [22]
        norms_pts = torch.norm(pts_feats_masked - centers_masked_3d, p=2, dim=1)  # [19]

        # original code
        # # Calculate the sum of norms for each query
        # loss_center_perquery = norms_img + norms_pts
        # # Sum over queries
        # loss_center = loss_center_perquery.sum()
        
        # modified code
        loss_center = norms_img.sum() + norms_pts.sum()
        # num_queries = (img_feats_masked.shape[0] + pts_feats_masked.shape[0]) / 2
        
        return loss_center / num_queries * 10

    def _calculate_geomed_loss(
        self, img_feats_masked, pts_feats_masked, centers_masked_2d, centers_masked_3d, num_queries=200
    ):
        numerator_img = img_feats_masked - centers_masked_2d  # [22, 128]
        numerator_pts = pts_feats_masked - centers_masked_3d  # [19, 128]

        denominator_img = torch.norm(numerator_img, p=2, dim=1, keepdim=True)  # [22, 1]
        denominator_pts = torch.norm(numerator_pts, p=2, dim=1, keepdim=True)    # [19, 1]

        normalized_img = numerator_img / denominator_img  # [19, 128]
        normalized_pts = numerator_pts / denominator_pts  # [22, 128]

        # original code
        # combined_terms = normalized_img + normalized_pts  # [33, 128]
        # sum_combined_terms = combined_terms.sum(dim=0)  # [128]
        
        
        # modified code
        sum_combined_terms = normalized_img.sum(dim=0) + normalized_pts.sum(dim=0) # [128]

        loss_geomed = torch.norm(sum_combined_terms, p=2, dim=0) ** 2  # []: constant
        
        # num_queries = (img_feats_masked.shape[0] + pts_feats_masked.shape[0]) / 2
        return loss_geomed / num_queries * 3
    
    def _calculate_separate_loss(self, self_centers, num_queries):
        lambda_val = 0.1
        num_classes = self_centers.size(0)
        C1 = self_centers.unsqueeze(1).expand(num_classes, num_classes, 128)
        C2 = self_centers.unsqueeze(0).expand(num_classes, num_classes, 128)
        loss_sep = -lambda_val * (C1 - C2).pow(2).sum(dim=2).triu(1).sum()
        return loss_sep / num_queries / 7
