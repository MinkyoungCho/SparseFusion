sleep 4h;
bash tools/dist_train.sh configs/impfusion_nusc_voxel_LC_2d_3d_Cross_full.py 4 --work-dir work_dirs/impfusion_nusc_voxel_LC_2d_3d_Cross_cameraSE_focal_fuseProj_catImage_proj2D_ColAttnHeatmapW0.5_fuseSelf_imgHeatmap2_PointAug_ImgAug_maskrcnnCOCO_regLayer2woBN_fullset --resume-from /media/msc-auto/HDD/yichen/TransFusion/work_dirs/impfusion_nusc_voxel_LC_2d_3d_Cross_cameraSE_focal_fuseProj_catImage_proj2D_ColAttnHeatmapW0.5_fuseSelf_imgHeatmap2_PointAug_ImgAug_maskrcnnCOCO_regLayer2woBN_fullset/epoch_1.pth;