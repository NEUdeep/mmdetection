
cd /workspace/mnt/storage/kanghaidong/new_video_project/mmdetection/
sh install.sh
cd /workspace/mnt/storage/kanghaidong/new_video_project/mmdetection/
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"

# ./tools/dist_train.sh configs/swin/mask_rcnn_swin_small_patch4_window7_mstrain_480-800_adamw_3x_coco.py 2 --cfg-options model.pretrained=/workspace/mnt/storage/kanghaidong/cloud_project/basic_files/swin_small_patch4_window7_224.pth
# ./tools/dist_train.sh configs/swin/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 --cfg-options model.pretrained=/workspace/mnt/storage/kanghaidong/cloud_project/basic_files/swin_small_patch4_window7_224.pth
# ./tools/dist_train.sh configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py 1
# ./tools/dist_train.sh configs/detectors/detectors_htc_r101_20e_coco.py 1
./tools/dist_train.sh configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py 8 #--resume-from work_dirs/detectors_cascade_rcnn_r50_1x_coco/epoch_12.pth 
# ./tools/dist_train.sh configs/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco.py 8
# ./tools/dist_train.sh configs/detectors/cascade_rcnn_r50_sac_1x_coco.py 8