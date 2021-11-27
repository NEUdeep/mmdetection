./tools/dist_test.sh ./detectors_cascade_rcnn_r50_1x_coco_B.py work_dirs/detectors_cascade_rcnn_r50_1x_coco/epoch_12.pth 2 --format-only --options "jsonfile_prefix=work_dirs/detectors_cascade_rcnn_r50_1x_coco/openBrand_result_12"

./tools/dist_test.sh ./detectors_cascade_rcnn_r50_1x_coco_A.py work_dirs/detectors_cascade_rcnn_r50_1x_coco/epoch_150.pth 1 --format-only --options "jsonfile_prefix=work_dirs/detectors_cascade_rcnn_r50_1x_coco/openBrand_result_150"

./tools/dist_test.sh ./detectors_cascade_rcnn_r50_1x_coco_A.py work_dirs/detectors_cascade_rcnn_r50_1x_coco/epoch_150.pth 1 --eval bbox #用于评估检测map; allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']

./tools/dist_test.sh ./detectors_cascade_rcnn_r50_1x_coco_B.py work_dirs/detectors_cascade_rcnn_r50_1x_coco_B12/epoch_24.pth 1 --format-only --options "jsonfile_prefix=work_dirs/detectors_cascade_rcnn_r50_1x_coco_B12/B12_openBrand_result_24"

./tools/dist_test.sh configs/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py work_dirs/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/epoch_29.pth 1 --format-only --options "jsonfile_prefix=work_dirs/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparsercnn_openBrand_result_29"

./tools/dist_test.sh cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py work_dirs/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco/epoch_12.pth 1 --format-only --options "jsonfile_prefix=work_dirs/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_openBrand_result_12"

./tools/dist_test.sh best/detectors_cascade_rcnn_r50_1x_coco_C.py best/C_epoch_12.pth 1 --format-only --options "jsonfile_prefix=best/0.1-C_openBrand_result_12"