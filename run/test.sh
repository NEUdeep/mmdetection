./tools/dist_test.sh ./detectors_cascade_rcnn_r50_1x_coco_B.py work_dirs/detectors_cascade_rcnn_r50_1x_coco/epoch_12.pth 2 --format-only --options "jsonfile_prefix=work_dirs/detectors_cascade_rcnn_r50_1x_coco/openBrand_result_12"

./tools/dist_test.sh ./detectors_cascade_rcnn_r50_1x_coco_A.py work_dirs/detectors_cascade_rcnn_r50_1x_coco/epoch_150.pth 1 --format-only --options "jsonfile_prefix=work_dirs/detectors_cascade_rcnn_r50_1x_coco/openBrand_result_150"

./tools/dist_test.sh ./detectors_cascade_rcnn_r50_1x_coco_A.py work_dirs/detectors_cascade_rcnn_r50_1x_coco/epoch_150.pth 1 --eval bbox #用于评估检测map; allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']