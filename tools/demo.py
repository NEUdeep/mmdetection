import mmcv
import os
import numpy as np
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector


config_file = 'detectors_cascade_rcnn_r50_1x_coco_A.py'
checkpoint_file = 'work_dirs/detectors_cascade_rcnn_r50_1x_coco/epoch_12.pth'

model = init_detector(config_file,checkpoint_file)

img_dir = '/root/public/Datasets/2021-industry-quality-inspection-competition/test/'
out_dir = 'results/'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

img =img_dir + '004931.jpeg'
result = inference_detector(model,img)
model.show_result(img, result, model.CLASSES)

print(result)