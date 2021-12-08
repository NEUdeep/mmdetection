import mmcv
import os
import cv2
from mmcv import Timer
import time
import numpy as np
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector


config_file = 'detectors_cascade_rcnn_r50_1x_coco_C.py'
checkpoint_file = 'best/C_epoch_12.pth'

model = init_detector(config_file,checkpoint_file)

img_dir = '/root/neu-lab/'
out_dir = 'results/'
test= 'test'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

img =img_dir + '001107.jpeg'
s = time.time()
infer_time = Timer()
result = inference_detector(model,img)
e = time.time()
average_time = infer_time.since_start()
# model.show_result(img, result)
img = model.show_result(img, result)
cv2.imwrite("{}/{}.jpg".format(out_dir, test), img)

print("Using python time is:",e-s, "s")
print("Using mmcv is:",average_time, "s")