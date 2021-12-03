## Robust-detection

` Based:`[mmedetection](./docs/MMDETECTION_README.md)

### new feature

- support swa training
- support mixup
- support albu augmentation
- and so on...


### 1.install

`pytorch=1.6.0`

`bash install.sh or pip install  according install.sh content` 


### 2.data preparation

`python generate_coco_test.py` or `python imagestococo.py`
and you will get coco formate `test.json`


### 3.generate predict json

- checkpoint
  ` you can find best/C_epoch_12.pth`
- config
  `you can find detectors_cascade_rcnn_r50_1x_coco_C.py`
- generate predict json
  `./tools/dist_test.sh detectors_cascade_rcnn_r50_1x_coco_C.py best/C_epoch_12.pth 1 --format-only --options "jsonfile_prefix=best/best-flip-soft_nms-0.1-0.001-C_openBrand_result_12.bbox"`

and you will get [best-flip-soft_nms-0.1-0.001-C_openBrand_result_12.bbox.json](./best)

- new test path, you just modified config like:

```data = dict(
 test=dict(
        type='OpenBrandDataset',
        ann_file='/root/neu-lab/mmdetection/test.json', #/root/neu-lab/mmdetection/test.json
        img_prefix=
        '/root/public/Datasets/2021-industry-quality-inspection-competition/test/',
)
```

`ann_file: your test.json path`
`img_prefix: your test image path`


### 4.json2csv

`python ./tools/coco2voc.csv.py`


### 5.tools

- visualization train data

`python tools/vis_coco_datasets.py`

- generate anchor_ratio
  `python tools/coco_anchor_ratio.py`
  you can find [anchor_ratio](./anchor_ratio/anchor_ratio.png)


### 6.train

`./tools/dist_train.sh detectors_cascade_rcnn_r50_1x_coco_C.py num_gpus`

- num_gpus
  ` number of gpus: you can set: num_gpus=1`


- new train path, you just modified config like:

```data = dict(
train=dict(
        type='OpenBrandDataset',
        ann_file='/root/neu-lab/train.json',
        img_prefix=
        '/root/public/Datasets/2021-industry-quality-inspection-competition/train/',
)
```

`ann_file: your train.json path`
`img_prefix: your train image path`


### 7.evaluation

- evaluation train datasets
  `./tools/dist_test.sh detectors_cascade_rcnn_r50_1x_coco_C.py best/C_epoch_12.pth 1 --eval bbox`