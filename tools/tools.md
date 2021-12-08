### tools

[统计训练集的anchor-ratio](mmdetection/tools/coco_anchor_ratio.py);

[生成coco格式的test集合](mmdetection/tools/generate_coco_test.py);

[生成coco格式的test集合](mmdetection/tools/imagestococo.py);

[可视化coco格式数据集](mmdetection/tools/vis_coco_datasets.py);

[单张图片检测](mmdetection/tools/demo.py);

[单张图片推理的profile耗时](mmdetection/tools/demo.py);

[虚拟数据forward测试](mmdetection/tools/infertime_analyze.py);

[可视化测试结果json的结果](mmdetection/tools/vis_json.py);

[测试结果json转为比赛可提交的csv](mmdetection/tools/coco2voc.csv.py)注：转成的csv带有表头，可能因为编码原因提交会报错，需要删掉表头进行提交。