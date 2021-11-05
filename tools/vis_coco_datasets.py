import json
import os

import cv2

# json文件路径
path = "/root/neu-lab/train.json"
file = open(path, "r", encoding='utf-8')
fileJson = json.load(file)
field = fileJson["annotations"]
images = fileJson["images"]
# 图片路径
ori_pic = '/root/public/Datasets/2021-industry-quality-inspection-competition/train'
# 图片保存路径
save_path = '/root/neu-lab/mmali/mmdetection/save'

name = []
result = {}
for line in field:
    # 获取图片名字
    # img_name = str(line['image_id']).strip() + '.jpeg'
    id = line['image_id']
    for i in range(len(images)):
        img_id = images[i]["id"]
        if img_id == id:
            img_name = images[i]["file_name"]
    # 获取目标框信息
    bbox = line['bbox']

    if img_name not in name:
        name.append(img_name)
        result[img_name] = [bbox]
    else:
        result[img_name].append(bbox)

for img_name in result.keys():
    img = cv2.imread(os.path.join(ori_pic, img_name))
    print(img_name)
    for bouding_box in result[img_name]:
        print(bouding_box)
        x1 = int(bouding_box[0])
        y1 = int(bouding_box[1])
        x2 = int(bouding_box[0]) + int(bouding_box[2])
        y2 = int(bouding_box[1]) + int(bouding_box[3])
        img2 = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255))
    cv2.imwrite(os.path.join(save_path, img_name), img2)