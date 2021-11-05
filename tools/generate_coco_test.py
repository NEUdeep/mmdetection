import os
import PIL.Image as Image
import json

root = '/root/public/Datasets/2021-industry-quality-inspection-competition'

test_json = os.path.join(root, 'test')  # test image root
out_file = os.path.join('./test.json')  # test json output path 生成json的位置

data = {}
# 这部分如果不同数据集可以替换
# data['categories'] = [{"id": 1, "name": "Pedestrain", "supercategory": "none"},
#                       {"id": 2, "name": "People", "supercategory": "none"},
#                       {"id": 3, "name": "Bicycle", "supercategory": "none"},
#                       {"id": 4, "name": "Car", "supercategory": "none"},
#                       {"id": 5, "name": "Van", "supercategory": "none"},
#                       {"id": 6, "name": "Truck", "supercategory": "none"},
#                       {"id": 7, "name": "Tricycle", "supercategory": "none"},
#                       {"id": 8, "name": "Awning-tricycle", "supercategory": "none"},
#                       {"id": 9, "name": "Bus", "supercategory": "none"},
#                       {"id": 10, "name": "Motor", "supercategory": "none"}]  # 数据集的类别

# for 缺陷检测比赛
data['categories'] = [{"id": 1, "name": "lk", "supercategory": "none"},
                      {"id": 2, "name": "sy", "supercategory": "none"},
                      {"id": 3, "name": "gy", "supercategory": "none"}]  # 数据集的类别






images = []
for name in os.listdir(test_json):
    file_path = os.path.join(test_json, name)
    file = Image.open(file_path)
    tmp = dict()
    tmp['id'] = name[:-4]

    # idx += 1
    tmp['width'] = file.size[0]
    tmp['height'] = file.size[1]
    tmp['file_name'] = name
    images.append(tmp)

data['images'] = images
with open(out_file, 'w') as f:
    json.dump(data, f)

# with open(out_file, 'r') as f:
#     test = json.load(f)
#     for i in test['categories']:
#         print(i['id'])
print('finish')