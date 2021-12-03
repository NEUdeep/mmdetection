import os
import json
import pandas as pd
result_csv = '/Users/kanghaidong/Downloads/best-flip-soft_nms-0.1-0.001-C_openBrand_result_12.bbox.bbox.csv'



input = "/Users/kanghaidong/Downloads/best-flip-soft_nms-0.1-0.001-C_openBrand_result_12.bbox.bbox.json"
f = open(input, encoding='utf-8')
cotent = f.read()
data = json.loads(cotent)

print(len(data), type(data), data)

url_txt = []

"""

'sy':0
'gy':1
'lk':2

上面是train标注里面的格式；
而在生成的测试结果里面，基本上是从1开始的；故需要注意
"""

category = ['sy','gy','lk']

img=[]
label=[]
confidence=[]
boxes=[]
x1=[]
y1=[]
x2=[]
y2=[]

index = 0
for i in range(len(data)):
    img.append(data[i]["image_id"] + 'jpeg')
    label.append(category[data[i]["category_id"]-1])
    confidence.append(data[i]["score"])
    boxes = data[i]["bbox"]
    x1.append(int(boxes[0]))
    y1.append(int(boxes[1]))
    x2.append(int(boxes[0]+boxes[2]))
    y2.append(int(boxes[1]+boxes[3]))


    res_dict = {
            'img': img,
            'label': label,
            'confidence': confidence,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
        }


    df = pd.DataFrame(res_dict)
    df.to_csv(result_csv, index=False)