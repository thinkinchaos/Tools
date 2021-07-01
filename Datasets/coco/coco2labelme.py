# -*- coding: utf-8 -*-
"""
@author: Taoting
将用coco格式的json转化成labeime标注格式的json
"""

import json
import cv2
import numpy as np
import os

# label_num = {'person': 0, 'bicycle': 0, 'car': 0, 'motorcycle': 0, 'bus': 0, 'train': 0, 'truck': 0}  # 根据你的数据来修改
label_num = {}

# 用一个labelme格式的json作为参考，因为很多信息都是相同的，不需要修改。
def reference_labelme_json():
    ref_json_path = '../reference_labelme.json'
    data = json.load(open(ref_json_path))
    return data


def labelme_shapes(data, data_ref):
    shapes = []

    for ann in data['annotations']:
        shape = {}
        class_name = [i['name'] for i in data['categories'] if i['id'] == ann['category_id']]
        # label要对应每一类从_1开始编号
        label_num[class_name[0]] += 1
        shape['label'] = class_name[0] + '_' + str(label_num[class_name[0]])
        shape['line_color'] = data_ref['shapes'][0]['line_color']
        shape['fill_color'] = data_ref['shapes'][0]['fill_color']

        shape['points'] = []
        # ~ print(ann['segmentation'])
        if not type(ann['segmentation']) == list:
            continue
        else:
            x = ann['segmentation'][0][::2]  # 奇数个是x的坐标
            y = ann['segmentation'][0][1::2]  # 偶数个是y的坐标
            for j in range(len(x)):
                shape['points'].append([x[j], y[j]])

            shape['shape_type'] = data_ref['shapes'][0]['shape_type']
            # shape['flags'] = data_ref['shapes'][0]['flags']
            shapes.append(shape)
    return shapes


def Coco2labelme(json_path, data_ref):
    with open(json_path, 'r') as fp:
        data = json.load(fp)  # 加载json文件
        data_labelme = {}
        data_labelme['version'] = data_ref['version']
        data_labelme['flags'] = data_ref['flags']

        data_labelme['shapes'] = labelme_shapes(data, data_ref)

        data_labelme['lineColor'] = data_ref['lineColor']
        data_labelme['fillColor'] = data_ref['fillColor']
        data_labelme['imagePath'] = data['images'][0]['file_name']

        data_labelme['imageData'] = None
        # ~ data_labelme['imageData'] = data_ref['imageData']

        data_labelme['imageHeight'] = data['images'][0]['height']
        data_labelme['imageWidth'] = data['images'][0]['width']

        return data_labelme


if __name__ == '__main__':
    root_dir = 'D:/DATA/datasets/adas_4_3/coco2labelme_result'
    json_list = os.listdir(root_dir)
    # 参考的json
    data_ref = reference_labelme_json()

    for json_path in json_list:
        if json_path.split('.')[-1] == 'json':
            print('当前文件： ', json_path)

            with open(os.path.join(root_dir, json_path), 'r') as f:
                data = json.load(f)
            for cat in data['categories']:
                label_num.setdefault(cat['name'], cat['id'])
            # print(label_num)

            data_labelme = Coco2labelme(os.path.join(root_dir, json_path), data_ref)
            file_name = data_labelme['imagePath']
            # 保存json文件
            json.dump(data_labelme, open('%s.json' % file_name.split('.')[0], 'w'), indent=4)