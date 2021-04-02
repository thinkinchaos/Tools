#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/28 9:45
# @Author  : Sillet
# @Function: 根据voc格式list文件，拷贝数据
import json
import math
import os

import cv2

from collections import OrderedDict
from tqdm import tqdm

image_root = '/mnt/ssd_disk/streamax/adas_data/VehiclePersonV2/JPEGImages'
json_root = '/mnt/ssd_disk/streamax/adas_data/VehiclePersonV2/parsing_ann'
voc_list_file = '/mnt/ssd_disk/streamax/adas_data/VehiclePersonV2/ImageSets/Main/trainval.txt'

save_root = '/mnt/ssd_disk/cv_storage_tmp/dataset/adas_data'

batch_size = 10000


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def del_color(data):
    if 'lineColor' in data:
        del data['lineColor']
    if 'fillColor' in data:
        del data['fillColor']

    for n, shape in enumerate(data['shapes']):
        if 'line_color' in shape:
            del shape['line_color']
        if 'fill_color' in shape:
            del shape['fill_color']

        shape['shape_type'] = 'polygon'
        data['shapes'][n] = shape

    return data


def _main():
    with open(voc_list_file, 'r') as f:
        voc_list = f.readlines()

    list_len = len(voc_list)
    batch_num = int(math.ceil(list_len / batch_size))
    for n in range(batch_num):
        save_folder = os.path.join(save_root, '%d' % n)
        mkdir(save_folder)

        print('%d / %d -> prcessing %s:' % (n + 1, batch_num, save_folder))
        begin_pos = n * batch_size
        end_pos = begin_pos + batch_size if n < batch_num - 1 else None
        for line in tqdm(voc_list[begin_pos:end_pos]):
            img_name = '%s.jpg' % line[:-1]
            json_name = '%s.json' % line[:-1]

            img_path = os.path.join(image_root, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            json_path = os.path.join(json_root, json_name)
            with open(os.path.join(json_path), 'r') as fp:
                json_data = json.load(fp, object_pairs_hook=OrderedDict)
            if json_data is None:
                continue

            # 删除不必要的项目
            del_color(json_data)

            cv2.imwrite(os.path.join(save_folder, img_name), img)
            with open(os.path.join(save_folder, json_name), 'w') as fp:
                json.dump(json_data, fp)
        print('done!')


if __name__ == '__main__':
    _main()
    exit()
