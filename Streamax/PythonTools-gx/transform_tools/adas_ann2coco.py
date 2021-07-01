#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/24 16:03
# @Author  : Sillet
# @Function: adas数据转换为coco数据格式

import argparse
import glob
import os
import random

from labelImgToCOCO import labelme2coco

# ADAS Object categories
AdasCatName = {
    'roads': 'roads',
    'ground_mark': 'ground_mark',
    'zebra_crossing_groups': 'zebra-crs',

    'vehicle': 'vehicle',
    'car': 'vehicle',
    'bus': 'vehicle',
    'truck': 'vehicle',

    'non-motor': 'non-motor',
    'motorcycle': 'non-motor',
    'bicycle': 'non-motor',

    'person': 'person',

    'sign': 'sign',
    'stop sign': 'sign'
}

adasCatId = [
    'roads',
    'ground_mark',
    'zebra-crs',
    'vehicle',
    'non-motor',
    'person',
    'sign'
]


class AdasAnn2COCO(labelme2coco):
    def __init__(self, labelme_json=None, save_json_path='./train.json'):
        super(AdasAnn2COCO, self).__init__(labelme_json, save_json_path)
        self.catDict = AdasCatName
        self.creat_categorie_list()

    def creat_categorie_list(self):
        for val in adasCatId:
            if val not in self.label:
                self.categories.append(self.categorie(val))
                self.label.append(val)


def parser():
    args = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    args.add_argument(
        "labelme_label_path",
        help="path to labelme labels folder",
        type=str,
    )
    args.add_argument(
        "coco_label_path",
        help="save path to labelme labels file-tools",
        type=str,
    )

    return args.parse_args()


def _main():
    args = parser()
    labelme_jsons = glob.glob(args.labelme_label_path + '/*.json')
    random.shuffle(labelme_jsons)
    split_pos = int(len(labelme_jsons) * 0.9)

    print('transfer to train data:')
    AdasAnn2COCO(labelme_jsons[:split_pos], os.path.join(args.coco_label_path, 'train.json')).save_json()

    print('transfer to train test-env:')
    AdasAnn2COCO(labelme_jsons[split_pos + 1:], os.path.join(args.coco_label_path, 'val.json')).save_json()


if __name__ == '__main__':
    _main()
    exit()
