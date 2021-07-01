#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/17 15:32
# @Author  : Sillet
# @Function: 从labelImg标注转换为COCO通用标注

# -*- coding:utf-8 -*-
# !/usr/bin/env python

import argparse
import json
import io
import os
import random

import cv2
import base64
import glob
import copy
from PIL import Image, ImageDraw
from tqdm import tqdm

from pycocotools.coco import COCO
import numpy as np


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def polygons_to_box(img_shape, polygons):
    height, width = img_shape
    polygons = np.array(polygons)
    x_min, y_min = [int(i) for i in polygons.min(axis=0)]
    x_max, y_max = [int(i) for i in polygons.max(axis=0)]

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width - 1, x_max)
    y_max = min(height - 1, y_max)

    return [x_min, y_min, x_max - x_min, y_max - y_min]  # [x1,y1,w,h] 对应COCO的bbox格式


def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def mask2box(mask):
    """
    从mask反算出其边框
    mask：[h,w]  0、1组成的图片
    1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
    """
    # np.where(mask==1)
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]

    try:
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)
    except:
        return None

    # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
    # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
    # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
    return [left_top_c, left_top_r, right_bottom_c - left_top_c,
            right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式


class labelme2coco(object):
    def __init__(self, labelme_json=None, save_json_path='./train.json'):
        """
        :param labelme_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        """
        if labelme_json is None:
            labelme_json = []
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.catDict = None

    def get_label_name(self, label):
        for key, value in self.catDict.items():
            if label in key:
                return value
        return 'unknown'

    def creat_categorie_list(self):
        """
        创建self.categories， self.label 确保label id 稳定
        :return:
        """
        if self.catDict is None:
            return

    def data_transfer(self):
        for num, json_file in enumerate(tqdm(self.labelme_json)):
            with open(json_file, 'r') as fp:
                data = json.load(fp)  # 加载json文件
                self.images.append(self.image(data, num))
                for shapes in data['shapes']:
                    label = shapes['label'] if self.catDict is None else self.get_label_name(shapes['label'])
                    if label == "unknown":
                        continue
                    if label not in self.label:
                        self.categories.append(self.categorie(label))
                        self.label.append(label)
                    points = shapes['points']  # 这里的point是用rectangle标注得到的，只有两个点，需要转成四个点
                    if len(points) == 2:
                        points.append([points[0][0], points[1][1]])
                        points.append([points[1][0], points[0][1]])
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1

    def image(self, data, num):
        image_infor = {}
        if data['imageData'] is None:
            height, width = data['imageHeight'], data['imageWidth']
        else:
            img = img_b64_to_arr(data['imageData'])  # 解析原图片数据
            # img=io.imread(data['imagePath']) # 通过图片路径打开图片
            # img = cv2.imread(data['imagePath'], 0)
            height, width = img.shape[:2]

        img = None
        image_infor['height'] = height
        image_infor['width'] = width
        image_infor['id'] = num + 1
        image_infor['file_name'] = data['imagePath'].split('/')[-1]

        self.height = height
        self.width = width

        return image_infor

    def categorie(self, label):
        categorie = {
            'supercategory': 'component',
            'id': len(self.label) + 1,
            'name': label}

        return categorie

    def annotation(self, points, label, num):
        annotation = dict(
            segmentation=[list(np.asarray(points).flatten())],
            iscrowd=0,
            image_id=num + 1,
            bbox=list(map(float, self.getbbox(points))))
        # annotation['bbox'] = str(self.getbbox(points)) # 使用list保存json文件时报错（不知道为什么）
        # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 使用该方式转成list
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        annotation['category_id'] = self.getcatid(label)
        annotation['id'] = self.annID
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return 1

    def getbbox(self, points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points

        box = polygons_to_box([self.height, self.width], polygons)

        # mask = polygons_to_mask([self.height, self.width], polygons)
        # box = mask2box(mask)

        return box

    def data2coco(self):
        data_coco = {
            'images': self.images,
            'categories': self.categories,
            'annotations': self.annotations
        }

        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=None,
                  cls=MyEncoder)  # , cls=MyEncoder indent=4 更加美观显示


def img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(Image.open(f))
    return img_arr


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


def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """

    palette = np.array([3 ** 25 - 1, 3 ** 15 - 1, 3 ** 11 - 1])
    colors = np.array(labels).reshape(-1, 1) * palette
    colors = (colors % 255).astype("uint8")
    return colors


def show_mask_label(img, points, labels):
    colors = compute_colors_for_labels(labels).tolist()
    mask_image = copy.deepcopy(img)
    for point, color in zip(points, colors):
        point = point[:, np.newaxis, :]
        mask_image = cv2.drawContours(mask_image, [point.astype("int32")], -1, color, cv2.FILLED)

    alpha = 0.7
    beta = 1 - alpha
    gamma = 0

    composite = cv2.addWeighted(img, alpha, mask_image, beta, gamma)

    return composite


def _main():
    args = parser()
    labelme_jsons = glob.glob(args.labelme_label_path + '/*.json')
    random.shuffle(labelme_jsons)
    split_pos = int(len(labelme_jsons) * 0.9)

    print('transfer to train data:')
    labelme2coco(labelme_jsons[:split_pos], os.path.join(args.coco_label_path, 'train.json')).save_json()

    print('transfer to train test-env:')
    labelme2coco(labelme_jsons[split_pos + 1:], os.path.join(args.coco_label_path, 'val.json')).save_json()


def test_coco():
    args = parser()
    json_file = args.coco_label_path
    image_root = '/mnt/ssd_disk/test_image'
    image_save = '/home/gaoxiang/workspace/ret_img'

    coco = COCO(json_file)

    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()

    imgInfs = coco.loadImgs(imgIds)

    for imgInf in tqdm(imgInfs):
        img_path = '%s/%s' % (image_root, imgInf['file_name'])
        I = cv2.imread(img_path)
        # I = Image.open(img_path)
        annIds = coco.getAnnIds(imgIds=imgInf['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        points = []
        labels = []
        for ann in anns:
            # bbox = ann['bboxs']
            # cv2.rectangle(I,bbox[])
            # if ann['category_id'] != 3:
            #     continue
            labels.append(ann['category_id'])
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                points.append(poly)
        rle_img = show_mask_label(I, points, labels)

        cv2.imwrite(os.path.join(image_save, imgInf['file_name']), rle_img)

    return


if __name__ == '__main__':
    _main()
    # test_coco()
    exit()
