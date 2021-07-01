#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/20 16:50
# @Author  : Sillet
# @Function: 校验语义分割标注数据

import os
import cv2
import argparse
import json
import numpy as np
import pathlib
from tqdm import tqdm

CATEGORIES = [
    'roads',
    'ground_mark',
    'zebra-crs',
    'vehicle',
    'non-motor',
    'person',
    'sign'
]


class ClsColor(object):
    def __init__(self, cls_list):
        self.palette = [
            [0, 41, 223],
            [189, 43, 91],
            [135, 14, 236],
            [160, 107, 0],
            [178, 191, 0],
            [110, 180, 66],
            [31, 144, 81],
            [21, 81, 33],
            [244, 0, 249]
        ]
        self.cls_id = {c: n % len(self.palette) for n, c in enumerate(cls_list)}

    def __call__(self, label):
        return self.palette[self.cls_id[label]]

    def add_legend(self, image):
        roi_height = image.shape[0] / 6
        raw_height = roi_height / len(self.cls_id)

        raw_radio = 0.6
        font_radio = 0.027 * raw_height

        anchor_x = 20
        anchor_y = 0

        for key, val in self.cls_id.items():
            anchor_y += raw_height
            color = tuple(self.palette[val])
            ax = int(anchor_x + raw_height * raw_radio)
            ay = int(anchor_y + raw_height * raw_radio)
            cv2.rectangle(image, (int(anchor_x), int(anchor_y)), (ax, ay), color, -1)

            cv2.putText(image, key, (int(anchor_x + raw_height), int(anchor_y + raw_height / 2)),
                        cv2.FONT_HERSHEY_COMPLEX, font_radio, self.palette[val], 2)


def get_args():
    parser = argparse.ArgumentParser(description="Parsing Label View")
    parser.add_argument(
        "--image_path",
        default="",
        help="path to image file-tools",
        type=str,
    )
    parser.add_argument(
        "--label_path",
        default="",
        help="path to label file-tools",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        default="./ret",
        help="path to result file-tools",
        type=str,
    )

    return parser.parse_args()


def draw_label(img_path, json_path, dst_path, cls_color):
    image = cv2.imread(img_path)
    with open(json_path, 'r') as fp:
        json_data = json.load(fp)
    ret_img = np.zeros(image.shape, dtype=image.dtype)

    for shape in json_data['shapes']:
        if shape['shape_type'] != "polygon":
            continue

        color = cls_color(shape['label'])
        contours = [np.array(shape['points'])]
        ret_img = cv2.drawContours(ret_img, contours, -1, color, cv2.FILLED)
    cls_color.add_legend(ret_img)

    # out_win = 'ret image'
    # cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
    # cv2.imshow(out_win, ret_img)
    # cv2.resizeWindow(out_win, 1920, 1080)
    # cv2.waitKey()

    cv2.imwrite(dst_path, ret_img)


def prcess_folder(img_folder, label_folder, dst_folder):
    pathlib.Path(dst_folder).mkdir(parents=True, exist_ok=True)

    image_files = [os.path.splitext(f)[0] for f in os.listdir(img_folder) if '.jpg' in f]
    label_files = [os.path.splitext(f)[0] for f in os.listdir(label_folder) if '.json' in f]

    file_names = list(set(image_files).intersection(set(label_files)))

    cls_color = ClsColor(CATEGORIES)

    for file_name in tqdm(file_names):
        image_path = os.path.join(img_folder, file_name + '.jpg')
        label_path = os.path.join(label_folder, file_name + '.json')
        dst_path = os.path.join(dst_folder, file_name + '.jpg')

        draw_label(image_path, label_path, dst_path, cls_color)


if __name__ == '__main__':
    args = get_args()
    prcess_folder(args.image_path, args.label_path, args.save_path)
    print('done!')
