#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/21 11:39
# @Author  : Sillet
# @Function: 选取斑马线区域

import os
import cv2
import logging
import xml.etree.ElementTree as ET
# from ML import mkdir
from tqdm import tqdm

JPG = '.jpg'
XML = '.xml'
logging.basicConfig(filename='../log.txt')
expand_pixel = 100

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_boxes(xml_file):
    parser = ET.parse(xml_file)
    xml_root = parser.getroot()
    bboxes = []
    for obj in xml_root.findall('object'):
        if obj.find('name').text == 'zebra_crossing':
            box_root = obj.find('bndbox')
            box = [
                int(box_root.find('xmin').text),
                int(box_root.find('ymin').text),
                int(box_root.find('xmax').text),
                int(box_root.find('ymax').text)
            ]
            bboxes.append(box)

    return bboxes


def save_img(img, save_path, boxes):
    h, w, _ = img.shape
    for n, box in enumerate(boxes):
        img_path = '%s_%02d%s' % (save_path, n, JPG)
        if box[3] - box[1] > h / 8 and box[2] - box[0] > w / 3:
            x1 = box[0] - expand_pixel if (box[0] - expand_pixel) > 0 else 0
            y1 = box[1] - expand_pixel if (box[1] - expand_pixel) > 0 else 0
            x2 = box[2] + expand_pixel if (box[2] + expand_pixel) < w else w
            y2 = box[3] + expand_pixel if (box[3] + expand_pixel) < h else h

            cv2.imwrite(img_path, img[y1:y2, x1:x2, :])


def file_prcess(image_root, xml_root, dst_root):
    mkdir(dst_root)
    img_files = [(os.path.splitext(n)[0], os.path.join(r, n))
                 for r, d, f in os.walk(image_root) for n in f if JPG in n]
    # print(img_files)

    for file_basename, img_f in tqdm(img_files):
        xml_file = os.path.join(xml_root, file_basename + XML)
        if not os.path.exists(xml_file):
            logging.warning('%s%s is not exists!' % (file_basename, XML))
            continue

        img = cv2.imread(img_f)
        boxes = get_boxes(xml_file)

        save_path = os.path.join(dst_root, file_basename)
        save_img(img, save_path, boxes)


if __name__ == '__main__':
    img = cv2.imread('../test-env.jpg')
    xml_file = '../test-env.xml'

    boxes = get_boxes(xml_file)
    save_img(img, '../test', boxes)

    # image_path = 'X:\\gaoxiang\\dataset\\zebra_crossing\\2.7mmZebra_carPerson\\JPEGImages'
    # xml_path = 'X:\\gaoxiang\\dataset\\zebra_crossing\\2.7mmZebra_carPerson\\Annotations_CarPersonZebra'
    # save_path = '.\\test-env'
    # file_prcess(image_path, xml_path, save_path)
