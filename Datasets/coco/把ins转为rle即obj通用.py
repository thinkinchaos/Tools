from pycocotools.coco import COCO
import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path
import json
import pycocotools.mask as mask
import json
import os
import cv2
from tqdm import tqdm
from pycocotools import mask as maskUtils
import copy
import itertools


def poly2rle(mode='train'):
    dataset = 'D:/DATA/adas/adas_1907/'
    # img_path = dataset + mode + '2017'
    instance_json_file = dataset + 'instances_' + mode + '2017.json'
    save_json = dataset + 'adas_obj_' + mode + '.json'

    if not os.path.exists(str(Path(save_json).parent)):
        os.makedirs(str(Path(save_json).parent))

    coco = COCO(instance_json_file)
    with open(instance_json_file, 'r') as f:
        d = json.load(f)
        for ann in d['annotations']:
            rle = coco.annToRLE(coco.loadAnns(ann['id'])[0])
            rle['counts'] = str(rle['counts'], encoding="utf-8")
            ann['segmentation'] = rle
    cats = set()
    for i in d['annotations']:
        cats.add(i['category_id'])
    print(cats)

    with open(save_json, 'w') as f4:
        json.dump(d, f4)


def remove_overlap(mode='val'):
    dataset = 'D:/DATA/adas/adas_1907/'
    save_json = dataset + 'adas_obj_' + mode + '.json'

    with open(save_json, 'r') as f:
        d = json.load(f)

    dd = d.copy()
    for j in range(1, len(dd['images']) + 1):
        masks = [maskUtils.decode(i['segmentation']) for i in dd['annotations'] if i['image_id'] == j]
        pairs = list(itertools.combinations(masks, 2))
        overlap = np.zeros((1080, 1920), dtype='uint8')
        for pair in pairs:
            overlap += (pair[0] & pair[1])

        for mask in masks:
            mask *= ~(overlap.astype('bool'))

        cnt = 0
        for i in dd['annotations']:
            if i['image_id'] == j:
                rle = maskUtils.encode(masks[cnt])
                rle['counts'] = str(rle['counts'], encoding="utf-8")
                i['segmentation'] = rle
                cnt += 1

        print('Processed img',j)

    save_json_new = dataset + 'adas_obj_no_overlap_' + mode + '.json'
    with open(save_json_new, 'w') as ff:
        json.dump(dd, ff)


if __name__ == '__main__':
    remove_overlap('train')
