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
import numpy as np
from tqdm import tqdm
from pycocotools import mask as maskUtils


# {'supercategory': 'component', 'id': 1, 'name': 'roads'}
# {'supercategory': 'component', 'id': 2, 'name': 'ground_mark'}
# {'supercategory': 'component', 'id': 3, 'name': 'zebra-crs'}
# {'supercategory': 'component', 'id': 4, 'name': 'vehicle'}
# {'supercategory': 'component', 'id': 5, 'name': 'non-motor'}
# {'supercategory': 'component', 'id': 6, 'name': 'person'}
# {'supercategory': 'component', 'id': 7, 'name': 'sign'}

def del_stuffs_in_instances(save_file, instance_json_file):
    with open(instance_json_file, 'r') as f:
        src_d = json.load(f)
    ins_d = src_d.copy()
    ins_d['annotations'] = [i for i in ins_d['annotations'] if i['category_id'] != 1]
    ins_d['categories'] = [i for i in ins_d['categories'] if i['id'] != 1]
    for i in ins_d['categories']:
        i['id'] -= 1
    for i in ins_d['annotations']:
        i['category_id'] -= 1

    for i in ins_d['categories']:
        print(i)

    with open(save_file, 'w') as f1:
        json.dump(ins_d, f1)


def create_stuff_json_from_instances(save_file, instance_json_file):
    with open(instance_json_file, 'r') as f:
        src_d = json.load(f)
    stf_d = src_d.copy()
    del stf_d['categories']
    stf_d.setdefault('categories', [{'supercategory': 'component', 'id': 7, 'name': 'roads'}])
    stf_d['annotations'] = [i for i in stf_d['annotations'] if i['category_id'] == 1]
    for ann in stf_d['annotations']:
        ann['category_id'] = 7

    with open(save_file, 'w') as f2:
        json.dump(stf_d, f2)

    with open(save_file, 'r') as f3:
        stf_d2 = json.load(f3)
    coco = COCO(save_file)
    for ann1 in stf_d2['annotations']:
        annIds = coco.getAnnIds(imgIds=ann1['image_id'])
        rle = coco.annToRLE(coco.loadAnns(annIds)[0])
        rle['counts'] = str(rle['counts'], encoding="utf-8")
        ann1['segmentation'] = rle

    with open(save_file, 'w') as f4:
        json.dump(stf_d2, f4)


def show_rle_stuff_seg(json_path):
    with open(json_path) as anno_:
        d = json.load(anno_)
    annotations = d['annotations']

    for i in range(len(annotations)):
        annotation = annotations[i]
        segmentation = annotation['segmentation']
        segmentation['counts'] = bytes(segmentation['counts'], encoding="utf8")
        mask = maskUtils.decode(segmentation)  # 分割解码
        mask=Image.fromarray(mask.astype('uint8'))
        plt.imshow(mask)
        plt.show()

if __name__ == '__main__':
    mode = 'val'

    instance_json_file = 'D:/DATA/adas/adas_4_3/' + mode + '.json'
    img_path = 'D:/DATA/adas/adas_4_3/' + mode
    save_dir = 'D:/DATA/adas/adas_pan'

    save_stuff_json = save_dir + '/adas_stuff/adas_stuff_' + mode + '.json'
    save_instance_json = save_dir + '/adas_ins/adas_ins_' + mode + '.json'

    if not os.path.exists(str(Path(save_stuff_json).parent)):
        os.makedirs(str(Path(save_stuff_json).parent))
    if not os.path.exists(str(Path(save_instance_json).parent)):
        os.makedirs(str(Path(save_instance_json).parent))

    ################################################################
    # del_stuffs_in_instances(save_instance_json, instance_json_file)
    # create_stuff_json_from_instances(save_stuff_json, instance_json_file)
    ################################################################

    show_rle_stuff_seg(save_stuff_json)
    # show_rle_stuff_seg(img_path, 'C:/Users/Gigabyte/Downloads/Compressed/stuff_annotations_trainval2017/annotations/stuff_val2017.json')
