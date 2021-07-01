import copy
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
import itertools
import numpy as np
import json
import shutil
import os


def modify_ins_json_of_cat_and_segtype(root, mode, rle_flag):
    root = root.replace('adas_parsing_data_1907', 'panoptic')
    file = root + '/instances_' + mode + '2017.json'
    with open(file, 'r') as f:
        d = json.load(f)

    global coco
    if rle_flag == 0:
        coco = COCO(file)

    new_d_img_ids = set()
    new_d_ann = []
    for it in d['annotations']:
        if it['category_id'] == 1 or it['category_id'] == 3:
            if rle_flag == 0:
                rle = coco.annToRLE(coco.loadAnns(it['id'])[0])
                rle['counts'] = str(rle['counts'], encoding="utf-8")
                it['segmentation'] = rle
            else:
                pass

            new_d_ann.append(it)
            new_d_img_ids.add(it['image_id'])

    new_d_img = []
    for it in d['images']:
        if it['id'] in new_d_img_ids:
            new_d_img.append(it)

    new_d_cat = [{"id": 1, "name": "zebra-crs", "supercategory": "adas"}, {"id": 2, "name": "roads", "supercategory": "adas"}]
    # new_d_cat = [{"id": 1, "name": "zhatu", "supercategory": "adas"}]

    new_d = copy.deepcopy(d)
    new_d["images"] = new_d_img
    new_d["categories"] = new_d_cat
    new_d["annotations"] = new_d_ann

    for it in new_d['annotations']:
        if it['category_id'] == 3:
            it['category_id'] = 1
        elif it['category_id'] == 1:
            it['category_id'] = 2

    global save_file
    if rle_flag == 1:
        save_file = root + '/instances_' + mode + '2017.json'
    elif rle_flag == 0:
        save_file = root + '/objects_' + mode + '2017.json'
    with open(save_file, 'w') as f:
        json.dump(new_d, f)


def remove_overlaps_in_obj(root, mode):
    root = root.replace('adas_parsing_data_1907', 'panoptic')
    file = root + '/objects_' + mode + '2017.json'

    with open(file, 'r') as f:
        d = json.load(f)

    img_ids = set()
    for it in d['annotations']:
        img_ids.add(it['image_id'])

    for img_id in img_ids:
        masks = [maskUtils.decode(i['segmentation']) for i in d['annotations'] if i['image_id'] == img_id]
        pairs = list(itertools.combinations(masks, 2))
        overlap = np.zeros((1080, 1920), dtype='uint8')
        for pair in pairs:
            overlap += (pair[0] & pair[1])

        for mask in masks:
            mask *= ~(overlap.astype('bool'))

        cnt = 0
        for i in d['annotations']:
            if i['image_id'] == img_id:
                rle = maskUtils.encode(masks[cnt])
                rle['counts'] = str(rle['counts'], encoding="utf-8")
                i['segmentation'] = rle
                cnt += 1

        # print('Processed img', img_id)

    save_json_new = root + '/objects_no_overlap_' + mode + '2017.json'
    with open(save_json_new, 'w') as ff:
        json.dump(d, ff)


def remove_xml_and_create_ins(root, mode):
    file = root + '/coco_v1910_' + mode + '.json'

    with open(file, 'r') as f:
        d = json.load(f)

    del_ids = []
    for it in d['images']:
        if 'xml' in it['file_name']:
            del_ids.append(it['id'])
    d['images'] = [i for i in d['images'] if i['id'] not in del_ids]

    d['annotations'] = [i for i in d['annotations'] if i['image_id'] not in del_ids]

    save_file = root.replace('adas_parsing_data_1907', 'panoptic')+ '/instances_'+mode+'2017.json'
    with open(save_file, 'w') as f:
        json.dump(d, f)


def divide_images(root, mode):
    file = root + '/coco_v1910_' + mode + '.json'
    print(file)
    with open(file, 'r') as f:
        d = json.load(f)

    save_dir = root.replace('adas_parsing_data_1907', 'panoptic')+ '/'+mode+'2017'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for it in d['images']:
        shutil.copy(root + '/image/' + it['file_name'], save_dir)

def create_pan_cat_json(root):
    root = root.replace('adas_parsing_data_1907', 'panoptic')
    cat = [{"supercategory": "adas", "color": [224, 0, 249], "isthing": 1, "id": 1, "name": "zebra-crs"},
           {"supercategory": "adas", "color": [0, 41, 223], "isthing": 0, "id": 2, "name": "roads"}]
    with open(root + '/panoptic_coco_categories.json', 'w') as f:
        json.dump(cat, f)

if __name__ == '__main__':
    modes = ['train', 'val']
    root = 'D:/DATA/adas/adas1907/adas_parsing_data_1907'
    for mode in modes:
        # divide_images(root, mode)
        # remove_xml_and_create_ins(root, mode)
        # modify_ins_json_of_cat_and_segtype(root, mode, rle_flag=0)
        # modify_ins_json_of_cat_and_segtype(root, mode, rle_flag=1)
        # remove_overlaps_in_obj(root, mode)
        # create_pan_cat_json(root)
        pass

# python converters/detection2panoptic_coco_format.py --input_json_file mydata/objects_no_overlap_train2017.json --output_json_file mydata/panoptic_train2017.json --categories_json_file mydata/panoptic_coco_categories.json

# python converters/detection2panoptic_coco_format.py --input_json_file mydata/objects_no_overlap_val2017.json --output_json_file mydata/panoptic_val2017.json --categories_json_file mydata/panoptic_coco_categories.json

# python converters/panoptic2semantic_segmentation.py --input_json_file mydata/panoptic_val2017_stff.json --segmentations_folder mydata/panoptic_val2017 --semantic_seg_folder mydata/panoptic_val2017_semantic_trainid_stff --categories_json_file mydata/panoptic_coco_categories.json

# python converters/panoptic2semantic_segmentation.py --input_json_file mydata/panoptic_train2017_stff.json --segmentations_folder mydata/panoptic_train2017 --semantic_seg_folder mydata/panoptic_train2017_semantic_trainid_stff --categories_json_file mydata/panoptic_coco_categories.json
