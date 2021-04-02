import json
from pycocotools.coco import COCO
import os
import cv2
import numpy as np
import shutil
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path


def pan(in_d):
    for t in in_d['categories']:
        if t['name'] == 'roads':
            t.setdefault('isthing', 0)
        else:
            t.setdefault('isthing', 1)

    annotations = []
    for tmp in in_d['images']:
        annotations_tmp = {}
        segments_info = []
        for tmp2 in in_d['annotations']:
            if tmp2['image_id'] == tmp['id']:
                tmp3 = {}
                tmp3.setdefault('id', tmp2['id'])
                tmp3.setdefault('category_id', tmp2['category_id'])
                tmp3.setdefault('iscrowd', tmp2['iscrowd'])
                tmp3.setdefault('bbox', tmp2['bbox'])
                tmp3.setdefault('area', tmp2['area'])
                segments_info.append(tmp3)
        annotations_tmp.setdefault('file_name', tmp['file_name'][:-3] + 'png')
        annotations_tmp.setdefault('image_id', tmp['id'])
        annotations_tmp.setdefault('segments_info', segments_info)
        annotations.append(annotations_tmp)

    del in_d['annotations']
    in_d.setdefault('annotations', annotations)

    return in_d


def pan_stff(in_d):
    for t in in_d['categories']:
        if t['name'] == 'roads':
            t.setdefault('isthing', 0)
        else:
            t.setdefault('isthing', 1)

        if t['name'] == 'roads':
            t.setdefault('color', [0, 41, 223])
        elif t['name'] == 'ground_mark':
            t.setdefault('color', [135, 14, 236])
        elif t['name'] == 'zebra-crs':
            t.setdefault('color', [224, 0, 249])
        elif t['name'] == 'vehicle':
            t.setdefault('color', [187, 43, 91])
        elif t['name'] == 'non-motor':
            t.setdefault('color', [160, 107, 0])
        elif t['name'] == 'person':
            t.setdefault('color', [178, 191, 0])
        elif t['name'] == 'sign':
            t.setdefault('color', [110, 180, 66])

    for t in in_d['images']:
        t['file_name'] = t['file_name'][:-3] + 'png'

    annotations = []
    for tmp in in_d['images']:
        annotations_tmp = {}
        segments_info = []
        for tmp2 in in_d['annotations']:
            if tmp2['image_id'] == tmp['id']:
                tmp3 = {}
                tmp3.setdefault('id', tmp2['id'])
                tmp3.setdefault('category_id', tmp2['category_id'])
                tmp3.setdefault('iscrowd', tmp2['iscrowd'])
                tmp3.setdefault('bbox', tmp2['bbox'])
                tmp3.setdefault('area', tmp2['area'])
                segments_info.append(tmp3)
        annotations_tmp.setdefault('file_name', tmp['file_name'][:-3] + 'png')
        annotations_tmp.setdefault('image_id', tmp['id'])
        annotations_tmp.setdefault('segments_info', segments_info)
        annotations.append(annotations_tmp)

    del in_d['annotations']
    in_d.setdefault('annotations', annotations)

    return in_d


def pan_cats_stff(in_d):
    for t in in_d['categories']:
        if t['name'] == 'roads':
            t.setdefault('isthing', 0)
        else:
            t.setdefault('isthing', 1)
        # t.setdefault('isthing', 1)

        if t['name'] == 'roads':
            t.setdefault('color', [0, 41, 223])
        elif t['name'] == 'ground_mark':
            t.setdefault('color', [135, 14, 236])
        elif t['name'] == 'zebra-crs':
            t.setdefault('color', [224, 0, 249])
        elif t['name'] == 'vehicle':
            t.setdefault('color', [187, 43, 91])
        elif t['name'] == 'non-motor':
            t.setdefault('color', [160, 107, 0])
        elif t['name'] == 'person':
            t.setdefault('color', [178, 191, 0])
        elif t['name'] == 'sign':
            t.setdefault('color', [110, 180, 66])

    out_d = []
    for k in in_d['categories']:
        out_d.append(k)

    return out_d


def convert_jsons(anns_dir, mode):
    coco_json = anns_dir + '/coco_v1910_' + mode + '.json'
    with open(coco_json, 'r') as f:
        d = json.load(f)

    with open(anns_dir + '/instances_' + mode + '2017.json', 'w') as f:
        json.dump(d, f)

    with open(anns_dir + '/panoptic_' + mode + '2017.json', 'w') as f:
        tmp1 = d.copy()
        tmp = pan(tmp1)
        json.dump(tmp, f)

    with open(anns_dir + '/panoptic_' + mode + '2017_stff.json', 'w') as f:
        tmp1 = d.copy()
        tmp = pan_stff(tmp1)
        json.dump(tmp, f)

    with open(anns_dir + '/panoptic_coco_categories_stff.json', 'w') as f:
        tmp1 = d.copy()
        tmp = pan_cats_stff(tmp1)
        json.dump(tmp, f)


def convert_imgs_ch3(root, mode, classes):
    coco_json = root + '/coco_v1910_' + mode + '.json'
    imgs_dir = root + '/image'

    save_dir = root + '/panoptic_' + mode + '2017'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    coco = COCO(coco_json)
    with open(coco_json, 'r') as f:
        d = json.load(f)
    imgIds = list(range(1, len(d['images']) + 1))
    for id in imgIds:
        img_info = coco.loadImgs(id)[0]
        image = cv2.imread(os.path.join(imgs_dir, img_info['file_name']))
        h, w, _ = image.shape
        mask = np.zeros((h, w, 3), dtype='uint8')
        anns_this_img = []
        for j, cls in enumerate(classes):
            catIds = coco.getCatIds(catNms=cls)
            annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=0)
            anns = coco.loadAnns(annIds)
            for ann in anns:
                anns_this_img.append(ann)
        for ann_idx in range(len(anns_this_img)):
            mask_tmp = coco.annToMask(anns_this_img[ann_idx])
            a = np.where(mask_tmp > 0, np.random.randint(50, 255, size=1), 0).astype('uint8')
            b = np.where(mask_tmp > 0, np.random.randint(50, 255, size=1), 0).astype('uint8')
            c = np.where(mask_tmp > 0, np.random.randint(50, 255, size=1), 0).astype('uint8')
            mask_color = np.stack([a, b, c], axis=0).transpose((1, 2, 0))
            mask += mask_color
        # print(id)
        cv2.imwrite(os.path.join(save_dir, img_info['file_name'][:-3] + 'png'), mask)

def convert_imgs_ch1(root):
    modes = ['train', 'val']
    for mode in modes:
        instance_json = root + '/coco_v1910_' + mode + '.json'
        imgs_dir = root + '/image'

        save_dir = root + '/panoptic_' + mode + '2017_semantic_trainid_stff'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        coco = COCO(instance_json)

        with open(instance_json, 'r') as f:
            d = json.load(f)
        imgIds = list(range(1, len(d['images']) + 1))

        for id in imgIds:
            img_info = coco.loadImgs(id)[0]
            image = cv2.imread(os.path.join(imgs_dir, img_info['file_name']))
            h, w, _ = image.shape
            mask = np.zeros((h, w), dtype='uint8')

            cat_ids_this_img = set()
            for t in d['annotations']:
                cat_ids_this_img.add(t['category_id'])

            for cat_id in cat_ids_this_img:
                annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=cat_id, iscrowd=0)
                anns = coco.loadAnns(annIds)
                mask_this_cls = np.zeros((h, w))
                for k in range(len(anns)):
                    mask_tmp = coco.annToMask(anns[k])
                    mask_this_cls += mask_tmp
                mask_this_cls= np.where(mask_this_cls > 0, cat_id, 0).astype('uint8')
                mask_bool = ~((mask & mask_this_cls).astype('bool'))
                # mask_bool = ~mask_bool
                mask += mask_this_cls
                mask *= mask_bool
                # if 1 in mask:
                #     print(mask)
            mask[mask==0] = 255

            mask = Image.fromarray(mask.astype('uint8'))
            mask.save(os.path.join(save_dir, img_info['file_name'][:-3] + 'png'))
            # plt.imshow(mask)
            # plt.show()
            # p = set()
            # width = mask.size[0]
            # height = mask.size[1]
            # for h in range(0, height):
            #     for w in range(0, width):
            #         pixel = mask.getpixel((w, h))
            #         p.add(pixel)
            # print(p)



def divide_imgs(root):
    coco_json_train = root + '/coco_v1910_' + 'train' + '.json'
    coco_json_val = root + '/coco_v1910_' + 'val' + '.json'
    imgs_dir = root + '/image'

    save_dir1 = root + '/train2017'
    save_dir2 = root + '/val2017'
    if not os.path.exists(save_dir1):
        os.makedirs(save_dir1)
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)

    with open(coco_json_train, 'r') as f:
        d = json.load(f)
        for i in d['images']:
            filename = i['file_name']
            shutil.copy(os.path.join(imgs_dir, filename), save_dir1)

    with open(coco_json_val, 'r') as f:
        d = json.load(f)
        for i in d['images']:
            filename = i['file_name']
            shutil.copy(os.path.join(imgs_dir, filename), save_dir2)


if __name__ == '__main__':
    root = 'D:/DATA/adas/adas_parsing_data_1907'
    cat_json = 'D:/DATA/adas/adas_parsing_data_1907/panoptic_coco_categories_stff.json'
    convert_imgs_ch1(root)
    # classes = ['roads', 'ground_mark', 'zebra-crs', 'vehicle', 'non-motor', 'person', 'sign']
    # convert_imgs_ch3(root, 'val', classes)
    # convert_imgs_ch3(root, 'train', classes)
    #################################################
    # convert_jsons(root, 'train')
    # convert_jsons(root, 'val')
    #################################################
    # divide_imgs(root)
