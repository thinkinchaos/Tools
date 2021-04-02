#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/11 16:04
# @Author  : Sillet
# @Function: 处理adas专有数据

import os
import json
import logging

from collections import OrderedDict
from tqdm import tqdm

folder_dict = {
    'Night': 'night',
    'Daytime': 'daytime',
    'Indoor': 'indoor'
}

path = '/mnt/ssd_disk/streamax/train_dataset/adas_parsing_data_1911/raw'
save_path = '/mnt/ssd_disk/streamax/train_dataset/adas_parsing_data_1911/json'

root_path = '/home/gaoxiang/workspace/ssd_disk_1/train_dataset/adas_parsing_data_1911'


def add_property(json_file, dst_file, add_propery):
    with open(json_file, 'r') as f:
        json_data = json.load(f, object_pairs_hook=OrderedDict)
        json_data.update(add_propery)

    with open(dst_file, 'w') as f:
        json.dump(json_data, f, indent=4)


def add_folder(root_path, dst_folder):
    add_pro = {
        'environment': ''
    }
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)
        if not os.path.isdir(folder_path) or folder_name not in folder_dict:
            continue

        file_list = [f for f in os.listdir(folder_path) if '.json' in f]
        add_pro['environment'] = folder_dict[folder_name]
        print('processing %s:' % folder_name)
        for file_name in tqdm(file_list):
            root_file = os.path.join(folder_path, file_name)
            dst_file = os.path.join(dst_folder, file_name)
            add_property(root_file, dst_file, add_pro)
        print('done')


def cp_image_with_json(image_root, image_dst, json_root, json_bak):
    # logging.basicConfig(filename='logger.log', level=logging.DEBUG)
    name_list = [os.path.splitext(f)[0] for f in os.listdir(json_root) if '.json' in f]

    for name in tqdm(name_list):
        org_image = os.path.join(image_root, name + '.jpg')
        if os.path.exists(org_image):
            # os.system('cp %s %s/' % (org_image, image_dst))
            continue
        else:
            # logging.DEBUG('%s is not exists' % name)
            # os.system('mv %s/%s.json %s/' % (json_root, name, json_bak))
            print('/n%s/%s.json' % (json_root, name))
            # os.system('rm %s/%s.json' % (json_root, name))


if __name__ == '__main__':
    # add_folder(path, save_path)

    cp_image_with_json(
        os.path.join(root_path, 'image_raw'),
        os.path.join(root_path, 'image'),
        os.path.join(root_path, 'sv191120'),
        os.path.join(root_path, 'json_bak'),
    )
