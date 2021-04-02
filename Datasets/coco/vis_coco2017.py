from pathlib import Path
import json
from PIL import Image
from matplotlib import pyplot as plt

root = 'D:/DATA/panoptic_annotations_trainval2017_tiny'
imgs_dir = root + '/images'
anns_dir = root + '/annotations'

instances_val2017 = anns_dir + '/instances_val2017.json'  # 特别之处：iscrowd 有0有1
panoptic_val2017 = anns_dir + '/panoptic_val2017.json'  # 特别之处，cat增加了isthing 1
panoptic_coco_categories_stff = anns_dir + '/panoptic_coco_categories_stff.json'  # 有supercat，多个name. isthing=0:stuff
panoptic_val2017_stff = anns_dir + '/panoptic_val2017_stff.json'

# im=Image.open(root+'/annotations/panoptic_val2017/000000000785.png')
# p=set()
# width = im.size[0]
# height = im.size[1]
# for h in range(0, height):
#   for w in range(0, width):
#     pixel = im.getpixel((w, h))
#     p.add(pixel)
# print(p)

with open('instances_train2017.json', 'r') as f:
    d = json.load(f)

imgs = [i for i in d['images'] if 'xml' not in i['file_name']]
d['images'] = imgs
ids = [i['id'] for i in imgs]

# anns = [i for i in d['annotations'] if 'xml' not in i['file_name']]
anns = [i for i in d['annotations'] if i['image_id'] in ids]

d['annotations'] = anns


# for i in d['annotations']:
#     # print(i)
#     if i['image_id'] == 3535:
#         print(i['id'])
with open('instances_train2017_.json', 'w') as ff:
    json.dump(d, ff)


# # # file = '1.json'
# with open(panoptic_val2017_stff, 'r') as f:
#     d1 = json.load(f)
# with open(panoptic_val2017, 'r') as f:
#     d2 = json.load(f)
# for i in d2['categories']:
#     print(i)
# break
# #     # for i,j in d.items():
# #     #     # print(i)
# #     #     pass
# #     # for i in d:
# #     #     print(i)
# #     for i,j in enumerate(d['images']):
# #         print(j)
# #     for i,j in enumerate(d['annotations']):
# #         print('annotations.',j)
#     ids=[1,35,159,187]
#     # ids=[53,83,25,39]
#     for i,j in enumerate(d):
#         print(j)
#         # if j['id'] in ids:
#         #     print(j)


def simplify_instance_json(dir):
    file = dir + '/instances_val217.json'
    with open(file, 'r') as f:
        d = json.load(f)

        categories = d['categories']
        annotations = d['annotations']
        images = d['images']
        if 'licenses' in images:
            del images['license']
        if 'coco_url' in images:
            del images['coco_url']
        if 'flickr_url' in images:
            del images['flickr_url']
        if 'date_captured' in images:
            del images['date_captured']

        new_d = {}
        new_d.setdefault('images', images)
        new_d.setdefault('categories', categories)
        new_d.setdefault('annotations', annotations)

        for k, v in new_d.items():
            print(k, v)

        with open(file, 'w') as ff:
            json.dump(new_d, ff)


def simplify_panoptic_json(dir):
    file = dir + '/panoptic_val2017.json'
    with open(file, 'r') as f:
        d = json.load(f)

        categories0 = d['categories'][0]
        annotations0 = d['annotations'][0]
        images0 = d['images'][0]

        if 'licenses' in images0:
            del images0['license']
        if 'coco_url' in images0:
            del images0['coco_url']
        if 'flickr_url' in images0:
            del images0['flickr_url']
        if 'date_captured' in images0:
            del images0['date_captured']

        new_d = {}
        new_d.setdefault('images', images0)
        new_d.setdefault('categories', categories0)
        new_d.setdefault('annotations', annotations0)

        for k in d['images']:
            print(k)

        # with open(file, 'w') as ff:
        #     json.dump(new_d, ff)


def simplify_panoptic_stff_json(dir):
    file = dir + '/panoptic_val2017_stff.json'
    with open(file, 'r') as f:
        d = json.load(f)

        categories0 = d['categories'][0]
        annotations0 = d['annotations'][0]
        images0 = d['images'][0]

        if 'licenses' in images0:
            del images0['license']
        if 'coco_url' in images0:
            del images0['coco_url']
        if 'flickr_url' in images0:
            del images0['flickr_url']
        if 'date_captured' in images0:
            del images0['date_captured']

        new_d = {}
        new_d.setdefault('images', images0)
        new_d.setdefault('categories', categories0)
        new_d.setdefault('annotations', annotations0)

        for k, v in new_d.items():
            print(k, v)

        with open(file, 'w') as ff:
            json.dump(new_d, ff)


def simplify_panoptic_categories_stff_json(dir):
    file = dir + '/panoptic_coco_categories_stff.json'
    with open(file, 'r') as f:
        d = json.load(f)

        with open(file, 'w') as ff:
            json.dump(d[0], ff)

# simplify_instance_json(anns_dir)
# simplify_panoptic_json(anns_dir)
# simplify_panoptic_stff_json(anns_dir)
# simplify_panoptic_categories_stff_json(anns_dir)
