import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from pathlib import Path


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(root, image_id, classes):
    in_file = open(root + '/Annotations/%s.xml' % image_id, errors='ignore')
    out_file = open(root + '/labels/%s.txt' % image_id, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    # size = root.find('size')
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)
    w = 720
    h = 576

    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        difficult = 0
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


root = 'E:/datasets/nuclear_obj_real/'
save_dir = root + '/labels'
if not os.path.exists(save_dir):  # 改成自己建立的myData
    os.makedirs(save_dir)

classes = ["robot", "battery", "wood", 'brick', 'barrel', 'box', 'sack', 'motor']  # 改成自己的类别

for set_name in Path(root + '/ImageSets/Main').glob('*.txt'):
    name = set_name.name[:-4]

    image_ids = open(root + '/ImageSets/Main/%s.txt' % name).read().strip().split()

    list_file = open(root + '/labels/%s.txt' % name, 'w')

    for image_id in image_ids:
        list_file.write(root + '/JPEGImages/%s.jpg\n' % image_id)
        print(image_id)
        convert_annotation(root, image_id, classes)
    list_file.close()
