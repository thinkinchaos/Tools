from pathlib import Path
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import cv2
import shutil
import copy
import re
import random
from collections import Counter


def mkdir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def pretty_xml(element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


def mod_text(root_, name_, text_):
    item = root_.find(name_)
    if item is None:
        item = ET.SubElement(root_, name_)
        item.text = text_
    else:
        item.text = text_


def extract_voc(voc_dir, save_voc_dir, extract_classes):
    save_dir = save_voc_dir + '/Annotations'
    mkdir(save_dir)
    save_dir2 = save_voc_dir + '/JPEGImages'
    mkdir(save_dir2)
    save_dir3 = save_voc_dir + '/ImageSets/Main'
    mkdir(save_dir3)
    for xml in tqdm(Path(voc_dir + '/Annotations').glob('*.xml')):

        root = ET.parse(str(xml)).getroot()
        new_root = copy.deepcopy(root)

        for item in list(new_root.findall('object')):  # 删除所有obj
            new_root.remove(item)
        assert len(new_root.findall('object')) == 0  # 确保所有obj都已删除

        for obj in root.findall('object'):
            if obj.find('name').text in extract_classes:
                new_root.append(obj)

        if len(new_root.findall('object')) == 0:  # 如果依然没有obj，说明该图是负样本，添加一个负目标
            object = ET.SubElement(new_root, 'object')
            mod_text(object, 'name', 'neg')
            mod_text(object, 'pose', 'Unspecified')
            mod_text(object, 'truncated', '1')
            mod_text(object, 'difficult', '0')
            bndbox = ET.SubElement(object, 'bndbox')
            mod_text(bndbox, 'xmin', '0')
            mod_text(bndbox, 'ymin', '0')
            mod_text(bndbox, 'xmax', '20')
            mod_text(bndbox, 'ymax', '20')

        pretty_xml(new_root, '\t', '\n')
        tree = ET.ElementTree(new_root)

        tree.write(save_dir + '/' + xml.name, encoding='utf-8')
        shutil.copy(voc_dir + '/JPEGImages/' + xml.name[:-3] + 'jpg', save_dir2)


def get_annotations(xml_path):
    pattens = ['xmin', 'ymin', 'xmax', 'ymax']
    bbox = []
    root = ET.parse(str(xml_path)).getroot()
    objs = root.findall('object')
    if len(objs) == 0:
        print('Empty Annotation', xml_path)
    for obj in objs:
        tmp = []
        tmp.append(obj.find('name').text)
        bndbox = obj.find('bndbox')
        for patten in pattens:
            tmp.append(int(bndbox.find(patten).text))
        bbox.append(tmp)
    return bbox


def vis_a_xml(classes, image_path, xml_path, save_path, show_or_write=False):
    if 'neg' not in classes:
        classes.append('neg')
    colors = [(128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
              (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
              (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
              (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
    assert len(colors) >= len(classes)
    adopt_colors = colors[:len(classes)]
    class_color_dict = dict(zip(classes, adopt_colors))
    bbox = get_annotations(xml_path)
    image = cv2.imread(image_path)
    for info in bbox:
        cv2.rectangle(image, (info[1], info[2]), (info[3], info[4]), class_color_dict[info[0]], thickness=2)
        cv2.putText(image, info[0], (info[1] + (info[3] - info[1]) // 2, info[2] + (info[4] - info[2]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color_dict[info[0]], 2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if show_or_write:
        cv2.imshow('ss', image)
        cv2.waitKey()
    else:
        cv2.imwrite(os.path.join(save_path, Path(image_path).name), image)


def vis_voc(classes, voc_dir, txt, show):
    if txt in ['train', 'val', 'test', 'trainval', 'neg', 'trainval_withneg']:
        save_dir = voc_dir + '/Visualization/vis_' + txt
        mkdir(save_dir)
        with open(voc_dir + '/ImageSets/Main/' + txt + '.txt', 'r') as f:
            files = [i.strip() for i in f.readlines() if len(i.strip()) > 0]
        for file in tqdm(files):
            xml_path = voc_dir + '/Annotations/' + file + '.xml'
            img_path = voc_dir + '/JPEGImages/' + file + '.jpg'
            vis_a_xml(classes, img_path, xml_path, save_dir, show_or_write=show)
    else:
        save_dir = voc_dir + '/Visualization/vis_all'
        mkdir(save_dir)
        xml_list = os.listdir(voc_dir + '/Annotations')
        for i in tqdm(xml_list):
            xml_path = os.path.join(voc_dir + '/Annotations', i)
            img_path = os.path.join(voc_dir + '/JPEGImages', i.replace('.xml', '.jpg'))
            vis_a_xml(classes, img_path, xml_path, save_dir, show_or_write=show)


def find_names_in_xml(xml_file):
    parser = ET.parse(str(xml_file))
    xml_root = parser.getroot()
    objs = xml_root.findall('object')
    names = set()
    for obj in objs:
        names.add(obj.find('name').text)
    return names


def generate_image_sets(voc_dir):
    txt_dir = voc_dir + '/ImageSets/Main'

    pos_files, neg_files = [], []
    for xml in Path(voc_dir + '/Annotations').glob('*.xml'):
        names = find_names_in_xml(str(xml))
        if 'neg' in names and len(names) == 1:
            neg_files.append(xml.name[:-4])
        elif 'neg' in names and len(names) > 1:
            print('Problem Annotation file!', str(xml))
        else:
            pos_files.append(xml.name[:-4])

    random.shuffle(neg_files)
    random.shuffle(pos_files)

    trainval_percent = 0.9
    trainval_num = int(len(pos_files) * trainval_percent)
    trainval_files = pos_files[:trainval_num]
    test_files = pos_files[trainval_num:]

    if trainval_num < len(neg_files):
        trainval_files_withneg = trainval_files + neg_files[:trainval_num]
    else:
        trainval_files_withneg = trainval_files + neg_files
    random.shuffle(trainval_files_withneg)

    with open(txt_dir + '/neg.txt', 'w') as f:
        for file in neg_files:
            f.write(file)
            f.write('\n')

    with open(txt_dir + '/trainval_withneg.txt', 'w') as f:
        for file in trainval_files_withneg:
            f.write(file)
            f.write('\n')

    with open(txt_dir + '/trainval.txt', 'w') as f:
        for file in trainval_files:
            f.write(file)
            f.write('\n')

    with open(txt_dir + '/test.txt', 'w') as f:
        for file in test_files:
            f.write(file)
            f.write('\n')


def image_set_to_video(voc_dir, set_name):
    video_w, video_h = 640, 480
    video_path = voc_dir + '/' + set_name + '.avi'
    vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 1, (video_w, video_h))
    print(video_path)
    with open(voc_dir + '/ImageSets/Main/' + set_name + '.txt', 'r') as f:
        samples = [i.strip() for i in f.readlines() if len(i.strip()) > 0]
    for name in tqdm(samples):
        frame = cv2.imread(voc_dir + '/JPEGImages/' + name + '.jpg')
        frame = cv2.resize(frame, (video_w, video_h))
        # cv2.imshow('ss', frame)
        # cv2.waitKey()
        vw.write(frame)
    vw.release()


def show_image_sets_info(voc_dir):
    for txt in Path(voc_dir + '/ImageSets/Main').glob('*.txt'):
        classes = []
        class_set = set()
        with open(str(txt), 'r') as f:
            samples = [i.strip() for i in f.readlines() if len(i.strip()) > 0]
            for i in samples:
                tree = ET.parse(voc_dir + '/Annotations/' + i + '.xml')
                root = tree.getroot()
                objs = root.findall('object')
                for obj in objs:
                    name = obj.find('name').text
                    class_set.add(name)
                    classes.append(name)
        print(txt.name, '\tfile_num:{}\tclass_num:{}\t'.format(len(samples), len(class_set)), Counter(classes))


def find_empty_xml(voc_dir):
    for xml in Path(voc_dir).rglob('*.xml'):
        parser = ET.parse(str(xml))
        xml_root = parser.getroot()
        objs = xml_root.findall('object')
        # if len(objs) == 0 or objs is None:
        #     print(xml)
        names = set()
        for obj in objs:
            names.add(obj.find('name').text)
        print(names)


def online_find_good_ann(voc_dir):
    # good_ann_dir = 'E:/datasets/nuclear'
    # mkdir(problem_ann_dir)
    # for img in Path(voc_dir + '/Visualization/vis_all').glob('*.jpg'):
    #     image = cv2.imread(str(img))
    #     cv2.imshow('vis', image)
    #     k = cv2.waitKey()
    #     if k == ord('f'):
    #         os.remove(str(img))
    #         shutil.move(voc_dir + '/Annotations/' + img.name[:-3] + 'xml', problem_ann_dir)
    #         shutil.move(voc_dir + '/JPEGImages/' + img.name, problem_ann_dir)
    #         print(img)
    #         # cv2.waitKey()
    pass


if __name__ == '__main__':
    voc_dirs = [
        # 'E:/datasets/voc/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007',
        # 'E:/datasets/voc/VOCtest_06-Nov-2007/VOCdevkit/VOC2007'
        'E:/datasets/nuclear_cold'
    ]
    save_voc_dir = 'E:/datasets/nuclear_cold_tiny'
    extract_classes = [
        'robot',
        'battery',
        # 'wood',
        'brick',
        'barrel',
        'box',
        # 'sack',
        # 'motor'
    ]
    # extract_classes = [
    #     'aeroplane',
    #     'bicycle',
    #     'bird',
    #     'boat',
    #     'bottle',
    #     'bus',
    #     'car',
    #     'cat',
    #     'chair',
    #     'cow',
    #     'diningtable',
    #     'dog',
    #     'horse',
    #     'motorbike',
    #     'person',
    #     'pottedplant',
    #     'sheep',
    #     'sofa',
    #     'train',
    #     'tvmonitor',
    # ]

    # extract_classes = [
    #     'cylinder',
    #     'robot',
    #     'wagon',
    #     'hook',
    #     'sack'
    # ]

    # extract_classes = [
    #     'robot',
    #     'battery',
    #     'wood',
    #     'brick',
    #     'barrel',
    #     'box',
    #     'sack',
    #     'motor'
    # ]

    # extract_classes = [
    #     'cylinder',
    #     'robot',
    #     'wagon',
    #     'hook',
    #     'sack'
    # ]

    # extract_classes = [
    #     'aeroplane',
    #     'bicycle',
    #     # 'bird',
    #     'boat',
    #     # 'bottle',
    #     'bus',
    #     'car',
    #     # 'cat'
    # ]

    # 提出VOC，删除多余的类别，找出纯负样本
    for voc_dir in voc_dirs:
        extract_voc(voc_dir, save_voc_dir, extract_classes)

    # 生成ImageSets
    generate_image_sets(save_voc_dir)

    # 可视化VOC，验证处理结果
    vis_voc(extract_classes, save_voc_dir, txt='all', show=False)

    show_image_sets_info(save_voc_dir)

    image_set_to_video(save_voc_dir, 'test')

    find_empty_xml(save_voc_dir)

    online_find_good_ann(save_voc_dir)
