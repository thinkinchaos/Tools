from pathlib import Path
import json
import xml.etree.ElementTree as ET
# from voc.crop_zebra import get_boxes
import os
import numpy as np
new_json_dir = 'D:/DATA/shenzhen_rsycnzebra_crossing_done_527/full_json_fast/save'
# if os.path.exists(new_json_dir):
#     os.makedirs(new_json_dir)
labelme_root = 'D:/DATA/shenzhen_rsycnzebra_crossing_done_527/done'
# labelme_image_names = [i.name for i in Path(labelme_root).rglob('*.jpg')]
# labelme_jsons = [i for i in Path(labelme_root).rglob('*.json')]
# print(len(labelme_images), len(labelme_jsons))

xml_dir = 'D:/DATA/shenzhen_rsycnzebra_crossing_done_527/Annotations_CarPersonZebra'
xmls = [i for i in Path(xml_dir).rglob('*.xml')]
# print(len(xmls))

# cord_info = []

for xml in xmls:
    parser = ET.parse(str(xml))
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
    filename = xml_root.find('filename').text[:-4]
    h = 1080
    w = 1920
    expand_pixel = 100
    for n, box in enumerate(bboxes):
        croped_img_name = '%s_%02d%s' % (filename, n, '.jpg')
        if box[3] - box[1] > h / 8 and box[2] - box[0] > w / 3:
            x1 = box[0] - expand_pixel if (box[0] - expand_pixel) > 0 else 0
            y1 = box[1] - expand_pixel if (box[1] - expand_pixel) > 0 else 0
            # x2 = box[2] + expand_pixel if (box[2] + expand_pixel) < w else w
            # y2 = box[3] + expand_pixel if (box[3] + expand_pixel) < h else h
            # new_box_tmp = [x1, y1, x2, y2]
            # cord_info.append((croped_img_name, x1, y1))

            croped_json_name = 'D:/DATA/shenzhen_rsycnzebra_crossing_done_527/full_json_fast/jsons/' + croped_img_name[:-4]+'.json'
            if os.path.exists(croped_json_name):
                with open(str(croped_json_name), 'r') as f:
                    dict = json.load(f)

                dict['imageHeight'] = 1080
                dict['imageWidth'] = 1920
                shapes = dict['shapes']
                for shape in shapes:
                    points = shape['points']

                    for point in points:
                        # print('before',point)
                        point[0] += x1
                        point[1] += y1
                        # print('after',point)
                        if point[0]>1920:
                            point[0]=1920
                        if point[1]>1080:
                            point[1]=1080

                with open(os.path.join(new_json_dir, croped_img_name[:-4]+'.json'), 'w') as f2:
                    json.dump(dict, f2)
