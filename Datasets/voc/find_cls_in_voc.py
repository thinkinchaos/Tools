import glob
import xml.etree.ElementTree as ET
import numpy as np

ANNOTATIONS_PATH = 'E:/datasets/nuclear_obj_cold/Annotations'
# names=set()
from pathlib import Path
all=[]
train=[]
val=[]
for xml_file in Path(ANNOTATIONS_PATH).glob('*.xml'):
    tree = ET.parse(str(xml_file))
    xml_root = tree.getroot()
    names_this_xml=[]
    for obj in xml_root.findall('object'):
        # names_this_xml.append(obj.find('name').text)
        if obj.find('name').text == 'dog':
            all.append(xml_file.name)
    #         obj.find('name').text = 'brick'
    # tree.write(xml_file, encoding="utf-8", xml_declaration=True, method='xml')


    # print(names_this_xml)
    # if 'robat' in names_this_xml:

        # element = ET.Element('object')
        #
        # sub_element1 = ET.Element('name')
        # sub_element1.text = obj_info[1]
        # element.append(sub_element1)
        #
        # sub_element2 = ET.Element('bndbox')
        # xmin = ET.Element('xmin')
        # xmin.text = str(obj_info[2])
        # ymin = ET.Element('ymin')
        # ymin.text = str(obj_info[3])
        # xmax = ET.Element('xmax')
        # xmax.text = str(obj_info[4])
        # ymax = ET.Element('ymax')
        # ymax.text = str(obj_info[5])
        # sub_element2.append(xmin)
        # sub_element2.append(ymin)
        # sub_element2.append(xmax)
        # sub_element2.append(ymax)
        #
        # element.append(sub_element2)
        # root.append(element)




print(all)

# # from pathlib import Path
# import random
# # xmls=[i.name for i in Path('Annotations_CarPersonZebra').iterdir() if i.is_file()]
# random.shuffle(all)
# percent=0.9
# train_n = int(len(all)*percent)
# train=all[:train_n]
# val=all[train_n:]
#
# with open('//192.168.133.15/workspace/sunxin/DATA/2.7_Zebra_HardNeg/Annotations_CarPersonZebra/zebra_train.txt','w') as f:
#     for i in train:
#         f.write(i)
#         f.write('\n')
#
# with open('//192.168.133.15/workspace/sunxin/DATA/2.7_Zebra_HardNeg/Annotations_CarPersonZebra/zebra_val.txt','w') as f:
#     for i in val:
#         f.write(i)
#         f.write('\n')


