from pathlib import Path
import json
import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2

data_dir = 'D:/DATA/H264_New/疲劳'
xml_paths = [i for i in Path(data_dir).rglob('*.xml')]

info_set=set()

for xml in xml_paths:
    with open(str(xml),'r')as f:
        info = f.read().replace('<?xml version="1.0" encoding="utf-8"?>','')

    new_xml = 'D:/tmp.xml'
    with open(new_xml, 'w') as f2:
        f2.write("<root>{}</root>".format(info))

    parser = ET.parse(new_xml)
    xml_root = parser.getroot()
    frames = parser.findall('frame')

    for frame in frames:
        frameindex = int(frame.find('frameindex').text)

        pts = frame.find('aidata').find('pts').text
        data = (str(frame.find('aidata').find('data').text).strip()).split(',')

        for i in data[-3:]:
            # if i=='DRIVER_OK' or i=='SMKSW:ON' or i == 'CALLSW:ON' or i == 'SHELTERSW:ON' or i =='NODRVSW:ON' or
            # print(i)

            if 'yts'not in i and 'fcnt' not in i and 'ats' not in i and 'fts' not in i:
                info_set.add(i)

            # if 'DRIVER_FATIGUE_LIGHT' in i:
            #     print(str(xml))

            # if i == 'DRIVER_OK':
                # print('ok')
        # for i in data[-1:]:
            # print(i)
            # info_set.add(i)

print(info_set)


