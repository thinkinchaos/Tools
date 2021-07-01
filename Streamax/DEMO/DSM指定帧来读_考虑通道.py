from pathlib import Path
import json
import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2

import dlib

# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
# conda config --set show_channel_urls yes
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# method=1
bias = 0
# {'16711680', '16776960', '16777215', '255'}
data_dir = 'D:/DATA/H264_New/'
labels = []
# labels.append({'idx': 0, 'dir': '抽烟', 'flag': 'SMOKING_DETECTED'})
# labels.append({'idx': 1, 'dir': '打电话/1', 'flag': 'PHONE_DETECTED'})
# labels.append({'idx': 2, 'dir': '打电话/2', 'flag': 'PHONE_DETECTED'})
# labels.append({'idx': 3, 'dir': '打电话/3', 'flag': 'PHONE_DETECTED'})
labels.append({'idx': 4, 'dir': '打哈欠', 'flag': 'DRIVER_FATIGUE_LIGHT'})
# labels.append({'idx': 5, 'dir': '疲劳', 'flag': 'FATIGUE'})

for label_idx in range(len(labels)):
    box_cls = set()
    images = []
    xml_paths = [i for i in Path(data_dir + labels[label_idx]['dir']).rglob('*.xml')]
    video_path = [i for i in Path(data_dir + labels[label_idx]['dir']).glob('*.avi')][0]

    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cnt = 0
    while True:
        rval, frame_ = cap.read()
        if rval:
            cnt += 1
            images.append(frame_)
        else:
            break
    total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(cnt, total_frame_num)

    save_dir = data_dir + '/demo'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    out = cv2.VideoWriter(save_dir + '/' + labels[label_idx]['flag'] + str(label_idx) + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frameWidth, frameHeight))

    all_xml_frames = []
    for xml_path in xml_paths:
        with open(str(xml_path), 'r')as f:
            info = f.read().replace('<?xml version="1.0" encoding="utf-8"?>', '')
        new_xml = save_dir + '/tmp.xml'
        with open(new_xml, 'w') as f2:
            f2.write("<root>{}</root>".format(info))

        parser = ET.parse(new_xml)
        xml_root = parser.getroot()
        xml_frames = parser.findall('frame')
        all_xml_frames.append((xml_frames))

    for xml_frames in all_xml_frames:
        for xml_frame in xml_frames:
            frameindex = int(xml_frame.find('frameindex').text)
            if frameindex < cnt - abs(bias):
                data = xml_frame.find('aidata').find('data')
                dataObj = xml_frame.find('aidata').find('dataObj')
                if data is not None:
                    if labels[label_idx]['flag'] in data.text:
                        images[frameindex] = cv2.putText(images[frameindex], labels[label_idx]['flag'], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if dataObj is not None:
                    o = dataObj.text
                    o = o.split(',')
                    for info in o:
                        if info[:4] == 'pt-3':
                            info = info.replace('pt-3:', '').split(':16711680')
                            for pt in info:
                                if len(pt) > 6:
                                    pt = pt.strip(':').split(':')
                                    images[frameindex + bias] = cv2.circle(images[frameindex + bias], center=(int(pt[0]), int(pt[1])), radius=3, color=(0, 125, 255), thickness=-1)
                #
                #         elif info[:4] == 'rect':
                #             info = info.replace('rect-', '').split(':')
                #             # print(info)
                #             assert len(info) == 5
                #             cls = int(info[4])
                #             pt1 = (int(info[0]), int(info[1]))
                #             pt2 = (int(info[2]), int(info[3]))
                #             if cls == 16711680:
                #                 images[frameindex + bias] = cv2.rectangle(images[frameindex + bias], pt1, pt2, color=(0, 0, 255), thickness=2)
                #             elif cls == 16776960:
                #                 images[frameindex + bias] = cv2.rectangle(images[frameindex + bias], pt1, pt2, color=(0, 0, 255), thickness=2)
                #             elif cls == 16777215:
                #                 images[frameindex + bias] = cv2.rectangle(images[frameindex + bias], pt1, pt2, color=(0, 0, 255), thickness=2)
                #             elif cls == 255:
                #                 images[frameindex + bias] = cv2.rectangle(images[frameindex + bias], pt1, pt2, color=(0, 0, 255), thickness=2)

    # pt_x_pre, pt_y_pre=0,0
    p1_x_pre, p1_y_pre = 0, 0
    p2_x_pre, p2_y_pre = 1, 1
    for i, img in enumerate(images):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        rects = detector(img_gray, 0)

        pts=[]
        if len(rects)>0:
            max_area=0
            max_area_idx=0
            for j in range(len(rects)):
                area=(rects[j].right()-rects[j].left())*(rects[j].bottom()-rects[j].top())
                # prjnt(rects[j].rjght()-rects[j].left(), rects[j].bottom()-rects[j].top())
                if area>max_area:
                    max_area_idx=j
                    max_area = area
            rect = rects[max_area_idx]
            # print(max_area)
            if max_area > 50000:
                landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])
                for idx, point in enumerate(landmarks):
                    pos = (point[0][0, 0], point[0][0, 1])
                    pts.append(pos)
                p1_x_cur,  p1_y_cur = rect.left(), rect.top()
                p2_x_cur, p2_y_cur = rect.right(), rect.bottom()
            else:
                # print(max_area)
                # print(p1_x_pre, p1_y_pre, p2_x_pre, p2_y_pre)
                # print(p1_x_cur, p1_y_cur, p2_x_cur, p2_y_cur)
                # break
                p1_x_cur, p1_y_cur = p1_x_pre, p1_y_pre
                p2_x_cur, p2_y_cur = p2_x_pre, p2_y_pre

        else:
            p1_x_cur,  p1_y_cur = p1_x_pre, p1_y_pre
            p2_x_cur, p2_y_cur = p2_x_pre, p2_y_pre
        # print(pts)

        p1_x,  p1_y = int((p1_x_pre + p1_x_cur)/2),int((p1_y_pre + p1_y_cur)/2)
        p2_x, p2_y = int((p2_x_pre + p2_x_cur)/2),int((p2_y_pre + p2_y_cur)/2)

        p1_x_pre, p1_y_pre = p1_x,  p1_y
        p2_x_pre, p2_y_pre = p2_x, p2_y

        if i>10:
            # print(p1_x_pre, p1_y_pre, p2_x_pre, p2_y_pre)
            # print(p1_x_cur, p1_y_cur, p2_x_cur, p2_y_cur)
            # print(p1_x, p1_y, p2_x, p2_y)

            # print(pts)
            # for p in pts:
            #     cv2.circle(img, p, 3, color=(0, 125, 252), thickness=-1)

            cv2.rectangle(img, (p1_x, p1_y), (p2_x, p2_y), (125, 255, 0), 2)

            out.write(img)

            # cv2.imshow('ss', img)
            # cv2.waitKey(1)




