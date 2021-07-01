from pathlib import Path
import json
import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2

data_dir = 'D:/DATA/H264_New/'
labels = []
labels.append({'idx': 0, 'dir': '抽烟', 'flag': 'SMOKING_DETECTED'})
labels.append({'idx': 1, 'dir': '打电话/1', 'flag': 'PHONE_DETECTED'})
labels.append({'idx': 2, 'dir': '打电话/2', 'flag': 'PHONE_DETECTED'})
labels.append({'idx': 3, 'dir': '打电话/3', 'flag': 'PHONE_DETECTED'})
labels.append({'idx': 4, 'dir': '打哈欠', 'flag': 'DRIVER_FATIGUE_LIGHT'})
labels.append({'idx': 5, 'dir': '疲劳', 'flag': 'FATIGUE'})

for label_idx in range(len(labels)):

    xml_path = [i for i in Path(data_dir + labels[label_idx]['dir']).glob('*.xml')][0]
    video_path = [i for i in Path(data_dir + labels[label_idx]['dir']).glob('*.avi')][0]

    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    save_dir = data_dir + '/demo'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    out = cv2.VideoWriter(save_dir + '/' + labels[label_idx]['flag'] + str(label_idx) + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frameWidth, frameHeight))

    with open(str(xml_path), 'r')as f:
        info = f.read().replace('<?xml version="1.0" encoding="utf-8"?>', '')

    new_xml = save_dir + '/tmp.xml'
    with open(new_xml, 'w') as f2:
        f2.write("<root>{}</root>".format(info))

    frame_info = []

    parser = ET.parse(new_xml)
    xml_root = parser.getroot()
    xml_frames = parser.findall('frame')

    for xml_frame in xml_frames:
        frameindex = int(xml_frame.find('frameindex').text)
        if frameindex < total_frame_num:
            data = (str(xml_frame.find('aidata').find('data').text).strip()).split(',')
            dataObj = (str(xml_frame.find('aidata').find('dataObj').text)).strip('pt-3:').split(',')
            face_align_pts = dataObj[0].split('16711680')
            face_boxes = dataObj[1:]

            boxes = []
            for face_box in face_boxes:
                face_box = face_box.replace(':16711680', '').replace('rect-', '').split(':')
                if len(face_box) >= 4:
                    pt1 = (int(face_box[0]), int(face_box[1]))
                    pt2 = (int(face_box[2]), int(face_box[3]))
                    boxes.append((pt1, pt2))

            pts = []
            for pt in face_align_pts:
                pt = pt.strip(':').split(':')
                if len(pt) == 2:
                    pts.append((int(pt[0]), int(pt[1])))

            ok = bool()
            if labels[label_idx]['flag'] in data:
                ok = False
            else:
                ok = True

            frame_info.append({'label': label_idx, 'file': str(xml_path), 'idx': frameindex, 'pts': pts, 'boxes': boxes, 'ok': ok, 'data': dataObj})

    for i in range(total_frame_num):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = cap.read()

        alert_cnt = 0
        # ok_cnt = 0
        alert_flag = False

        for info in frame_info:
            if info['idx'] == i:
                max_area = 0
                min_area = 1000 * 1000
                for box in info['boxes']:
                    area = (box[1][0] - box[0][0]) * (box[1][1] - box[0][1])
                    if area > max_area:
                        max_area = area
                    if area < min_area:
                        min_area = area
                for box in info['boxes']:
                    area = (box[1][0] - box[0][0]) * (box[1][1] - box[0][1])
                    if area == max_area:
                        frame = cv2.rectangle(frame, box[0], box[1], color=(0, 0, 255), thickness=2)
                    elif area == min_area:
                        frame = cv2.rectangle(frame, box[0], box[1], color=(177, 156, 242), thickness=2)
                    else:
                        frame = cv2.rectangle(frame, box[0], box[1], color=(177, 156, 242), thickness=2)

                for pt in info['pts']:
                    frame = cv2.circle(frame, center=pt, radius=2, color=(0, 0, 255), thickness=2)

                # if info['ok'] and alert_flag:
                #     ok_cnt += 1

                if not info['ok']:
                    alert_flag = True

        if alert_flag:
            alert_cnt += 1
        if alert_cnt > 40:
            alert_flag = False
            alert_cnt = 0

        if alert_flag:
            frame = cv2.putText(frame, labels[label_idx]['flag'], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        out.write(frame)
#########################################################################################################################################################
#                 alert_flag = True
#                 # start_flag = False
#                 around_bin = 7
#                 around_flags = []
#                 for j in range(i, i + around_bin):
#                     if j < len(frame_imgs):
#                         around_flags.append(frame_imgs[j][1])
#                 for k in range(i - 1, i - around_bin, -1):
#                     if k > 0:
#                         around_flags.append(frame_imgs[k][1])
#
#                 if frame_imgs[i][1] == 1:
#                     alert_flag = True
#                     start_flag = True
#                 elif frame_imgs[i][1] == 0 and (1 not in around_flags):
#                     alert_flag = False
#                 else:
#                     alert_flag = True
#
#                 if alert_flag:
#                     frame_imgs[i][0] = cv2.putText(frame_imgs[i][0], labels[label_idx]['flag'], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
#                 else:
#                     frame_imgs[i][0] = cv2.putText(frame_imgs[i][0], 'DRIVER_OK', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
#
#                 if write_flag:
#                     out.write(frame_imgs[i][0])
#                 else:
#                     cv2.imshow('ss', frame_imgs[i][0])
#                     cv2.waitKey(0)
#
#
#             cv2.imshow('ss', frame)
#             cv2.waitKey(0)
#
#             print(xml.name, frameindex, face_box)
#
#
#                 frame_imgs[frameindex][1] = 1
#
# for i in range(len(frame_imgs)):
#     alert_flag = True
#     # start_flag = False
#     around_bin = 7
#     around_flags = []
#     for j in range(i, i + around_bin):
#         if j < len(frame_imgs):
#             around_flags.append(frame_imgs[j][1])
#     for k in range(i - 1, i - around_bin, -1):
#         if k > 0:
#             around_flags.append(frame_imgs[k][1])
#
#     if frame_imgs[i][1] == 1:
#         alert_flag = True
#         start_flag = True
#     elif frame_imgs[i][1] == 0 and (1 not in around_flags):
#         alert_flag = False
#     else:
#         alert_flag = True
#
#     if alert_flag:
#         frame_imgs[i][0] = cv2.putText(frame_imgs[i][0], labels[label_idx]['flag'], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
#     else:
#         frame_imgs[i][0] = cv2.putText(frame_imgs[i][0], 'DRIVER_OK', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
#
#     if write_flag:
#         out.write(frame_imgs[i][0])
#     else:
#         cv2.imshow('ss', frame_imgs[i][0])
#         cv2.waitKey(0)
