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

    sub_dir = [i for i in Path(data_dir + labels[label_idx]['dir']).iterdir() if i.is_dir()][0]
    xml_paths = [i for i in Path(data_dir + labels[label_idx]['dir']).glob('*.xml')]
    for append_xml in Path(str(sub_dir)).glob('*.xml'):
        xml_paths.append(append_xml)
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

    frame_info = []
    already_precess_frame_idx = []
    for xml_path in xml_paths:
        with open(str(xml_path), 'r')as f:
            info = f.read().replace('<?xml version="1.0" encoding="utf-8"?>', '')
        print(str(xml_path))
        new_xml = save_dir + '/tmp.xml'
        with open(new_xml, 'w') as f2:
            f2.write("<root>{}</root>".format(info))

        parser = ET.parse(new_xml)
        xml_root = parser.getroot()
        xml_frames = parser.findall('frame')

        for xml_frame in xml_frames:
            frameindex = int(xml_frame.find('frameindex').text)
            # if frameindex in already_precess_frame_idx:
            # if
            #     continue
            # else:

            if frameindex < total_frame_num:

                data = (str(xml_frame.find('aidata').find('data').text).strip()).split(',')

                ok = bool()
                if labels[label_idx]['flag'] in data:
                    ok = False
                else:
                    ok = True


                if xml_frame.find('aidata').find('dataObj') is not None:
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

                    # if frameindex in already_precess_frame_idx:
                    #     del_idx = []
                    #     for i, info in enumerate(frame_info):
                    #         if info['idx'] == frameindex:
                    #             del_idx.append(i)
                    #     # print(del_idx, len(frame_info))
                    #     for d in del_idx:
                    #         del frame_info[d]

                    if 'other' not in str(xml_path):
                        frame_info.append({'label': label_idx, 'file': str(xml_path), 'idx': frameindex, 'pts': pts, 'boxes': boxes, 'ok': ok, 'data': dataObj})
                        already_precess_frame_idx.append(frameindex)
                    else:
                        if frameindex not in already_precess_frame_idx:
                            frame_info.append({'label': label_idx, 'file': str(xml_path), 'idx': frameindex, 'pts': pts, 'boxes': boxes, 'ok': ok, 'data': dataObj})
                            already_precess_frame_idx.append(frameindex)
                        else:
                            pass

                else:
                    if frameindex not in already_precess_frame_idx:
                        frame_info.append({'label': label_idx, 'file': str(xml_path), 'idx': frameindex, 'pts': None, 'boxes': None, 'ok': ok, 'data': None})
                        already_precess_frame_idx.append(frameindex)



    print(len(frame_info), total_frame_num)
    # print(frame_info)
    for info in frame_info:
        print(info)

    for i in range(total_frame_num):
        ret, frame = cap.read()
        # cnt+=1
        frame = cv2.putText(frame, 'video frame:'+str(i), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        alert_cnt = 0
        alert_flag = False

        for info in frame_info:
            if info['idx'] == i:

                frame = cv2.putText(frame, str(info['pts']), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                frame = cv2.putText(frame, str(info['boxes']), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                frame = cv2.putText(frame, str(info['data']), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                frame = cv2.putText(frame, str(info['file']), (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                frame = cv2.putText(frame, 'xml frame:'+str(info['idx']), (0, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                if info['boxes'] is not None:
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

                if info['pts'] is not None:
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
            frame = cv2.putText(frame, labels[label_idx]['flag'], (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)


        # cv2.imshow('ss', frame)
        # cv2.waitKey()
        out.write(frame)

    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #     # cv2.imshow('ss', frame)
    #     # cv2.waitKey(1)
    #







    # print(cnt, total_frame_num)




