import cv2
from pathlib import Path
from tqdm import tqdm
import numpy as np
import itertools
import random
import os
import math
import json


def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def gen_data_from_videos(vid_dir, save_root, n_num, hw_range):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for i, vid_path in enumerate(Path(vid_dir).glob('*.avi')):
        save_dir = save_root + '/' + vid_path.name[:-4]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cap = cv2.VideoCapture(str(vid_path))
        total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print('获得不可用区域的掩码，通过均值获得基准图，并保存。')
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, first = cap.read()
        first = first[hw_range[0]:hw_range[1], hw_range[2]:hw_range[3]]
        mean = np.zeros(first.shape, dtype='float64')
        mask = np.zeros(first.shape, dtype='bool')
        first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        draw = np.zeros(first.shape, dtype='float64')
        for id in tqdm(range(1, total_frame_num - 1)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, id)
            _, frame = cap.read()
            frame = frame[hw_range[0]:hw_range[1], hw_range[2]:hw_range[3]]
            mean = mean + frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(frame, first)
            _, diff = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(draw, contours, -1, 255, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        draw = cv2.dilate(draw, kernel)
        draw = ~(draw.astype('bool'))
        for c in range(3):
            mask[:, :, c] = draw
            mean[:, :, c] /= (total_frame_num - 2)
        mean = mean.astype('uint8') * mask
        cv2.imwrite(save_root + '/' + vid_path.name[:-4] + '.png', mean, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        print('对视频帧和基准图求PSNR，并排序，以获得前N个噪声严重的样本。')
        id_psnr_pairs = []
        for id in range(1, total_frame_num - 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, id)
            _, frame2 = cap.read()
            frame2 = frame2[hw_range[0]:hw_range[1], hw_range[2]:hw_range[3]] * mask
            PSNR = psnr(frame2, mean)
            id_psnr_pairs.append([id, PSNR])
        id_psnr_pairs = np.array(id_psnr_pairs)
        id_psnr_pairs = id_psnr_pairs[np.argsort(id_psnr_pairs[:, 1]), :]
        sample_pairs = []
        for t, sample_pair in enumerate(id_psnr_pairs):
            sample_pairs.append(sample_pair)
            if t > n_num:
                break

        print('抠出样本中不可用部分，并保存样本')
        sample_id = [p[0] for p in sample_pairs]
        print([p[1] for p in sample_pairs])
        for j, id in enumerate(sample_id):
            cap.set(cv2.CAP_PROP_POS_FRAMES, id)
            _, sample = cap.read()
            sample = sample[hw_range[0]:hw_range[1], hw_range[2]:hw_range[3]] * mask
            j += 1
            cv2.imwrite(save_dir + '/' + '{:03d}'.format(j) + '.png', sample, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def val_psnr_from_dataset(dataset_dir):
    info_dict = {}
    sub_dirs = [i for i in Path(dataset_dir).iterdir() if i.is_dir()]
    for sub_dir in tqdm(sub_dirs):
        GT = cv2.imread(dataset_dir + '/' + sub_dir.name + '.png')
        PSNRs = []
        for path in Path(str(sub_dir)).glob('*.png'):
            img = cv2.imread(str(path))
            PSNRs.append(psnr(img, GT))
        PSNR = np.mean(PSNRs)
        print(str(sub_dir), PSNR)
        info_dict.setdefault(sub_dir.name, PSNR)

    with open(dataset_dir + '/info.json', 'w') as f:
        json.dump(info_dict, f, indent=2)


if __name__ == '__main__':
    noise_img_num = 35
    hw_range = (69, 452, 83, 597)
    vid_dir = r'E:\infread\infrared_videos'
    save_root = r'E:\infread\infrared_dataset'
    gen_data_from_videos(vid_dir, save_root, noise_img_num, hw_range)
    val_psnr_from_dataset(save_root)
