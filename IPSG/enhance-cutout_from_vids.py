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


def gen_data_from_videos(data_root='E:/datasets/infread/videos', hw_range=(69,480,0,640)):
    H_dir = data_root + '/H'
    L_dir = data_root + '/L'
    dirs = [H_dir, L_dir]
    frame_nums = []
    for dir in dirs:
        assert 'videos' in str(dir)
        for vid_path in Path(dir).glob('*.avi'):
            cap = cv2.VideoCapture(str(vid_path))
            total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            frame_nums.append(total_frame_num)
    frame_num = min(frame_nums)
    if frame_num > 999:
        frame_num = 999

    for dir in dirs:
        for vid_path in Path(dir).glob('*.avi'):
            save_dir1 = str(dir).replace('videos', 'images') + '/' + vid_path.name[:-4]
            if not os.path.exists(save_dir1):
                os.makedirs(save_dir1)
            cap = cv2.VideoCapture(str(vid_path))
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > frame_num
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, first = cap.read()
            first = first[hw_range[0]:hw_range[1], hw_range[2]:hw_range[3]]
            mean = np.zeros(first.shape, dtype='float64')
            for id in tqdm(range(frame_num)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, id)
                _, frame = cap.read()
                frame = frame[hw_range[0]:hw_range[1], hw_range[2]:hw_range[3]]
                cv2.imwrite(save_dir1 + '/noised_' + vid_path.name[:-4] +'_'+'{:03d}'.format(id)+ '_.png', frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                mean = mean + frame
            for c in range(3):
                mean[:, :, c] /= (frame_num)
            mean = mean.astype('uint8')
            cv2.imwrite(save_dir1 + '/clean_' + vid_path.name[:-4] + '.png', mean, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def val_psnr_from_dataset(dataset_dir):
    info_dict = {}
    sub_dirs = [i for i in Path(dataset_dir).iterdir() if i.is_dir()]
    for sub_dir in tqdm(sub_dirs):
        PSNRs = []
        for img_path in Path(str(sub_dir)).glob('*.png'):
            if 'clean' in img_path.name:
                continue
            else:
                clean_name = img_path.name.replace('noised', 'clean')[:11] + '.png'
                clean = cv2.imread(str(img_path.parent) + '/' + clean_name)
                noised = cv2.imread(str(img_path))
            PSNRs.append(psnr(clean, noised))
        PSNR = np.mean(PSNRs)
        print(str(sub_dir), PSNR)
        info_dict.setdefault(sub_dir.name, PSNR)

    with open(dataset_dir + '/info.json', 'w') as f:
        json.dump(info_dict, f, indent=2)


if __name__ == '__main__':
    # gen_data_from_videos()
    val_psnr_from_dataset(r'E:\datasets\infread\images\L')
