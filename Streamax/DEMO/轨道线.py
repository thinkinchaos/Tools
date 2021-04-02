from pathlib import Path
import cv2
import numpy as np


def show(img):
    cv2.imshow('ss', img)
    cv2.waitKey()


video_w, video_h = 1920, 1080
fourcc = cv2.VideoWriter_fourcc(*'XVID')
vw = cv2.VideoWriter('D:/DATA/guijiao/done/DEMO.avi', fourcc, 25, (video_w, video_h))
for bin_path in Path('D:/DATA/guijiao/done/guidao/binary').glob('*.png'):
    binary = cv2.imread(str(bin_path))
    src = cv2.imread('D:/DATA/guijiao/done/guidao/detect/' + bin_path.name[:-3] + 'jpg')
    binarygray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(binarygray, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    src_bg = cv2.bitwise_and(src, src, mask=mask_inv)
    binary_fg = cv2.bitwise_and(binary, binary, mask=mask)
    crop_num = 8
    height = 1080 // crop_num
    crop = binary_fg[0:height, :]
    for t in range(1, crop_num - 1, 1):
        tmp = binary_fg[height * t:height * (t + 1), :]
        k = np.ones((2 * t + 1, 2 * t + 1), np.uint8)
        tmp = cv2.dilate(tmp, k, iterations=1)
        crop = cv2.vconcat([crop, tmp])
    crop_last = binary_fg[height * (crop_num - 1):, :]
    k2 = np.ones((2 * (crop_num - 1) + 1, 2 * (crop_num - 1) + 1), np.uint8)
    crop_last = cv2.dilate(crop_last, k2, iterations=1)
    binary_fg = cv2.vconcat([crop, crop_last])
    binary_fg[:, :, 0] = np.where(binary_fg[:, :, 0] > 0, 53, 0)
    binary_fg[:, :, 1] = np.where(binary_fg[:, :, 1] > 0, 158, 0)
    binary_fg[:, :, 2] = np.where(binary_fg[:, :, 1] > 0, 255, 0)
    tmp = cv2.cvtColor(binary_fg, cv2.COLOR_BGR2GRAY)
    ret, mask2 = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask2)
    src_bg = cv2.bitwise_and(src, src, mask=mask_inv)
    binary_fg = cv2.addWeighted(src, 0.4, binary_fg, 0.6, 0)
    dst = cv2.add(src_bg, binary_fg)
    print(bin_path.name)
    vw.write(dst)
vw.release()
