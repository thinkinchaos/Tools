#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/8 14:32
# @Author  : Sillet
# @Function: 添加公司标志在视频上

import copy
import cv2
import argparse
import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Parsing Label View")
    parser.add_argument(
        "--logo_path",
        default="./logo.jpg",
        type=str,
    )
    parser.add_argument(
        "--video_path",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        default="./ret.mp4",
        help="path to result file-tools",
        type=str,
    )

    return parser.parse_args()


class OverLogo(object):
    def __init__(self, logo_path, image_w):
        self.logo_data = cv2.imread(logo_path)
        self.h2w_radio = 1.0 * self.logo_data.shape[0] / self.logo_data.shape[1]
        self.pose = 20
        self.size_radio = 0.15
        self.mask_thr = 100

        self.logo_w = int(image_w * self.size_radio)
        self.logo_h = int(self.logo_w * self.h2w_radio)

        self.logo_data = cv2.resize(self.logo_data, (self.logo_w, self.logo_h))
        self.logo_mask = copy.deepcopy(self.logo_data)
        for r in range(self.logo_h):
            for c in range(self.logo_w):
                data = self.logo_data[r, c]
                if np.sum(data) < self.mask_thr:
                    self.logo_mask[r, c] = np.array([1 for _ in data], dtype=data.dtype)
                    self.logo_data[r, c] = np.array([0 for _ in data], dtype=data.dtype)
                else:
                    self.logo_mask[r, c] = np.array([0 for _ in data], dtype=data.dtype)

    def over_logo(self, img):
        img[self.pose:self.pose + self.logo_h, self.pose:self.pose + self.logo_w] = \
            img[self.pose:self.pose + self.logo_h, self.pose:self.pose + self.logo_w] * self.logo_mask
        img[self.pose:self.pose + self.logo_h, self.pose:self.pose + self.logo_w] += self.logo_data
        # alpha = 0.5
        # beta = 1 - alpha
        # gamma = 0
        #
        # composite = cv2.addWeighted(image, alpha, mask_image, beta, gamma)

        return img


if __name__ == '__main__':
    args = get_args()

    cam = cv2.VideoCapture(args.video_path)
    cam_frame_count = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    cam_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    cam_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cam_fps = cam.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    write_cap = cv2.VideoWriter(args.save_path, fourcc, cam_fps, (int(cam_w), int(cam_h)))
    logo = OverLogo(args.logo_path, cam_w)

    with tqdm(total=cam_frame_count) as pbar:
        while True:
            ret_val, img = cam.read()
            if not ret_val:
                break

            logo.over_logo(img)
            pbar.update(1)
            write_cap.write(img)

            # cv2.imshow('ret', img)
            # cv2.waitKey(10)

    write_cap.release()
